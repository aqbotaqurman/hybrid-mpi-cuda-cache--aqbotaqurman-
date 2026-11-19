import os
import sys
import json
import math
import time
import hashlib
import platform
import subprocess
import numpy as np
from mpi4py import MPI
from numba import cuda, njit, float64, uint64
import numba.types as types

TILE_DIM = 16
SHARED_DIM = TILE_DIM + 2  # 18x18 shared tile
GLOBAL_NX = 512
GLOBAL_NY = 512
LOCAL_NX = 256
LOCAL_NY = 256
PADDED_NX = 258
PADDED_NY = 258
STENCIL_STEPS = 300
SHM_ITERS = 40_000_000
LAMBDA_DAMP = 0.15

def get_gpu_uuid() -> str:
    out = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=uuid", "--format=csv,noheader"],
        encoding="utf-8",
    )
    return out.strip().splitlines()[0].strip()

def detect_cpu_line_size() -> int:
    path = "/sys/devices/system/cpu/cpu0/cache/index0/coherency_line_size"
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            return int(f.read().strip())
    return 64

# ==============================================================================
# TASK 1: Hybrid MPI Cartesian Grid + Node-Local MPI-3 Shared-Memory Window
# ==============================================================================
def init_hybrid_context(github_user: str) -> dict:
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    if size != 4:
        raise RuntimeError(f"Requires exactly 4 MPI ranks (2x2 grid), got {size}.")

    cuda.select_device(0)
    gpu_uuid = comm.bcast(get_gpu_uuid() if rank == 0 else None, root=0)
    cpu_sig = f"{platform.machine()}::{os.cpu_count()}"
    line_bytes = detect_cpu_line_size()

    digest = hashlib.sha256(f"{github_user.strip()}::{gpu_uuid}::{cpu_sig}".encode("utf-8")).hexdigest()
    instance_seed = int(digest[:16], 16)

    # Non-periodic 2x2 Cartesian communicator
    cart_comm = comm.Create_cart(dims=[2, 2], periods=[False, False], reorder=False)
    coords = cart_comm.Get_coords(rank)  # [row, col]
    up, down = cart_comm.Shift(direction=0, disp=1)
    left, right = cart_comm.Shift(direction=1, disp=1)

    # Node-local shared memory communicator & 64-byte aligned window (64 uint64 slots = 512 bytes)
    node_comm = comm.Split_type(MPI.COMM_TYPE_SHARED)
    node_rank = node_comm.Get_rank()
    
    # Allocate extra bytes to guarantee 64-byte alignment
    n_slots = 64
    itemsize = 8
    alloc_bytes = (n_slots * itemsize + 64) if node_rank == 0 else 0
    win = MPI.Win.Allocate_shared(alloc_bytes, itemsize, comm=node_comm)
    
    buf, _ = win.Shared_query(0)
    raw_arr = np.ndarray(buffer=buf, dtype=np.uint8, shape=(n_slots * itemsize + 64,))
    offset = (64 - (raw_arr.ctypes.data % 64)) % 64
    shm_view = np.ndarray(buffer=raw_arr[offset : offset + n_slots * itemsize], dtype=np.uint64, shape=(n_slots,))

    return {
        "comm": comm,
        "cart_comm": cart_comm,
        "node_comm": node_comm,
        "win": win,
        "shm_view": shm_view,
        "rank": rank,
        "coords": coords,
        "neighbors": {"up": up, "down": down, "left": left, "right": right},
        "gpu_uuid": gpu_uuid,
        "cpu_sig": cpu_sig,
        "line_bytes": line_bytes,
        "instance_seed": instance_seed,
    }

# ==============================================================================
# TASK 2: Rank-Local GPU Subdomain & Halo Pack/Unpack Kernels
# ==============================================================================
def initialize_fields(ctx: dict):
    row, col = ctx["coords"]
    dx = 1.0 / (GLOBAL_NX + 1)
    
    # Global coordinates for interior [1..256, 1..256]
    gy = (row * LOCAL_NY + np.arange(1, LOCAL_NY + 1)) * dx
    gx = (col * LOCAL_NX + np.arange(1, LOCAL_NX + 1)) * dx
    GX, GY = np.meshgrid(gx, gy)

    u_host = np.zeros((PADDED_NY, PADDED_NX), dtype=np.float64)
    k_host = np.ones((PADDED_NY, PADDED_NX), dtype=np.float64)
    s_host = np.zeros((PADDED_NY, PADDED_NX), dtype=np.float64)

    # Deterministic initial condition & heterogeneous diffusivity
    rng = np.random.default_rng(ctx["instance_seed"] + ctx["rank"])
    noise = 1e-4 * rng.standard_normal((LOCAL_NY, LOCAL_NX))
    
    u_host[1:-1, 1:-1] = np.sin(math.pi * GX) * np.sin(math.pi * GY) + noise
    k_host[1:-1, 1:-1] = 0.5 + 0.4 * np.cos(2.0 * math.pi * GX) * np.cos(2.0 * math.pi * GY)
    s_host[1:-1, 1:-1] = 0.1 * np.sin(2.0 * math.pi * GX) * np.sin(2.0 * math.pi * GY)

    # Pad k_host edges so boundary derivatives remain well-defined
    k_host[0, :] = k_host[1, :]
    k_host[-1, :] = k_host[-2, :]
    k_host[:, 0] = k_host[:, 1]
    k_host[:, -1] = k_host[:, -2]

    kappa_max_local = float(np.max(k_host[1:-1, 1:-1]))
    kappa_max = ctx["cart_comm"].allreduce(kappa_max_local, op=MPI.MAX)
    dt = 0.8 * (dx * dx) / (4.0 * kappa_max)

    return u_host, k_host, s_host, dx, dt, kappa_max

@cuda.jit
def pack_halos_kernel(u, send_up, send_down, send_left, send_right):
    idx = cuda.grid(1)
    if idx < LOCAL_NX:
        i = idx + 1
        send_up[idx] = u[1, i]
        send_down[idx] = u[LOCAL_NY, i]
        send_left[idx] = u[i, 1]
        send_right[idx] = u[i, LOCAL_NX]

@cuda.jit
def unpack_halos_kernel(u, recv_up, recv_down, recv_left, recv_right, has_up, has_down, has_left, has_right):
    idx = cuda.grid(1)
    if idx < LOCAL_NX:
        i = idx + 1
        if has_up:
            u[0, i] = recv_up[idx]
        if has_down:
            u[PADDED_NY - 1, i] = recv_down[idx]
        if has_left:
            u[i, 0] = recv_left[idx]
        if has_right:
            u[i, PADDED_NX - 1] = recv_right[idx]

# ==============================================================================
# TASK 3: 2D Shared-Memory Tiled Stencil & Dual-Stream Pipeline
# ==============================================================================
@cuda.jit
def tiled_stencil_kernel(u_in, u_out, k_arr, s_arr, dt, inv_dx2, y_start, y_end, x_start, x_end):
    tile = cuda.shared.array(shape=(SHARED_DIM, SHARED_DIM), dtype=float64)
    
    tx = cuda.threadIdx.x
    ty = cuda.threadIdx.y
    gx = cuda.blockIdx.x * TILE_DIM + tx + 1
    gy = cuda.blockIdx.y * TILE_DIM + ty + 1

    # Load center element into shared tile
    if gx <= LOCAL_NX and gy <= LOCAL_NY:
        tile[ty + 1, tx + 1] = u_in[gy, gx]
    
    # Load halo apron into shared tile
    if tx == 0 and gx <= LOCAL_NX and gy <= LOCAL_NY:
        tile[ty + 1, 0] = u_in[gy, gx - 1]
    if tx == TILE_DIM - 1 and gx <= LOCAL_NX and gy <= LOCAL_NY:
        tile[ty + 1, SHARED_DIM - 1] = u_in[gy, gx + 1]
    if ty == 0 and gx <= LOCAL_NX and gy <= LOCAL_NY:
        tile[0, tx + 1] = u_in[gy - 1, gx]
    if ty == TILE_DIM - 1 and gx <= LOCAL_NX and gy <= LOCAL_NY:
        tile[SHARED_DIM - 1, tx + 1] = u_in[gy + 1, gx]

    cuda.syncthreads()

    if x_start <= gx <= x_end and y_start <= gy <= y_end:
        c = tile[ty + 1, tx + 1]
        up = tile[ty, tx + 1]
        dn = tile[ty + 2, tx + 1]
        lt = tile[ty + 1, tx]
        rt = tile[ty + 1, tx + 2]

        kc = k_arr[gy, gx]
        lap = kc * (up + dn + lt + rt - 4.0 * c) * inv_dx2
        u_out[gy, gx] = c + dt * (lap + s_arr[gy, gx] - LAMBDA_DAMP * c)

def exchange_halos_and_step(ctx, d_u, d_u_next, d_k, d_s, bufs, dt, inv_dx2, overlapped: bool):
    cart = ctx["cart_comm"]
    nb = ctx["neighbors"]
    grid_1d = (LOCAL_NX + 127) // 128
    block_1d = 128
    grid_2d = (LOCAL_NX // TILE_DIM, LOCAL_NY // TILE_DIM)
    block_2d = (TILE_DIM, TILE_DIM)

    stream_int = bufs["stream_int"] if overlapped else cuda.default_stream()
    stream_bnd = bufs["stream_bnd"] if overlapped else cuda.default_stream()

    # 1. Pack halos on boundary stream
    pack_halos_kernel[grid_1d, block_1d, stream_bnd](
        d_u, bufs["d_send_up"], bufs["d_send_down"], bufs["d_send_left"], bufs["d_send_right"]
    )

    # 2. D2H transfers to pinned host buffers
    bufs["d_send_up"].copy_to_host(bufs["h_send_up"], stream=stream_bnd)
    bufs["d_send_down"].copy_to_host(bufs["h_send_down"], stream=stream_bnd)
    bufs["d_send_left"].copy_to_host(bufs["h_send_left"], stream=stream_bnd)
    bufs["d_send_right"].copy_to_host(bufs["h_send_right"], stream=stream_bnd)

    # If overlapped, launch deep interior [2..255, 2..255] concurrently with MPI/PCIe
    if overlapped:
        tiled_stencil_kernel[grid_2d, block_2d, stream_int](
            d_u, d_u_next, d_k, d_s, dt, inv_dx2, 2, LOCAL_NY - 1, 2, LOCAL_NX - 1
        )

    stream_bnd.synchronize()

    # 3. Non-blocking MPI halo exchange
    reqs = []
    if nb["up"] != MPI.PROC_NULL:
        reqs.append(cart.Irecv(bufs["h_recv_up"], source=nb["up"], tag=10))
        reqs.append(cart.Isend(bufs["h_send_up"], dest=nb["up"], tag=11))
    if nb["down"] != MPI.PROC_NULL:
        reqs.append(cart.Irecv(bufs["h_recv_down"], source=nb["down"], tag=11))
        reqs.append(cart.Isend(bufs["h_send_down"], dest=nb["down"], tag=10))
    if nb["left"] != MPI.PROC_NULL:
        reqs.append(cart.Irecv(bufs["h_recv_left"], source=nb["left"], tag=20))
        reqs.append(cart.Isend(bufs["h_send_left"], dest=nb["left"], tag=21))
    if nb["right"] != MPI.PROC_NULL:
        reqs.append(cart.Irecv(bufs["h_recv_right"], source=nb["right"], tag=21))
        reqs.append(cart.Isend(bufs["h_send_right"], dest=nb["right"], tag=20))

    MPI.Request.Waitall(reqs)

    # 4. H2D transfers & unpack halos
    bufs["d_recv_up"].copy_to_device(bufs["h_recv_up"], stream=stream_bnd)
    bufs["d_recv_down"].copy_to_device(bufs["h_recv_down"], stream=stream_bnd)
    bufs["d_recv_left"].copy_to_device(bufs["h_recv_left"], stream=stream_bnd)
    bufs["d_recv_right"].copy_to_device(bufs["h_recv_right"], stream=stream_bnd)

    unpack_halos_kernel[grid_1d, block_1d, stream_bnd](
        d_u,
        bufs["d_recv_up"], bufs["d_recv_down"], bufs["d_recv_left"], bufs["d_recv_right"],
        int(nb["up"] != MPI.PROC_NULL),
        int(nb["down"] != MPI.PROC_NULL),
        int(nb["left"] != MPI.PROC_NULL),
        int(nb["right"] != MPI.PROC_NULL),
    )

    if overlapped:
        # Update 4 outer border strips on stream_bnd
        tiled_stencil_kernel[grid_2d, block_2d, stream_bnd](d_u, d_u_next, d_k, d_s, dt, inv_dx2, 1, 1, 1, LOCAL_NX)
        tiled_stencil_kernel[grid_2d, block_2d, stream_bnd](d_u, d_u_next, d_k, d_s, dt, inv_dx2, LOCAL_NY, LOCAL_NY, 1, LOCAL_NX)
        tiled_stencil_kernel[grid_2d, block_2d, stream_bnd](d_u, d_u_next, d_k, d_s, dt, inv_dx2, 2, LOCAL_NY - 1, 1, 1)
        tiled_stencil_kernel[grid_2d, block_2d, stream_bnd](d_u, d_u_next, d_k, d_s, dt, inv_dx2, 2, LOCAL_NY - 1, LOCAL_NX, LOCAL_NX)
        stream_int.synchronize()
        stream_bnd.synchronize()
    else:
        # Full interior [1..256, 1..256] serialized
        tiled_stencil_kernel[grid_2d, block_2d, stream_bnd](
            d_u, d_u_next, d_k, d_s, dt, inv_dx2, 1, LOCAL_NY, 1, LOCAL_NX
        )
        stream_bnd.synchronize()

# ==============================================================================
# TASK 4: Multi-Rank Host Cache-Line False Sharing Sweep
# ==============================================================================
@njit(nogil=True)
def host_shm_accumulate(shm_arr, idx, iters):
    # Prevent register hoisting by forcing memory read-modify-write via volatile-like dependency
    val = uint64(1)
    for i in range(iters):
        cur = shm_arr[idx]
        cur = (cur ^ val) + uint64(1)
        shm_arr[idx] = cur
        val = (cur & uint64(1)) + uint64(1)

def benchmark_shm_strides(ctx: dict) -> dict:
    node_comm = ctx["node_comm"]
    rank = ctx["rank"]
    shm_view = ctx["shm_view"]
    results = {}

    # Warmup JIT
    shm_view[:] = 0
    host_shm_accumulate(shm_view, rank * 16, 1000)
    node_comm.Barrier()

    for stride in [1, 8, 16]:
        shm_view[:] = 0
        node_comm.Barrier()
        t0 = time.perf_counter()
        host_shm_accumulate(shm_view, rank * stride, SHM_ITERS)
        node_comm.Barrier()
        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        max_ms = node_comm.allreduce(elapsed_ms, op=MPI.MAX)
        results[f"stride_{stride}_ms"] = round(max_ms, 3)

    results["false_sharing_slowdown_ratio"] = round(
        results["stride_1_ms"] / max(results["stride_8_ms"], 1e-6), 3
    )
    return results

# ==============================================================================
# TASK 5: Cross-Tier Verification & Manifest Export
# ==============================================================================
def run_stencil_simulation(ctx, u_init, k_host, s_host, dx, dt, overlapped: bool):
    d_u = cuda.to_device(u_init)
    d_u_next = cuda.to_device(u_init)
    d_k = cuda.to_device(k_host)
    d_s = cuda.to_device(s_host)

    bufs = {
        "d_send_up": cuda.device_array(LOCAL_NX, dtype=np.float64),
        "d_send_down": cuda.device_array(LOCAL_NX, dtype=np.float64),
        "d_send_left": cuda.device_array(LOCAL_NX, dtype=np.float64),
        "d_send_right": cuda.device_array(LOCAL_NX, dtype=np.float64),
        "d_recv_up": cuda.device_array(LOCAL_NX, dtype=np.float64),
        "d_recv_down": cuda.device_array(LOCAL_NX, dtype=np.float64),
        "d_recv_left": cuda.device_array(LOCAL_NX, dtype=np.float64),
        "d_recv_right": cuda.device_array(LOCAL_NX, dtype=np.float64),
        "h_send_up": cuda.pinned_array(LOCAL_NX, dtype=np.float64),
        "h_send_down": cuda.pinned_array(LOCAL_NX, dtype=np.float64),
        "h_send_left": cuda.pinned_array(LOCAL_NX, dtype=np.float64),
        "h_send_right": cuda.pinned_array(LOCAL_NX, dtype=np.float64),
        "h_recv_up": cuda.pinned_array(LOCAL_NX, dtype=np.float64),
        "h_recv_down": cuda.pinned_array(LOCAL_NX, dtype=np.float64),
        "h_recv_left": cuda.pinned_array(LOCAL_NX, dtype=np.float64),
        "h_recv_right": cuda.pinned_array(LOCAL_NX, dtype=np.float64),
        "stream_int": cuda.stream(),
        "stream_bnd": cuda.stream(),
    }

    inv_dx2 = 1.0 / (dx * dx)
    energy_history = {}

    def record_energy(step):
        u_curr = d_u.copy_to_host()
        local_e = float(np.sum(u_curr[1:-1, 1:-1] ** 2) * (dx * dx))
        global_e = ctx["cart_comm"].allreduce(local_e, op=MPI.SUM)
        energy_history[f"step_{step}"] = round(global_e, 10)

    record_energy(0)
    ctx["cart_comm"].Barrier()
    t0 = time.perf_counter()

    for step in range(1, STENCIL_STEPS + 1):
        exchange_halos_and_step(ctx, d_u, d_u_next, d_k, d_s, bufs, dt, inv_dx2, overlapped)
        d_u, d_u_next = d_u_next, d_u
        if step in (100, 200, 300):
            record_energy(step)

    ctx["cart_comm"].Barrier()
    elapsed_ms = (time.perf_counter() - t0) * 1000.0
    max_ms = ctx["cart_comm"].allreduce(elapsed_ms, op=MPI.MAX)

    return d_u.copy_to_host(), round(max_ms, 3), energy_history

def main():
    if len(sys.argv) < 2:
        raise ValueError("Usage: python3 hybrid_mpi_cuda_cache.py <GITHUB_USERNAME>")
    github_user = sys.argv[1]

    ctx = init_hybrid_context(github_user)
    rank = ctx["rank"]
    u_init, k_host, s_host, dx, dt, kappa_max = initialize_fields(ctx)

    # Run serialized & overlapped modes
    u_ser, gpu_serialized_ms, energy_ser = run_stencil_simulation(ctx, u_init, k_host, s_host, dx, dt, overlapped=False)
    u_ovl, gpu_overlapped_ms, energy_ovl = run_stencil_simulation(ctx, u_init, k_host, s_host, dx, dt, overlapped=True)

    # Verify cross-rank parity (< 1e-12)
    local_diff = float(np.max(np.abs(u_ser[1:-1, 1:-1] - u_ovl[1:-1, 1:-1])))
    max_diff = ctx["cart_comm"].allreduce(local_diff, op=MPI.MAX)
    assert max_diff < 1e-12, f"Parity check failed: max_diff={max_diff}"

    # Host SHM false sharing benchmark
    shm_metrics = benchmark_shm_strides(ctx)

    if rank == 0:
        # PTX Assembly Audit
        ptx_dict = tiled_stencil_kernel.inspect_asm()
        ptx_code = next(iter(ptx_dict.values()))
        ptx_audit = {
            "st.shared.f64": ptx_code.count("st.shared.f64"),
            "ld.shared.f64": ptx_code.count("ld.shared.f64"),
            "bar.sync": ptx_code.count("bar.sync"),
        }

        # CPU LLVM IR Audit
        llvm_dict = host_shm_accumulate.inspect_llvm()
        llvm_ir = next(iter(llvm_dict.values()))
        llvm_audit = {
            "load_i64_count": llvm_ir.count("load i64"),
            "store_i64_count": llvm_ir.count("store i64"),
            "has_memory_ops": ("load i64" in llvm_ir and "store i64" in llvm_ir),
        }

        manifest = {
            "github_username": github_user,
            "gpu_uuid": ctx["gpu_uuid"],
            "cpu_sig": ctx["cpu_sig"],
            "cpu_cache_line_bytes": ctx["line_bytes"],
            "instance_seed": hex(ctx["instance_seed"]),
            "grid_params": {
                "global_nx": GLOBAL_NX,
                "global_ny": GLOBAL_NY,
                "dx": dx,
                "dt": dt,
                "kappa_max": kappa_max,
                "dt_max_cfl": (dx * dx) / (4.0 * kappa_max),
                "steps": STENCIL_STEPS,
            },
            "gpu_timings": {
                "gpu_serialized_ms": gpu_serialized_ms,
                "gpu_overlapped_ms": gpu_overlapped_ms,
                "speedup_ratio": round(gpu_serialized_ms / max(gpu_overlapped_ms, 1e-6), 3),
            },
            "parity_max_diff": max_diff,
            "global_energy_history": energy_ovl,
            "shm_coherency_benchmark": shm_metrics,
            "ptx_audit": ptx_audit,
            "llvm_ir_audit": llvm_audit,
        }

        with open("hybrid_lab_manifest.json", "w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2)

        print("=== HYBRID MPI-CUDA & CACHE-LINE COHERENCY TELEMETRY ===")
        print(json.dumps(manifest, indent=2))

    ctx["win"].Free()

if __name__ == "__main__":
    main()
