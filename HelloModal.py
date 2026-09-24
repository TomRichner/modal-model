"""Hello World for Modal: factorize a matrix locally on CPU and remotely on an L4.

    uv run modal run HelloModal.py            # both local CPU and remote L4
    uv run modal run HelloModal.py --n 8192   # larger matrix

Credentials come from `modal setup` (stored in ~/.modal.toml, outside the repo).
"""

import time

import modal

app = modal.App("hello-modal")

# Remote container: Debian + CUDA-enabled PyTorch wheel.
image = modal.Image.debian_slim(python_version="3.12").pip_install("torch")


def factorize(n: int, device: str) -> dict:
    """QR and SVD of a random n-by-n matrix; report timings and reconstruction error."""
    import torch

    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but not available")
    name = torch.cuda.get_device_name(0) if device == "cuda" else "cpu"

    torch.manual_seed(0)
    A = torch.randn(n, n, device=device, dtype=torch.float32)

    def timed(fn):
        if device == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        out = fn()
        if device == "cuda":
            torch.cuda.synchronize()
        return out, time.perf_counter() - t0

    # Warm up (CUDA context, cuSOLVER handles) so timings reflect the factorization.
    torch.linalg.qr(A[:64, :64])

    (Q, R), t_qr = timed(lambda: torch.linalg.qr(A))
    (U, S, Vh), t_svd = timed(lambda: torch.linalg.svd(A, full_matrices=False))
    scale = torch.linalg.norm(A)
    return {
        "device": name,
        "torch": torch.__version__,
        "n": n,
        "qr_s": t_qr,
        "svd_s": t_svd,
        "qr_rel_err": (torch.linalg.norm(Q @ R - A) / scale).item(),
        "svd_rel_err": (torch.linalg.norm(U @ torch.diag(S) @ Vh - A) / scale).item(),
    }


@app.function(image=image, gpu="L4", timeout=600)
def factorize_gpu(n: int) -> dict:
    return factorize(n, "cuda")


def report(label: str, r: dict) -> None:
    print(f"[{label}] {r['device']} (torch {r['torch']}), n={r['n']}")
    print(f"    QR  {r['qr_s']:8.3f} s   rel err {r['qr_rel_err']:.2e}")
    print(f"    SVD {r['svd_s']:8.3f} s   rel err {r['svd_rel_err']:.2e}")


@app.local_entrypoint()
def main(n: int = 4096):
    report("local", factorize(n, "cpu"))
    report("modal", factorize_gpu.remote(n))
