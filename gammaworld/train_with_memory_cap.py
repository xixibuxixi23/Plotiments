"""Training entry point with a per-rank PyTorch CUDA allocator limit."""

import os

import torch


def _set_memory_limit() -> None:
    limit_mib = int(os.environ.get("PYTORCH_PROCESS_MEMORY_LIMIT_MIB", "73728"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    torch.cuda.set_device(local_rank)
    total_bytes = torch.cuda.get_device_properties(local_rank).total_memory
    limit_bytes = limit_mib * 1024 * 1024
    if limit_bytes >= total_bytes:
        raise ValueError(
            f"PYTORCH_PROCESS_MEMORY_LIMIT_MIB={limit_mib} must be below "
            f"physical GPU memory ({total_bytes / 1024**2:.0f} MiB)"
        )
    torch.cuda.set_per_process_memory_fraction(limit_bytes / total_bytes, device=local_rank)
    print(
        f"[rank {local_rank}] PyTorch CUDA allocator capped at "
        f"{limit_mib / 1024:.1f} GiB",
        flush=True,
    )


if __name__ == "__main__":
    _set_memory_limit()
    from cosmos_oss.scripts.train import main

    main()
