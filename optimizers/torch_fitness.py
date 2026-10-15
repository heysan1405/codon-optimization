import time
import torch


def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def batch_gc_content(sequences, device=None):
    if not sequences:
        return torch.empty(0, dtype=torch.float32)

    if len({len(s) for s in sequences}) != 1:
        raise ValueError("All sequences must have equal length.")

    device = torch.device(device) if device else get_device()
    encoded = torch.tensor(
        [[1 if base in "GC" else 0 for base in s.upper()]
         for s in sequences],
        dtype=torch.float32,
        device=device,
    )
    return encoded.mean(dim=1)


def benchmark(batch_size=50, sequence_length=717, repeats=200):
    sequences = [
        ("ACGT" * ((sequence_length + 3) // 4))[:sequence_length]
        for _ in range(batch_size)
    ]

    print("PyTorch:", torch.__version__)
    print("CUDA available:", torch.cuda.is_available())
    if torch.cuda.is_available():
        print("GPU:", torch.cuda.get_device_name(0))

    for name, device in [
        ("CPU", "cpu"),
        ("CUDA", "cuda" if torch.cuda.is_available() else "cpu"),
    ]:
        start = time.perf_counter()
        for _ in range(repeats):
            batch_gc_content(sequences, device)
        if device == "cuda":
            torch.cuda.synchronize()
        print(f"{name}: {time.perf_counter() - start:.4f}s")


if __name__ == "__main__":
    benchmark()
