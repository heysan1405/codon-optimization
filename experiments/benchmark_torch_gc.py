import time
import torch

from core.sequence import gc_content

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def make_sequences(count=2048, length=1653):
    pattern = "ATGCGCAT"
    return [
        (pattern * ((length + len(pattern) - 1) // len(pattern)))[:length]
        for _ in range(count)
    ]


def cpu_gc(sequences):
    return [gc_content(s) for s in sequences]


def torch_gc(sequences, device):
    encoded = torch.tensor(
        [[1 if base in "GC" else 0 for base in s] for s in sequences],
        dtype=torch.float32,
        device=device,
    )
    return encoded.mean(dim=1)


def main():
    sequences = make_sequences()

    print("PyTorch:", torch.__version__)
    print("CUDA available:", torch.cuda.is_available())
    if torch.cuda.is_available():
        print("GPU:", torch.cuda.get_device_name(0))

    start = time.perf_counter()
    expected = cpu_gc(sequences)
    cpu_time = time.perf_counter() - start

    results = {}
    for device in ["cpu"] + (["cuda"] if torch.cuda.is_available() else []):
        torch_gc(sequences[:10], device)
        if device == "cuda":
            torch.cuda.synchronize()

        start = time.perf_counter()
        for _ in range(3):
            values = torch_gc(sequences, device)
            if device == "cuda":
                torch.cuda.synchronize()
        elapsed = (time.perf_counter() - start) / 3

        results[device] = values.cpu().tolist()
        print(f"Torch {device.upper()}: {elapsed:.4f}s")

    print(f"Existing CPU function: {cpu_time:.4f}s")
    print(
        "Maximum numerical difference:",
        max(abs(a - b) for a, b in zip(expected, results["cpu"])),
    )


if __name__ == "__main__":
    main()
