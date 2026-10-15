from pathlib import Path
import math
import sys
import time
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.cai import calculate_cai
from core.sequence import get_codons
from data.codon_tables import get_codon_table


def prepare_table(organism):
    table = get_codon_table(organism)
    relative = {}
    for aa, codons in table.items():
        maximum = max(codons.values())
        for codon, frequency in codons.items():
            relative[codon] = frequency / maximum
    return relative


def cai_torch_batch(sequences, relative, device):
    codon_index = {c: i for i, c in enumerate(relative)}
    values = torch.tensor(
        [relative[c] for c in relative],
        dtype=torch.float64,
        device=device,
    )
    rows = []
    for sequence in sequences:
        codons = get_codons(sequence)
        rows.append([codon_index[c] for c in codons])

    if len({len(row) for row in rows}) != 1:
        raise ValueError("Sequences must have equal lengths.")

    indices = torch.tensor(rows, dtype=torch.long, device=device)
    logs = torch.log(values[indices].clamp_min(1e-10))
    return torch.exp(logs.mean(dim=1))


def main():
    fasta = ROOT / "data/genes/fasta/gene08_lacz.fasta"
    sequence = "".join(
        line.strip() for line in fasta.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.startswith(">")
    ).upper()

    sequences = [sequence] * 2048
    relative = prepare_table("ecoli")

    print("CUDA available:", torch.cuda.is_available())
    if torch.cuda.is_available():
        print("GPU:", torch.cuda.get_device_name(0))

    start = time.perf_counter()
    expected = [calculate_cai(s, "ecoli") for s in sequences]
    cpu_time = time.perf_counter() - start

    for device in ["cpu"] + (["cuda"] if torch.cuda.is_available() else []):
        cai_torch_batch(sequences[:2], relative, device)
        if device == "cuda":
            torch.cuda.synchronize()

        start = time.perf_counter()
        for _ in range(3):
            actual = cai_torch_batch(sequences, relative, device)
            if device == "cuda":
                torch.cuda.synchronize()
        elapsed = (time.perf_counter() - start) / 3

        max_diff = max(
            abs(a - b)
            for a, b in zip(expected, actual.cpu().tolist())
        )
        print(f"Torch {device.upper()}: {elapsed:.4f}s")
        print(f"Maximum CAI difference: {max_diff:.3e}")

    print(f"Existing CAI function: {cpu_time:.4f}s")


if __name__ == "__main__":
    main()
