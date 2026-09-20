# Biological Dataset v1

This folder contains the metadata and downloader for the first 10 real
biological proteins selected for the codon-optimization project.

## Source policy

The nucleotide accessions are from NCBI GenBank/RefSeq. The script extracts
the annotated CDS rather than using a manually copied protein sequence.

## Dataset

1. GFP — *Aequorea victoria* — M62653.1
2. Firefly luciferase — *Photinus pyralis* — AB644228.1
3. HBB — *Homo sapiens* — NM_000518.5
4. INS — *Homo sapiens* — NM_000207.3
5. LYZ — *Homo sapiens* — NM_000239.3
6. ACTB — *Homo sapiens* — NM_001101.5
7. TP53 — *Homo sapiens* — NM_000546.6
8. lacZ — *Escherichia coli* K-12 MG1655 — NC_000913.3
9. groL — *Escherichia coli* K-12 MG1655 — NC_000913.3
10. sodA — *Escherichia coli* K-12 MG1655 — NC_000913.3

Target host for the first optimization experiments: *E. coli* K-12.

## Download

From the project root:

```bash
pip install biopython
python data/genes/download_dataset.py
```

The script creates:

```text
data/genes/
├── dataset_metadata.csv
├── download_dataset.py
├── validation_report.csv
└── fasta/
    ├── gene01_gfp.fasta
    ├── gene01_gfp_protein.fasta
    └── ...
```

Do not manually modify the downloaded FASTA files. The accession number is the
reproducibility key for the biological source.
