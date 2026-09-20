"""
Download and validate the real biological CDS dataset.

Run from the project root:

    python data\\genes\\download_dataset.py

Requirements:

    pip install biopython
"""

from pathlib import Path
import csv
import sys

from Bio import Entrez, SeqIO
from Bio.Seq import Seq


# ============================================================
# NCBI configuration
# ============================================================

Entrez.email = "codon.optimization.student@example.com"


# ============================================================
# Paths
# ============================================================

ROOT = Path(__file__).resolve().parent

META = ROOT / "dataset_metadata.csv"

FASTA_DIR = ROOT / "fasta"

FASTA_DIR.mkdir(exist_ok=True)


# ============================================================
# E. coli genes
# ============================================================
#
# Coordinates are 1-based inclusive coordinates from
# NC_000913.3.
#
# strand = -1 means reverse complement.
# strand = +1 means forward strand.
# ============================================================

BACTERIAL_GENES = {
    "gene08_lacz": {
        "gene": "lacZ",
        "start": 363231,
        "stop": 366305,
        "strand": -1,
    },

    "gene09_grol": {
        "gene": "groL",
        "start": 4371025,
        "stop": 4372671,
        "strand": 1,
    },

    "gene10_soda": {
        "gene": "sodA",
        "start": 4100810,
        "stop": 4101430,
        "strand": 1,
    },
}


# ============================================================
# Download a GenBank record
# ============================================================

def fetch_genbank(accession):
    print(f"Downloading {accession} ...")

    with Entrez.efetch(
        db="nuccore",
        id=accession,
        rettype="gb",
        retmode="text",
    ) as handle:

        return SeqIO.read(handle, "genbank")


# ============================================================
# Download the complete genome FASTA
# ============================================================

def fetch_genome_fasta(accession):
    """
    Download the nucleotide FASTA separately.

    This avoids the 'Sequence content is undefined' problem
    that can occur when using the large GenBank record.
    """

    print(f"Downloading genome FASTA {accession} ...")

    with Entrez.efetch(
        db="nuccore",
        id=accession,
        rettype="fasta",
        retmode="text",
    ) as handle:

        return SeqIO.read(handle, "fasta")


# ============================================================
# Find CDS for normal mRNA records
# ============================================================

def find_mrna_cds(record):
    cds_features = [
        feature
        for feature in record.features
        if feature.type == "CDS"
    ]

    if not cds_features:
        raise RuntimeError(
            f"No CDS feature found in {record.id}"
        )

    return cds_features[0]


# ============================================================
# Extract E. coli CDS using genomic coordinates
# ============================================================

def extract_bacterial_cds(genome_record, record_id):

    if record_id not in BACTERIAL_GENES:
        raise RuntimeError(
            f"Unknown bacterial dataset ID: {record_id}"
        )

    info = BACTERIAL_GENES[record_id]

    # NCBI coordinates are 1-based inclusive.
    # Python slicing is 0-based and stop-exclusive.
    start = info["start"] - 1
    stop = info["stop"]

    sequence = genome_record.seq[start:stop]

    if info["strand"] == -1:
        sequence = sequence.reverse_complement()

    return str(sequence).upper()


# ============================================================
# Validate DNA
# ============================================================

def validate_dna(cds, record_id):

    invalid_bases = set(cds) - set("ACGT")

    if invalid_bases:
        raise ValueError(
            f"{record_id}: invalid DNA bases "
            f"{sorted(invalid_bases)}"
        )

    if len(cds) == 0:
        raise ValueError(
            f"{record_id}: CDS is empty"
        )

    if len(cds) % 3 != 0:
        raise ValueError(
            f"{record_id}: CDS length "
            f"{len(cds)} is not divisible by 3"
        )


# ============================================================
# Translate CDS
# ============================================================

def translate_cds(cds):

    translated = str(
        Seq(cds).translate(
            to_stop=False
        )
    )

    terminal_stop = translated.endswith("*")

    protein = translated.rstrip("*")

    return protein, terminal_stop


# ============================================================
# Write FASTA
# ============================================================

def write_fasta(path, header, sequence):

    with path.open(
        "w",
        encoding="utf-8"
    ) as handle:

        handle.write(
            f">{header}\n"
        )

        for i in range(
            0,
            len(sequence),
            80
        ):

            handle.write(
                sequence[i:i + 80] + "\n"
            )


# ============================================================
# Main
# ============================================================

def main():

    # --------------------------------------------------------
    # Read metadata
    # --------------------------------------------------------

    if not META.exists():
        raise FileNotFoundError(
            f"Metadata file not found:\n{META}"
        )

    with META.open(
        encoding="utf-8"
    ) as handle:

        rows = list(
            csv.DictReader(handle)
        )

    if not rows:
        raise RuntimeError(
            "dataset_metadata.csv is empty."
        )


    # --------------------------------------------------------
    # Download E. coli genome once
    # --------------------------------------------------------

    bacterial_ids = set(
        BACTERIAL_GENES.keys()
    )

    genome_record = None

    if any(
        row["id"] in bacterial_ids
        for row in rows
    ):

        genome_record = fetch_genome_fasta(
            "NC_000913.3"
        )


    # --------------------------------------------------------
    # Validation results
    # --------------------------------------------------------

    report = []


    # --------------------------------------------------------
    # Process all proteins
    # --------------------------------------------------------

    for row in rows:

        dataset_id = row["id"]

        gene_name = row["gene_name"]

        accession = row["accession"]

        organism = row["organism"]


        print(
            f"\nProcessing "
            f"{gene_name} "
            f"({accession}) ..."
        )


        # ----------------------------------------------------
        # E. coli genomic genes
        # ----------------------------------------------------

        if dataset_id in bacterial_ids:

            cds = extract_bacterial_cds(
                genome_record,
                dataset_id
            )


        # ----------------------------------------------------
        # mRNA records
        # ----------------------------------------------------

        else:

            record = fetch_genbank(
                accession
            )

            feature = find_mrna_cds(
                record
            )

            cds = str(
                feature.extract(
                    record.seq
                )
            ).upper()


        # ----------------------------------------------------
        # Validate DNA
        # ----------------------------------------------------

        validate_dna(
            cds,
            dataset_id
        )


        # ----------------------------------------------------
        # Translate
        # ----------------------------------------------------

        protein, terminal_stop = translate_cds(
            cds
        )


        # ----------------------------------------------------
        # FASTA header
        # ----------------------------------------------------

        header = (
            f"{dataset_id}|"
            f"{gene_name}|"
            f"{accession}|"
            f"{organism}"
        )


        # ----------------------------------------------------
        # DNA FASTA
        # ----------------------------------------------------

        dna_path = (
            FASTA_DIR /
            f"{dataset_id}.fasta"
        )

        write_fasta(
            dna_path,
            header,
            cds
        )


        # ----------------------------------------------------
        # Protein FASTA
        # ----------------------------------------------------

        protein_path = (
            FASTA_DIR /
            f"{dataset_id}_protein.fasta"
        )

        write_fasta(
            protein_path,
            header,
            protein
        )


        # ----------------------------------------------------
        # Validation report
        # ----------------------------------------------------

        report.append(
            {
                "id": dataset_id,

                "gene_name": gene_name,

                "accession": accession,

                "organism": organism,

                "cds_length_bp": len(cds),

                "protein_length_aa": len(protein),

                "terminal_stop_present":
                    terminal_stop,

                "valid_dna": True,

                "divisible_by_3":
                    len(cds) % 3 == 0,

                "status": "PASS",
            }
        )


        print(
            f"  PASS | "
            f"{len(cds)} bp | "
            f"{len(protein)} aa"
        )


    # --------------------------------------------------------
    # Save validation report
    # --------------------------------------------------------

    report_path = (
        ROOT /
        "validation_report.csv"
    )

    with report_path.open(
        "w",
        newline="",
        encoding="utf-8"
    ) as handle:

        writer = csv.DictWriter(
            handle,
            fieldnames=report[0].keys()
        )

        writer.writeheader()

        writer.writerows(report)


    # --------------------------------------------------------
    # Final summary
    # --------------------------------------------------------

    passed = sum(
        row["status"] == "PASS"
        for row in report
    )

    print()
    print("=" * 60)
    print("DATASET VALIDATION COMPLETE")
    print("=" * 60)

    print(
        f"Sequences processed : {len(report)}"
    )

    print(
        f"Sequences passed    : {passed}"
    )

    print(
        f"FASTA directory     : {FASTA_DIR}"
    )

    print(
        f"Validation report   : {report_path}"
    )

    print("=" * 60)


# ============================================================
# Entry point
# ============================================================

if __name__ == "__main__":

    try:

        main()

    except ImportError:

        print(
            "Biopython is required."
        )

        print(
            "Run:"
        )

        print(
            "    pip install biopython"
        )

        sys.exit(1)

    except Exception as error:

        print()
        print("ERROR:")
        print(error)

        sys.exit(1)