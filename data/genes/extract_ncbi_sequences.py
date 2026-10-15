from pathlib import Path
import zipfile
import shutil


PROJECT_ROOT = Path(__file__).resolve().parents[2]

INPUT_ROOT = PROJECT_ROOT / "data" / "ncbi_downloads"
OUTPUT_ROOT = PROJECT_ROOT / "data" / "genes" / "ncbi_extracted"

FNA_ROOT = OUTPUT_ROOT / "fna"
FAA_ROOT = OUTPUT_ROOT / "faa"


def clean_name(name):
    return name.replace(" ", "_").lower()


def extract_all():

    FNA_ROOT.mkdir(parents=True, exist_ok=True)
    FAA_ROOT.mkdir(parents=True, exist_ok=True)

    zip_files = sorted(
        INPUT_ROOT.rglob("*.zip")
    )

    print("=" * 75)
    print("NCBI SEQUENCE EXTRACTION")
    print("=" * 75)

    print(f"Input folder : {INPUT_ROOT}")
    print(f"ZIP files    : {len(zip_files)}")
    print(f"FNA output   : {FNA_ROOT}")
    print(f"FAA output   : {FAA_ROOT}")
    print("=" * 75)

    if not zip_files:
        print("\nERROR: No ZIP files found.")
        return

    extracted_fna = 0
    extracted_faa = 0
    failed = 0

    for index, zip_path in enumerate(zip_files, 1):

        # Expected:
        # ncbi_downloads/ecoli/adk/adk_datasets.zip

        relative = zip_path.relative_to(INPUT_ROOT)

        parts = relative.parts

        if len(parts) >= 3:

            organism = parts[0]
            gene = parts[1]

        else:

            organism = "unknown"
            gene = zip_path.stem

        organism = clean_name(organism)
        gene = clean_name(gene)

        print(
            f"\n[{index}/{len(zip_files)}] "
            f"{organism} / {gene}"
        )

        try:

            with zipfile.ZipFile(
                zip_path,
                "r"
            ) as z:

                names = z.namelist()

                # -----------------------------------------
                # Find gene.fna
                # -----------------------------------------

                fna_files = [
                    name
                    for name in names
                    if name.endswith("/gene.fna")
                    or name.endswith("\\gene.fna")
                    or name == "gene.fna"
                ]

                # -----------------------------------------
                # Find protein.faa
                # -----------------------------------------

                faa_files = [
                    name
                    for name in names
                    if name.endswith("/protein.faa")
                    or name.endswith("\\protein.faa")
                    or name == "protein.faa"
                ]

                # -----------------------------------------
                # Extract FNA
                # -----------------------------------------

                if fna_files:

                    source = fna_files[0]

                    output = (
                        FNA_ROOT
                        / organism
                        / f"{gene}.fna"
                    )

                    output.parent.mkdir(
                        parents=True,
                        exist_ok=True,
                    )

                    with z.open(source) as src:
                        with output.open(
                            "wb"
                        ) as dst:

                            shutil.copyfileobj(
                                src,
                                dst
                            )

                    extracted_fna += 1

                    print(
                        f"  FNA: {output}"
                    )

                else:

                    print(
                        "  WARNING: gene.fna not found"
                    )

                # -----------------------------------------
                # Extract FAA
                # -----------------------------------------

                if faa_files:

                    source = faa_files[0]

                    output = (
                        FAA_ROOT
                        / organism
                        / f"{gene}.faa"
                    )

                    output.parent.mkdir(
                        parents=True,
                        exist_ok=True,
                    )

                    with z.open(source) as src:
                        with output.open(
                            "wb"
                        ) as dst:

                            shutil.copyfileobj(
                                src,
                                dst
                            )

                    extracted_faa += 1

                    print(
                        f"  FAA: {output}"
                    )

                else:

                    print(
                        "  WARNING: protein.faa not found"
                    )

        except zipfile.BadZipFile:

            failed += 1

            print(
                "  ERROR: Invalid ZIP file"
            )

        except Exception as exc:

            failed += 1

            print(
                f"  ERROR: {exc}"
            )

    print()
    print("=" * 75)
    print("EXTRACTION COMPLETE")
    print("=" * 75)

    print(
        f"ZIP files found : {len(zip_files)}"
    )

    print(
        f"FNA extracted   : {extracted_fna}"
    )

    print(
        f"FAA extracted   : {extracted_faa}"
    )

    print(
        f"Failed          : {failed}"
    )

    print()
    print(
        f"FNA directory   : {FNA_ROOT}"
    )

    print(
        f"FAA directory   : {FAA_ROOT}"
    )

    print("=" * 75)


if __name__ == "__main__":
    extract_all()