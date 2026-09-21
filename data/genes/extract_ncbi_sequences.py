from pathlib import Path
import zipfile


# ============================================================
# Paths
# ============================================================

SCRIPT_DIR = Path(__file__).resolve().parent

# ZIPs are actually in:
# bio/ncbi_downloads/
ZIP_ROOT = SCRIPT_DIR.parent.parent / "ncbi_downloads"

# Extracted files will be placed in:
# bio/data/genes/ncbi_extracted/
OUTPUT_ROOT = SCRIPT_DIR / "ncbi_extracted"

FNA_ROOT = OUTPUT_ROOT / "fna"
FAA_ROOT = OUTPUT_ROOT / "faa"