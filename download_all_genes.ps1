$ErrorActionPreference = "Stop"

# ============================================================
# DOWNLOAD ALL 4 ORGANISMS
# 15 genes per organism
# Each organism -> separate folder
# Each gene -> separate folder
# Each gene -> individual ZIP
# ============================================================

$baseDir = "..\..\data\ncbi_downloads"
$datasetsExe = Join-Path $baseDir "datasets.exe"

New-Item -ItemType Directory -Force -Path $baseDir | Out-Null

# ============================================================
# DOWNLOAD NCBI DATASETS CLI
# ============================================================

if (-not (Test-Path $datasetsExe)) {

    Write-Host ""
    Write-Host "Downloading NCBI Datasets CLI..." -ForegroundColor Cyan

    $url = "https://ftp.ncbi.nlm.nih.gov/pub/datasets/command-line/v2/win64/datasets.exe"

    Invoke-WebRequest `
        -Uri $url `
        -OutFile $datasetsExe

    Write-Host "NCBI Datasets CLI downloaded." -ForegroundColor Green
}

# ============================================================
# ORGANISMS AND GENES
# ============================================================

$organisms = @(

    # --------------------------------------------------------
    # E. COLI
    # --------------------------------------------------------

    @{
        Folder = "ecoli"
        Taxon = "Escherichia coli str. K-12 substr. MG1655"

        Genes = @(
            "groES"
            "adk"
            "fur"
            "lacI"
            "recA"
            "ftsN"
            "zapE"
            "lacY"
            "groL"
            "rpoD"
            "dnaK"
            "htpG"
            "lacZ"
            "recC"
            "rpoC"
        )
    }

    # --------------------------------------------------------
    # BACILLUS SUBTILIS
    # --------------------------------------------------------

    @{
        Folder = "bacillus_subtilis"
        Taxon = "Bacillus subtilis subsp. subtilis str. 168"

        Genes = @(
            "groES"
            "adk"
            "recA"
            "groEL"
            "dnaK"
            "htpG"
            "recO"
            "rpoC"
            "gyrA"
            "gyrB"
            "infB"
            "tufA"
            "rplB"
            "rpsJ"
            "ftsZ"
        )
    }

    # --------------------------------------------------------
    # PSEUDOMONAS AERUGINOSA
    # --------------------------------------------------------

    @{
        Folder = "pseudomonas_aeruginosa"
        Taxon = "Pseudomonas aeruginosa PAO1"

        Genes = @(
            "groES"
            "adk"
            "recA"
            "groEL"
            "rpoD"
            "dnaK"
            "htpG"
            "recC"
            "rpoC"
            "gyrA"
            "gyrB"
            "infB"
            "tufA"
            "rplB"
            "ftsZ"
        )
    }

    # --------------------------------------------------------
    # MYCOBACTERIUM TUBERCULOSIS
    # --------------------------------------------------------

    @{
        Folder = "mycobacterium_tuberculosis"
        Taxon = "Mycobacterium tuberculosis H37Rv"

        Genes = @(
            "groES"
            "adk"
            "recA"
            "groEL2"
            "dnaK"
            "htpG"
            "rpoC"
            "gyrA"
            "gyrB"
            "infB"
            "tuf"
            "rplB"
            "rpsJ"
            "ftsZ"
            "recG"
        )
    }
)

# ============================================================
# COUNTERS
# ============================================================

$totalGenes = 0

foreach ($org in $organisms) {
    $totalGenes += $org.Genes.Count
}

$current = 0
$success = 0
$failed = 0

# ============================================================
# START
# ============================================================

Write-Host ""
Write-Host "============================================================"
Write-Host "        NCBI BIOLOGICAL DATA DOWNLOAD"
Write-Host "============================================================"
Write-Host "Organisms : $($organisms.Count)"
Write-Host "Genes     : $totalGenes"
Write-Host "============================================================"
Write-Host ""

# ============================================================
# DOWNLOAD
# ============================================================

foreach ($org in $organisms) {

    $organismDir = Join-Path $baseDir $org.Folder

    New-Item `
        -ItemType Directory `
        -Force `
        -Path $organismDir | Out-Null

    Write-Host ""
    Write-Host "############################################################"
    Write-Host " ORGANISM: $($org.Taxon)"
    Write-Host "############################################################"

    foreach ($gene in $org.Genes) {

        $current++

        # ----------------------------------------------------
        # Gene folder
        # ----------------------------------------------------

        $geneDir = Join-Path $organismDir $gene

        New-Item `
            -ItemType Directory `
            -Force `
            -Path $geneDir | Out-Null

        # ----------------------------------------------------
        # ZIP file
        # ----------------------------------------------------

        $outputZip = Join-Path `
            $geneDir `
            "${gene}_datasets.zip"

        Write-Host ""
        Write-Host "[$current/$totalGenes] $($org.Folder) -> $gene" `
            -ForegroundColor Cyan

        # ----------------------------------------------------
        # Skip existing
        # ----------------------------------------------------

        if (Test-Path $outputZip) {

            Write-Host "Already exists - skipping" `
                -ForegroundColor Yellow

            $success++
            continue
        }

        # ----------------------------------------------------
        # Download
        # ----------------------------------------------------

        try {

            & $datasetsExe download gene symbol $gene `
                --taxon $org.Taxon `
                --include gene,cds,protein `
                --filename $outputZip `
                --no-progressbar

            if ($LASTEXITCODE -eq 0 -and (Test-Path $outputZip)) {

                Write-Host "SUCCESS" -ForegroundColor Green

                $success++
            }
            else {

                Write-Host "FAILED" -ForegroundColor Red

                $failed++
            }

        }
        catch {

            Write-Host "FAILED" -ForegroundColor Red
            Write-Host $_.Exception.Message -ForegroundColor Red

            $failed++
        }
    }
}

# ============================================================
# FINAL SUMMARY
# ============================================================

Write-Host ""
Write-Host "============================================================"
Write-Host "                  DOWNLOAD COMPLETE"
Write-Host "============================================================"

Write-Host ""
Write-Host "Successful : $success" -ForegroundColor Green
Write-Host "Failed     : $failed" -ForegroundColor Red
Write-Host "Total      : $totalGenes" -ForegroundColor Cyan

Write-Host ""
Write-Host "ZIP FILES:"
Write-Host ""

Get-ChildItem `
    $baseDir `
    -Recurse `
    -Filter "*_datasets.zip" |
    Select-Object FullName, Length