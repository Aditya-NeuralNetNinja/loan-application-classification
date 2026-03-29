#!/bin/bash
# ============================================================================
# download_data.sh — Download HMDA 2023 Loan Application Register (LAR) Data
# ============================================================================
# Source: CFPB / FFIEC Snapshot National Loan-Level Dataset
# URL:    https://ffiec.cfpb.gov/data-publication/snapshot-national-loan-level-dataset/2023
# 
# This downloads the 2023 HMDA Snapshot LAR dataset (~1.5M+ records) used for
# the loan denial prediction pipeline.
# ============================================================================

set -euo pipefail

# --- Configuration ---
YEAR="2023"
DATA_DIR="./data"
BASE_URL="https://s3.amazonaws.com/cfpb-hmda-public/prod/snapshot-data"

# File URLs (Snapshot National Loan-Level Dataset)
LAR_CSV_URL="${BASE_URL}/${YEAR}/${YEAR}_public_lar_csv.zip"
TS_CSV_URL="${BASE_URL}/${YEAR}/${YEAR}_public_ts_csv.zip"
PANEL_CSV_URL="${BASE_URL}/${YEAR}/${YEAR}_public_panel_csv.zip"

# --- Helper functions ---
info()  { echo -e "\033[1;34m[INFO]\033[0m  $*"; }
ok()    { echo -e "\033[1;32m[OK]\033[0m    $*"; }
warn()  { echo -e "\033[1;33m[WARN]\033[0m  $*"; }
error() { echo -e "\033[1;31m[ERROR]\033[0m $*"; exit 1; }

check_deps() {
    for cmd in curl unzip; do
        if ! command -v "$cmd" &>/dev/null; then
            error "'$cmd' is required but not installed."
        fi
    done
}

download_file() {
    local url="$1"
    local dest="$2"
    local filename
    filename=$(basename "$dest")

    if [[ -f "$dest" ]]; then
        warn "$filename already exists, skipping download."
        return 0
    fi

    info "Downloading $filename ..."
    curl -L --progress-bar --fail -o "$dest" "$url" || {
        rm -f "$dest"
        error "Failed to download $filename from $url"
    }
    ok "$filename downloaded ($(du -h "$dest" | cut -f1))."
}

unzip_file() {
    local zip_file="$1"
    local dest_dir="$2"

    if [[ ! -f "$zip_file" ]]; then
        warn "Zip file not found: $zip_file"
        return 1
    fi

    info "Extracting $(basename "$zip_file") ..."
    unzip -o -q "$zip_file" -d "$dest_dir"
    ok "Extracted to $dest_dir/"
}

# --- Main ---
main() {
    echo "=============================================="
    echo "  HMDA ${YEAR} Dataset Downloader"
    echo "=============================================="
    echo ""

    check_deps

    # Create data directory
    mkdir -p "$DATA_DIR"

    # Download LAR (Loan Application Register) — the main dataset
    download_file "$LAR_CSV_URL" "${DATA_DIR}/${YEAR}_public_lar_csv.zip"
    unzip_file "${DATA_DIR}/${YEAR}_public_lar_csv.zip" "$DATA_DIR"

    # Download Transmittal Sheet (institution metadata) — optional but useful
    download_file "$TS_CSV_URL" "${DATA_DIR}/${YEAR}_public_ts_csv.zip"
    unzip_file "${DATA_DIR}/${YEAR}_public_ts_csv.zip" "$DATA_DIR"

    # Download Panel (institution details) — optional
    download_file "$PANEL_CSV_URL" "${DATA_DIR}/${YEAR}_public_panel_csv.zip"
    unzip_file "${DATA_DIR}/${YEAR}_public_panel_csv.zip" "$DATA_DIR"

    echo ""
    echo "=============================================="
    info "Downloaded files:"
    echo "----------------------------------------------"
    ls -lh "$DATA_DIR"/*.csv 2>/dev/null || warn "No CSV files found."
    echo "=============================================="
    echo ""
    ok "Done! LAR data is ready at: ${DATA_DIR}/"
    echo ""
    echo "  Primary dataset (LAR):  ${DATA_DIR}/${YEAR}_public_lar.csv"
    echo "  Transmittal sheet (TS): ${DATA_DIR}/${YEAR}_public_ts.csv"
    echo "  Institution panel:      ${DATA_DIR}/${YEAR}_public_panel.csv"
    echo ""
    echo "  NOTE: The LAR CSV is large (~2-3 GB unzipped, ~20M+ rows)."
    echo "  Your PySpark pipeline handles this via Spark's distributed reader."
    echo ""
}

main "$@"