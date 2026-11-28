#!/bin/bash

#SBATCH -J h5_decompress
#SBATCH --output=logs/normal/%j_decompress_output.log
#SBATCH --error=logs/errors/%j_decompress_error.log

# Default LRZ Linux Cluster serial partition (adjust --time / --partition if needed)
#SBATCH --clusters=serial
#SBATCH --partition=serial_std
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=0-01:30:00
#SBATCH --mem=8G

# =============================================================================
# Lossless zstd decompression of .h5.zst files in a target directory
#
# Usage:
#   sbatch decompress_h5.sh /path/to/h5_directory
#   (or via ./submit_decompress.sh /path/to/h5_directory)
#
# Safety guarantees:
#   - Decompresses one file at a time into the same folder
#   - Starts with the smallest .h5.zst files first
#   - Writes to a temporary .h5.tmp file, atomically renames to .h5,
#     and ONLY THEN deletes the .h5.zst archive
#   - Safe to re-run if interrupted
# =============================================================================

set -euo pipefail

mkdir -p logs/normal logs/errors

# --- Parse arguments ---------------------------------------------------------
if [ $# -lt 1 ]; then
    echo "Usage: $0 <target_directory>" >&2
    exit 1
fi

TARGET_DIR="${1%/}"
THREADS="${SLURM_CPUS_PER_TASK:-4}"

if [ ! -d "$TARGET_DIR" ]; then
    echo "Error: Target directory does not exist: $TARGET_DIR" >&2
    exit 1
fi

# --- Ensure zstd is available ------------------------------------------------
if ! command -v zstd >/dev/null 2>&1; then
    if command -v module >/dev/null 2>&1; then
        module load miniforge3 >/dev/null 2>&1 || true
    fi
fi

if ! command -v zstd >/dev/null 2>&1; then
    echo "Error: 'zstd' not found in PATH. Load a module or conda env with zstd." >&2
    exit 1
fi

start_time=$(date +"%Y-%m-%d %H:%M:%S")
echo "================================================================="
echo "HDF5 Decompression Job started at: $start_time"
echo "Target directory : $TARGET_DIR"
echo "Threads          : $THREADS"
echo "zstd version     : $(zstd --version)"
echo "================================================================="

# --- Collect .h5.zst files sorted from smallest to largest -------------------
mapfile -d '' FILES < <(
    find "$TARGET_DIR" -maxdepth 1 -type f -name "*.h5.zst" -printf "%s\t%p\0" \
        | sort -z -n -t $'\t' -k1,1 \
        | cut -z -f2-
)

TOTAL_FILES=${#FILES[@]}
if [ "$TOTAL_FILES" -eq 0 ]; then
    echo "No .h5.zst files found in $TARGET_DIR. Nothing to do."
    exit 0
fi

echo "Found $TOTAL_FILES .h5.zst file(s). Decompressing smallest -> largest..."
echo ""

TOTAL_BEFORE=0
TOTAL_AFTER=0
SUCCEEDED=0
FAILED=0

for i in "${!FILES[@]}"; do
    f="${FILES[$i]}"
    idx=$((i + 1))
    fname=$(basename "$f")
    comp_bytes=$(stat -c%s "$f")

    final_out="${f%.zst}"       # strips trailing .zst -> original .h5 path
    tmp_out="${final_out}.tmp"

    rm -f "$tmp_out"

    # Decompress -> rename -> delete .zst archive
    if zstd -q -d -f "-T${THREADS}" "$f" -o "$tmp_out"; then
        mv -f "$tmp_out" "$final_out"
        decomp_bytes=$(stat -c%s "$final_out")
        rm -f "$f"

        TOTAL_BEFORE=$((TOTAL_BEFORE + comp_bytes))
        TOTAL_AFTER=$((TOTAL_AFTER + decomp_bytes))
        SUCCEEDED=$((SUCCEEDED + 1))

        comp_mb=$(awk -v a="$comp_bytes" 'BEGIN { printf "%.2f", a/1048576 }')
        decomp_mb=$(awk -v b="$decomp_bytes" 'BEGIN { printf "%.2f", b/1048576 }')

        echo "[$idx/$TOTAL_FILES] OK   $fname: ${comp_mb} MiB -> ${decomp_mb} MiB"
    else
        echo "[$idx/$TOTAL_FILES] FAIL $fname: decompression failed; archive kept." >&2
        rm -f "$tmp_out"
        FAILED=$((FAILED + 1))
    fi
done

# --- Summary -----------------------------------------------------------------
end_time=$(date +"%Y-%m-%d %H:%M:%S")
before_gb=$(awk -v a="$TOTAL_BEFORE" 'BEGIN { printf "%.3f", a/1073741824 }')
after_gb=$(awk -v b="$TOTAL_AFTER" 'BEGIN { printf "%.3f", b/1073741824 }')

echo ""
echo "================================================================="
echo "Finished at      : $end_time"
echo "Succeeded        : $SUCCEEDED / $TOTAL_FILES (Failed: $FAILED)"
echo "Total compressed : ${before_gb} GiB"
echo "Total restored   : ${after_gb} GiB"
echo "================================================================="

# Remove empty error log file if running under SLURM
if [ -n "${SLURM_JOB_ID:-}" ]; then
    ERR_FILE="logs/errors/${SLURM_JOB_ID}_decompress_error.log"
    if [ -f "$ERR_FILE" ] && [ ! -s "$ERR_FILE" ]; then
        rm -f "$ERR_FILE"
    fi
fi

if [ "$FAILED" -gt 0 ]; then
    exit 1
fi
