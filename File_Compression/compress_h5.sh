#!/bin/bash

#SBATCH -J h5_compress
#SBATCH --output=logs/normal/%j_compress_output.log
#SBATCH --error=logs/errors/%j_compress_error.log

# Default LRZ Linux Cluster serial partition (adjust --time / --partition if needed)
#SBATCH --clusters=serial
#SBATCH --partition=serial_std
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=0-02:00:00
#SBATCH --mem=8G

# =============================================================================
# Lossless zstd compression of .h5 files in a target directory (smallest first)
#
# Usage:
#   sbatch compress_h5.sh /path/to/h5_directory [compression_level]
#   (or via ./submit_compress.sh /path/to/h5_directory [compression_level])
#
# Safety guarantees:
#   - Compresses one file at a time (only needs free space for 1 compressed file)
#   - Starts with the smallest .h5 files first to free space quickly
#   - Writes to a temporary .zst.tmp file, verifies integrity with `zstd -t`,
#     atomically renames to .h5.zst, and ONLY THEN deletes the original .h5
#   - Safe to re-run if interrupted by walltime or quota
# =============================================================================

set -euo pipefail

mkdir -p logs/normal logs/errors

# --- Parse arguments ---------------------------------------------------------
if [ $# -lt 1 ]; then
    echo "Usage: $0 <target_directory> [compression_level (default: 3)]" >&2
    exit 1
fi

TARGET_DIR="${1%/}"
COMP_LEVEL="${2:-3}"
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
echo "HDF5 Compression Job started at: $start_time"
echo "Target directory : $TARGET_DIR"
echo "Compression level: -$COMP_LEVEL"
echo "Threads          : $THREADS"
echo "zstd version     : $(zstd --version)"
echo "================================================================="

# --- Collect .h5 files sorted from smallest to largest -----------------------
# Non-recursive (-maxdepth 1) so subdirectories are not touched unexpectedly.
mapfile -d '' FILES < <(
    find "$TARGET_DIR" -maxdepth 1 -type f -name "*.h5" -printf "%s\t%p\0" \
        | sort -z -n -t $'\t' -k1,1 \
        | cut -z -f2-
)

TOTAL_FILES=${#FILES[@]}
if [ "$TOTAL_FILES" -eq 0 ]; then
    echo "No .h5 files found in $TARGET_DIR. Nothing to do."
    exit 0
fi

echo "Found $TOTAL_FILES .h5 file(s). Processing smallest -> largest..."
echo ""

TOTAL_BEFORE=0
TOTAL_AFTER=0
SUCCEEDED=0
FAILED=0

for i in "${!FILES[@]}"; do
    f="${FILES[$i]}"
    idx=$((i + 1))
    fname=$(basename "$f")
    orig_bytes=$(stat -c%s "$f")

    if [ "$orig_bytes" -eq 0 ]; then
        echo "[$idx/$TOTAL_FILES] WARNING: $fname is 0 bytes (still compressing)."
    fi

    tmp_out="${f}.zst.tmp"
    final_out="${f}.zst"

    # Remove stale temp file if a previous run was interrupted mid-file
    rm -f "$tmp_out"

    # Compress -> verify integrity -> rename -> delete original
    if zstd -q -f "-${COMP_LEVEL}" "-T${THREADS}" "$f" -o "$tmp_out" \
        && zstd -q -t "$tmp_out"; then
        mv -f "$tmp_out" "$final_out"
        comp_bytes=$(stat -c%s "$final_out")
        rm -f "$f"

        TOTAL_BEFORE=$((TOTAL_BEFORE + orig_bytes))
        TOTAL_AFTER=$((TOTAL_AFTER + comp_bytes))
        SUCCEEDED=$((SUCCEEDED + 1))

        if [ "$orig_bytes" -gt 0 ]; then
            saved_pct=$(awk -v a="$orig_bytes" -v b="$comp_bytes" 'BEGIN { printf "%.1f", (1 - b/a)*100 }')
        else
            saved_pct="0.0"
        fi
        orig_mb=$(awk -v a="$orig_bytes" 'BEGIN { printf "%.2f", a/1048576 }')
        comp_mb=$(awk -v b="$comp_bytes" 'BEGIN { printf "%.2f", b/1048576 }')

        echo "[$idx/$TOTAL_FILES] OK   $fname: ${orig_mb} MiB -> ${comp_mb} MiB (-${saved_pct}%)"
    else
        echo "[$idx/$TOTAL_FILES] FAIL $fname: compression or integrity check failed; original kept." >&2
        rm -f "$tmp_out"
        FAILED=$((FAILED + 1))
    fi
done

# --- Summary -----------------------------------------------------------------
end_time=$(date +"%Y-%m-%d %H:%M:%S")
before_gb=$(awk -v a="$TOTAL_BEFORE" 'BEGIN { printf "%.3f", a/1073741824 }')
after_gb=$(awk -v b="$TOTAL_AFTER" 'BEGIN { printf "%.3f", b/1073741824 }')
freed_gb=$(awk -v a="$TOTAL_BEFORE" -v b="$TOTAL_AFTER" 'BEGIN { printf "%.3f", (a-b)/1073741824 }')

echo ""
echo "================================================================="
echo "Finished at   : $end_time"
echo "Succeeded     : $SUCCEEDED / $TOTAL_FILES (Failed: $FAILED)"
echo "Total before  : ${before_gb} GiB"
echo "Total after   : ${after_gb} GiB"
echo "Space freed   : ${freed_gb} GiB"
echo "================================================================="

# Remove empty error log file if running under SLURM
if [ -n "${SLURM_JOB_ID:-}" ]; then
    ERR_FILE="logs/errors/${SLURM_JOB_ID}_compress_error.log"
    if [ -f "$ERR_FILE" ] && [ ! -s "$ERR_FILE" ]; then
        rm -f "$ERR_FILE"
    fi
fi

if [ "$FAILED" -gt 0 ]; then
    exit 1
fi
