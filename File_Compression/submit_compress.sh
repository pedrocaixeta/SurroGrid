#!/bin/bash
# Helper wrapper to submit the HDF5 compression SLURM job safely.
# Usage: ./submit_compress.sh /path/to/h5_directory [compression_level] [extra sbatch args...]

set -euo pipefail

if [ $# -lt 1 ]; then
    echo "Usage: $0 <target_directory> [compression_level (default: 3)] [extra sbatch flags...]" >&2
    exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TARGET_DIR="$1"
shift

# Optional 2nd positional arg if it is a positive integer (1..19)
COMP_LEVEL="3"
if [ $# -ge 1 ] && [[ "$1" =~ ^[0-9]+$ ]]; then
    COMP_LEVEL="$1"
    shift
fi

if [ ! -d "$TARGET_DIR" ]; then
    echo "Error: Target directory does not exist: $TARGET_DIR" >&2
    exit 1
fi

cd "$SCRIPT_DIR"
mkdir -p logs/normal logs/errors

echo "Submitting compression job for: $TARGET_DIR (zstd level -$COMP_LEVEL)"
sbatch "$@" compress_h5.sh "$TARGET_DIR" "$COMP_LEVEL"
