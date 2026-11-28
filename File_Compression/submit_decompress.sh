#!/bin/bash
# Helper wrapper to submit the HDF5 decompression SLURM job safely.
# Usage: ./submit_decompress.sh /path/to/h5_directory [extra sbatch args...]

set -euo pipefail

if [ $# -lt 1 ]; then
    echo "Usage: $0 <target_directory> [extra sbatch flags...]" >&2
    exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TARGET_DIR="$1"
shift

if [ ! -d "$TARGET_DIR" ]; then
    echo "Error: Target directory does not exist: $TARGET_DIR" >&2
    exit 1
fi

cd "$SCRIPT_DIR"
mkdir -p logs/normal logs/errors

echo "Submitting decompression job for: $TARGET_DIR"
sbatch "$@" decompress_h5.sh "$TARGET_DIR"
