#!/bin/bash
# ==============================================================================
# Self-submitting SLURM job-array runner for all raw grids belonging to one DSO.
#
# Usage (run directly on the LRZ login node — do NOT wrap with `sbatch`):
#   ./infer_all_grids_from_dso.sh                          # uses hardcoded DSO_SHORT below
#   ./infer_all_grids_from_dso.sh SWU                      # override DSO short code
#   ./infer_all_grids_from_dso.sh Nrgie --dry-run          # preview counts & sbatch command
#   ./infer_all_grids_from_dso.sh Nrgie --limit 5          # smoke-test first 5 pending grids
#   ./infer_all_grids_from_dso.sh ByWrk --grids-per-task 5 --max-concurrent 50
#
# Valid DSO_SHORT codes (from Grid_Importing/gridreadout/2_sample_grids.ipynb):
#   AllgN (1,651) | ByWrk (38,871) | LEW (7,011) | MFrak (841) | Nrgie (6,446)
#   RegB    (858) | SWA      (880) | SWM (4,230) | SWU   (356) | nVNB (24,469)
# ==============================================================================

set -euo pipefail

# ---------------------------------------------------------------------------- #
# ------------------------- User Configuration ------------------------------- #
# ---------------------------------------------------------------------------- #
# Default DSO short code (can also be passed as the first positional CLI argument)
DSO_SHORT="${DSO_SHORT:-SWU}"

# Input raw grids folder and flat output folder
RAW_GRID_DIR="${RAW_GRID_DIR:-/dss/dssfs05/lwp-dss-0003/pn98cu/pn98cu-dss-0001/PedroC/INFERENCE/0.Raw_grids}"
OUTPUT_DIR="${OUTPUT_DIR:-/dss/dssfs05/lwp-dss-0003/pn98cu/pn98cu-dss-0001/PedroC/INFERENCE/Directly_Inferred}"
OUTPUT_SUFFIX="_direc_Inferred"

# SLURM cluster / resource defaults (per array task)
# Defaulting to serial_long (--qos=cm4_serial_long) because serial_std is frequently congested
# and serial_long has idle nodes + a 7-day walltime limit. Override via --partition / --qos.
CLUSTER="${CLUSTER:-serial}"
PARTITION="${PARTITION:-serial_long}"
QOS="${QOS:-cm4_serial_long}"
CPUS_PER_TASK="${CPUS_PER_TASK:-4}"
MEM="${MEM:-4G}"
TIME_PER_GRID_MIN="${TIME_PER_GRID_MIN:-5}"    # Walltime budget per grid (minutes; avg ~3 min/grid)

# Array scaling defaults
MAX_CONCURRENT="${MAX_CONCURRENT:-50}"         # Max simultaneously running array tasks (%N)
GRIDS_PER_TASK="${GRIDS_PER_TASK:-10}"         # 10 grids/task -> 50 min shared pool (36 tasks for SWU)
MAX_GRIDS="${MAX_GRIDS:-0}"                    # 0 = all pending grids; >0 = cap for testing

# Resolve script directory (handles both direct invocation and SLURM spool copies)
if [ -n "${SLURM_SUBMIT_DIR:-}" ]; then
  SCRIPT_DIR="$SLURM_SUBMIT_DIR"
else
  SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
fi
SCRIPT_PATH="${SCRIPT_DIR}/$(basename "${BASH_SOURCE[0]}")"

# Guard against accidental `sbatch infer_all_grids_from_dso.sh` without array submission
if [ -n "${SLURM_JOB_ID:-}" ] && [ -z "${SLURM_ARRAY_TASK_ID:-}" ]; then
  echo "ERROR: Run this script directly (./infer_all_grids_from_dso.sh), not via sbatch." >&2
  exit 1
fi

# ============================================================================ #
# MODE 2: SLURM ARRAY WORKER (runs inside compute node when submitted)         #
# ============================================================================ #
if [ -n "${SLURM_ARRAY_TASK_ID:-}" ]; then
  cd "$SCRIPT_DIR"
  mkdir -p logs/normal logs/errors

  module load miniforge3
  module list

  eval "$(conda shell.bash hook)"
  conda activate inference
  conda env list

  export HDF5_USE_FILE_LOCKING=FALSE
  export PYTHONDONTWRITEBYTECODE=1

  start_time=$(date +"%Y-%m-%d %H:%M:%S")
  echo "========================================================================"
  echo "Array Worker Started : $start_time"
  echo "SLURM Job / Task     : ${SLURM_ARRAY_JOB_ID:-N/A}_${SLURM_ARRAY_TASK_ID}"
  echo "DSO                  : ${DSO_SHORT}"
  echo "Manifest             : ${MANIFEST_FILE}"
  echo "Grids per task       : ${GRIDS_PER_TASK}"
  echo "========================================================================"

  if [ ! -f "$MANIFEST_FILE" ]; then
    echo "ERROR: Manifest file not found: $MANIFEST_FILE" >&2
    exit 1
  fi

  TOTAL_LINES=$(wc -l < "$MANIFEST_FILE")
  START_LINE=$(( (SLURM_ARRAY_TASK_ID - 1) * GRIDS_PER_TASK + 1 ))
  END_LINE=$(( SLURM_ARRAY_TASK_ID * GRIDS_PER_TASK ))
  if [ "$END_LINE" -gt "$TOTAL_LINES" ]; then
    END_LINE="$TOTAL_LINES"
  fi

  if [ "$START_LINE" -gt "$TOTAL_LINES" ]; then
    echo "Task $SLURM_ARRAY_TASK_ID has no assigned grids (start=$START_LINE > total=$TOTAL_LINES)."
    exit 0
  fi

  # Extract this task's slice of "GRID_ID<TAB>RAW_STEM" lines
  mapfile -t TASK_ENTRIES < <(sed -n "${START_LINE},${END_LINE}p" "$MANIFEST_FILE")

  FAILED_IDS=()
  SUCCEEDED=0
  SKIPPED=0

  for entry in "${TASK_ENTRIES[@]}"; do
    [ -z "$entry" ] && continue
    GRID_ID="${entry%%	*}"
    RAW_STEM="${entry#*	}"
    EXPECTED_OUT="${OUTPUT_DIR}/${RAW_STEM}${OUTPUT_SUFFIX}.h5"

    echo ""
    echo "------------------------------------------------------------------------"
    echo "Grid ID: $GRID_ID (raw stem: $RAW_STEM)"
    # Auto-skip temporarily disabled so existing *_direc_Inferred.h5 files get overwritten
    # while adapting the NAP 2045 scenario. Uncomment to resume without re-running completed grids:
    # if [ -f "$EXPECTED_OUT" ]; then
    #   echo "Already inferred -> skipping: $EXPECTED_OUT"
    #   SKIPPED=$((SKIPPED + 1))
    #   continue
    # fi

    if srun python3 main.py "$GRID_ID" \
         --n_cpu "${SLURM_CPUS_PER_TASK:-1}" \
         --input_dir "$RAW_GRID_DIR" \
         --output_dir "$OUTPUT_DIR"; then
      SUCCEEDED=$((SUCCEEDED + 1))
    else
      echo "ERROR: Direct inference failed for Grid ID $GRID_ID ($RAW_STEM)" >&2
      FAILED_IDS+=("$GRID_ID")
    fi
  done

  end_time=$(date +"%Y-%m-%d %H:%M:%S")
  echo ""
  echo "========================================================================"
  echo "Array Worker Finished: $end_time"
  echo "Summary: succeeded=$SUCCEEDED, skipped=$SKIPPED, failed=${#FAILED_IDS[@]}"
  echo "========================================================================"

  # Delete error log file if empty; otherwise append DSO & grid IDs for quick triage
  if [ -n "${SLURM_ARRAY_JOB_ID:-}" ]; then
    ERR_FILE="logs/errors/${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}_error.log"
    if [ -f "$ERR_FILE" ]; then
      if [ ! -s "$ERR_FILE" ]; then
        rm "$ERR_FILE"
      else
        echo "DSO=${DSO_SHORT} | Failed Grid IDs: ${FAILED_IDS[*]:-none}" >> "$ERR_FILE"
      fi
    fi
  fi

  if [ "${#FAILED_IDS[@]}" -gt 0 ]; then
    exit 1
  fi
  exit 0
fi

# ============================================================================ #
# MODE 1: LAUNCHER (scans 0.Raw_grids, filters completed, submits job array)   #
# ============================================================================ #
DRY_RUN=0

usage() {
  cat <<EOF
Usage: ./$(basename "$0") [DSO_SHORT] [options]

Options:
  --dry-run                  Scan directories and print sbatch command without submitting
  --limit N                  Process at most N pending grids (default: 0 = all)
  --grids-per-task K         Grids executed sequentially per array task (default: ${GRIDS_PER_TASK})
  --max-concurrent M         Max concurrent array tasks via SLURM %M (default: ${MAX_CONCURRENT})
  --time-per-grid MIN        Walltime minutes allocated per grid (default: ${TIME_PER_GRID_MIN})
  --cluster NAME             SLURM cluster (default: ${CLUSTER})
  --partition NAME           SLURM partition (default: ${PARTITION})
  --qos NAME                 SLURM QoS; pass "" to omit (default: ${QOS})
  --input-dir PATH           Override RAW_GRID_DIR
  --output-dir PATH          Override OUTPUT_DIR
  -h, --help                 Show this help message
EOF
}

while [ "$#" -gt 0 ]; do
  case "$1" in
    --dry-run)
      DRY_RUN=1; shift ;;
    --limit)
      MAX_GRIDS="$2"; shift 2 ;;
    --grids-per-task)
      GRIDS_PER_TASK="$2"; shift 2 ;;
    --max-concurrent)
      MAX_CONCURRENT="$2"; shift 2 ;;
    --time-per-grid)
      TIME_PER_GRID_MIN="$2"; shift 2 ;;
    --cluster)
      CLUSTER="$2"; shift 2 ;;
    --partition)
      PARTITION="$2"; shift 2 ;;
    --qos)
      QOS="$2"; shift 2 ;;
    --input-dir)
      RAW_GRID_DIR="$2"; shift 2 ;;
    --output-dir)
      OUTPUT_DIR="$2"; shift 2 ;;
    -h|--help)
      usage; exit 0 ;;
    -*)
      echo "ERROR: Unknown option '$1'" >&2; usage; exit 1 ;;
    *)
      DSO_SHORT="$1"; shift ;;
  esac
done

# Resolve any relative --input-dir / --output-dir against caller's cwd before cd'ing into SCRIPT_DIR
RAW_GRID_DIR="$(realpath -m "$RAW_GRID_DIR")"
OUTPUT_DIR="$(realpath -m "$OUTPUT_DIR")"

cd "$SCRIPT_DIR"
mkdir -p logs/normal logs/errors logs/manifests

if [ ! -d "$RAW_GRID_DIR" ]; then
  echo "ERROR: Raw grid directory does not exist: $RAW_GRID_DIR" >&2
  exit 1
fi

TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
MANIFEST_FILE="${SCRIPT_DIR}/logs/manifests/${DSO_SHORT}_${TIMESTAMP}.txt"

# Scan RAW_GRID_DIR and OUTPUT_DIR via Python os.scandir (fast even for 80k+ files on DSS)
SCAN_STATS=$(PYTHONDONTWRITEBYTECODE=1 python3 - "$RAW_GRID_DIR" "$OUTPUT_DIR" "$DSO_SHORT" "$OUTPUT_SUFFIX" "$MAX_GRIDS" "$MANIFEST_FILE" "$DRY_RUN" <<'PYEOF'
import os, sys

raw_dir, out_dir, dso_short, out_suffix, max_grids, manifest_path, dry_run = (
    sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4], int(sys.argv[5]), sys.argv[6], int(sys.argv[7])
)

exclude_suffixes = ("_DmdAllocated.h5", "_Features.h5", "_Predicted.h5", f"{out_suffix}.h5", "_pwrflw.h5")

# Auto-skip temporarily disabled so existing *_direc_Inferred.h5 files get overwritten
# while adapting the NAP 2045 scenario. Uncomment the block below (and switch `pending`
# back to filtering against `done_files`) once you want resumable skip-existing behavior:
# done_files = set()
# if os.path.isdir(out_dir):
#     with os.scandir(out_dir) as it:
#         for entry in it:
#             if entry.name.endswith(f"{out_suffix}.h5"):
#                 done_files.add(entry.name)

# Single fast scan of raw grids matching exact second token == dso_short
matched = []
with os.scandir(raw_dir) as it:
    for entry in it:
        name = entry.name
        if not name.endswith(".h5") or name.endswith(exclude_suffixes):
            continue
        parts = name.split("_")
        if len(parts) >= 2 and parts[1] == dso_short:
            matched.append((parts[0], os.path.splitext(name)[0]))

matched.sort(key=lambda x: x[1])
total = len(matched)
# pending = [
#     (gid, stem) for gid, stem in matched
#     if f"{stem}{out_suffix}.h5" not in done_files
# ]
pending = list(matched)
already_done = total - len(pending)

if max_grids > 0:
    pending = pending[:max_grids]

if not dry_run and pending:
    with open(manifest_path, "w", encoding="utf-8") as f:
        for gid, stem in pending:
            f.write(f"{gid}\t{stem}\n")

preview = ",".join(gid for gid, _ in pending[:10])
print(f"{total} {already_done} {len(pending)} {preview}")
PYEOF
)

read -r TOTAL_MATCHED ALREADY_DONE PENDING_COUNT PREVIEW_IDS <<< "$SCAN_STATS"

echo "========================================================================"
echo "DSO Batch Direct Inference Launcher"
echo "  DSO_SHORT       : $DSO_SHORT"
echo "  RAW_GRID_DIR    : $RAW_GRID_DIR"
echo "  OUTPUT_DIR      : $OUTPUT_DIR"
echo "  Total matching  : $TOTAL_MATCHED"
echo "  Already done    : $ALREADY_DONE"
echo "  Pending to run  : $PENDING_COUNT"
echo "========================================================================"

if [ "$PENDING_COUNT" -eq 0 ]; then
  echo "Nothing to submit for DSO '$DSO_SHORT'."
  exit 0
fi

# Compute number of array tasks and per-task walltime (D-HH:MM:SS)
NUM_TASKS=$(( (PENDING_COUNT + GRIDS_PER_TASK - 1) / GRIDS_PER_TASK ))
TOTAL_MIN=$(( GRIDS_PER_TASK * TIME_PER_GRID_MIN ))
DAYS=$(( TOTAL_MIN / 1440 ))
HOURS=$(( (TOTAL_MIN % 1440) / 60 ))
MINS=$(( TOTAL_MIN % 60 ))
WALLTIME=$(printf "%d-%02d:%02d:00" "$DAYS" "$HOURS" "$MINS")

# Warn if computed walltime exceeds partition limit (serial_std: 1 day, serial_long: 7 days)
if [ "$TOTAL_MIN" -gt 1440 ] && [ "$PARTITION" = "serial_std" ]; then
  echo "WARNING: Computed walltime ($WALLTIME) exceeds serial_std's 1-day limit. Lower --grids-per-task." >&2
elif [ "$TOTAL_MIN" -gt 10080 ] && [ "$PARTITION" = "serial_long" ]; then
  echo "WARNING: Computed walltime ($WALLTIME) exceeds serial_long's 7-day limit. Lower --grids-per-task." >&2
fi

SBATCH_CMD=(
  sbatch
  -J "inf_${DSO_SHORT}"
  --clusters="$CLUSTER"
  --partition="$PARTITION"
)
if [ -n "$QOS" ]; then
  SBATCH_CMD+=(--qos="$QOS")
fi
SBATCH_CMD+=(
  --ntasks=1
  --cpus-per-task="$CPUS_PER_TASK"
  --mem="$MEM"
  --time="$WALLTIME"
  --array="1-${NUM_TASKS}%${MAX_CONCURRENT}"
  --output="logs/normal/%A_%a_output.log"
  --error="logs/errors/%A_%a_error.log"
  --export="ALL,MANIFEST_FILE=${MANIFEST_FILE},DSO_SHORT=${DSO_SHORT},GRIDS_PER_TASK=${GRIDS_PER_TASK},RAW_GRID_DIR=${RAW_GRID_DIR},OUTPUT_DIR=${OUTPUT_DIR}"
  "$SCRIPT_PATH"
)

echo "  Array tasks     : 1-${NUM_TASKS} (max ${MAX_CONCURRENT} concurrent, ${GRIDS_PER_TASK} grid(s)/task)"
echo "  Per-task res.   : ${CPUS_PER_TASK} CPUs, ${MEM} RAM, walltime ${WALLTIME} (${CLUSTER}/${PARTITION}${QOS:+, qos=${QOS}})"
echo "  First grid IDs  : ${PREVIEW_IDS}"

if [ "$DRY_RUN" -eq 1 ]; then
  echo ""
  echo "[DRY RUN] Would execute:"
  printf '  %q ' "${SBATCH_CMD[@]}"
  echo ""
  exit 0
fi

echo "  Manifest saved  : $MANIFEST_FILE"
echo ""
"${SBATCH_CMD[@]}"
