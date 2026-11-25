#!/bin/bash
# ==============================================================================
# Self-submitting SLURM job-array runner for Step 1 (1.Cheap_Ground_Truth)
# across all raw grids belonging to one DSO on LRZ's `inter` (`cm4_inter`) queue.
#
# Usage (run directly on the LRZ login node — do NOT wrap with `sbatch`):
#   ./run_dso_inter.sh SWU --dry-run            # preview counts & sbatch command
#   ./run_dso_inter.sh SWU                      # submit all pending SWU grids
#   ./run_dso_inter.sh SWU --limit 5            # smoke-test 5 smallest pending grids
#   ./run_dso_inter.sh nVNB                     # auto-caps at <=199 tasks; re-run for next wave
#   ./run_dso_inter.sh nVNB --part 1/2          # or submit explicit halves (153 tasks each)
#   ./run_dso_inter.sh ByWrk --part 1/3         # or submit explicit thirds (162 tasks each)
#
# Valid DSO_SHORT codes (ordered smallest -> largest):
#   SWU    (356) | MFrak   (841) | RegB    (858) | SWA     (880) | AllgN (1,651)
#   SWM  (4,230) | Nrgie (6,446) | LEW   (7,011) | nVNB (24,469) | ByWrk (38,871)
#
# Notes:
# - Grids are sorted by raw .h5 file size ascending (smallest grids run first).
# - Already completed `*_DmdAllocated.h5` files are automatically skipped (use
#   `--overwrite` to force re-running existing outputs).
# - Each array task runs ONE persistent Python process with a 4-worker pool
#   (ProcessPoolExecutor, n_cpu=1 per grid) using the optimized Step 1 code
#   (~2-7s wall/grid on 4 CPUs).
# - With GRIDS_PER_TASK=450 and TIME_PER_GRID_SEC=40, each full task requests
#   5h 00m (under cm4_inter's 8h limit), and ALL 10 DSOs combined (85,613 grids)
#   fit in 194 tasks total (< 200 tasks)!
# - Submissions are strictly capped at MAX_TASKS=199 (< 200 tasks).
# ==============================================================================

set -euo pipefail

# ---------------------------------------------------------------------------- #
# ------------------------- User Configuration ------------------------------- #
# ---------------------------------------------------------------------------- #
DSO_SHORT="${DSO_SHORT:-SWU}"

# Input raw grids folder and Step 1 output folder (matches pipeline_config.py)
RAW_GRID_DIR="${RAW_GRID_DIR:-/dss/dssfs05/lwp-dss-0003/pn98cu/pn98cu-dss-0001/PedroC/INFERENCE/0.Raw_grids}"
OUTPUT_DIR="${OUTPUT_DIR:-/dss/dssfs05/lwp-dss-0003/pn98cu/pn98cu-dss-0001/PedroC/INFERENCE/Step_by_Step/1.Cheap_Ground_Truthed}"
OUTPUT_SUFFIX="_DmdAllocated"

# SLURM cluster / resource defaults (per array task)
# cm4_inter has an 8h max walltime limit; no --qos needed on inter cluster.
CLUSTER="${CLUSTER:-inter}"
PARTITION="${PARTITION:-cm4_inter}"
QOS="${QOS:-}"
CPUS_PER_TASK="${CPUS_PER_TASK:-4}"
MEM="${MEM:-10G}"                              # 4 concurrent workers (~0.5-1.0 GB peak RSS each)
TIME_PER_GRID_SEC="${TIME_PER_GRID_SEC:-40}"   # Walltime budget per grid [s] (actual ~2-7s wall on 4 CPUs)

# Array scaling defaults
# 450 grids/task * 40s = 300 min = 5h 00m per task (under cm4_inter 8h limit; actual ~15-60 min).
# All 10 Bavarian DSOs combined (85,613 grids) total 194 tasks (< 200 tasks)!
MAX_CONCURRENT="${MAX_CONCURRENT:-20}"         # Max simultaneously running array tasks (%M)
MAX_TASKS="${MAX_TASKS:-199}"                  # Hard cap on submitted array tasks (< 200)
GRIDS_PER_TASK="${GRIDS_PER_TASK:-450}"        # Grids processed per array task (4-worker pool)
MAX_GRIDS="${MAX_GRIDS:-0}"                    # 0 = all pending grids; >0 = cap for testing
PART_SPEC="${PART_SPEC:-1/1}"                  # Split DSO into N parts, run part K (e.g. 1/2, 1/3)
OVERWRITE="${OVERWRITE:-0}"                    # 0 = skip already done; 1 = re-run all

# Resolve script directory (handles both direct invocation and SLURM spool copies)
if [ -n "${SLURM_SUBMIT_DIR:-}" ]; then
  SCRIPT_DIR="$SLURM_SUBMIT_DIR"
else
  SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
fi
SCRIPT_PATH="${SCRIPT_DIR}/$(basename "${BASH_SOURCE[0]}")"

# Guard against accidental `sbatch run_dso_inter.sh` without array submission
if [ -n "${SLURM_JOB_ID:-}" ] && [ -z "${SLURM_ARRAY_TASK_ID:-}" ]; then
  echo "ERROR: Run this script directly (./run_dso_inter.sh <DSO>), not via sbatch." >&2
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

  # Prevent nested BLAS/OpenMP threads from oversubscribing multiprocessing workers
  export OMP_NUM_THREADS=1
  export MKL_NUM_THREADS=1
  export OPENBLAS_NUM_THREADS=1
  export VECLIB_MAXIMUM_THREADS=1
  export NUMEXPR_NUM_THREADS=1
  export HDF5_USE_FILE_LOCKING=FALSE
  export PYTHONDONTWRITEBYTECODE=1

  start_time=$(date +"%Y-%m-%d %H:%M:%S")
  echo "========================================================================"
  echo "Step 1 Array Worker Started : $start_time"
  echo "SLURM Job / Task            : ${SLURM_ARRAY_JOB_ID:-N/A}_${SLURM_ARRAY_TASK_ID}"
  echo "DSO / Part                  : ${DSO_SHORT} (${PART_SPEC})"
  echo "Manifest                    : ${MANIFEST_FILE}"
  echo "Grids per task              : ${GRIDS_PER_TASK}"
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

  # Run this task's manifest slice inside ONE persistent Python process using a
  # grid-level ProcessPoolExecutor (SLURM_CPUS_PER_TASK workers, n_cpu=1 per grid).
  # This eliminates repeated Python/pandapower/pvlib imports and intra-grid Pool overhead.
  BATCH_CMD=(
    srun python3 main.py
    --manifest "$MANIFEST_FILE"
    --start_line "$START_LINE"
    --end_line "$END_LINE"
    --n_cpu "${SLURM_CPUS_PER_TASK:-1}"
    --input_dir "$RAW_GRID_DIR"
    --output_dir "$OUTPUT_DIR"
  )
  if [ "${OVERWRITE:-0}" -eq 1 ]; then
    BATCH_CMD+=(--overwrite)
  fi

  BATCH_EXIT=0
  "${BATCH_CMD[@]}" || BATCH_EXIT=$?

  end_time=$(date +"%Y-%m-%d %H:%M:%S")
  echo ""
  echo "========================================================================"
  echo "Step 1 Array Worker Finished: $end_time (exit_code=${BATCH_EXIT})"
  echo "========================================================================"

  # Delete error log file if empty; otherwise append DSO & slice metadata for quick triage
  if [ -n "${SLURM_ARRAY_JOB_ID:-}" ]; then
    ERR_FILE="logs/errors/${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}_error.log"
    if [ -f "$ERR_FILE" ]; then
      if [ ! -s "$ERR_FILE" ]; then
        rm "$ERR_FILE"
      else
        echo "DSO=${DSO_SHORT} (${PART_SPEC}) | Manifest lines ${START_LINE}-${END_LINE} | Exit=${BATCH_EXIT}" >> "$ERR_FILE"
      fi
    fi
  fi

  exit "$BATCH_EXIT"
fi

# ============================================================================ #
# MODE 1: LAUNCHER (scans 0.Raw_grids, sorts smallest first, submits array)    #
# ============================================================================ #
DRY_RUN=0

usage() {
  cat <<EOF
Usage: ./$(basename "$0") [DSO_SHORT] [options]

Options:
  --part K/N                 Split DSO's matched grids into N equal chunks and run part K (default: 1/1)
  --dry-run                  Scan directories and print sbatch command without submitting
  --limit N                  Process at most N pending grids (default: 0 = all)
  --overwrite                Re-run grids even if *_DmdAllocated.h5 already exists
  --grids-per-task K         Grids processed per array task (default: ${GRIDS_PER_TASK})
  --max-tasks T              Max array tasks per submission, < 200 (default: ${MAX_TASKS})
  --max-concurrent M         Max concurrent array tasks via SLURM %M (default: ${MAX_CONCURRENT})
  --time-per-grid-sec SEC    Walltime seconds allocated per grid (default: ${TIME_PER_GRID_SEC})
  --time-per-grid MIN        Walltime minutes allocated per grid (sets --time-per-grid-sec = MIN*60)
  --cluster NAME             SLURM cluster (default: ${CLUSTER})
  --partition NAME           SLURM partition (default: ${PARTITION})
  --qos NAME                 SLURM QoS; empty string omits --qos (default: "${QOS}")
  --input-dir PATH           Override RAW_GRID_DIR
  --output-dir PATH          Override OUTPUT_DIR
  -h, --help                 Show this help message
EOF
}

while [ "$#" -gt 0 ]; do
  case "$1" in
    --part)
      PART_SPEC="$2"; shift 2 ;;
    --dry-run)
      DRY_RUN=1; shift ;;
    --limit)
      MAX_GRIDS="$2"; shift 2 ;;
    --overwrite)
      OVERWRITE=1; shift ;;
    --grids-per-task)
      GRIDS_PER_TASK="$2"; shift 2 ;;
    --max-tasks)
      MAX_TASKS="$2"; shift 2 ;;
    --max-concurrent)
      MAX_CONCURRENT="$2"; shift 2 ;;
    --time-per-grid-sec)
      TIME_PER_GRID_SEC="$2"; shift 2 ;;
    --time-per-grid)
      TIME_PER_GRID_SEC=$(( $2 * 60 )); shift 2 ;;
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

# Validate --part K/N
if ! [[ "$PART_SPEC" =~ ^[0-9]+/[0-9]+$ ]]; then
  echo "ERROR: --part must have the form K/N (e.g. 1/2 or 2/2), got '$PART_SPEC'." >&2
  exit 1
fi
PART_IDX="${PART_SPEC%/*}"
PART_TOTAL="${PART_SPEC#*/}"
if [ "$PART_TOTAL" -lt 1 ] || [ "$PART_IDX" -lt 1 ] || [ "$PART_IDX" -gt "$PART_TOTAL" ]; then
  echo "ERROR: Invalid --part '$PART_SPEC' (require 1 <= K <= N)." >&2
  exit 1
fi

# Resolve any relative paths against caller's cwd before cd'ing into SCRIPT_DIR
RAW_GRID_DIR="$(realpath -m "$RAW_GRID_DIR")"
OUTPUT_DIR="$(realpath -m "$OUTPUT_DIR")"

cd "$SCRIPT_DIR"
mkdir -p logs/normal logs/errors logs/manifests

if [ ! -d "$RAW_GRID_DIR" ]; then
  echo "ERROR: Raw grid directory does not exist: $RAW_GRID_DIR" >&2
  exit 1
fi

TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
PART_TAG=""
if [ "$PART_TOTAL" -gt 1 ]; then
  PART_TAG="_p${PART_IDX}of${PART_TOTAL}"
fi
MANIFEST_FILE="${SCRIPT_DIR}/logs/manifests/${DSO_SHORT}${PART_TAG}_${TIMESTAMP}.txt"

# Compute maximum grids allowed in a single submission to stay <= MAX_TASKS
if [ "$MAX_TASKS" -lt 1 ] || [ "$GRIDS_PER_TASK" -lt 1 ]; then
  echo "ERROR: --max-tasks and --grids-per-task must both be >= 1." >&2
  exit 1
fi
MAX_GRIDS_PER_SUBMIT=$(( MAX_TASKS * GRIDS_PER_TASK ))

# Scan RAW_GRID_DIR & OUTPUT_DIR via Python os.scandir (fast on DSS), sort smallest .h5 first
SCAN_STATS=$(PYTHONDONTWRITEBYTECODE=1 python3 - \
  "$RAW_GRID_DIR" "$OUTPUT_DIR" "$DSO_SHORT" "$OUTPUT_SUFFIX" \
  "$PART_IDX" "$PART_TOTAL" "$MAX_GRIDS" "$MAX_GRIDS_PER_SUBMIT" "$OVERWRITE" "$MANIFEST_FILE" "$DRY_RUN" <<'PYEOF'
import math, os, sys

(
    raw_dir, out_dir, dso_short, out_suffix,
    part_idx, part_total, max_grids, max_grids_per_submit, overwrite, manifest_path, dry_run
) = (
    sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4],
    int(sys.argv[5]), int(sys.argv[6]), int(sys.argv[7]), int(sys.argv[8]),
    int(sys.argv[9]), sys.argv[10], int(sys.argv[11])
)

exclude_suffixes = ("_DmdAllocated.h5", "_Features.h5", "_Predicted.h5", "_direc_Inferred.h5", "_pwrflw.h5")

done_files = set()
if not overwrite and os.path.isdir(out_dir):
    with os.scandir(out_dir) as it:
        for entry in it:
            if entry.name.endswith(f"{out_suffix}.h5"):
                done_files.add(entry.name)

# Scan raw grids matching second underscore token == dso_short, recording file size
matched = []
with os.scandir(raw_dir) as it:
    for entry in it:
        name = entry.name
        if not name.endswith(".h5") or name.endswith(exclude_suffixes):
            continue
        parts = name.split("_")
        if len(parts) >= 2 and parts[1] == dso_short:
            try:
                fsize = entry.stat().st_size
            except OSError:
                fsize = 0
            matched.append((fsize, parts[0], os.path.splitext(name)[0]))

# Sort ascending by file size (smallest grids first), tie-break by filename for deterministic splits
matched.sort(key=lambda x: (x[0], x[2]))
total_dso = len(matched)

# Slice into deterministic part K/N before filtering already-done files so resuming part K is stable
chunk_size = math.ceil(total_dso / part_total) if total_dso > 0 else 0
start_i = (part_idx - 1) * chunk_size
end_i = min(total_dso, part_idx * chunk_size)
part_slice = matched[start_i:end_i]

pending_all = [
    (gid, stem) for _, gid, stem in part_slice
    if overwrite or (f"{stem}{out_suffix}.h5" not in done_files)
]
already_done = len(part_slice) - len(pending_all)

if max_grids > 0:
    pending_all = pending_all[:max_grids]

total_pending = len(pending_all)
wave_pending = pending_all[:max_grids_per_submit]
remaining_after_wave = total_pending - len(wave_pending)

if not dry_run and wave_pending:
    with open(manifest_path, "w", encoding="utf-8") as f:
        for gid, stem in wave_pending:
            f.write(f"{gid}\t{stem}\n")

preview = ",".join(gid for gid, _ in wave_pending[:10])
print(f"{total_dso} {len(part_slice)} {already_done} {total_pending} {len(wave_pending)} {remaining_after_wave} {preview}")
PYEOF
)

read -r TOTAL_DSO PART_MATCHED ALREADY_DONE TOTAL_PENDING WAVE_COUNT REMAINING_AFTER_WAVE PREVIEW_IDS <<< "$SCAN_STATS"

echo "========================================================================"
echo "Step 1 (Cheap Ground Truth) DSO Array Launcher"
echo "  DSO_SHORT       : $DSO_SHORT (part ${PART_SPEC})"
echo "  RAW_GRID_DIR    : $RAW_GRID_DIR"
echo "  OUTPUT_DIR      : $OUTPUT_DIR"
echo "  Sort order      : Smallest .h5 file size first"
echo "  Total in DSO    : $TOTAL_DSO"
echo "  In this part    : $PART_MATCHED"
echo "  Already done    : $ALREADY_DONE"
echo "  Total pending   : $TOTAL_PENDING"
echo "  This wave       : $WAVE_COUNT (cap: ${MAX_TASKS} tasks x ${GRIDS_PER_TASK} grids/task = ${MAX_GRIDS_PER_SUBMIT})"
if [ "$REMAINING_AFTER_WAVE" -gt 0 ]; then
  echo "  Remaining later : $REMAINING_AFTER_WAVE (re-run this exact command after this wave finishes!)"
fi
echo "========================================================================"

if [ "$WAVE_COUNT" -eq 0 ]; then
  echo "Nothing to submit for DSO '$DSO_SHORT' (part ${PART_SPEC})."
  exit 0
fi

# Compute array task count and walltime per task (D-HH:MM:SS)
NUM_TASKS=$(( (WAVE_COUNT + GRIDS_PER_TASK - 1) / GRIDS_PER_TASK ))
EFFECTIVE_GRIDS_PER_TASK="$GRIDS_PER_TASK"
if [ "$WAVE_COUNT" -lt "$GRIDS_PER_TASK" ]; then
  EFFECTIVE_GRIDS_PER_TASK="$WAVE_COUNT"
fi
TOTAL_MIN=$(( (EFFECTIVE_GRIDS_PER_TASK * TIME_PER_GRID_SEC + 59) / 60 ))
if [ "$TOTAL_MIN" -lt 15 ]; then
  TOTAL_MIN=15
fi

# Enforce walltime caps: <= 8h on cm4_inter, <= 12h everywhere else
MAX_ALLOWED_MIN=720
if [ "$PARTITION" = "cm4_inter" ]; then
  MAX_ALLOWED_MIN=480
fi
if [ "$TOTAL_MIN" -gt "$MAX_ALLOWED_MIN" ]; then
  echo "ERROR: Computed per-task walltime (${TOTAL_MIN} min) exceeds ${MAX_ALLOWED_MIN} min limit for ${PARTITION}." >&2
  echo "       Lower --grids-per-task or --time-per-grid." >&2
  exit 1
fi

DAYS=$(( TOTAL_MIN / 1440 ))
HOURS=$(( (TOTAL_MIN % 1440) / 60 ))
MINS=$(( TOTAL_MIN % 60 ))
WALLTIME=$(printf "%d-%02d:%02d:00" "$DAYS" "$HOURS" "$MINS")

JOB_NAME="cgt_${DSO_SHORT}${PART_TAG}"

SBATCH_CMD=(
  sbatch
  -J "$JOB_NAME"
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
  --export="ALL,MANIFEST_FILE=${MANIFEST_FILE},DSO_SHORT=${DSO_SHORT},PART_SPEC=${PART_SPEC},GRIDS_PER_TASK=${GRIDS_PER_TASK},RAW_GRID_DIR=${RAW_GRID_DIR},OUTPUT_DIR=${OUTPUT_DIR},OVERWRITE=${OVERWRITE}"
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
