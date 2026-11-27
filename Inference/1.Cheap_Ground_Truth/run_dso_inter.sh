#!/bin/bash
# ==============================================================================
# Self-submitting SLURM job-array runner for Step 1 (1.Cheap_Ground_Truth)
# on the LRZ Linux Cluster (CoolMUC-4).
#
# Why `cm4_tiny` is the default preset (per LRZ CoolMUC-4 docs):
#   - `inter / cm4_inter` is an interactive partition capped at 1-2 submitted
#     jobs per user (`AssocMaxSubmitJobLimit` if you submit multiple arrays!).
#   - `cm4 / cm4_tiny` (mandatory `--qos=cm4_tiny`) is LRZ's official single-node
#     parallel batch / task-farming partition: 17-112 cores/job, up to 4 running
#     jobs simultaneously, up to 24h walltime.
#   - With 56 CPUs/task and 22,000 grids/task, ALL 85,613 Bavarian grids fit
#     into ONLY 4 parallel tasks (~1-4h actual walltime, <=6h 00m requested)!
#
# Usage (run directly on the LRZ login node — do NOT wrap with `sbatch`):
#   ./run_dso_inter.sh ALL --dry-run               # preview ALL pending grids on cm4_tiny
#   ./run_dso_inter.sh ALL                         # submit ALL pending grids in <=4 tasks (56 CPUs each)
#   ./run_dso_inter.sh MFrak,RegB,SWA              # submit a custom comma-separated group of DSOs
#   ./run_dso_inter.sh SWU --limit 5               # smoke-test 5 smallest pending grids
#   ./run_dso_inter.sh ALL --preset cm4_inter      # run 1 single 56-CPU job on inter/cm4_inter (<=8h)
#
# Valid DSO_SHORT codes (ordered smallest -> largest), or `ALL`, or comma-separated list:
#   SWU    (356) | MFrak   (841) | RegB    (858) | SWA     (880) | AllgN (1,651)
#   SWM  (4,230) | Nrgie (6,446) | LEW   (7,011) | nVNB (24,469) | ByWrk (38,871)
#
# Notes:
# - Grids are sorted by raw .h5 file size ascending (smallest grids run first).
# - Already completed `*_DmdAllocated.h5` files are automatically skipped (use
#   `--overwrite` to force re-running existing outputs).
# - Each array task runs ONE persistent Python process with a ProcessPoolExecutor
#   of `CPUS_PER_TASK` workers (n_cpu=1 per grid).
# ==============================================================================

set -euo pipefail

# ---------------------------------------------------------------------------- #
# ------------------------- User Configuration ------------------------------- #
# ---------------------------------------------------------------------------- #
DSO_SHORT="${DSO_SHORT:-ALL}"
PRESET="${PRESET:-cm4_tiny}"   # cm4_tiny (default, 4x56 CPUs) | cm4_inter (1x56 CPUs, <=8h)

# Input raw grids folder and Step 1 output folder (matches pipeline_config.py)
RAW_GRID_DIR="${RAW_GRID_DIR:-/dss/dssfs05/lwp-dss-0003/pn98cu/pn98cu-dss-0001/PedroC/INFERENCE/0.Raw_grids}"
OUTPUT_DIR="${OUTPUT_DIR:-/dss/dssfs05/lwp-dss-0003/pn98cu/pn98cu-dss-0001/PedroC/INFERENCE/Step_by_Step/1.Cheap_Ground_Truthed}"
OUTPUT_SUFFIX="_DmdAllocated"

# SLURM cluster / resource overrides (empty => populated from PRESET below)
CLUSTER="${CLUSTER:-}"
PARTITION="${PARTITION:-}"
QOS="${QOS:-__PRESET__}"
CPUS_PER_TASK="${CPUS_PER_TASK:-}"
MEM="${MEM:-__PRESET__}"                       # Empty string omits --mem (uses LRZ ~2.1 GiB/core default)
CPU_SEC_PER_GRID="${CPU_SEC_PER_GRID:-55}"     # CPU-seconds budgeted per grid (4x SWU mean 13.7s; walltime = grids*cpu_sec/cpus)
TIME_PER_GRID_SEC_OVERRIDE=""                  # Optional explicit wall-seconds/grid override

# Array scaling overrides (empty => populated from PRESET below)
MAX_CONCURRENT="${MAX_CONCURRENT:-}"
MAX_TASKS="${MAX_TASKS:-}"
GRIDS_PER_TASK="${GRIDS_PER_TASK:-}"
MAX_GRIDS="${MAX_GRIDS:-0}"                    # 0 = all pending grids; >0 = cap for testing
PART_SPEC="${PART_SPEC:-1/1}"                  # Split matched grids into N parts, run part K (e.g. 1/2)
OVERWRITE="${OVERWRITE:-0}"                    # 0 = skip already done; 1 = re-run all
EST_MB_PER_GRID="${EST_MB_PER_GRID:-0}"        # 0 = auto-estimate from existing outputs (fallback: 25 MiB/grid)
CLEAN_PARTIALS="${CLEAN_PARTIALS:-0}"          # 1 = delete leftover partial .h5 files in OUTPUT_DIR before launch
SKIP_STORAGE_CHECK="${SKIP_STORAGE_CHECK:-0}"  # 1 = bypass pre-flight storage space abort

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
Usage: ./$(basename "$0") [DSO_SHORT|ALL|DSO1,DSO2,...] [options]

Presets (--preset NAME):
  cm4_tiny   (default) --clusters=cm4 --partition=cm4_tiny --qos=cm4_tiny
             56 CPUs/task, 22,000 grids/task, max 4 tasks (ALL 85,613 grids in 4 parallel jobs!)
  cm4_inter  --clusters=inter --partition=cm4_inter
             56 CPUs/task, 18,000 grids/task, max 1 task (fits inter's 1-2 job limit, <=8h)

Options:
  --preset NAME              Cluster/partition preset: cm4_tiny (default) or cm4_inter
  --part K/N                 Split matched grids into N equal chunks and run part K (default: 1/1)
  --dry-run                  Scan directories and print sbatch command without submitting
  --limit N                  Process at most N pending grids (default: 0 = all)
  --overwrite                Re-run grids even if *_DmdAllocated.h5 already exists
  --cpus N                   CPUs (parallel workers) per array task (preset default: 56)
  --mem SIZE                 SLURM --mem per task; empty omits --mem (default: "")
  --grids-per-task K         Grids processed per array task (preset default: 22000 or 18000)
  --max-tasks T              Max array tasks per submission (preset default: 4 or 1)
  --max-concurrent M         Max concurrent array tasks via SLURM %M (preset default: 4 or 1)
  --cpu-sec-per-grid SEC     CPU-seconds budgeted per grid (default: ${CPU_SEC_PER_GRID})
  --time-per-grid-sec SEC    Explicit walltime seconds per grid (overrides --cpu-sec-per-grid)
  --time-per-grid MIN        Explicit walltime minutes per grid
  --cluster NAME             Override SLURM cluster
  --partition NAME           Override SLURM partition
  --qos NAME                 Override SLURM QoS (empty string omits --qos)
  --input-dir PATH           Override RAW_GRID_DIR
  --output-dir PATH          Override OUTPUT_DIR
  --est-mb-per-grid MB       Estimated MiB per completed grid for storage check (0 = auto)
  --clean-partials           Remove leftover partial .h5 files in OUTPUT_DIR before submitting
  --skip-storage-check       Warn instead of aborting if estimated output exceeds free space
  -h, --help                 Show this help message
EOF
}

while [ "$#" -gt 0 ]; do
  case "$1" in
    --preset)
      PRESET="$2"; shift 2 ;;
    --part)
      PART_SPEC="$2"; shift 2 ;;
    --dry-run)
      DRY_RUN=1; shift ;;
    --limit)
      MAX_GRIDS="$2"; shift 2 ;;
    --overwrite)
      OVERWRITE=1; shift ;;
    --cpus)
      CPUS_PER_TASK="$2"; shift 2 ;;
    --mem)
      MEM="$2"; shift 2 ;;
    --grids-per-task)
      GRIDS_PER_TASK="$2"; shift 2 ;;
    --max-tasks)
      MAX_TASKS="$2"; shift 2 ;;
    --max-concurrent)
      MAX_CONCURRENT="$2"; shift 2 ;;
    --cpu-sec-per-grid)
      CPU_SEC_PER_GRID="$2"; shift 2 ;;
    --time-per-grid-sec)
      TIME_PER_GRID_SEC_OVERRIDE="$2"; shift 2 ;;
    --time-per-grid)
      TIME_PER_GRID_SEC_OVERRIDE=$(( $2 * 60 )); shift 2 ;;
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
    --est-mb-per-grid)
      EST_MB_PER_GRID="$2"; shift 2 ;;
    --clean-partials)
      CLEAN_PARTIALS=1; shift ;;
    --skip-storage-check)
      SKIP_STORAGE_CHECK=1; shift ;;
    -h|--help)
      usage; exit 0 ;;
    -*)
      echo "ERROR: Unknown option '$1'" >&2; usage; exit 1 ;;
    *)
      DSO_SHORT="$1"; shift ;;
  esac
done

# Populate defaults from selected LRZ preset for any unset resource parameters
case "$PRESET" in
  cm4_tiny)
    CLUSTER="${CLUSTER:-cm4}"
    PARTITION="${PARTITION:-cm4_tiny}"
    [ "$QOS" = "__PRESET__" ] && QOS="cm4_tiny"
    CPUS_PER_TASK="${CPUS_PER_TASK:-56}"
    [ "$MEM" = "__PRESET__" ] && MEM=""
    GRIDS_PER_TASK="${GRIDS_PER_TASK:-22000}"
    MAX_TASKS="${MAX_TASKS:-4}"
    MAX_CONCURRENT="${MAX_CONCURRENT:-4}"
    ;;
  cm4_inter)
    CLUSTER="${CLUSTER:-inter}"
    PARTITION="${PARTITION:-cm4_inter}"
    [ "$QOS" = "__PRESET__" ] && QOS=""
    CPUS_PER_TASK="${CPUS_PER_TASK:-56}"
    [ "$MEM" = "__PRESET__" ] && MEM=""
    GRIDS_PER_TASK="${GRIDS_PER_TASK:-18000}"
    MAX_TASKS="${MAX_TASKS:-1}"
    MAX_CONCURRENT="${MAX_CONCURRENT:-1}"
    ;;
  *)
    echo "ERROR: Unknown --preset '$PRESET' (expected 'cm4_tiny' or 'cm4_inter')." >&2
    exit 1
    ;;
esac

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
SAFE_DSO_TAG="${DSO_SHORT//,/_}"
MANIFEST_FILE="${SCRIPT_DIR}/logs/manifests/${SAFE_DSO_TAG}${PART_TAG}_${TIMESTAMP}.txt"

# Compute maximum grids allowed in a single submission to stay <= MAX_TASKS
if [ "$MAX_TASKS" -lt 1 ] || [ "$GRIDS_PER_TASK" -lt 1 ]; then
  echo "ERROR: --max-tasks and --grids-per-task must both be >= 1." >&2
  exit 1
fi
MAX_GRIDS_PER_SUBMIT=$(( MAX_TASKS * GRIDS_PER_TASK ))

SITE_DATA_FILE="${SCRIPT_DIR}/gridalloc/data/statistics/general/site_data.txt"

# Scan RAW_GRID_DIR & OUTPUT_DIR via Python os.scandir (fast on DSS), sort smallest .h5 first,
# and run pre-flight checks (storage space, leftover partial .h5 files, and site_data.txt PLZ coverage).
SCAN_STATS=$(PYTHONDONTWRITEBYTECODE=1 python3 - \
  "$RAW_GRID_DIR" "$OUTPUT_DIR" "$DSO_SHORT" "$OUTPUT_SUFFIX" \
  "$PART_IDX" "$PART_TOTAL" "$MAX_GRIDS" "$MAX_GRIDS_PER_SUBMIT" "$OVERWRITE" \
  "$MANIFEST_FILE" "$DRY_RUN" "$EST_MB_PER_GRID" "$CLEAN_PARTIALS" "$SITE_DATA_FILE" <<'PYEOF'
import math, os, shutil, sys

(
    raw_dir, out_dir, dso_short, out_suffix,
    part_idx, part_total, max_grids, max_grids_per_submit, overwrite,
    manifest_path, dry_run, est_mb_override, clean_partials, site_data_file
) = (
    sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4],
    int(sys.argv[5]), int(sys.argv[6]), int(sys.argv[7]), int(sys.argv[8]),
    int(sys.argv[9]), sys.argv[10], int(sys.argv[11]), float(sys.argv[12]),
    int(sys.argv[13]), sys.argv[14]
)

exclude_suffixes = ("_DmdAllocated.h5", "_Features.h5", "_Predicted.h5", "_direc_Inferred.h5", "_pwrflw.h5")
target_suffix = f"{out_suffix}.h5"

done_files = set()
partial_files = []
sampled_out_sizes = []
if os.path.isdir(out_dir):
    with os.scandir(out_dir) as it:
        for entry in it:
            name = entry.name
            if not name.endswith(".h5"):
                continue
            if name.endswith(target_suffix):
                if not overwrite:
                    done_files.add(name)
                if est_mb_override <= 0 and len(sampled_out_sizes) < 200:
                    try:
                        sampled_out_sizes.append(entry.stat().st_size)
                    except OSError:
                        pass
            elif not name.endswith(exclude_suffixes):
                partial_files.append(name)

partial_removed = 0
if clean_partials and partial_files and not dry_run:
    for pname in partial_files:
        try:
            os.remove(os.path.join(out_dir, pname))
            partial_removed += 1
        except OSError:
            pass

# Scan raw grids matching DSO_SHORT (`ALL`, single DSO, or comma-separated DSOs), recording file size
dso_filter = None if dso_short.upper() == "ALL" else {s.strip() for s in dso_short.split(",") if s.strip()}
matched = []
with os.scandir(raw_dir) as it:
    for entry in it:
        name = entry.name
        if not name.endswith(".h5") or name.endswith(exclude_suffixes):
            continue
        parts = name.split("_")
        if dso_filter is None or (len(parts) >= 2 and parts[1] in dso_filter):
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

# Storage estimate & available disk/quota space on OUTPUT_DIR (or nearest existing parent)
if est_mb_override > 0:
    est_mb = est_mb_override
    est_src = "override"
elif len(sampled_out_sizes) >= 10:
    est_mb = (sum(sampled_out_sizes) / len(sampled_out_sizes)) / (1024.0 * 1024.0)
    est_src = f"avg of {len(sampled_out_sizes)} existing outputs"
else:
    est_mb = 25.0
    est_src = "default"

est_bytes = int(len(wave_pending) * est_mb * 1024.0 * 1024.0)
probe_dir = out_dir
while probe_dir and not os.path.exists(probe_dir):
    parent = os.path.dirname(probe_dir)
    if parent == probe_dir:
        break
    probe_dir = parent
try:
    avail_bytes = shutil.disk_usage(probe_dir).free
except OSError:
    avail_bytes = -1

# Pre-flight PLZ coverage check against site_data.txt
missing_plzs = {}
if os.path.isfile(site_data_file) and wave_pending:
    known_plzs = set()
    with open(site_data_file, "r", encoding="utf-8", errors="ignore") as sf:
        header = sf.readline()
        for line in sf:
            line = line.strip()
            if not line:
                continue
            known_plzs.add(line.split("\t", 1)[0].strip().zfill(5))
    for _, stem in wave_pending:
        parts = stem.split("_")
        # Filename format: <id>_<dso>_<coords>_<plz>_<kcid>_<bcid>
        if len(parts) >= 4 and len(parts[3]) == 5 and parts[3].isdigit():
            plz = parts[3]
            if plz not in known_plzs:
                missing_plzs[plz] = missing_plzs.get(plz, 0) + 1

if not dry_run and wave_pending:
    with open(manifest_path, "w", encoding="utf-8") as f:
        for gid, stem in wave_pending:
            f.write(f"{gid}\t{stem}\n")

preview = ",".join(gid for gid, _ in wave_pending[:10]) or "none"
partial_sample = ",".join(sorted(partial_files)[:5]) or "none"
missing_plz_count = sum(missing_plzs.values())
missing_plz_list = ",".join(f"{k}({v})" for k, v in sorted(missing_plzs.items())[:10]) or "none"

print("\t".join([
    str(total_dso),
    str(len(part_slice)),
    str(already_done),
    str(total_pending),
    str(len(wave_pending)),
    str(remaining_after_wave),
    preview,
    str(len(partial_files)),
    str(partial_removed),
    partial_sample,
    f"{est_mb:.1f}",
    est_src,
    str(est_bytes),
    str(avail_bytes),
    str(missing_plz_count),
    missing_plz_list,
]))
PYEOF
)

IFS=$'\t' read -r \
  TOTAL_DSO PART_MATCHED ALREADY_DONE TOTAL_PENDING WAVE_COUNT REMAINING_AFTER_WAVE PREVIEW_IDS \
  PARTIAL_COUNT PARTIAL_REMOVED PARTIAL_SAMPLE EST_MB_USED EST_MB_SRC EST_BYTES AVAIL_BYTES \
  MISSING_PLZ_GRIDS MISSING_PLZ_LIST <<< "$SCAN_STATS"

EST_GIB=$(awk -v b="$EST_BYTES" 'BEGIN { printf "%.2f", b / 1073741824 }')
if [ "$AVAIL_BYTES" -ge 0 ]; then
  AVAIL_GIB=$(awk -v b="$AVAIL_BYTES" 'BEGIN { printf "%.2f", b / 1073741824 }')
  AVAIL_DISPLAY="${AVAIL_GIB} GiB"
else
  AVAIL_DISPLAY="unknown"
fi

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
echo "  Est. storage    : ~${EST_GIB} GiB (~${EST_MB_USED} MiB/grid, ${EST_MB_SRC}) | Free: ${AVAIL_DISPLAY}"
if [ "$PARTIAL_COUNT" -gt 0 ]; then
  if [ "$CLEAN_PARTIALS" -eq 1 ] && [ "$DRY_RUN" -eq 0 ]; then
    echo "  Partial .h5     : Cleaned ${PARTIAL_REMOVED}/${PARTIAL_COUNT} leftover partial file(s)"
  else
    echo "  Partial .h5     : WARNING: ${PARTIAL_COUNT} leftover partial .h5 file(s) in OUTPUT_DIR (e.g. ${PARTIAL_SAMPLE})"
    echo "                    Re-run with --clean-partials or run:"
    echo "                    find \"${OUTPUT_DIR}\" -maxdepth 1 -name '*.h5' ! -name '*${OUTPUT_SUFFIX}.h5' -delete"
  fi
else
  echo "  Partial .h5     : none"
fi
if [ "$MISSING_PLZ_GRIDS" -gt 0 ]; then
  echo "  Missing PLZs    : WARNING: ${MISSING_PLZ_GRIDS} grid(s) use PLZ(s) absent from site_data.txt: ${MISSING_PLZ_LIST}"
else
  echo "  Missing PLZs    : none (all wave PLZs present in site_data.txt)"
fi
echo "========================================================================"

if [ "$WAVE_COUNT" -eq 0 ]; then
  echo "Nothing to submit for DSO '$DSO_SHORT' (part ${PART_SPEC})."
  exit 0
fi

# Pre-flight storage check: abort submission if estimated output exceeds available space
if [ "$AVAIL_BYTES" -ge 0 ] && [ "$EST_BYTES" -gt "$AVAIL_BYTES" ]; then
  if [ "$DRY_RUN" -eq 1 ] || [ "$SKIP_STORAGE_CHECK" -eq 1 ]; then
    echo "WARNING: Estimated storage (~${EST_GIB} GiB) exceeds available space (${AVAIL_GIB} GiB) on OUTPUT_DIR!" >&2
  else
    rm -f "$MANIFEST_FILE"
    echo "ERROR: Estimated storage (~${EST_GIB} GiB for ${WAVE_COUNT} grids @ ${EST_MB_USED} MiB/grid)" >&2
    echo "       exceeds available space (${AVAIL_GIB} GiB) on ${OUTPUT_DIR}." >&2
    echo "       Free disk space, lower --limit / --grids-per-task, override --est-mb-per-grid, or pass --skip-storage-check." >&2
    exit 1
  fi
elif [ "$AVAIL_BYTES" -gt 0 ] && awk -v e="$EST_BYTES" -v a="$AVAIL_BYTES" 'BEGIN { exit !(e > 0.85 * a) }'; then
  echo "WARNING: Estimated storage (~${EST_GIB} GiB) uses >85% of available space (${AVAIL_GIB} GiB) on OUTPUT_DIR." >&2
fi

# Compute array task count and walltime per task (D-HH:MM:SS)
NUM_TASKS=$(( (WAVE_COUNT + GRIDS_PER_TASK - 1) / GRIDS_PER_TASK ))
EFFECTIVE_GRIDS_PER_TASK="$GRIDS_PER_TASK"
if [ "$WAVE_COUNT" -lt "$GRIDS_PER_TASK" ]; then
  EFFECTIVE_GRIDS_PER_TASK="$WAVE_COUNT"
fi
if [ -n "$TIME_PER_GRID_SEC_OVERRIDE" ]; then
  TOTAL_MIN=$(( (EFFECTIVE_GRIDS_PER_TASK * TIME_PER_GRID_SEC_OVERRIDE + 59) / 60 ))
else
  # Scale walltime automatically with CPUS_PER_TASK (wall_sec = grids * cpu_sec_per_grid / cpus)
  TOTAL_MIN=$(( (EFFECTIVE_GRIDS_PER_TASK * CPU_SEC_PER_GRID + CPUS_PER_TASK * 60 - 1) / (CPUS_PER_TASK * 60) ))
fi
if [ "$TOTAL_MIN" -lt 15 ]; then
  TOTAL_MIN=15
fi

# Enforce walltime caps: <= 8h on cm4_inter, <= 12h everywhere else (e.g. cm4_tiny allows 24h)
MAX_ALLOWED_MIN=720
if [ "$PARTITION" = "cm4_inter" ]; then
  MAX_ALLOWED_MIN=480
fi
if [ "$TOTAL_MIN" -gt "$MAX_ALLOWED_MIN" ]; then
  echo "ERROR: Computed per-task walltime (${TOTAL_MIN} min) exceeds ${MAX_ALLOWED_MIN} min limit for ${PARTITION}." >&2
  echo "       Lower --grids-per-task or --cpu-sec-per-grid, or increase --cpus." >&2
  exit 1
fi

DAYS=$(( TOTAL_MIN / 1440 ))
HOURS=$(( (TOTAL_MIN % 1440) / 60 ))
MINS=$(( TOTAL_MIN % 60 ))
WALLTIME=$(printf "%d-%02d:%02d:00" "$DAYS" "$HOURS" "$MINS")

JOB_NAME="cgt_${SAFE_DSO_TAG:0:18}${PART_TAG}"

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
)
if [ -n "$MEM" ]; then
  SBATCH_CMD+=(--mem="$MEM")
fi
SBATCH_CMD+=(
  --time="$WALLTIME"
  --array="1-${NUM_TASKS}%${MAX_CONCURRENT}"
  --output="logs/normal/%A_%a_output.log"
  --error="logs/errors/%A_%a_error.log"
  --export="ALL,MANIFEST_FILE=${MANIFEST_FILE},DSO_SHORT=${SAFE_DSO_TAG},PART_SPEC=${PART_SPEC},GRIDS_PER_TASK=${GRIDS_PER_TASK},RAW_GRID_DIR=${RAW_GRID_DIR},OUTPUT_DIR=${OUTPUT_DIR},OVERWRITE=${OVERWRITE}"
  "$SCRIPT_PATH"
)

MEM_DISPLAY="${MEM:-default (~2.1G/core)}"
echo "  Array tasks     : 1-${NUM_TASKS} (max ${MAX_CONCURRENT} concurrent, ${GRIDS_PER_TASK} grid(s)/task)"
echo "  Per-task res.   : ${CPUS_PER_TASK} CPUs, RAM ${MEM_DISPLAY}, walltime ${WALLTIME} (${CLUSTER}/${PARTITION}${QOS:+, qos=${QOS}})"
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
