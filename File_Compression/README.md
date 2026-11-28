# File_Compression

SLURM scripts for lossless `zstd` compression and decompression of `.h5` files on the LRZ Linux Cluster (DSS).

## Files

- `compress_h5.sh` — SLURM job script that compresses all `*.h5` files in a given directory (non-recursive), **smallest file first**, into `*.h5.zst` in the same folder.
- `decompress_h5.sh` — SLURM job script that restores all `*.h5.zst` files in a given directory back to `*.h5` in the same folder.
- `submit_compress.sh` / `submit_decompress.sh` — Convenience wrappers that validate the path, create `logs/`, and call `sbatch`.

## Usage (on LRZ login node)

```bash
cd File_Compression

# Compress all .h5 files in a folder (default zstd level 3)
./submit_compress.sh /dss/dssfs05/.../your_folder

# Compress with a custom level (e.g. level 9) or extra sbatch flags
./submit_compress.sh /dss/dssfs05/.../your_folder 9 --time=0-04:00:00

# Decompress all .h5.zst files in a folder back to .h5
./submit_decompress.sh /dss/dssfs05/.../your_folder
```

You can also call `sbatch` directly from inside `File_Compression/`:
```bash
sbatch compress_h5.sh /dss/dssfs05/.../your_folder 3
sbatch decompress_h5.sh /dss/dssfs05/.../your_folder
```

## Safety Behavior

- Processes **one file at a time**, starting from the **smallest** file so disk space is freed immediately.
- Writes to a `.tmp` file first, verifies integrity (`zstd -t` during compression), atomically renames to the final name, and **only then** deletes the source file.
- Safe to re-run if a job hits walltime or quota: already-processed files are skipped and any partial `.tmp` file is cleaned up automatically.
- Logs are written to `File_Compression/logs/normal/` and `File_Compression/logs/errors/` (empty error logs are automatically removed).
