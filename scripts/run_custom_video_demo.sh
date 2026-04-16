#!/bin/bash
set -euo pipefail

if [ "$#" -ne 1 ]; then
  echo "Usage: bash scripts/run_custom_video_demo.sh INPUT_DIR"
  exit 1
fi

input_dir="$1"

seq_name="$(
python - "$input_dir" <<'PY'
from pathlib import Path
import sys

input_dir = Path(sys.argv[1])
video_files = sorted(input_dir.glob("*.color.mp4"))
if len(video_files) != 1:
    raise SystemExit(f"Expected exactly one *.color.mp4 in {input_dir}, found {len(video_files)}")

name = video_files[0].name
suffix = ".color.mp4"
if not name.endswith(suffix):
    raise SystemExit(f"Unexpected input video name: {name}")

base = name[:-len(suffix)]
parts = base.split(".")
if len(parts) > 1 and parts[-1].isdigit():
    base = ".".join(parts[:-1])

print(base)
PY
)"

echo "Resolved custom sequence name: ${seq_name}"
echo "Converting custom video package from: ${input_dir}"
bash scripts/convert_custom_video_demo.sh "${input_dir}" "${seq_name}"

echo "Running VisTracker custom demo on: demo_data/${seq_name}"
bash scripts/demo_custom.sh "demo_data/${seq_name}"
