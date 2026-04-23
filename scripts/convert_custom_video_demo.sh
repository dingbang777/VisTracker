#!/bin/bash
set -euo pipefail

if [ "$#" -lt 1 ] || [ "$#" -gt 2 ]; then
  echo "Usage: bash scripts/convert_custom_video_demo.sh INPUT_DIR [SEQ_NAME]"
  exit 1
fi

input_dir="$1"
seq_name="${2:-}"

cmd=(
  python
  tools/convert_custom_video_to_vistracker.py
  --input-dir "$input_dir"
  --output-root demo_data
  --force
)

if [ -n "$seq_name" ]; then
  cmd+=(--sequence-name "$seq_name")
fi

"${cmd[@]}"
