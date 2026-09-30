#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$0")/env.sh"
mkdir -p "$HW1_CODE/.tmp/probe" "$HOME/hw1_kenlm_probe"
head -n 1000 "$HW1_CODE/results/methods/train_sample.txt" > "$HW1_CODE/.tmp/probe/tiny.txt"
set +e
"$HW1_TOOLS/kenlm/bin/lmplz" -o 2 -S 1G -T "$HW1_CODE/.tmp/probe" --text "$HW1_CODE/.tmp/probe/tiny.txt" --arpa "$HW1_CODE/.tmp/probe/drive.arpa" > "$HW1_CODE/.tmp/probe/drive.stdout" 2> "$HW1_CODE/.tmp/probe/drive.stderr"
echo "drive_exit=$?"
cp "$HW1_CODE/.tmp/probe/tiny.txt" "$HOME/hw1_kenlm_probe/tiny.txt"
"$HW1_TOOLS/kenlm/bin/lmplz" -o 2 -S 1G -T "$HOME/hw1_kenlm_probe" --text "$HOME/hw1_kenlm_probe/tiny.txt" --arpa "$HOME/hw1_kenlm_probe/ext4.arpa" > "$HW1_CODE/.tmp/probe/ext4.stdout" 2> "$HW1_CODE/.tmp/probe/ext4.stderr"
echo "ext4_exit=$?"
