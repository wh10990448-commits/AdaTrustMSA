#!/usr/bin/env bash
set -euo pipefail

destination="${1:-data/mosei_csd}"
minimum_bytes=$((20 * 1024 * 1024 * 1024))
mkdir -p "$destination"

available_bytes=$(df -PB1 "$destination" | awk 'NR==2 {print $4}')
if (( available_bytes < minimum_bytes )) && [[ "${ALLOW_LOW_DISK:-0}" != "1" ]]; then
  echo "Refusing download: df reports only ${available_bytes} bytes available." >&2
  echo "Free at least 20 GiB, or set ALLOW_LOW_DISK=1 only after checking root-reserved blocks." >&2
  exit 2
fi

if command -v hf >/dev/null 2>&1; then
  downloader=(hf download)
elif command -v huggingface-cli >/dev/null 2>&1; then
  downloader=(huggingface-cli download)
else
  echo "Install huggingface_hub first: python -m pip install huggingface_hub" >&2
  exit 3
fi

endpoint="${HF_ENDPOINT:-https://huggingface.co}"
HF_ENDPOINT="$endpoint" "${downloader[@]}" reeha-parkar/cmu-mosei-comp-seq \
  --repo-type dataset \
  --include \
    'data/CMU_MOSEI_Labels.csd' \
    'data/CMU_MOSEI_TimestampedWords.csd' \
    'data/CMU_MOSEI_COVAREP.csd' \
    'data/CMU_MOSEI_VisualFacet42.csd' \
  --local-dir "$destination"

cd "$destination"
sha256sum -c <<'EOF'
e0242228010773d8cf6cff660229126da5746f02545429abca0df93ffc36013b  data/CMU_MOSEI_COVAREP.csd
fba2589e91b6f325e82a524e0ce958efd565715f9aec1a490c4a542f5073fb4e  data/CMU_MOSEI_Labels.csd
8cc97323b100d9147fbfb7de9d66114443668b652810c56337b17128117a678f  data/CMU_MOSEI_TimestampedWords.csd
a42fc9eb6f2f010ea22697dae3e5490d2286871d6d353f685df2cc5498387cd3  data/CMU_MOSEI_VisualFacet42.csd
EOF

echo "Verified CMU-MOSEI computational sequences in $(pwd)/data"
