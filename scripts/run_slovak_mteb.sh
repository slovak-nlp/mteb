#!/usr/bin/env bash
set -euo pipefail

# Simple helper to rerun the Slovak subset of MTEB for a fixed list of models.
# Usage:
#   bash scripts/run_slovak_mteb.sh
# Environment overrides:
#   BENCHMARK="MTEB(slk, v1)"  # default benchmark identifier
#   BATCH_SIZE=1               # passed to --batch-size
#   OVERWRITE_STRATEGY=always  # passed to --overwrite-strategy
#   EXTRA_ARGS="--device cuda" # appended to every mteb invocation, optional

readonly BENCHMARK="${BENCHMARK:-MTEB(slk, v1)}"
readonly BATCH_SIZE="${BATCH_SIZE:-1}"
readonly OVERWRITE_STRATEGY="${OVERWRITE_STRATEGY:-only-missing}"
readonly EXTRA_ARGS="${EXTRA_ARGS:-}"

timestamp="$(date -u +"%Y%m%dT%H%M%SZ")"
log_dir="results/slk_runs"
mkdir -p "${log_dir}"
summary_file="${log_dir}/run_${timestamp}.log"

cat <<EOF | tee "${summary_file}"
Slovak MTEB rerun
Benchmark   : ${BENCHMARK}
Batch size  : ${BATCH_SIZE}
Overwrite   : ${OVERWRITE_STRATEGY}
Extra args  : ${EXTRA_ARGS:-<none>}
Timestamp   : ${timestamp} (UTC)
Log file    : ${summary_file}
EOF

declare -a MODELS=(
  "multilingual-e5-large-instruct=intfloat/multilingual-e5-large-instruct"
  "jina-embeddings-v3=jinaai/jina-embeddings-v3"
  "multilingual-e5-large=intfloat/multilingual-e5-large"
  "bge-m3=BAAI/bge-m3"
  "nomic-embed-text-v2-moe=nomic-ai/nomic-embed-text-v2-moe"
  "multilingual-e5-base=intfloat/multilingual-e5-base"
  "jina-embeddings-v4=jinaai/jina-embeddings-v4"
  "gte-multilingual-base=Alibaba-NLP/gte-multilingual-base"
  "multilingual-e5-small=intfloat/multilingual-e5-small"
  "Qwen3-Embedding-0.6B=Qwen/Qwen3-Embedding-0.6B"
  "paraphrase-multilingual-mpnet-base-v2=sentence-transformers/paraphrase-multilingual-mpnet-base-v2"
  "embeddinggemma-300m=google/embeddinggemma-300m"
  "paraphrase-multilingual-MiniLM-L12-v2=sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
  "LaBSE=sentence-transformers/LaBSE"
  "granite-embedding-278m-multilingual=ibm-granite/granite-embedding-278m-multilingual"
  "slovakbert-sts-stsb=kinit/slovakbert-sts-stsb"
  "granite-embedding-107m-multilingual=ibm-granite/granite-embedding-107m-multilingual"
  "slovakbert-skquad-mnlr=TUKE-DeutscheTelekom/slovakbert-skquad-mnlr"
  "static-similarity-mrl-multilingual-v1=sentence-transformers/static-similarity-mrl-multilingual-v1"
  "nomic-embed-text-v1.5=nomic-ai/nomic-embed-text-v1.5"
)

echo -e "\nStarting runs...\n" | tee -a "${summary_file}"

failures=0

for entry in "${MODELS[@]}"; do
  label="${entry%%=*}"
  hf_model="${entry#*=}"

  {
    printf '===== %s (%s) =====\n' "${label}" "${hf_model}"
    printf 'Command: mteb run -b %q -m %q --batch-size %s --overwrite-strategy %s %s\n' \
      "${BENCHMARK}" "${hf_model}" "${BATCH_SIZE}" "${OVERWRITE_STRATEGY}" "${EXTRA_ARGS}"
  } | tee -a "${summary_file}"

  if mteb run \
      -b "${BENCHMARK}" \
      -m "${hf_model}" \
      --batch-size "${BATCH_SIZE}" \
      --overwrite-strategy "${OVERWRITE_STRATEGY}" \
      ${EXTRA_ARGS} 2>&1 | tee -a "${summary_file}"; then
    echo "Status: SUCCESS (${label})" | tee -a "${summary_file}"
  else
    echo "Status: FAILURE (${label})" | tee -a "${summary_file}"
    failures=$((failures + 1))
  fi

  echo | tee -a "${summary_file}"
done

if [[ "${failures}" -eq 0 ]]; then
  echo "All runs completed successfully." | tee -a "${summary_file}"
else
  echo "${failures} model(s) failed. Check the log above for details." | tee -a "${summary_file}"
fi
