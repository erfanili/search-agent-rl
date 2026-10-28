# Sourced by scripts/*.sh (through gpus.sh). Sets where all data lives:
#   DATA           corpus, embeddings, index, eval sets, training data, run logs, HF/vLLM/Ray caches
# Resolution order: $DATA_ROOT from the environment, then DATA_ROOT=... in configs/paths.local.env
# (gitignored, per machine), then <repo>/data. agent/paths.py resolves it the same way for Python.

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
if [[ -z ${DATA_ROOT:-} && -f $ROOT/configs/paths.local.env ]]; then
    # shellcheck source=/dev/null
    source "$ROOT/configs/paths.local.env"
fi
export DATA_ROOT=${DATA_ROOT:-$ROOT/data}
DATA=$DATA_ROOT

# Latest snapshot dir of a hub repo in $HF_HOME (default $DATA/hf_cache), e.g. hf_snapshot models Qwen/Qwen2.5-3B
hf_snapshot() {
    local d
    d=$(ls -d "${HF_HOME:-$DATA/hf_cache}/hub/$1--${2//\//--}/snapshots/"* 2>/dev/null | tail -1)
    echo "${d%/}"
}
