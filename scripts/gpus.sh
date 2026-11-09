# Sourced by scripts/*.sh. Loads configs/gpus.env and derives everything GPU-related from it:
#   GPUS           comma list, e.g. 0,1,2,3      GPU_LIST  bash array of the same
#   N_GPUS         how many                      RETRIEVAL_GPU  a GPU index or cpu
#   cpus_for LIST  CPU list on the NUMA node(s) of those GPUs, for taskset
#   ncpus_for LIST how many CPUs that is
#   server_running true if retrieval_server.py is up
#   WORK_GPUS      GPU_LIST minus RETRIEVAL_GPU: the GPUs training and inference may use
#   require_retrieval_server   exit unless the server answers /health
#   verl_env       environment every verl run needs on this machine

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# shellcheck source=../configs/gpus.env
source "$ROOT/configs/gpus.env"
# shellcheck source=paths.sh
source "$ROOT/scripts/paths.sh"   # DATA and hf_snapshot

if [[ ! $GPUS =~ ^[0-9]+(,[0-9]+)*$ ]]; then
    echo "configs/gpus.env: GPUS='$GPUS' must look like 0,1,2,3" >&2
    exit 1
fi
IFS=, read -ra GPU_LIST <<<"$GPUS"
N_GPUS=${#GPU_LIST[@]}
RETRIEVAL_GPU=${RETRIEVAL_GPU:-${GPU_LIST[-1]}}
if [[ $RETRIEVAL_GPU != cpu && ,$GPUS, != *,$RETRIEVAL_GPU,* ]]; then
    echo "configs/gpus.env: RETRIEVAL_GPU=$RETRIEVAL_GPU is not in GPUS=$GPUS (or cpu)" >&2
    exit 1
fi

WORK_GPUS=()
for _g in "${GPU_LIST[@]}"; do [[ $_g == "$RETRIEVAL_GPU" ]] || WORK_GPUS+=("$_g"); done

# Machine fact (notes/compute.md): GPUs 0-3 sit on NUMA node 0 (CPUs 0-23,48-71),
# GPUs 4-7 on node 1 (CPUs 24-47,72-95). numactl isn't installed, so scripts pin with taskset.
cpus_for() {
    local lo=0 hi=0 g
    IFS=, read -ra _gs <<<"$1"
    for g in "${_gs[@]}"; do
        if ((g < 4)); then lo=1; else hi=1; fi
    done
    if ((lo && hi)); then echo "0-95"
    elif ((lo)); then echo "0-23,48-71"
    else echo "24-47,72-95"; fi
}

ncpus_for() {
    if [[ $(cpus_for "$1") == 0-95 ]]; then echo 96; else echo 48; fi
}

server_running() {
    pgrep -f "python[^ ]* [^ ]*retrieval_server.py" >/dev/null
}

require_retrieval_server() {
    curl -sf localhost:8000/health >/dev/null || { echo "retrieval server not up: bash scripts/serve_retrieval.sh" >&2; exit 1; }
}

verl_env() {
    export RAY_TMPDIR=${RAY_TMPDIR:-$DATA/rt}        # short name: Ray AF_UNIX socket paths must be <=107 bytes
    export NCCL_P2P_DISABLE=${NCCL_P2P_DISABLE:-1}   # P2P hangs on this machine (notes/smoke_test.md)
    export VLLM_USE_V1=1   # verl async server needs vLLM V1; unset, vLLM falls back to V0 (engine built off the main thread)
    export PYTHONPATH=$ROOT${PYTHONPATH:+:$PYTHONPATH}  # Ray workers import agent/ from the repo root
    mkdir -p "$RAY_TMPDIR"
}
