# Sourced by scripts/*.sh. Loads configs/gpus.env and derives everything GPU-related from it:
#   GPUS           comma list, e.g. 0,1,2,3      GPU_LIST  bash array of the same
#   N_GPUS         how many                      RETRIEVAL_GPU
#   cpus_for LIST  CPU list on the NUMA node(s) of those GPUs, for taskset
#   ncpus_for LIST how many CPUs that is
#   server_running true if retrieval_server.py is up

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# shellcheck source=../configs/gpus.env
source "$ROOT/configs/gpus.env"

if [[ ! $GPUS =~ ^[0-9]+(,[0-9]+)*$ ]]; then
    echo "configs/gpus.env: GPUS='$GPUS' must look like 0,1,2,3" >&2
    exit 1
fi
IFS=, read -ra GPU_LIST <<<"$GPUS"
N_GPUS=${#GPU_LIST[@]}
RETRIEVAL_GPU=${RETRIEVAL_GPU:-${GPU_LIST[-1]}}
if [[ ,$GPUS, != *,$RETRIEVAL_GPU,* ]]; then
    echo "configs/gpus.env: RETRIEVAL_GPU=$RETRIEVAL_GPU is not in GPUS=$GPUS" >&2
    exit 1
fi

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
