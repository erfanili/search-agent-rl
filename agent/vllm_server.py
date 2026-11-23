"""verl async vLLM server variants (E005 garble diagnosis, 2026-11-22).

verl 0.5.0's AsyncvLLMServer hard-codes enable_prefix_caching=True. AsyncvLLMServerNoPrefixCache is the same server
with prefix caching off and nothing else changed (it edits the engine args just before verl builds the vLLM config).
Use: actor_rollout_ref.rollout.agent.custom_async_server.path=pkg://agent.vllm_server
     actor_rollout_ref.rollout.agent.custom_async_server.name=AsyncvLLMServerNoPrefixCache
(scripts/train_grpo_3b.sh: PREFIX_CACHING=False).
"""
import ray
from verl.workers.rollout.vllm_rollout.vllm_async_server import AsyncvLLMServer as _ActorClass

_Base = _ActorClass.__ray_metadata__.modified_class.__mro__[1]  # the undecorated AsyncvLLMServer
assert _Base.__name__ == "AsyncvLLMServer", _Base


@ray.remote(num_cpus=1)
class AsyncvLLMServerNoPrefixCache(_Base):
    def _create_engine_config(self, engine_args):
        engine_args.enable_prefix_caching = False
        print("AsyncvLLMServerNoPrefixCache: enable_prefix_caching=False", flush=True)
        return super()._create_engine_config(engine_args)
