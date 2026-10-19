"""
verl v0.5.0 agent loop that runs Search-R1 episodes (P1.3).

How it plugs in (verl's async rollout, "agent loop" mode):
  actor_rollout_ref.rollout.name=vllm
  actor_rollout_ref.rollout.mode=async                 -> AgentLoopManager + vLLM async servers
  actor_rollout_ref.rollout.agent.agent_loop_config_path=configs/agent_loop.yaml
  data.return_raw_chat=True                            -> each row's "prompt" messages reach run()
and every training row has agent_name="search_r1_agent" (agent/make_check_data.py writes it).

configs/agent_loop.yaml names this class and passes its settings (retrieval URL, turns, top-k,
dump_dir) as keyword arguments; verl hands them to init_class once per worker process.

verl calls run() once per sampled trajectory and trains on what it returns:
  prompt_ids     the prompt, tokenized once
  response_ids   model and env segments, concatenated
  response_mask  1 on model tokens, 0 on env tokens: the policy loss, KL and entropy use only the 1s

Runs inside verl's Ray workers, so the repo root must be on PYTHONPATH (the launch scripts set it).
Talks to the retrieval server over HTTP and imports nothing from retrieval/ (notes/environments.md).
"""

import json
import logging
import os
import time
from typing import Any
from uuid import uuid4

import requests

from agent.search_r1_rollout import STOP_STRINGS, RolloutConfig, prompt_ids_for, run_episode
from verl.experimental.agent_loop.agent_loop import AgentLoopBase, AgentLoopOutput

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


class SearchR1AgentLoop(AgentLoopBase):
    @classmethod
    def init_class(
        cls,
        config,
        tokenizer,
        retrieval_url="http://127.0.0.1:8000",
        max_turns=4,
        topk=3,
        max_obs_tokens=500,
        invalid_action="retry",
        chat_template=True,
        retrieval_timeout=60,
        retrieval_retries=3,
        dump_dir=None,
        dump_max=200,
        **kwargs,
    ):
        if cls._class_initialized:
            return
        cls._class_initialized = True
        cls.tokenizer = tokenizer
        cls.cfg = RolloutConfig(
            max_turns=max_turns,
            topk=topk,
            max_obs_tokens=max_obs_tokens,
            response_length=config.actor_rollout_ref.rollout.response_length,
            invalid_action=invalid_action,
        )
        cls.chat_template = chat_template
        cls.retrieval_url = retrieval_url.rstrip("/")
        cls.retrieval_timeout = retrieval_timeout
        cls.retrieval_retries = retrieval_retries
        cls.dump_dir = dump_dir
        cls.dump_max = dump_max
        cls.n_dumped = 0
        if dump_dir:
            os.makedirs(dump_dir, exist_ok=True)
        print(f"SearchR1AgentLoop: {cls.cfg}, chat_template={chat_template}, retrieval={cls.retrieval_url}, dump_dir={dump_dir}")

    def _prompt_ids(self, messages):
        return prompt_ids_for(self.tokenizer, messages, self.chat_template)

    def _retrieve_blocking(self, query, topk):
        last = None
        for attempt in range(self.retrieval_retries):
            try:
                r = requests.post(
                    f"{self.retrieval_url}/retrieve",
                    json={"queries": [query], "topk": topk, "return_scores": False},
                    timeout=self.retrieval_timeout,
                )
                r.raise_for_status()
                return r.json()["result"][0]
            except requests.RequestException as e:
                last = e
                time.sleep(2**attempt)
        raise RuntimeError(f"retrieval failed after {self.retrieval_retries} attempts: {last}")

    async def run(self, messages: list[dict[str, Any]], sampling_params: dict[str, Any]) -> AgentLoopOutput:
        times = {"generate_sequences": 0.0, "tool_calls": 0.0}  # summed over the episode's turns
        request_id = uuid4().hex  # same id every turn: verl routes it to the same server (prefix cache)
        prompt_ids = await self.loop.run_in_executor(None, self._prompt_ids, messages)
        # stop at the end of an action; max_tokens is set by verl's server (max_model_len - context)
        sp = dict(sampling_params, stop=STOP_STRINGS, include_stop_str_in_output=True)

        async def generate(context_ids):
            t = time.time()
            out = await self.server_manager.generate(request_id=request_id, prompt_ids=list(context_ids), sampling_params=sp)
            times["generate_sequences"] += time.time() - t
            return out

        async def retrieve(query, topk):
            t = time.time()
            out = await self.loop.run_in_executor(None, self._retrieve_blocking, query, topk)
            times["tool_calls"] += time.time() - t
            return out

        ep = await run_episode(prompt_ids, generate, retrieve, self.tokenizer, self.cfg)

        if self.dump_dir and SearchR1AgentLoop.n_dumped < self.dump_max:
            SearchR1AgentLoop.n_dumped += 1
            self._dump(ep)

        n_model_turns = sum(s.role == "model" for s in ep.segments)
        n_env = sum(s.role == "env" for s in ep.segments)
        return AgentLoopOutput(
            prompt_ids=ep.prompt_ids,
            response_ids=ep.response_ids,
            response_mask=ep.response_mask,
            num_turns=1 + n_model_turns + n_env,  # same convention as verl's ToolAgentLoop
            metrics=times,
        )

    def _dump(self, ep):
        """One JSON line per episode, ids and text per segment, for scripts/view_rollouts.py."""
        rec = {
            "pid": os.getpid(),
            "finish": ep.finish,
            "n_search": ep.n_search,
            "n_invalid": ep.n_invalid,
            "n_close_tag_appended": ep.n_close_tag_appended,
            "queries": ep.queries,
            "prompt_ids": ep.prompt_ids,
            "prompt_text": self.tokenizer.decode(ep.prompt_ids, skip_special_tokens=False),
            "segments": [
                {"role": s.role, "kind": s.kind, "ids": s.ids,
                 "text": self.tokenizer.decode(s.ids, skip_special_tokens=False),
                 "tokens": self.tokenizer.convert_ids_to_tokens(s.ids)}
                for s in ep.segments
            ],
            "response_mask": ep.response_mask,
        }
        path = os.path.join(self.dump_dir, f"rollouts_{os.getpid()}.jsonl")
        with open(path, "a") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")