"""Agent-R1 adapter for the FinQA episode loop.

This is the only module in the package that imports ``agent_r1`` / ``verl``.
It packages every turn produced by :func:`finagent_r1.core.run_episode` into an
``AgentFlowStep`` and delegates padding, masks and terminal reward computation
to ``AgentFlowBase._postprocess``.

Reward contract:
* intermediate tool-use turns carry ``reward_score=0.0``;
* the terminal turn carries ``reward_score=None``, which makes the framework
  call ``custom_reward_function`` on that turn's response.
"""

from __future__ import annotations

import json
import logging
import os
from typing import Any
from uuid import uuid4

import numpy as np

from agent_r1.agent_flow.agent_flow import (
    AgentFlowBase,
    AgentFlowOutput,
    AgentFlowStep,
    register,
)

from finagent_r1.core import EpisodeConfig, Generation, run_episode
from finagent_r1.parsing import coerce_mapping, extract_query, merge_scope, merge_tools_kwargs
from finagent_r1.prompts import FINQA_TOOL_SCHEMAS
from finagent_r1.tools import FinancialRetrieverTool, PythonFinancialCalcTool

logger = logging.getLogger(__name__)


@register("finqa_agent")
class FinQAAgentFlow(AgentFlowBase):
    """Multi-step FinQA agent with a scratchpad and sliding observation window."""

    def __init__(
        self,
        trainer_config,
        server_manager,
        reward_loop_worker,
        tokenizer,
        processor,
        dataset_cls,
        dataset_config,
        **kwargs,
    ):
        super().__init__(
            trainer_config,
            server_manager,
            reward_loop_worker,
            tokenizer,
            processor,
            dataset_cls,
            dataset_config,
            **kwargs,
        )

        rollout = self.config.actor_rollout_ref.rollout
        self.prompt_length = int(getattr(rollout, "prompt_length", 4096))
        self.response_length = int(getattr(rollout, "response_length", 768))

        self.episode_config = EpisodeConfig(
            max_steps=max(1, int(kwargs.get("max_steps", 6))),
            max_parallel_calls=max(1, int(kwargs.get("max_parallel_calls", 2))),
            observation_window=max(1, int(kwargs.get("observation_window", 2))),
            prompt_length=self.prompt_length,
            response_length=self.response_length,
            max_obs_chars=max(400, int(kwargs.get("max_obs_chars", 6000))),
            max_observation_chars=max(300, int(kwargs.get("max_observation_chars", 3500))),
            max_scratchpad_items=max(1, int(kwargs.get("max_scratchpad_items", 8))),
            max_history_actions=max(1, int(kwargs.get("max_history_actions", 10))),
            context_mode=str(kwargs.get("context_mode", "scratchpad_window")),
        )

        self.corpus_path = kwargs.get("corpus_path") or os.getenv("FINQA_CORPUS_PATH")
        self.index_path = kwargs.get("index_path") or os.getenv("FINQA_INDEX_PATH")
        self.allow_scope_fallback = bool(kwargs.get("allow_scope_fallback", True))

        self.tools = [
            FinancialRetrieverTool(
                corpus_path=self.corpus_path,
                index_path=self.index_path,
                allow_scope_fallback=self.allow_scope_fallback,
            ),
            PythonFinancialCalcTool(),
        ]

    @staticmethod
    def _normalize_raw_prompt(raw_prompt: Any, tokenizer) -> list[dict[str, str]]:
        """Normalize the dataset's ``raw_prompt`` to a chat-message list."""

        if raw_prompt is None:
            return []
        if hasattr(raw_prompt, "tolist"):
            raw_prompt = raw_prompt.tolist()
        if isinstance(raw_prompt, str):
            return [{"role": "user", "content": raw_prompt}]
        if isinstance(raw_prompt, dict):
            return [dict(raw_prompt)]
        if (
            isinstance(raw_prompt, (list, tuple))
            and raw_prompt
            and all(isinstance(item, int) for item in raw_prompt)
        ):
            decoded = tokenizer.decode(list(raw_prompt), skip_special_tokens=True)
            return [{"role": "user", "content": decoded}]
        if isinstance(raw_prompt, (list, tuple)):
            return [dict(item) for item in raw_prompt if isinstance(item, dict)]
        return []

    async def run(self, sampling_params: dict[str, Any], **kwargs) -> AgentFlowOutput:
        """Execute one step-level rollout and package it for Agent-R1."""

        raw_prompt = kwargs.get("raw_prompt", kwargs.get("prompt"))
        messages = self._normalize_raw_prompt(raw_prompt, self.tokenizer)
        kwargs["raw_prompt"] = messages
        question = extract_query(messages)

        scope = merge_scope(kwargs)
        tools_kwargs = merge_tools_kwargs(kwargs)
        if not tools_kwargs.get("ticker"):
            tools_kwargs["ticker"] = scope.get("ticker", "")
        if tools_kwargs.get("year") is None:
            tools_kwargs["year"] = scope.get("year")

        async def generate_fn(prompt_ids: list[int], params: dict[str, Any]) -> Generation:
            output = await self.server_manager.generate(
                request_id=uuid4().hex,
                prompt_ids=prompt_ids,
                sampling_params=params,
            )
            routed = getattr(output, "routed_experts", None)
            if routed is not None:
                routed = routed[: len(prompt_ids) + self.response_length]
                if hasattr(routed, "detach"):
                    routed = routed.detach().cpu().numpy()
                elif not hasattr(routed, "shape"):
                    routed = np.asarray(routed)
            return Generation(
                token_ids=[int(token) for token in output.token_ids],
                logprobs=getattr(output, "log_probs", None),
                routed_experts=routed,
            )

        async def render_fn(msgs: list[dict], tool_schemas: list[dict] | None) -> list[int]:
            return await self.apply_chat_template(msgs, tools=tool_schemas or FINQA_TOOL_SCHEMAS)

        result = await run_episode(
            question=question,
            ticker=str(scope.get("ticker") or ""),
            year=scope.get("year"),
            generate_fn=generate_fn,
            render_fn=render_fn,
            decode_fn=lambda ids: self.tokenizer.decode(ids, skip_special_tokens=True),
            tools=self.tools,
            config=self.episode_config,
            sampling_params=sampling_params,
            tools_kwargs=tools_kwargs,
        )

        steps: list[AgentFlowStep] = []
        base_extra_info = coerce_mapping(kwargs.get("extra_info"))
        base_extra_info.setdefault("ground_truth", scope.get("ground_truth", ""))
        base_extra_info.setdefault("ticker", scope.get("ticker", ""))
        base_extra_info.setdefault("year", scope.get("year"))

        for turn in result.turns:
            step_extra_info = dict(base_extra_info)
            step_extra_info.update(result.stats)
            step_extra_info["step_count"] = turn.step_index
            step_extra_info["answer_found"] = 1 if turn.action.has_answer else 0
            step_extra_info["tool_names"] = [call.name for call in turn.action.tool_calls]

            step = AgentFlowStep(
                prompt_ids=turn.prompt_ids,
                response_ids=turn.response_ids,
                response_logprobs=turn.response_logprobs,
                routed_experts=turn.routed_experts,
                reward_score=None if turn.stop else 0.0,
                num_turns=2,
                extra_fields={
                    "step_count": turn.step_index,
                    "invalid_tool_calls": result.stats["invalid_tool_calls"],
                    "tool_errors": result.stats["tool_errors"],
                    "tool_call_count": result.stats["tool_call_count"],
                    "answer_found": int(turn.action.has_answer),
                    "stop": int(turn.stop),
                },
            )
            step = await self._postprocess(
                step,
                **{**kwargs, "extra_info": step_extra_info},
            )
            steps.append(step)

        if not steps:  # defensive; run_episode raises before this in practice
            raise RuntimeError("FinQAAgentFlow produced no steps.")

        logger.debug(
            "[finagent] steps=%d stats=%s",
            len(steps),
            json.dumps(result.stats, ensure_ascii=False),
        )
        return AgentFlowOutput(steps=steps, metrics=result.metrics)


__all__ = ["FinQAAgentFlow"]
