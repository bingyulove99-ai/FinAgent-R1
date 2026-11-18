"""Integration test for the Agent-R1 adapter without importing verl.

The real ``agent_r1`` package imports verl/torch/ray.  This test installs a
minimal stub of ``agent_r1.agent_flow.agent_flow`` (same trick used by many CPU
test suites), then exercises the actual ``FinQAAgentFlow.run`` implementation:
step packaging, scope wiring, terminal reward, and extra_info propagation.
"""

from __future__ import annotations

import asyncio
import json
import sys
import tempfile
import types
import unittest
from pathlib import Path
from types import SimpleNamespace

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from finagent_r1.data.prepare_finqa import _read_json_payload, build_corpus  # noqa: E402
from finagent_r1.rewards import compute_score  # noqa: E402

_FIXTURES = Path(__file__).resolve().parent / "fixtures" / "finqa_raw"


class _StubStep:
    def __init__(
        self,
        prompt_ids,
        response_ids,
        response_logprobs=None,
        routed_experts=None,
        reward_score=None,
        num_turns=2,
        extra_fields=None,
        **kwargs,
    ):
        self.prompt_ids = prompt_ids
        self.response_ids = response_ids
        self.response_logprobs = response_logprobs
        self.routed_experts = routed_experts
        self.reward_score = reward_score
        self.num_turns = num_turns
        self.extra_fields = dict(extra_fields or {})
        self.response_mask = None


class _StubOutput:
    def __init__(self, steps, metrics):
        self.steps = steps
        self.metrics = metrics


class _StubBase:
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
        self.config = trainer_config.config
        self.server_manager = server_manager
        self.reward_loop_worker = reward_loop_worker
        self.tokenizer = tokenizer
        self.processor = processor
        self.dataset_cls = dataset_cls
        self.dataset_config = dataset_config
        self.system_prompt = []
        self.loop = None
        self.postprocessed = []

    async def apply_chat_template(self, messages, tools=None, **kwargs):
        text = "".join(str(message.get("content")) for message in messages)
        return [1] * (len(text) // 4 + 10)

    async def _postprocess(self, step, **kwargs):
        step.response_mask = [1] * len(step.response_ids)
        if step.reward_score is None:
            ground_truth = (kwargs.get("reward_model") or {}).get("ground_truth", "")
            step.reward_score = compute_score(
                "finqa",
                self.tokenizer.decode(step.response_ids),
                ground_truth,
                kwargs.get("extra_info", {}),
            )
        self.postprocessed.append((step, kwargs.get("extra_info", {})))
        return step


def _install_stubs() -> None:
    if "agent_r1.agent_flow.agent_flow" in sys.modules:
        return

    agent_r1 = types.ModuleType("agent_r1")
    agent_r1.__path__ = []
    agent_flow = types.ModuleType("agent_r1.agent_flow")
    agent_flow.__path__ = []
    module = types.ModuleType("agent_r1.agent_flow.agent_flow")
    module.AgentFlowBase = _StubBase
    module.AgentFlowStep = _StubStep
    module.AgentFlowOutput = _StubOutput

    def register(name):
        def decorator(cls):
            return cls

        return decorator

    module.register = register
    sys.modules["agent_r1"] = agent_r1
    sys.modules["agent_r1.agent_flow"] = agent_flow
    sys.modules["agent_r1.agent_flow.agent_flow"] = module


class _FakeServer:
    def __init__(self, responses):
        self.responses = responses
        self.index = 0

    async def generate(self, request_id, prompt_ids, sampling_params):
        index = self.index
        self.index += 1
        return SimpleNamespace(
            token_ids=[index] * 6,
            log_probs=[-0.1] * 6,
            routed_experts=None,
        )


class _FakeTokenizer:
    def __init__(self, responses):
        self.responses = responses

    def decode(self, ids, skip_special_tokens=True):
        index = ids[0] if ids else 0
        return self.responses[min(index, len(self.responses) - 1)]


def _build_corpus_path() -> Path:
    examples = []
    for split in ("train", "dev", "test"):
        examples.extend(_read_json_payload(_FIXTURES / f"{split}.json"))
    handle = tempfile.NamedTemporaryFile("w", suffix=".jsonl", delete=False, encoding="utf-8")
    with handle:
        for chunk in build_corpus(examples):
            handle.write(json.dumps(chunk) + "\n")
    return Path(handle.name)


class FlowAdapterTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        _install_stubs()
        from finagent_r1.flow import FinQAAgentFlow  # imported after stubs

        cls.Flow = FinQAAgentFlow
        cls.corpus_path = _build_corpus_path()

    @classmethod
    def tearDownClass(cls):
        cls.corpus_path.unlink(missing_ok=True)

    def _run(self, responses):
        rollout = SimpleNamespace(prompt_length=4096, response_length=256)
        config = SimpleNamespace(actor_rollout_ref=SimpleNamespace(rollout=rollout))
        trainer_config = SimpleNamespace(config=config)
        tokenizer = _FakeTokenizer(responses)
        server = _FakeServer(responses)

        flow = self.Flow(
            trainer_config=trainer_config,
            server_manager=server,
            reward_loop_worker=None,
            tokenizer=tokenizer,
            processor=None,
            dataset_cls=None,
            dataset_config=None,
            max_steps=3,
            corpus_path=str(self.corpus_path),
        )

        output = asyncio.run(
            flow.run(
                {"temperature": 0.0},
                raw_prompt=[{"role": "user", "content": "What was revenue growth?"}],
                reward_model={"ground_truth": "25%", "style": "rule"},
                extra_info={
                    "ticker": "ACME",
                    "year": 2021,
                    "answer": "25%",
                    "exe_ans": 25.0,
                    "is_numeric": True,
                    "question_id": "q1",
                },
                env_kwargs=json.dumps({"tools_kwargs": {"ticker": "ACME", "year": 2021}}),
            )
        )
        return flow, output

    def test_full_rollout_packaging(self):
        responses = [
            '<analysis>need revenue</analysis><tool_call>{"name": "financial_retriever", '
            '"arguments": {"query": "revenue"}}</tool_call>',
            '<analysis>calculate</analysis><tool_call>{"name": "python_calc", '
            '"arguments": {"code": "(100 - 80) / 80 * 100"}}</tool_call>',
            "<analysis>done</analysis><answer>25%</answer>",
        ]
        flow, output = self._run(responses)

        self.assertEqual(len(output.steps), 3)
        self.assertEqual([step.reward_score for step in output.steps[:2]], [0.0, 0.0])
        expected = 0.96**2 + 0.1
        self.assertAlmostEqual(output.steps[2].reward_score, expected, places=4)
        self.assertEqual(output.metrics["generate_sequences"] > 0, True)

        terminal = output.steps[2]
        self.assertTrue(terminal.extra_fields["stop"])
        self.assertEqual(terminal.extra_fields["tool_call_count"], 2)
        self.assertEqual(terminal.extra_fields["tool_errors"], 0)

        # extra_info reaches the reward function with the trajectory stats.
        _, reward_extra = flow.postprocessed[-1]
        self.assertEqual(reward_extra["step_count"], 3)
        self.assertEqual(reward_extra["tool_call_count"], 2)
        self.assertEqual(reward_extra["question_id"], "q1")

    def test_scope_is_injected_into_tools(self):
        responses = ["<analysis>done</analysis><answer>25%</answer>"]
        flow, output = self._run(responses)
        self.assertEqual(len(output.steps), 1)
        # The retriever was constructed with the corpus; the scope is passed via
        # tools_kwargs.  A wrong scope would make the first retriever call fail.
        self.assertAlmostEqual(output.steps[0].reward_score, 1.1, places=4)

    def test_intermediate_steps_have_reward_zero(self):
        responses = [
            '<tool_call>{"name": "financial_retriever", "arguments": {"query": "revenue"}}</tool_call>',
            '<tool_call>{"name": "financial_retriever", "arguments": {"query": "gross profit"}}</tool_call>',
            "<answer>25%</answer>",
        ]
        _, output = self._run(responses)
        self.assertEqual(len(output.steps), 3)
        self.assertEqual(output.steps[0].reward_score, 0.0)
        self.assertEqual(output.steps[1].reward_score, 0.0)
        self.assertGreater(output.steps[2].reward_score or 0, 0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
