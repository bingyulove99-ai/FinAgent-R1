"""FinAgent-R1: a FinQA agent recipe for the Agent-R1 framework.

The package is intentionally layered:

* :mod:`finagent_r1.parsing`, :mod:`finagent_r1.prompts`,
  :mod:`finagent_r1.tools`, :mod:`finagent_r1.rewards` and
  :mod:`finagent_r1.core` do **not** import ``verl`` / ``agent_r1`` and can be
  unit-tested on a CPU-only machine.
* :mod:`finagent_r1.flow` is the thin adapter that plugs the episode loop into
  Agent-R1's ``AgentFlowBase``.
* :mod:`finagent_r1.data` and :mod:`finagent_r1.evaluation` are standalone
  command-line tools.

Keeping the RL-agnostic core separate from the framework adapter is what makes
the same prompt/tool/reward code usable for training, offline evaluation and
unit tests without duplication.
"""

__version__ = "0.1.0"

__all__ = ["__version__"]
