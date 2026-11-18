#!/usr/bin/env bash
# Verify the Linux training environment.  This script does not install torch /
# verl automatically because the correct wheels depend on your CUDA driver.
set -euo pipefail

PYTHON_BIN="${PYTHON:-python3}"
if ! "$PYTHON_BIN" -c "import sys" >/dev/null 2>&1; then
    PYTHON_BIN="${PYTHON:-python}"
fi

echo "== Python =="
"$PYTHON_BIN" --version
"$PYTHON_BIN" - <<'PY'
import sys
if sys.version_info < (3, 10):
    raise SystemExit("Agent-R1/verl 0.7 requires Python >= 3.10")
print("Python version OK")
PY

echo
echo "== GPUs =="
if command -v nvidia-smi >/dev/null 2>&1; then
    nvidia-smi --query-gpu=index,name,memory.total,driver_version --format=csv
else
    echo "nvidia-smi not found; this machine is not the training node."
fi

echo
echo "== Python packages =="
"$PYTHON_BIN" - <<'PY'
modules = ["torch", "verl", "vllm", "ray", "transformers", "hydra", "pandas", "pyarrow"]
for name in modules:
    try:
        module = __import__(name)
        version = getattr(module, "__version__", "?")
        print(f"  {name:12s} {version}")
    except Exception as exc:
        print(f"  {name:12s} MISSING ({type(exc).__name__})")
PY

cat <<'EOF'

If packages are missing, follow the official verl 0.7.0 installation guide:
  https://verl.readthedocs.io/en/latest/start/install.html

Reference order:
  1. create a Python 3.10 conda environment
  2. install a CUDA-matched PyTorch (A40 = Ampere, sm_86)
  3. clone and install verl==0.7.0
  4. install vLLM / Ray versions required by that verl release
  5. pip install -r projects/finagent_r1/requirements.txt
  6. run: bash projects/finagent_r1/scripts/smoke_test.sh

Agent-R1 itself is NOT pip-installed.  The training scripts put
upstream/Agent-R1 and projects/finagent_r1 on PYTHONPATH.
EOF
