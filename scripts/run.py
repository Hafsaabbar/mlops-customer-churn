"""
Cross-platform command runner for Customer Churn MLOps project.
Enables running workflow tasks on both Windows (PowerShell/CMD) and Unix systems:
python scripts/run.py [install | test | train | gate | retrain | rollback | all | status]
"""

import sys
import subprocess

PYTHON_EXEC = sys.executable

def run_cmd(cmd_list, description):
    print(f"\n=======================================================")
    print(f"▶ [RUNNER] {description}")
    print(f"  Command: {' '.join(cmd_list)}")
    print(f"=======================================================")
    res = subprocess.run(cmd_list)
    if res.returncode != 0:
        print(f"❌ Error executing step: {description} (code {res.returncode})")
        sys.exit(res.returncode)

def main():
    target = sys.argv[1] if len(sys.argv) > 1 else "all"

    if target == "install":
        run_cmd([PYTHON_EXEC, "-m", "pip", "install", "-r", "requirements.txt"], "Installing pinned dependencies")

    elif target == "prepare":
        run_cmd([PYTHON_EXEC, "-m", "src.data.prepare"], "Preparing splits and MANIFEST.json")

    elif target == "validate":
        run_cmd([PYTHON_EXEC, "-m", "src.validation.validate"], "Validating data splits against Pandera schema")

    elif target == "test":
        run_cmd([PYTHON_EXEC, "-m", "pytest", "tests/", "-v"], "Executing Pytest test suite")

    elif target == "train":
        run_cmd([PYTHON_EXEC, "-m", "src.training.train"], "Training models, Baselines, and MLflow logging")

    elif target == "gate":
        run_cmd([PYTHON_EXEC, "-m", "src.lifecycle.registry", "promote", "1"], "Executing Quality Gate & promoting champion")

    elif target == "feedback":
        run_cmd([PYTHON_EXEC, "-m", "src.lifecycle.feedback", "feedback"], "Replaying delayed offline stream feedback")

    elif target == "retrain":
        run_cmd([PYTHON_EXEC, "-m", "src.lifecycle.feedback", "retrain"], "Retraining on accumulated data & gating candidate v2")

    elif target == "fairness":
        run_cmd([PYTHON_EXEC, "-m", "src.lifecycle.fairness"], "Executing algorithmic fairness and bias audit")

    elif target == "rollback":
        run_cmd([PYTHON_EXEC, "-m", "src.lifecycle.registry", "rollback"], "Executing rollback to previous champion model")

    elif target == "status":
        run_cmd([PYTHON_EXEC, "-m", "src.lifecycle.registry", "status"], "Inspecting Model Registry status and aliases")

    elif target == "all":
        print("\n🚀 [MLOps Pipeline] Executing full initial lifecycle chain: Data -> Validation -> Training -> Quality Gate -> Registry -> Fairness")
        run_cmd([PYTHON_EXEC, "-m", "src.data.prepare"], "1. Data preparation and SHA-256 Manifest generation")
        run_cmd([PYTHON_EXEC, "-m", "src.validation.validate"], "2. Pandera data validation")
        run_cmd([PYTHON_EXEC, "-m", "src.training.train"], "3. Model training, baselines, cost optimization, MLflow")
        run_cmd([PYTHON_EXEC, "-m", "src.lifecycle.registry", "promote", "1"], "4. Quality Gate evaluation and v1 promotion")
        run_cmd([PYTHON_EXEC, "-m", "src.lifecycle.fairness"], "5. Fairness and demographic parity audit")
        run_cmd([PYTHON_EXEC, "-m", "src.lifecycle.registry", "status"], "6. Final Registry status verification")
        print("\n✅ [MLOps Pipeline] Initial chain completed successfully!\n")

    else:
        print(f"Unknown target: '{target}'")
        print("Available targets: install, prepare, validate, test, train, gate, feedback, retrain, fairness, rollback, status, all")
        sys.exit(1)

if __name__ == "__main__":
    main()
