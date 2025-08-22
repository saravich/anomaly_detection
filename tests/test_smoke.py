import os, json, yaml, subprocess, sys

def run(cmd):
    print("RUN", cmd)
    assert subprocess.call(cmd, shell=True) == 0

def test_end_to_end(tmp_path):
    cfg_src = os.path.join(os.getcwd(), "configs", "classical_ssim.yaml")
    cfg_local = tmp_path / "classical_ssim.yaml"
    with open(cfg_src, "r") as f:
        y = yaml.safe_load(f)
    y["dataset"]["root"] = str(tmp_path / "data" / "synth")
    with open(cfg_local, "w") as f:
        yaml.safe_dump(y, f)

    model = tmp_path / "model.pkl"
    preds = tmp_path / "preds"
    metrics = tmp_path / "metrics.json"
    report = tmp_path / "report.pdf"

    run(f"python -m adlib.data --make-synth {tmp_path}/data/synth")
    run(f"python -m adlib.cli.fit --config {cfg_local} --out {model}")
    run(f"python -m adlib.cli.infer --config {cfg_local} --model {model} --out {preds}")
    run(f"python -m adlib.cli.eval --preds {preds} --out {metrics}")
    run(f"python -m adlib.cli.report --config {cfg_local} --metrics {metrics} --preds {preds} --out {report}")

    assert os.path.exists(metrics)
    assert os.path.exists(report)
