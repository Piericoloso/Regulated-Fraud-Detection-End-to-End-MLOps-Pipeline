import argparse
import yaml
import pandas as pd
import mlflow

from src.data.make_dataset import generate_synthetic
from src.features.subfolder.schema import FEATURES
from src.monitoring.drift import psi

def main(config_path: str):
    cfg = yaml.safe_load(open(config_path, "r", encoding="utf-8"))
    mlflow.set_tracking_uri(cfg["mlflow"]["tracking_uri"])
    exp = mlflow.get_experiment_by_name(cfg["mlflow"]["experiment_name"])
    if exp is None:
        raise ValueError("Experiment not found. Train first.")

    # baseline = "none", current = configured drift_mode
    base = generate_synthetic(n_rows=20000, seed=cfg["seed"], drift_mode="none")
    curr = generate_synthetic(n_rows=20000, seed=cfg["seed"] + 1, drift_mode=cfg.get("drift_mode", "mild"))

    rows = []
    for f in FEATURES:
        rows.append({"feature": f, "psi": psi(base[f], curr[f])})

    report = pd.DataFrame(rows).sort_values("psi", ascending=False)
    out_path = "docs/drift_report.csv"
    report.to_csv(out_path, index=False)
    print(f"Saved {out_path}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/train.yaml")
    args = ap.parse_args()
    main(args.config)
