import argparse
import yaml
import mlflow
import mlflow.sklearn
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

from src.common.logging import get_logger
from src.data.make_dataset import generate_synthetic_fraud
from src.data.validation import validate_dataset
from src.features.subfolder.build_features import split_xy
from src.models.evaluate import evaluate_binary

log = get_logger(__name__)

def build_model(C: float, max_iter: int) -> Pipeline:
    return Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(C=C, max_iter=max_iter, class_weight="balanced")),
        ]
    )

def main(config_path: str):
    cfg = yaml.safe_load(open(config_path, "r", encoding="utf-8"))

    mlflow.set_tracking_uri(cfg["mlflow"]["tracking_uri"])
    mlflow.set_experiment(cfg["mlflow"]["experiment_name"])

    df = generate_synthetic_fraud(
        n_rows=int(cfg["n_rows"]),
        seed=int(cfg["seed"]),
        drift_mode=str(cfg.get("drift_mode", "none")),
    )
    validate_dataset(df)

    X, y = split_xy(df)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=float(cfg["split"]["test_size"]), random_state=int(cfg["seed"]), stratify=y
    )

    model = build_model(C=float(cfg["model"]["C"]), max_iter=int(cfg["model"]["max_iter"]))

    with mlflow.start_run() as run:
        mlflow.log_params(
            {
                "seed": cfg["seed"],
                "n_rows": cfg["n_rows"],
                "drift_mode": cfg.get("drift_mode", "none"),
                "model_type": cfg["model"]["type"],
                "C": cfg["model"]["C"],
                "max_iter": cfg["model"]["max_iter"],
            }
        )

        model.fit(X_train, y_train)
        prob = model.predict_proba(X_test)[:, 1]
        m = evaluate_binary(y_test, prob)

        mlflow.log_metrics({"roc_auc": m.roc_auc, "pr_auc": m.pr_auc, "brier": m.brier, "pos_rate": m.pos_rate})

        # save evaluation artifact
        report = pd.DataFrame(
            [{"metric": "roc_auc", "value": m.roc_auc}, {"metric": "pr_auc", "value": m.pr_auc}, {"metric": "brier", "value": m.brier}]
        )
        report_path = "eval_report.csv"
        report.to_csv(report_path, index=False)
        mlflow.log_artifact(report_path)

        # register model
        registered_name = cfg["mlflow"]["registered_model_name"]
        mlflow.sklearn.log_model(
            sk_model=model,
            artifact_path="model",
            registered_model_name=registered_name,
            input_example=X_test.iloc[:3],
            signature=mlflow.models.infer_signature(X_test, model.predict_proba(X_test)[:, 1]),
        )

        log.info("training_complete", extra={"run_id": run.info.run_id, "roc_auc": m.roc_auc, "pr_auc": m.pr_auc})
        print(run.info.run_id)

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/train.yaml")
    args = ap.parse_args()
    main(args.config)
