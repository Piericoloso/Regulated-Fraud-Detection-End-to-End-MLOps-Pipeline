import mlflow

def get_model_uri(name: str, stage: str) -> str:
    # Example: models:/fincrime_suspicious_txn/Production
    return f"models:/{name}/{stage}"

def promote_latest_run_to_stage(registered_model_name: str, run_id: str, stage: str = "Staging"):
    client = mlflow.tracking.MlflowClient()
    # find latest version created from this run
    versions = client.search_model_versions(f"name='{registered_model_name}'")
    target = None
    for v in versions:
        if v.run_id == run_id:
            target = v
            break
    if target is None:
        raise ValueError("No model version found for run_id. Did you register the model in this run?")
    client.transition_model_version_stage(
        name=registered_model_name,
        version=target.version,
        stage=stage,
        archive_existing_versions=False,
    )
    return target.version
