from src.data.make_dataset import generate_synthetic_fraud
from src.data.validation import validate_dataset
from src.features.subfolder.schema import FEATURES, TARGET

def test_generated_dataset_validates():
    df = generate_synthetic_fraud(n_rows=1000, seed=1, drift_mode="none")
    validate_dataset(df)

def test_generated_columns_match_schema():
    df = generate_synthetic_fraud(n_rows=100, seed=2)
    assert list(df.columns) == FEATURES + [TARGET]
