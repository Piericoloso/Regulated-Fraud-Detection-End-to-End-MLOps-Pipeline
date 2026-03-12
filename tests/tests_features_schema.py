from src.features.subfolder.schema import FEATURES, TARGET

def test_schema_is_non_empty():
    assert len(FEATURES) > 0
    assert isinstance(TARGET, str) and TARGET