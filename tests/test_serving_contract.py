from src.serving.schema import PredictRequest

def test_request_schema():
    req = PredictRequest(
        amount=120.0,
        hour=12,
        country_risk=3,
        is_cross_border=1,
        customer_tenure_days=400,
        txns_7d=2,
        chargebacks_365d=0,
    )
    assert req.hour == 12
