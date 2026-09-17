import pytest
from utils.validation import validate_prediction_input
def test_validation_rejects_missing_fields():
    with pytest.raises(ValueError): validate_prediction_input({})
