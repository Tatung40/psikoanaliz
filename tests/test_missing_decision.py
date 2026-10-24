import pandas as pd
from core.missing_decision import assess_necessity

def test_no_missing_needs_no_imputation():
    data = pd.DataFrame({'I1':[0,1], 'I2':[1,0]})
    result = assess_necessity(data, ['I1','I2'])
    assert result['status'] == 'not_needed' and result['needed'] is False

def test_missing_supported_analysis_does_not_require_imputation():
    data = pd.DataFrame({'I1':[0,None], 'I2':[1,0]})
    result = assess_necessity(data, ['I1','I2'], supports_missing=True)
    assert result['status'] == 'not_required' and result['needed'] is False

def test_missing_unsupported_requires_review_not_automatic_imputation():
    data = pd.DataFrame({'I1':[0,None], 'I2':[1,0]})
    result = assess_necessity(data, ['I1','I2'], supports_missing=False)
    assert result['status'] == 'review' and result['needed'] is None
