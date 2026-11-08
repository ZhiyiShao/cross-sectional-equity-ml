"""Local artifact integration; only bounded final-training rows, never evaluation data."""
import numpy as np
import pandas as pd
import pytest

from equity_research.models import final_fit_v2 as final
from equity_research.evaluation.v2 import locked_bundle, preflight, predict_frame

pytestmark = pytest.mark.skipif(not (final.OUT/'manifest.json').exists(), reason='Local V2 final fits absent')


def test_frozen_model_preflight_never_opens_observations(monkeypatch):
    monkeypatch.setattr(pd, 'read_parquet', lambda *a, **k: pytest.fail('Preflight opened observations'))
    result = preflight()
    assert result['models'] == 6 and result['observations_read'] == 0
    ledger = final.read_json(final.OUT/'fit_ledger.json')
    assert len(ledger) == 6 and all(r['status']=='completed' for r in ledger)
    final.check_metadata(final.read_json(final.OUT/'protected_artifact_metadata.json'))


def test_prediction_entry_replays_training_references_without_refit():
    _, original, _ = final.load_plan()
    _, bundle, _ = locked_bundle()
    frame, mask, _ = final.load_training(original)
    dates = [frame.loc[mask, 'signal_date'].min(), frame.loc[mask, 'signal_date'].max()]
    # Include every signal-date candidate before label filtering.
    columns = ['PERMNO', 'signal_date', 'signal_eligible', 'industry']+original
    candidates = frame.loc[frame.signal_date.isin(dates), columns]
    pred = predict_frame(candidates, bundle, (final.START, final.END))
    reference = pd.read_parquet(final.OUT/'reload_reference_local.parquet', filters=[('signal_date', 'in', dates)])
    joined = reference.merge(pred, on=['PERMNO', 'signal_date'], validate='one_to_one', suffixes=('_saved', '_entry'))
    assert len(joined) == len(reference)
    for model in final.IDS:
        np.testing.assert_allclose(joined[model+'_saved'], joined[model+'_entry'], atol=1e-12, rtol=1e-12)
