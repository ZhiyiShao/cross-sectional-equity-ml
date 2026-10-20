"""Offline environment smoke check using synthetic data, not trading results."""
from importlib.metadata import version
from pathlib import Path
import json
import tempfile

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from scipy.stats import spearmanr
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


def main():
    rng = np.random.default_rng(42)
    x = pd.DataFrame(rng.normal(size=(400, 5)), columns=list("abcde"))
    y = 2 * x.a - x.b + 0.3 * x.c**2 + rng.normal(scale=0.1, size=400)
    models = {
        "ridge": make_pipeline(StandardScaler(), Ridge(alpha=1.0)),
        "lightgbm": LGBMRegressor(n_estimators=80, num_leaves=7,
                                  random_state=42, n_jobs=2, verbosity=-1),
    }
    metrics = {}
    for name, model in models.items():
        model.fit(x.iloc[:300], y.iloc[:300])
        prediction = model.predict(x.iloc[300:])
        assert prediction.shape == (100,) and np.isfinite(prediction).all()
        mse = mean_squared_error(y.iloc[300:], prediction)
        baseline = mean_squared_error(y.iloc[300:], np.full(100, y.iloc[:300].mean()))
        assert mse < baseline, f"{name} did not learn the synthetic relationship"
        metrics[name] = {"mse": float(mse), "spearman": float(spearmanr(y.iloc[300:], prediction).statistic)}
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory)
        x.to_parquet(path / "sample.parquet", index=False)
        pd.testing.assert_frame_equal(x, pd.read_parquet(path / "sample.parquet"))
        fig, ax = plt.subplots()
        ax.plot(y.iloc[300:].to_numpy())
        fig.savefig(path / "plot.png")
        plt.close(fig)
        assert (path / "plot.png").stat().st_size > 0
    packages = ["numpy", "pandas", "scipy", "scikit-learn", "lightgbm", "matplotlib", "pyarrow", "jupyterlab", "ipykernel", "pytest"]
    print(json.dumps({"status": "PASS", "synthetic_data_only": True,
                      "versions": {p: version(p) for p in packages}, "metrics": metrics}, indent=2))


if __name__ == "__main__":
    main()
