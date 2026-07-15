"""
Retrain the soil-parameter RandomForest model pack.

Combines the curated base dataset (final_training_base.csv) with any number of
exported training CSVs (semicolon-separated, e.g.
training_data-export-*.csv), cleans them, and rebuilds soil_model_pack_rf.pkl.

Model pack layout (consumed by main.py):
    {
        'scaler':  StandardScaler fit on X_1..X_18,
        'models':  {target_name: RandomForestRegressor, ...},
        'targets': [target_name, ...],
    }
"""
import glob
import pickle
import sys

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score
from sklearn.model_selection import cross_val_predict
from sklearn.preprocessing import StandardScaler

FEATURES = [f"X_{i}" for i in range(1, 19)]
TARGETS = ["N(ppm)", "P(ppm)", "K(ppm)", "OC_percent", "pH", "EC",
           "Fe", "Mn", "Cu", "Zn", "B", "S"]

# Map the exported (lowercase, snake) column names -> canonical base names.
EXPORT_RENAME = {
    **{f"x_{i}": f"X_{i}" for i in range(1, 19)},
    "n_ppm": "N(ppm)", "p_ppm": "P(ppm)", "k_ppm": "K(ppm)",
    "oc_percent": "OC_percent", "ph": "pH", "ec": "EC",
    "fe": "Fe", "mn": "Mn", "cu": "Cu", "zn": "Zn", "b": "B", "s": "S",
}

RANDOM_STATE = 42
MODEL_PACK = "soil_model_pack_rf.pkl"


def load_base(path="final_training_base.csv"):
    df = pd.read_csv(path)
    return df[FEATURES + TARGETS]


def load_export(path):
    # Exports vary between ';' and ',' separators; auto-detect.
    df = pd.read_csv(path, sep=None, engine="python")
    df = df.rename(columns=EXPORT_RENAME)
    df = df[FEATURES + TARGETS]
    return df


def clean(df):
    """Drop rows without a full set of labels and all-zero (garbage) rows."""
    before = len(df)
    df = df.dropna(subset=TARGETS)                       # need every target
    df = df[~(df[TARGETS] == 0).all(axis=1)]             # drop all-zero rows
    df = df.drop_duplicates()
    print(f"  cleaned: {before} -> {len(df)} usable rows")
    return df.reset_index(drop=True)


def main():
    frames = []

    base = clean(load_base())
    print(f"base rows: {len(base)}")
    frames.append(base)

    exports = sorted(set(glob.glob("training_data-export-*.csv"))
                     | set(glob.glob("training_data_*.csv")))
    for path in exports:
        print(f"export {path}:")
        frames.append(clean(load_export(path)))

    data = pd.concat(frames, ignore_index=True).drop_duplicates().reset_index(drop=True)
    print(f"\nTOTAL training rows: {len(data)}")

    X = data[FEATURES].values.astype(float)
    scaler = StandardScaler().fit(X)
    Xs = scaler.transform(X)

    models = {}
    print("\nPer-target 5-fold CV R^2 (sanity check):")
    n_folds = min(5, len(data))
    for t in TARGETS:
        y = data[t].values.astype(float)
        rf = RandomForestRegressor(n_estimators=100, random_state=RANDOM_STATE)
        try:
            pred = cross_val_predict(rf, Xs, y, cv=n_folds)
            r2 = r2_score(y, pred)
        except Exception:
            r2 = float("nan")
        rf.fit(Xs, y)                                    # final fit on all data
        models[t] = rf
        print(f"  {t:12s} R2={r2:6.3f}")

    pack = {"scaler": scaler, "models": models, "targets": TARGETS}
    with open(MODEL_PACK, "wb") as f:
        pickle.dump(pack, f)
    print(f"\nSaved {MODEL_PACK} ({len(models)} models, {len(data)} samples).")


if __name__ == "__main__":
    sys.exit(main())
