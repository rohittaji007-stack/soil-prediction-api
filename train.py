"""
Retrain the soil model pack -- HONEST version.

For every soil parameter we measure, by cross-validation on the real training
data, how well the 18-channel AS7265x spectrum actually predicts it. Each
parameter is then assigned a reliability tier:

    "measured"    regression R^2 >= R2_MIN         -> report a numeric value
    "screening"   3-class accuracy >= ACC_MIN      -> report Low / Med / High
    "unavailable" neither threshold met            -> report nothing (no signal)

The pack stores the fitted models AND the measured metrics/tiers/thresholds, so
main.py can only ever surface what the data supports.

Pack layout consumed by main.py:
    {
      'scaler':      StandardScaler (fit on L1-normalised spectra),
      'models':      {target: RandomForestRegressor},      # numeric predictors
      'classifiers': {target: RandomForestClassifier},     # low/med/high
      'thresholds':  {target: [low_cut, high_cut]},        # tertile boundaries
      'metrics':     {target: {'r2':..., 'acc':...}},      # CV performance
      'tiers':       {target: 'measured'|'screening'|'unavailable'},
      'targets':     [...],
      'sklearn_version': '...',
    }
"""
import glob
import pickle

import numpy as np
import pandas as pd
import sklearn
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import accuracy_score, r2_score
from sklearn.model_selection import cross_val_predict
from sklearn.preprocessing import StandardScaler

FEATURES = [f"X_{i}" for i in range(1, 19)]
TARGETS = ["N(ppm)", "P(ppm)", "K(ppm)", "OC_percent", "pH", "EC",
           "Fe", "Mn", "Cu", "Zn", "B", "S"]
UNITS = {"N(ppm)": "ppm", "P(ppm)": "ppm", "K(ppm)": "ppm", "OC_percent": "%",
         "pH": "", "EC": "dS/m", "Fe": "ppm", "Mn": "ppm", "Cu": "ppm",
         "Zn": "ppm", "B": "ppm", "S": "ppm"}

EXPORT_RENAME = {
    **{f"x_{i}": f"X_{i}" for i in range(1, 19)},
    "n_ppm": "N(ppm)", "p_ppm": "P(ppm)", "k_ppm": "K(ppm)",
    "oc_percent": "OC_percent", "ph": "pH", "ec": "EC",
    "fe": "Fe", "mn": "Mn", "cu": "Cu", "zn": "Zn", "b": "B", "s": "S",
}

R2_MIN = 0.20      # min cross-val R^2 to trust a numeric value
ACC_MIN = 0.50     # min 3-class accuracy to trust a Low/Med/High label
RANDOM_STATE = 42
MODEL_PACK = "soil_model_pack_rf.pkl"


def l1_normalise(X):
    """Per-scan normalisation: divide each 18-channel spectrum by its own sum.
    Cancels intensity/distance/LED drift so scans are comparable. Applied
    identically here and in main.py."""
    X = np.asarray(X, dtype=float)
    return X / (X.sum(axis=1, keepdims=True) + 1e-9)


def load_base(path="final_training_base.csv"):
    return pd.read_csv(path)[FEATURES + TARGETS]


def load_export(path):
    df = pd.read_csv(path, sep=None, engine="python").rename(columns=EXPORT_RENAME)
    return df[FEATURES + TARGETS]


def clean(df):
    df = df.dropna(subset=TARGETS)
    df = df[~(df[TARGETS] == 0).all(axis=1)]
    return df.drop_duplicates().reset_index(drop=True)


def main():
    frames = [clean(load_base())]
    exports = sorted(set(glob.glob("training_data-export-*.csv"))
                     | set(glob.glob("training_data_*.csv")))
    for path in exports:
        frames.append(clean(load_export(path)))
    data = pd.concat(frames, ignore_index=True).drop_duplicates().reset_index(drop=True)
    print(f"Training rows: {len(data)}\n")

    Xn = l1_normalise(data[FEATURES].values)
    scaler = StandardScaler().fit(Xn)
    Xs = scaler.transform(Xn)
    n_folds = min(5, len(data))

    models, classifiers, thresholds, metrics, tiers = {}, {}, {}, {}, {}

    header = f"{'param':11s}{'R2':>7s}{'3cls_acc':>10s}   tier"
    print(header); print("-" * len(header))
    for t in TARGETS:
        y = data[t].values.astype(float)

        # --- numeric predictor + honest CV R^2 ---
        reg = RandomForestRegressor(n_estimators=300, min_samples_leaf=3,
                                    max_depth=4, max_features=0.5,
                                    random_state=RANDOM_STATE)
        try:
            r2 = r2_score(y, cross_val_predict(reg, Xs, y, cv=n_folds))
        except Exception:
            r2 = float("nan")
        reg.fit(Xs, y)

        # --- low/med/high classifier + honest CV accuracy ---
        lo, hi = np.quantile(y, [1 / 3, 2 / 3])
        cls = np.digitize(y, [lo, hi])            # 0=Low 1=Med 2=High
        clf = RandomForestClassifier(n_estimators=300, min_samples_leaf=2,
                                     random_state=RANDOM_STATE)
        try:
            acc = accuracy_score(cls, cross_val_predict(clf, Xs, cls, cv=n_folds))
        except Exception:
            acc = float("nan")
        clf.fit(Xs, cls)

        if r2 >= R2_MIN:
            tier = "measured"
        elif acc >= ACC_MIN:
            tier = "screening"
        else:
            tier = "unavailable"

        models[t] = reg
        classifiers[t] = clf
        thresholds[t] = [float(lo), float(hi)]
        metrics[t] = {"r2": round(float(r2), 3), "acc": round(float(acc), 3)}
        tiers[t] = tier
        print(f"{t:11s}{r2:7.2f}{acc:10.2f}   {tier}")

    print("\nReported to users:")
    print("  measured   :", [t for t in TARGETS if tiers[t] == "measured"] or "none")
    print("  screening  :", [t for t in TARGETS if tiers[t] == "screening"] or "none")
    print("  unavailable:", [t for t in TARGETS if tiers[t] == "unavailable"] or "none")

    pack = {"scaler": scaler, "models": models, "classifiers": classifiers,
            "thresholds": thresholds, "metrics": metrics, "tiers": tiers,
            "units": UNITS, "targets": TARGETS,
            "sklearn_version": sklearn.__version__}
    with open(MODEL_PACK, "wb") as f:
        pickle.dump(pack, f)
    print(f"\nSaved {MODEL_PACK} (sklearn {sklearn.__version__}).")


if __name__ == "__main__":
    main()
