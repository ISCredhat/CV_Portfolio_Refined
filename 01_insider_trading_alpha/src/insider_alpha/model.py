"""Walk-forward ML ranking of insider purchases (no look-ahead).

At each retraining date T the model is fitted only on events whose label window
(entry + LABEL_HORIZON sessions) ended before T, then scores events filed in
[T, T + RETRAIN_MONTHS). The trading threshold (top TOP_FRACTION) is taken from
the score distribution on the most recent year of *training* data, so the rule
is fixed before any test event is seen.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import AdaBoostClassifier, HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

from . import config as C


def make_model(name: str):
    if name == "logit":
        return make_pipeline(SimpleImputer(strategy="median"), StandardScaler(),
                             LogisticRegression(C=0.05, max_iter=2000))
    if name == "adaboost":   # boosted shallow trees: a simple, classic baseline
        return make_pipeline(SimpleImputer(strategy="median"),
                             AdaBoostClassifier(DecisionTreeClassifier(max_depth=2), n_estimators=100,
                                                random_state=C.RANDOM_STATE))
    if name == "gbm":        # LightGBM-style histogram gradient boosting
        return HistGradientBoostingClassifier(max_iter=300, learning_rate=0.03, max_leaf_nodes=15,
                                              min_samples_leaf=200, l2_regularization=1.0,
                                              early_stopping=False, random_state=C.RANDOM_STATE)
    raise ValueError(name)


def walk_forward(ev: pd.DataFrame, features: list[str], label_col: str, models: list[str],
                 start: str = C.WALK_FORWARD_START, step_months: int = C.RETRAIN_MONTHS,
                 verbose: bool = True) -> tuple[pd.DataFrame, list[dict]]:
    """Returns (scores DataFrame indexed like ev, list of per-fold diagnostics)."""
    X_all = ev[features].astype("float64")
    y_all = (ev[label_col] > 0).astype(int)
    scores = pd.DataFrame(index=ev.index, columns=[f"score_{m}" for m in models] + [f"sel_{m}" for m in models]
                          + [f"sel_low_{m}" for m in models], dtype="float64")
    diags = []
    last = ev["filing_date"].max()
    T = pd.Timestamp(start)
    while T <= last:
        T_end = T + pd.DateOffset(months=step_months)
        train = ev["realise_date"].notna() & (ev["realise_date"] < T) & ev[label_col].notna()
        test = (ev["filing_date"] >= T) & (ev["filing_date"] < T_end)
        if test.sum() == 0 or train.sum() < 500:
            T = T_end
            continue
        recent = train & (ev["filing_date"] >= T - pd.DateOffset(years=1))
        for m in models:
            mdl = make_model(m)
            mdl.fit(X_all[train], y_all[train])
            s_test = mdl.predict_proba(X_all[test])[:, 1]
            s_recent = mdl.predict_proba(X_all[recent])[:, 1]
            thr = np.quantile(s_recent, 1 - C.TOP_FRACTION)
            thr_low = np.quantile(s_recent, C.TOP_FRACTION)
            scores.loc[test, f"score_{m}"] = s_test
            scores.loc[test, f"sel_{m}"] = (s_test >= thr).astype(float)
            scores.loc[test, f"sel_low_{m}"] = (s_test <= thr_low).astype(float)
            diags.append({"train_end": T, "model": m, "n_train": int(train.sum()),
                          "n_test": int(test.sum()), "threshold": thr, "threshold_low": thr_low})
        if verbose:
            print(f"  fold {T.date()}: train {train.sum():,}  test {test.sum():,}", flush=True)
        T = T_end
    return scores, diags


def oos_metrics(ev: pd.DataFrame, score_col: str, label_col: str) -> pd.DataFrame:
    """Per-year out-of-sample AUC, rank IC and top-minus-bottom quintile spread."""
    d = ev[[score_col, label_col, "filing_date"]].dropna()
    rows = []
    for yr, g in d.groupby(d["filing_date"].dt.year):
        if len(g) < 100 or g[label_col].gt(0).nunique() < 2:
            continue
        q = pd.qcut(g[score_col].rank(method="first"), 5, labels=False)
        rows.append({
            "year": yr, "n": len(g),
            "auc": roc_auc_score(g[label_col] > 0, g[score_col]),
            "rank_ic": g[score_col].corr(g[label_col], method="spearman"),
            "q5_minus_q1": g.loc[q == 4, label_col].mean() - g.loc[q == 0, label_col].mean(),
        })
    out = pd.DataFrame(rows)
    if len(out):
        q = pd.qcut(d[score_col].rank(method="first"), 5, labels=False)
        out.loc[len(out)] = {"year": "all", "n": len(d),
                             "auc": roc_auc_score(d[label_col] > 0, d[score_col]),
                             "rank_ic": d[score_col].corr(d[label_col], method="spearman"),
                             "q5_minus_q1": d.loc[q == 4, label_col].mean() - d.loc[q == 0, label_col].mean()}
    return out


def permutation_importance_last_fold(ev: pd.DataFrame, features: list[str], label_col: str,
                                     model_name: str = "gbm", n_repeats: int = 5) -> pd.DataFrame:
    """Permutation importance (AUC drop) of a model trained on all realised data before the
    final year and evaluated on the final year - an honest out-of-sample importance."""
    from sklearn.inspection import permutation_importance
    d = ev[ev[label_col].notna() & ev["realise_date"].notna()]
    cut = d["filing_date"].max() - pd.DateOffset(years=1)
    tr = d[d["realise_date"] < cut]
    te = d[d["filing_date"] >= cut]
    mdl = make_model(model_name).fit(tr[features].astype(float), (tr[label_col] > 0).astype(int))
    res = permutation_importance(mdl, te[features].astype(float), (te[label_col] > 0).astype(int),
                                 scoring="roc_auc", n_repeats=n_repeats, random_state=C.RANDOM_STATE)
    return (pd.DataFrame({"feature": features, "auc_drop": res.importances_mean, "std": res.importances_std})
            .sort_values("auc_drop", ascending=False).reset_index(drop=True))
