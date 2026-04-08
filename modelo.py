from __future__ import annotations

import numpy as np
import pandas as pd
from catboost import CatBoostRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split


def build_model_pipe(df: pd.DataFrame):
    d = df.copy()
    d["ingreso_mensual"] = pd.to_numeric(d["ingreso_mensual"], errors="coerce")
    d = d.dropna(subset=["ingreso_mensual"])
    d = d[d["ingreso_mensual"] > 0]

    drop_cols = {"ingreso_mensual", "salario_hora", "distrito"}
    feat_cols = [c for c in d.columns if c not in drop_cols]

    X = d[feat_cols].copy()
    y = d["ingreso_mensual"].astype(float)

    if "horas_semanales" in X.columns:
        X["horas_semanales"] = pd.to_numeric(X["horas_semanales"], errors="coerce").fillna(0.0)

    cat_cols = [c for c in X.columns if c != "horas_semanales"]
    for c in cat_cols:
        X[c] = pd.to_numeric(X[c], errors="coerce").fillna(0).astype(int).astype(str)

    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=42)
    ytr_log = np.log1p(ytr)
    yte_log = np.log1p(yte)

    cat_idx = [X.columns.get_loc(c) for c in cat_cols]
    model = CatBoostRegressor(
        loss_function="RMSE",
        depth=10,
        learning_rate=0.05,
        iterations=6000,
        random_seed=42,
        verbose=False,
        early_stopping_rounds=200,
    )
    model.fit(
        Xtr,
        ytr_log,
        cat_features=cat_idx,
        eval_set=(Xte, yte_log),
        use_best_model=True,
    )

    pred_log = model.predict(Xte)
    pred = np.expm1(pred_log)
    pred = np.clip(pred, 0, None)
    metrics = {
        "mae": float(mean_absolute_error(yte, pred)),
        "r2": float(r2_score(yte, pred)),
    }
    return model, ["horas_semanales"], cat_cols, metrics, feat_cols

