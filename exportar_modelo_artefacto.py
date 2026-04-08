"""
Ejecutar UNA VEZ en tu máquina (no en Streamlit) para generar:
  artifacts/modelo.cbm
  artifacts/modelo_meta.json

Así el dashboard solo carga el modelo entrenado y no vuelve a entrenar.
Requiere el mismo `modelo.py` que usaste para entrenar (no lo modifica).
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from modelo import build_model_pipe

BASE = Path(__file__).resolve().parent
CSV_PATH = BASE / "df_final.csv"
OUT_DIR = BASE / "artifacts"


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(CSV_PATH)
    model, _num_cols, cat_cols, metrics, feat_cols = build_model_pipe(df)
    cbm_path = OUT_DIR / "modelo.cbm"
    meta_path = OUT_DIR / "modelo_meta.json"
    model.save_model(str(cbm_path))
    meta = {
        "feat_cols": feat_cols,
        "cat_cols": cat_cols,
        "target_log": True,
        "metrics": metrics,
    }
    meta_path.write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Guardado: {cbm_path}")
    print(f"Guardado: {meta_path}")
    print(f"MAE={metrics['mae']:.2f}  R²={metrics['r2']:.4f}")


if __name__ == "__main__":
    main()
