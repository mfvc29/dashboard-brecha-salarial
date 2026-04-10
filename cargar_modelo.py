

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from catboost import CatBoostRegressor

ARTIFACT_DIR = "artifacts"
MODEL_FILE = "modelo.cbm"
META_FILE = "modelo_meta.json"


def rutas_artefacto(base: Path) -> tuple[Path, Path]:
    d = base / ARTIFACT_DIR
    return d / MODEL_FILE, d / META_FILE


def modelo_disponible(base: Path) -> bool:
    cbm, meta = rutas_artefacto(base)
    return cbm.is_file() and meta.is_file()


def intentar_cargar_modelo(base: Path) -> tuple[CatBoostRegressor, dict[str, Any]] | None:
    """Devuelve (modelo, meta) o None si faltan archivos o hay error al leer."""
    if not modelo_disponible(base):
        return None
    try:
        return cargar_modelo_entrenado(base)
    except Exception:
        return None


def cargar_modelo_entrenado(base: Path) -> tuple[CatBoostRegressor, dict[str, Any]]:
    """
    Lee modelo.cbm y modelo_meta.json. El JSON debe incluir al menos:
    feat_cols, cat_cols, target_log (bool), metrics (opcional).
    """
    cbm_path, meta_path = rutas_artefacto(base)
    if not modelo_disponible(base):
        raise FileNotFoundError(
            f"Faltan artefactos en {base / ARTIFACT_DIR}: "
            f"necesitas {MODEL_FILE} y {META_FILE}."
        )
    model = CatBoostRegressor()
    model.load_model(str(cbm_path))
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    return model, meta
