"""
Dashboard — Brecha salarial entre géneros (datathon).
Datos: egresados universitarios (df_final.csv).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from cargar_modelo import intentar_cargar_modelo

BASE = Path(__file__).resolve().parent
CSV_PATH = BASE / "df_final.csv"

# --- Paleta de diseño (datathon) ---
C = {
    "primario_m": "#2E86AB",
    "secundario_f": "#F18F01",
    "brecha": "#C73E1D",
    "neutro_oscuro": "#2F2F2F",
    "neutro_claro": "#F5F5F5",
    "blanco": "#FFFFFF",
    "borde": "#E0E0E0",
    "borde_input": "#CCCCCC",
    "hover": "#EAEAEA",
    "grid": "#EAEAEA",
    "subtitulo": "#555555",
    "texto": "#333333",
    "nota": "#777777",
}

GENERO_MAP = {1.0: "Masculino", 2.0: "Femenino"}
GENERO_INV = {"Masculino": 1.0, "Femenino": 2.0}

# Descripciones para la UI (variable renombrada → código encuesta + texto)
VAR_META: dict[str, dict[str, str]] = {
    "genero": {"cap": "GENERO", "desc": "Sexo del encuestado."},
    "ingreso_mensual": {"cap": "CAP400P436_MONE", "desc": "Ingreso mensual principal."},
    "horas_semanales": {"cap": "CAP400P415_TOT", "desc": "Horas trabajadas semanalmente."},
    "ingreso_primer_empleo": {"cap": "CAP400P472", "desc": "Ingreso en el primer empleo."},
    "categoria_ocupacional": {"cap": "CAP400P409", "desc": "Categoría ocupacional."},
    "tipo_contrato": {"cap": "CAP400P413", "desc": "Tipo de contrato."},
    "sector_laboral": {"cap": "CAP400P410", "desc": "Sector laboral."},
    "tamanio_empresa": {"cap": "CAP400P414", "desc": "Tamaño de la empresa."},
    "ocupacion_cod": {"cap": "CAP400P405ACOD", "desc": "Código de ocupación principal."},
    "actividad_economica_cod": {"cap": "CAP400P408COD", "desc": "Actividad económica de la empresa."},
    "carrera_id": {"cap": "CAP300P312", "desc": "Carrera o especialidad (identificador en datos procesados)."},
    "titulado": {"cap": "CAP300P330", "desc": "Título profesional."},
    "cuadro_merito": {"cap": "CAP300P324", "desc": "Pertenece al cuadro de méritos."},
    "postgrado": {"cap": "CAP300P335", "desc": "Estudios de postgrado realizados."},
    "idioma_extranjero": {"cap": "CAP600P603_10", "desc": "Dominio de idioma extranjero."},
    "practicas_preprof": {"cap": "CAP300P349", "desc": "Realizó prácticas profesionales."},
    "gestion_universidad": {"cap": "SELECT_UNI_GESTION", "desc": "Gestión de la universidad: 1 Pública, 2 Privada."},
    "departamento": {"cap": "NOMBRECCDD", "desc": "Departamento de residencia."},
    "departamento_id": {"cap": "NOMBRECCDD", "desc": "Departamento de residencia (identificador en datos procesados)."},
    "distrito": {"cap": "NOMBRECCDI", "desc": "Distrito de residencia."},
    "culmino_estudio": {"cap": "CULMEST", "desc": "¿Ha culminado sus estudios universitarios? 1 Sí, 2 No."},
    "salario_hora": {"cap": "—", "desc": "Derivado: ingreso mensual ÷ (horas semanales × 4,3)."},
    "genero_label": {"cap": "GENERO", "desc": "Etiqueta de sexo del encuestado (Masculino / Femenino)."},
}


def _var_help(col: str) -> str:
    m = VAR_META.get(col)
    if not m:
        return ""
    cap = m["cap"]
    if cap == "—":
        return m["desc"]
    return f"{cap} — {m['desc']}"


def _var_axis_label(col: str, short: str) -> str:
    m = VAR_META.get(col)
    if not m:
        return short
    return f"{short} ({m['cap']})" if m["cap"] != "—" else short


def _df_int_default(df: pd.DataFrame, col: str, fallback: int = 0) -> int:
    if col not in df.columns:
        return fallback
    s = pd.to_numeric(df[col], errors="coerce").dropna()
    if s.empty:
        return fallback
    return int(s.median())


def _cat_cols_from_feat_cols(feat_cols: list[str]) -> list[str]:
    """Igual que `modelo.build_model_pipe`: categóricas = todas las features salvo `horas_semanales`."""
    return [c for c in feat_cols if c != "horas_semanales"]


def _default_feature_value(df: pd.DataFrame, col: str):
    """Valor típico para completar una feature ausente (mediana numérica o moda categórica)."""
    if col not in df.columns:
        return 0.0
    s = df[col]
    if pd.api.types.is_numeric_dtype(s) or pd.api.types.is_bool_dtype(s):
        v = pd.to_numeric(s, errors="coerce").median()
        return 0.0 if pd.isna(v) else float(v)
    mode = s.dropna().mode()
    if len(mode) > 0:
        return mode.iloc[0]
    return 0.0


def _complete_prediction_row(df: pd.DataFrame, feat_cols: list[str], row_dict: dict) -> pd.DataFrame:
    """
    Una fila con exactamente las columnas y el orden de `feat_cols`, como en el entrenamiento
    (`build_model_pipe` arma X con esas columnas en ese orden).
    """
    out: dict = {}
    for c in feat_cols:
        if c in row_dict:
            val = row_dict[c]
            if val is None or (isinstance(val, float) and np.isnan(val)):
                out[c] = _default_feature_value(df, c)
            else:
                out[c] = val
        else:
            out[c] = _default_feature_value(df, c)
    return pd.DataFrame([out])[feat_cols]


def _prepare_features_for_catboost(row: pd.DataFrame, feat_cols: list[str]) -> pd.DataFrame:
    """Codifica como `modelo.build_model_pipe`: numérico solo `horas_semanales`; resto → string."""
    cat_cols = _cat_cols_from_feat_cols(feat_cols)
    X = row.copy()
    if "horas_semanales" in X.columns:
        X["horas_semanales"] = pd.to_numeric(X["horas_semanales"], errors="coerce").fillna(0.0)
    for c in cat_cols:
        if c == "horas_semanales":
            continue
        X[c] = pd.to_numeric(X[c], errors="coerce").fillna(0).astype(int).astype(str)
    return X


def _df_int_bounds(df: pd.DataFrame, col: str) -> tuple[int, int]:
    if col not in df.columns:
        return 0, 999_999
    s = pd.to_numeric(df[col], errors="coerce").dropna()
    if s.empty:
        return 0, 999_999
    return int(s.min()), int(s.max())


@st.cache_resource
def _load_ml_artifact():
    """CatBoost + metadatos desde `artifacts/`; no entrena ni importa `modelo.py`."""
    return intentar_cargar_modelo(BASE)


def _render_variable_glossary() -> None:
    """Lista las variables críticas con código de encuesta y descripción."""
    bloques = [
        (
            "1. Identidad y resultado",
            ["genero", "ingreso_mensual", "horas_semanales", "ingreso_primer_empleo", "salario_hora"],
        ),
        (
            "2. Condición laboral",
            [
                "categoria_ocupacional",
                "tipo_contrato",
                "sector_laboral",
                "tamanio_empresa",
                "ocupacion_cod",
                "actividad_economica_cod",
            ],
        ),
        (
            "3. Capital humano",
            ["carrera_id", "titulado", "cuadro_merito", "postgrado", "idioma_extranjero", "practicas_preprof"],
        ),
        ("4. Entorno institucional", ["gestion_universidad"]),
        ("5. Control geográfico", ["departamento", "departamento_id", "distrito"]),
        ("6. Estudios", ["culmino_estudio"]),
    ]
    lines: list[str] = []
    for titulo, keys in bloques:
        lines.append(f"**{titulo}**")
        for k in keys:
            if k not in VAR_META:
                continue
            meta = VAR_META[k]
            cap = meta["cap"]
            desc = meta["desc"]
            if cap == "—":
                lines.append(f"- **{k}**: {desc}")
            else:
                lines.append(f"- **{k}** (`{cap}`): {desc}")
        lines.append("")
    st.markdown("\n".join(lines).strip())


@st.cache_data(show_spinner=False)
def load_data() -> pd.DataFrame:
    df = pd.read_csv(CSV_PATH)
    df["salario_hora"] = pd.to_numeric(df["salario_hora"], errors="coerce")
    m = np.isinf(df["salario_hora"])
    df.loc[m, "salario_hora"] = np.nan
    mask_fill = df["salario_hora"].isna() & (df["horas_semanales"] > 0)
    df.loc[mask_fill, "salario_hora"] = (
        df.loc[mask_fill, "ingreso_mensual"] / df.loc[mask_fill, "horas_semanales"]
    )
    df["genero_label"] = df["genero"].map(GENERO_MAP).fillna("Sin dato")
    return df


def brecha_porcentual(media_m: float, media_f: float) -> float | None:
    if media_m is None or media_f is None or not np.isfinite(media_m) or not np.isfinite(media_f):
        return None
    if media_m <= 0:
        return None
    return (media_m - media_f) / media_m * 100.0


def agregar_brecha_por_grupo(df: pd.DataFrame, col: str, min_n: int = 30) -> pd.DataFrame:
    rows = []
    for val, g in df.groupby(col, dropna=False):
        gm = g.loc[g["genero"] == 1.0, "ingreso_mensual"]
        gf = g.loc[g["genero"] == 2.0, "ingreso_mensual"]
        if len(gm) < min_n or len(gf) < min_n:
            continue
        mm, mf = gm.mean(), gf.mean()
        rows.append(
            {
                col: val,
                "n_hombres": len(gm),
                "n_mujeres": len(gf),
                "media_m": mm,
                "media_f": mf,
                "brecha_pct": brecha_porcentual(mm, mf),
            }
        )
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    return out.sort_values("brecha_pct", ascending=False)


def _inject_theme_css() -> None:
    st.markdown(
        f"""
        <style>
            html, body, [class*="css"]  {{
                font-family: sans-serif !important;
            }}
            .main .block-container {{
                padding-top: 1.5rem;
                padding-bottom: 2rem;
                max-width: 1200px;
            }}
            h1 {{
                color: {C["neutro_oscuro"]} !important;
                font-weight: 700 !important;
                font-size: 1.85rem !important;
            }}
            h2, h3 {{
                color: {C["subtitulo"]} !important;
                font-weight: 600 !important;
            }}
            p, li, span, label {{
                color: {C["texto"]} !important;
            }}
            .stCaption, [data-testid="stCaption"] {{
                color: {C["nota"]} !important;
            }}
            section[data-testid="stSidebar"] {{
                background-color: {C["blanco"]} !important;
                border-right: 1px solid {C["borde"]} !important;
            }}
            div[data-testid="stMetric"] {{
                background-color: {C["blanco"]};
                border: 1px solid {C["borde"]};
                border-radius: 12px;
                padding: 0.75rem 1rem;
            }}
            div[data-testid="stMetric"] label {{
                color: {C["neutro_oscuro"]} !important;
            }}
            .stTabs [data-baseweb="tab-list"] {{
                gap: 8px;
            }}
            .stTabs [data-baseweb="tab"] {{
                border-radius: 8px 8px 0 0;
            }}
            div[data-testid="stExpander"], .element-container div[data-baseweb] {{
                border-radius: 8px;
            }}
            .info-box {{
                background: {C["blanco"]};
                border: 1px solid {C["borde"]};
                border-left: 4px solid {C["brecha"]};
                border-radius: 10px;
                padding: 1rem 1.25rem;
                color: {C["texto"]};
            }}
            section[data-testid="stSidebar"] [data-baseweb="select"] > div:first-child,
            section[data-testid="stSidebar"] [data-baseweb="input"] > div {{
                border-color: {C["borde_input"]} !important;
                border-radius: 8px !important;
                background-color: {C["blanco"]} !important;
            }}
            section[data-testid="stSidebar"] [data-baseweb="select"]:hover > div:first-child,
            section[data-testid="stSidebar"] [data-baseweb="input"]:hover > div {{
                background-color: {C["hover"]} !important;
            }}
        </style>
        """,
        unsafe_allow_html=True,
    )


def _plotly_base_layout(fig, title: str | None = None) -> None:
    layout_kwargs = dict(
        paper_bgcolor=C["blanco"],
        plot_bgcolor=C["blanco"],
        font=dict(family="sans-serif", color=C["texto"], size=12),
        margin=dict(l=48, r=24, t=56 if title else 48, b=48),
    )
    if title:
        layout_kwargs["title"] = dict(text=title, font=dict(color=C["neutro_oscuro"], size=16))
    fig.update_layout(**layout_kwargs)
    fig.update_xaxes(
        showgrid=True,
        gridcolor=C["grid"],
        gridwidth=1,
        zeroline=False,
        linecolor=C["borde"],
        tickfont=dict(color=C["texto"]),
    )
    fig.update_yaxes(
        showgrid=True,
        gridcolor=C["grid"],
        gridwidth=1,
        zeroline=False,
        linecolor=C["borde"],
        tickfont=dict(color=C["texto"]),
    )


COLOR_GENERO = {"Masculino": C["primario_m"], "Femenino": C["secundario_f"]}


def main():
    st.set_page_config(
        page_title="Brecha salarial por género",
        page_icon="📊",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    _inject_theme_css()

    st.title("Brecha salarial entre géneros")
    st.caption(
        "Panel exploratorio y modelo predictivo de **ingreso mensual** — datos de egresados "
        "(codificación INEI: 1 Masculino, 2 Femenino)."
    )
    with st.expander("Variables del estudio (códigos CAP y descripción)", expanded=False):
        _render_variable_glossary()

    df = load_data()

    with st.sidebar:
        st.header("Filtros")
        dep_col = "departamento" if "departamento" in df.columns else "departamento_id"
        deps = sorted(df[dep_col].dropna().unique().tolist())
        sel_dep = st.multiselect(
            "Departamento (vacío = todos)" if dep_col == "departamento" else "Departamento ID (vacío = todos)",
            deps,
            default=[],
            help=_var_help(dep_col) + " Si no eliges ninguno, se usan todos.",
        )
        ing_min, ing_max = float(df["ingreso_mensual"].min()), float(df["ingreso_mensual"].max())
        r_ing = st.slider(
            "Rango ingreso mensual (S/)",
            min_value=0.0,
            max_value=float(max(ing_max, 1)),
            value=(0.0, float(max(ing_max, 1))),
            help=_var_help("ingreso_mensual"),
        )
        r_horas = st.slider(
            "Horas semanales (máx.)",
            min_value=1,
            max_value=max(int(df["horas_semanales"].max()), 1),
            value=max(int(df["horas_semanales"].quantile(0.99)), 48),
            help=_var_help("horas_semanales"),
        )
        st.divider()
        st.markdown("**Dataset**")
        st.text(f"Filas: {len(df):,}")
        st.text(f"Columnas: {df.shape[1]}")

    dff = df.copy()
    dep_col = "departamento" if "departamento" in dff.columns else "departamento_id"
    if len(sel_dep) > 0:
        dff = dff[dff[dep_col].isin(sel_dep)]
    dff = dff[
        (dff["ingreso_mensual"] >= r_ing[0])
        & (dff["ingreso_mensual"] <= r_ing[1])
        & (dff["horas_semanales"] <= r_horas)
        & (dff["horas_semanales"] > 0)
    ]

    hom = dff[dff["genero"] == 1.0]["ingreso_mensual"]
    muj = dff[dff["genero"] == 2.0]["ingreso_mensual"]

    m_m, m_f = hom.mean(), muj.mean()
    med_m, med_f = hom.median(), muj.median()
    gap = brecha_porcentual(m_m, m_f)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric(
        "Media ingreso — Masculino",
        f"S/ {m_m:,.0f}" if len(hom) else "—",
        help=_var_help("ingreso_mensual") + " Promedio en submuestra masculina.",
    )
    c2.metric(
        "Media ingreso — Femenino",
        f"S/ {m_f:,.0f}" if len(muj) else "—",
        help=_var_help("ingreso_mensual") + " Promedio en submuestra femenina.",
    )
    if gap is not None:
        c3.metric(
            "Brecha (media)",
            f"{gap:.1f} %",
            help="(media M − media F) / media M × 100. Compara medias de "
            + _var_help("ingreso_mensual"),
        )
    else:
        c3.metric("Brecha (media)", "—")
    c4.metric(
        "Observaciones (filtradas)",
        f"{len(dff):,}",
        help="Filas que cumplen filtros de sidebar y con horas semanales > 0.",
    )

    if gap is not None and gap > 0:
        st.markdown(
            f'<div class="info-box">Con los filtros actuales, el ingreso mensual <strong>promedio</strong> '
            f'de mujeres es aproximadamente <strong style="color:{C["brecha"]}">{gap:.1f} %</strong> menor '
            f"que el de hombres (brecha relativa).</div>",
            unsafe_allow_html=True,
        )
    elif gap is not None and gap < 0:
        st.markdown(
            f'<div class="info-box">La media femenina supera a la masculina en magnitud relativa '
            f'(<strong style="color:{C["brecha"]}">{gap:.1f} %</strong>; interpretar con precaución).</div>',
            unsafe_allow_html=True,
        )

    tab1, tab2, tab3 = st.tabs(["Distribución y comparación", "Brecha por dimensiones", "Predicción (ML)"])

    with tab1:
        col_a, col_b = st.columns(2)
        with col_a:
            fig_box = px.box(
                dff,
                x="genero_label",
                y="ingreso_mensual",
                color="genero_label",
                points=False,
                labels={
                    "ingreso_mensual": _var_axis_label("ingreso_mensual", "Ingreso mensual (S/)"),
                    "genero_label": _var_axis_label("genero_label", "Género"),
                },
                color_discrete_map=COLOR_GENERO,
                category_orders={"genero_label": ["Masculino", "Femenino"]},
            )
            fig_box.update_layout(showlegend=False, height=420)
            _plotly_base_layout(fig_box, "Ingreso mensual por género")
            st.caption(_var_help("ingreso_mensual") + " " + _var_help("genero"))
            st.plotly_chart(fig_box, use_container_width=True)
        with col_b:
            sh = dff.dropna(subset=["salario_hora"])
            sh = sh[np.isfinite(sh["salario_hora"])]
            sh = sh[sh["salario_hora"] < sh["salario_hora"].quantile(0.995)]
            fig_v = px.violin(
                sh,
                x="genero_label",
                y="salario_hora",
                color="genero_label",
                box=True,
                labels={
                    "salario_hora": _var_axis_label("salario_hora", "Salario hora (S/)"),
                    "genero_label": _var_axis_label("genero_label", "Género"),
                },
                color_discrete_map=COLOR_GENERO,
                category_orders={"genero_label": ["Masculino", "Femenino"]},
            )
            fig_v.update_layout(showlegend=False, height=420)
            _plotly_base_layout(fig_v, "Salario por hora (S/) — extremos recortados al p99.5")
            st.caption(_var_help("salario_hora") + " Extremos recortados al percentil 99.5.")
            st.plotly_chart(fig_v, use_container_width=True)

        st.subheader("Histograma comparativo")
        fig_hist = go.Figure()
        for label, series in [("Masculino", hom), ("Femenino", muj)]:
            fig_hist.add_trace(
                go.Histogram(
                    x=series,
                    name=label,
                    opacity=0.72,
                    nbinsx=45,
                    marker_color=COLOR_GENERO[label],
                )
            )
        fig_hist.update_layout(
            barmode="overlay",
            xaxis_title=_var_axis_label("ingreso_mensual", "Ingreso mensual (S/)"),
            yaxis_title="Frecuencia",
            height=400,
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        )
        _plotly_base_layout(fig_hist, "Ingreso mensual")
        st.caption(_var_help("ingreso_mensual"))
        st.plotly_chart(fig_hist, use_container_width=True)

        st.caption(
            f"Medianas — Masculino: S/ {med_m:,.0f} · Femenino: S/ {med_f:,.0f} "
            f"(en submuestra filtrada). {_var_help('genero')}"
        )

    with tab2:
        st.subheader("Brecha por departamento")
        st.caption(_var_help(dep_col))
        bd = agregar_brecha_por_grupo(dff, dep_col, min_n=25)
        if bd.empty:
            st.warning("No hay suficientes observaciones por género en los grupos; afloja filtros.")
        else:
            bd_plot = bd.head(20).sort_values("brecha_pct", ascending=True)
            dep_lbl = _var_axis_label(dep_col, "Departamento")
            fig_d = px.bar(
                bd_plot,
                x="brecha_pct",
                y=dep_col,
                orientation="h",
                labels={"brecha_pct": "Brecha % (M vs F)", dep_col: dep_lbl},
                color="brecha_pct",
                color_continuous_scale=[
                    [0.0, C["neutro_claro"]],
                    [0.5, "#E8A598"],
                    [1.0, C["brecha"]],
                ],
            )
            fig_d.update_layout(height=520, yaxis={"categoryorder": "total ascending"})
            _plotly_base_layout(fig_d, "Top departamentos por brecha de media")
            st.caption(
                "Brecha relativa entre medias de "
                + _var_help("ingreso_mensual")
                + " "
                + _var_help("genero")
            )
            st.plotly_chart(fig_d, use_container_width=True)

        c21, c22 = st.columns(2)
        with c21:
            st.subheader("Por sector laboral (código)")
            st.caption(_var_help("sector_laboral"))
            bs = agregar_brecha_por_grupo(dff, "sector_laboral", min_n=40)
            if not bs.empty:
                fig_s = px.bar(
                    bs.sort_values("brecha_pct", ascending=False).head(12),
                    x="sector_laboral",
                    y="brecha_pct",
                    labels={
                        "brecha_pct": "Brecha %",
                        "sector_laboral": _var_axis_label("sector_laboral", "Sector (cód.)"),
                    },
                    color_discrete_sequence=[C["brecha"]],
                )
                _plotly_base_layout(fig_s, "Brecha por sector laboral")
                st.plotly_chart(fig_s, use_container_width=True)
        with c22:
            st.subheader("Por tamaño de empresa (código)")
            st.caption(_var_help("tamanio_empresa"))
            bt = agregar_brecha_por_grupo(dff, "tamanio_empresa", min_n=40)
            if not bt.empty:
                fig_t = px.bar(
                    bt.sort_values("brecha_pct", ascending=False),
                    x="tamanio_empresa",
                    y="brecha_pct",
                    labels={
                        "brecha_pct": "Brecha %",
                        "tamanio_empresa": _var_axis_label("tamanio_empresa", "Tamaño empresa (cód.)"),
                    },
                    color_discrete_sequence=[C["brecha"]],
                )
                _plotly_base_layout(fig_t, "Brecha por tamaño de empresa")
                st.plotly_chart(fig_t, use_container_width=True)

    with tab3:
        st.markdown(
            f'<p style="color:{C["texto"]};">Predicción con <strong>CatBoost</strong> ya entrenado '
            "(carga desde <code>artifacts/</code>). El objetivo del modelo es "
            "<strong>ingreso mensual</strong> (entrenamiento con <code>log1p</code> si así "
            "consta en los metadatos).</p>",
            unsafe_allow_html=True,
        )
        ml = _load_ml_artifact()
        if ml is None:
            st.warning(
                "No se encontró el modelo entrenado en esta carpeta. "
                "La app no vuelve a entrenar ni modifica `modelo.py`."
            )
            st.info(
                "Coloca junto a `app.py` los archivos **`artifacts/modelo.cbm`** y "
                "**`artifacts/modelo_meta.json`** (el mismo par que generaste al exportar tu modelo). "
                "Si aún no los tienes, puedes generarlos **una vez en tu PC** con: "
                "`python exportar_modelo_artefacto.py` (usa tu `modelo.py` tal cual para entrenar y guardar)."
            )
        else:
            pipe, meta = ml
            feat_cols = list(meta["feat_cols"])
            metrics = meta.get("metrics") or {}
            target_log = bool(meta.get("target_log", True))

            def _decode_pred(raw: float) -> float:
                return float(np.expm1(raw)) if target_log else float(raw)

            m1, m2 = st.columns(2)
            m1.metric(
                "MAE (validación)",
                f"S/ {metrics['mae']:,.0f}" if metrics.get("mae") is not None else "—",
            )
            m2.metric(
                "R² (validación)",
                f"{metrics['r2']:.3f}" if metrics.get("r2") is not None else "—",
            )
            with st.expander("Variables de entrada del modelo (orden = entrenamiento)", expanded=False):
                st.markdown(
                    f"**{len(feat_cols)} columnas.** Categóricas (todas salvo `horas_semanales`), "
                    "como en `modelo.build_model_pipe`."
                )
                st.code("\n".join(feat_cols), language=None)

            st.subheader("Estimar ingreso con tus valores")
            st.caption(
                "Cada campo corresponde a una variable del cuestionario (ver expander superior). "
                "Los códigos numéricos son las categorías INEI tal como vienen en los datos."
            )
            dep_col_model = "departamento" if "departamento" in df.columns else "departamento_id"
            deps_all = sorted(df[dep_col_model].dropna().unique().tolist())
            ocu_lo, ocu_hi = _df_int_bounds(df, "ocupacion_cod")
            act_lo, act_hi = _df_int_bounds(df, "actividad_economica_cod")
            car_lo, car_hi = _df_int_bounds(df, "carrera_id")
            gc1, gc2 = st.columns(2)
            with gc1:
                gen_sel = st.selectbox("Género", ["Masculino", "Femenino"], help=_var_help("genero"))
                horas = st.number_input(
                    "Horas semanales",
                    min_value=1,
                    max_value=100,
                    value=40,
                    help=_var_help("horas_semanales"),
                )
                dep_sel = st.selectbox(
                    "Departamento" if dep_col_model == "departamento" else "Departamento (ID)",
                    deps_all,
                    index=0,
                    help=_var_help(dep_col_model),
                )
            with gc2:
                culm = st.number_input(
                    "¿Culminó estudios? (cód.)",
                    min_value=0.0,
                    max_value=5.0,
                    value=float(_df_int_default(df, "culmino_estudio", 1)),
                    help=_var_help("culmino_estudio"),
                )
                ing_pe = st.number_input(
                    "Ingreso primer empleo (cód.)",
                    min_value=0.0,
                    max_value=20.0,
                    value=float(_df_int_default(df, "ingreso_primer_empleo", 2)),
                    help=_var_help("ingreso_primer_empleo"),
                )
                cat_oc = st.number_input(
                    "Categoría ocupacional (cód.)",
                    min_value=0.0,
                    max_value=10.0,
                    value=float(_df_int_default(df, "categoria_ocupacional", 3)),
                    help=_var_help("categoria_ocupacional"),
                )
            gc3, gc4 = st.columns(2)
            with gc3:
                t_con = st.number_input(
                    "Tipo de contrato (cód.)",
                    min_value=0.0,
                    max_value=10.0,
                    value=float(_df_int_default(df, "tipo_contrato", 2)),
                    help=_var_help("tipo_contrato"),
                )
                sec = st.number_input(
                    "Sector laboral (cód.)",
                    min_value=0.0,
                    max_value=10.0,
                    value=float(_df_int_default(df, "sector_laboral", 5)),
                    help=_var_help("sector_laboral"),
                )
                tam = st.number_input(
                    "Tamaño de empresa (cód.)",
                    min_value=0.0,
                    max_value=10.0,
                    value=float(_df_int_default(df, "tamanio_empresa", 3)),
                    help=_var_help("tamanio_empresa"),
                )
            with gc4:
                tit = st.number_input(
                    "Titulado (cód.)",
                    min_value=0.0,
                    max_value=5.0,
                    value=float(_df_int_default(df, "titulado", 2)),
                    help=_var_help("titulado"),
                )
                cm = st.number_input(
                    "Cuadro de méritos (cód.)",
                    min_value=0.0,
                    max_value=5.0,
                    value=float(_df_int_default(df, "cuadro_merito", 2)),
                    help=_var_help("cuadro_merito"),
                )
                pg = st.number_input(
                    "Postgrado (cód.)",
                    min_value=0.0,
                    max_value=5.0,
                    value=float(_df_int_default(df, "postgrado", 2)),
                    help=_var_help("postgrado"),
                )
            gc5, gc6 = st.columns(2)
            with gc5:
                idi = st.number_input(
                    "Idioma extranjero (cód.)",
                    min_value=0.0,
                    max_value=10.0,
                    value=float(_df_int_default(df, "idioma_extranjero", 2)),
                    help=_var_help("idioma_extranjero"),
                )
                pra = st.number_input(
                    "Prácticas preprofesionales (cód.)",
                    min_value=0.0,
                    max_value=10.0,
                    value=float(_df_int_default(df, "practicas_preprof", 1)),
                    help=_var_help("practicas_preprof"),
                )
            with gc6:
                ges = st.number_input(
                    "Gestión universidad (cód.)",
                    min_value=0.0,
                    max_value=10.0,
                    value=float(_df_int_default(df, "gestion_universidad", 1)),
                    help=_var_help("gestion_universidad"),
                )
                ocup_cod = st.number_input(
                    "Ocupación principal — código",
                    min_value=ocu_lo,
                    max_value=ocu_hi,
                    value=_df_int_default(df, "ocupacion_cod", ocu_lo),
                    help=_var_help("ocupacion_cod"),
                )
                act_cod = st.number_input(
                    "Actividad económica — código",
                    min_value=act_lo,
                    max_value=act_hi,
                    value=_df_int_default(df, "actividad_economica_cod", act_lo),
                    help=_var_help("actividad_economica_cod"),
                )
            carrera_id_val = st.number_input(
                "Carrera — identificador (datos)",
                min_value=car_lo,
                max_value=car_hi,
                value=_df_int_default(df, "carrera_id", car_lo),
                help=_var_help("carrera_id"),
            )
    
            row_dict = {
                "culmino_estudio": culm,
                "genero": GENERO_INV[gen_sel],
                "horas_semanales": float(horas),
                "ingreso_primer_empleo": ing_pe,
                "categoria_ocupacional": cat_oc,
                "tipo_contrato": t_con,
                "sector_laboral": sec,
                "tamanio_empresa": tam,
                "ocupacion_cod": float(ocup_cod),
                "actividad_economica_cod": float(act_cod),
                "titulado": tit,
                "cuadro_merito": cm,
                "postgrado": pg,
                "idioma_extranjero": idi,
                "practicas_preprof": pra,
                "gestion_universidad": ges,
                "carrera_id": float(carrera_id_val),
                dep_col_model: dep_sel,
            }
            row = _complete_prediction_row(df, feat_cols, row_dict)

            if st.button("Predecir ingreso mensual", type="primary"):
                Xp = _prepare_features_for_catboost(row, feat_cols)
                pred = _decode_pred(float(pipe.predict(Xp)[0]))
                st.success(f"**Ingreso mensual estimado:** S/ {pred:,.0f}")
                st.caption(
                    "La predicción es orientativa (error típico cercano al MAE mostrado). "
                    "El género entra como una variable más; no implica causalidad."
                )

            st.divider()
            st.subheader("Comparar predicción cambiando solo el género")
            row_m = row.copy()
            row_f = row.copy()
            row_m["genero"] = 1.0
            row_f["genero"] = 2.0
            Xm = _prepare_features_for_catboost(row_m, feat_cols)
            Xf = _prepare_features_for_catboost(row_f, feat_cols)
            pm, pf = _decode_pred(float(pipe.predict(Xm)[0])), _decode_pred(float(pipe.predict(Xf)[0]))
            st.write(f"Misma ficha con **Masculino**: S/ {pm:,.0f}")
            st.write(f"Misma ficha con **Femenino**: S/ {pf:,.0f}")
            if pm > 0:
                diff = (pm - pf) / pm * 100.0
                st.info(
                    f"Diferencia relativa del modelo entre ambas filas: **{diff:.1f} %** "
                    f"(asociación aprendida por el modelo, no efecto causal)."
                )


if __name__ == "__main__":
    main()
