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

# --- Paleta de diseño (actualizada) ---
C = {
    "primario_m": "#0A0F3C",      # Azul oscuro
    "secundario_f": "#CBB6E6",    # Morado suave
    "brecha": "#6C3FD1",          # Morado fuerte (accent)
    "neutro_oscuro": "#0A0F3C",   # Azul oscuro (headers)
    "neutro_claro": "#F5F5F5",
    "blanco": "#FFFFFF",          # Blanco
    "borde": "#BDBDBD",           # Gris claro
    "borde_input": "#BDBDBD",     # Gris claro
    "hover": "#EAEAEA",
    "grid": "#EAEAEA",
    "subtitulo": "#5A5A5A",       # Gris oscuro
    "texto": "#000000",           # Negro
    "nota": "#7A7A7A",            # Gris medio
}

GENERO_MAP = {1.0: "Masculino", 2.0: "Femenino"}
GENERO_INV = {"Masculino": 1.0, "Femenino": 2.0}

# Decimales mostrados en toda la interfaz
ND = 2

# Textos de ayuda (solo lenguaje claro; sin códigos de variable ni CAP)
VAR_DESC: dict[str, str] = {
    "genero": "Sexo del encuestado.",
    "ingreso_mensual": "Ingreso mensual principal.",
    "horas_semanales": "Horas trabajadas por semana.",
    "ingreso_primer_empleo": "Ingreso en el primer empleo (categoría de la encuesta).",
    "categoria_ocupacional": "Categoría ocupacional.",
    "tipo_contrato": "Tipo de contrato.",
    "sector_laboral": "Sector laboral.",
    "tamanio_empresa": "Tamaño de la empresa.",
    "ocupacion_cod": "Ocupación principal (categoría codificada).",
    "actividad_economica_cod": "Actividad económica de la empresa (categoría codificada).",
    "carrera_id": "Carrera o especialidad (según el catálogo usado en los datos).",
    "titulado": "Título profesional.",
    "cuadro_merito": "Pertenece al cuadro de méritos.",
    "postgrado": "Estudios de postgrado realizados.",
    "idioma_extranjero": "Dominio de idioma extranjero.",
    "practicas_preprof": "Prácticas profesionales realizadas.",
    "gestion_universidad": "Gestión de la universidad (pública o privada).",
    "departamento": "Departamento de residencia.",
    "departamento_id": "Departamento de residencia.",
    "departamento_nombre": "Departamento de residencia.",
    "distrito": "Distrito de residencia.",
    "culmino_estudio": "¿Culminó estudios universitarios? (categoría de la encuesta).",
    "salario_hora": "Salario por hora estimado a partir del ingreso mensual y las horas semanales.",
    "genero_label": "Género (Masculino / Femenino).",
}


def _var_help(col: str) -> str:
    return VAR_DESC.get(col, "")


def _fmt_round(x: float | None, nd: int = ND) -> str:
    if x is None or not np.isfinite(x):
        return "—"
    return f"{round(float(x), nd):.{nd}f}"


def _fmt_soles(x: float | None, nd: int = ND) -> str:
    if x is None or not np.isfinite(x):
        return "—"
    return f"S/ {_fmt_round(x, nd)}"


@st.cache_data(show_spinner=False)
def load_departamento_id_to_nombre() -> dict[int, str]:
    p = BASE / "maps" / "departamento_map.csv"
    if not p.is_file():
        return {}
    t = pd.read_csv(p)
    return {int(r["departamento_id"]): str(r["departamento"]).strip() for _, r in t.iterrows()}


def _departamento_nombre_from_id(dmap: dict[int, str], x) -> str:
    if pd.isna(x):
        return "Sin dato"
    try:
        xi = int(float(x))
    except (TypeError, ValueError):
        return str(x)
    return dmap.get(xi, f"ID {xi}")


def _columna_filtro_departamento(df: pd.DataFrame) -> str:
    if "departamento_nombre" in df.columns:
        return "departamento_nombre"
    if "departamento" in df.columns:
        return "departamento"
    return "departamento_id"


def _read_codigo_etiqueta_map(filename: str) -> dict[int, str]:
    """Mapa codigo→etiqueta para gráficos y selectores (no toca el modelo)."""
    p = BASE / "maps" / filename
    if not p.is_file():
        return {}
    t = pd.read_csv(p)
    if "codigo" not in t.columns or "etiqueta" not in t.columns:
        return {}
    out: dict[int, str] = {}
    for _, r in t.iterrows():
        try:
            out[int(float(r["codigo"]))] = str(r["etiqueta"]).strip()
        except (TypeError, ValueError, KeyError):
            continue
    return out


MAP_SI_NO = {1.0: "Sí", 2.0: "No", 0.0: "No"}
MAP_CAT_OCUPACIONAL = {
    1.0: "Empleador o patrono",
    2.0: "Trabajador independiente",
    3.0: "Empleado",
    4.0: "Obrero",
    5.0: "Trabajador familiar no remunerado",
    6.0: "Trabajador del hogar",
    7.0: "Otro"
}
MAP_TIPO_CONTRATO = {
    1.0: "Contrato indefinido / permanente",
    2.0: "Contrato a plazo fijo",
    3.0: "En período de prueba",
    4.0: "Convenios de formación laboral",
    5.0: "Locación de servicios / Honorarios",
    6.0: "Régimen especial (CAS)",
    7.0: "Sin contrato",
    8.0: "Otro"
}
MAP_GEST_UNIV = {1.0: "Pública", 2.0: "Privada"}
MAP_IDIOMA_EXTRANJERO = {1.0: "Excelente", 2.0: "Bueno", 3.0: "Regular", 4.0: "Malo", 5.0: "No sabe"}

def _safe_float(val, fallback: float = 0.0) -> float:
    try:
        return float(val)
    except (ValueError, TypeError):
        return fallback

def _get_opciones(df: pd.DataFrame, col: str) -> list[float]:
    if col not in df.columns:
        return [0.0]
    return sorted(df[col].dropna().unique().tolist())

@st.cache_data(show_spinner=False)
def load_sector_laboral_map() -> dict[int, str]:
    return _read_codigo_etiqueta_map("sector_laboral_map.csv")

@st.cache_data(show_spinner=False)
def load_tamanio_empresa_map() -> dict[int, str]:
    return _read_codigo_etiqueta_map("tamanio_empresa_map.csv")

@st.cache_data(show_spinner=False)
def load_ocupacion_map() -> dict[int, str]:
    return _read_codigo_etiqueta_map("ocupacion_map.csv")

@st.cache_data(show_spinner=False)
def load_actividad_economica_map() -> dict[int, str]:
    return _read_codigo_etiqueta_map("actividad_economica_map.csv")

@st.cache_data(show_spinner=False)
def load_carrera_map() -> dict[int, str]:
    p = BASE / "maps" / "carrera_map.csv"
    if not p.is_file():
        return {}
    t = pd.read_csv(p)
    return {int(r["carrera_id"]): str(r["carrera"]).split('(')[0].strip() for _, r in t.iterrows() if pd.notna(r.get("carrera_id")) and pd.notna(r.get("carrera"))}


def _etiqueta_codigo(m: dict[int, str], valor) -> str:
    try:
        k = int(float(valor))
    except (TypeError, ValueError):
        return str(valor)
    return m.get(k, f"Nivel {k}")


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
    """Glosario en lenguaje claro (sin códigos técnicos ni nombres de columnas internas)."""
    bloques = [
        (
            "Identidad y resultado",
            [
                "Sexo del encuestado.",
                "Ingreso mensual principal.",
                "Horas trabajadas por semana.",
                "Ingreso en el primer empleo.",
                "Salario por hora (derivado del ingreso y las horas).",
            ],
        ),
        (
            "Condición laboral",
            [
                "Categoría ocupacional, tipo de contrato, sector y tamaño de empresa.",
                "Ocupación y actividad económica (categorías codificadas).",
            ],
        ),
        (
            "Capital humano y formación",
            [
                "Carrera o especialidad.",
                "Título profesional, cuadro de méritos, postgrado.",
                "Idioma extranjero y prácticas profesionales.",
            ],
        ),
        ("Institución", ["Gestión de la universidad (pública o privada)."]),
        ("Ubicación", ["Departamento y distrito de residencia."]),
        ("Estudios", ["Si culminó o no la carrera universitaria."]),
    ]
    lines: list[str] = []
    for titulo, items in bloques:
        lines.append(f"**{titulo}**")
        for t in items:
            lines.append(f"- {t}")
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
    dmap = load_departamento_id_to_nombre()
    if "departamento_id" in df.columns and dmap:
        df["departamento_nombre"] = df["departamento_id"].apply(lambda x: _departamento_nombre_from_id(dmap, x))
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
    out = out.sort_values("brecha_pct", ascending=False)
    out["brecha_pct"] = pd.to_numeric(out["brecha_pct"], errors="coerce").round(ND)
    return out


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
            .stButton button p, .stButton button span, .stButton button div {{
                color: {C["blanco"]} !important;
            }}
            [data-baseweb="tag"] span {{
                color: {C["blanco"]} !important;
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
    with st.expander("Qué datos se usan en el panel", expanded=False):
        _render_variable_glossary()

    df = load_data()

    with st.sidebar:
        st.header("Filtros")
        dep_filt = _columna_filtro_departamento(df)
        deps = sorted(df[dep_filt].dropna().unique().tolist())
        sel_dep = st.multiselect(
            "Departamento (vacío = todos)",
            deps,
            default=[],
            help=_var_help(dep_filt) + " Si no eliges ninguno, se usan todos.",
        )
        
        opts_tit_raw = _get_opciones(df, "titulado")
        opts_tit_labels = list(dict.fromkeys([MAP_SI_NO.get(k, str(k)) for k in opts_tit_raw]))
        sel_tit_labels = st.multiselect(
            "Titulado (vacío = todos)",
            opts_tit_labels,
            default=[],
            help=_var_help("titulado"),
        )
        sel_tit = [k for k in opts_tit_raw if MAP_SI_NO.get(k, str(k)) in sel_tit_labels]
        
        map_sec_sidebar = load_sector_laboral_map()
        opts_sec = _get_opciones(df, "sector_laboral")
        sel_sec = st.multiselect(
            "Sector laboral (vacío = todos)",
            opts_sec,
            format_func=lambda k: _etiqueta_codigo(map_sec_sidebar, k),
            default=[],
            help=_var_help("sector_laboral"),
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
        
        st.divider()
        st.markdown("**Autores**")
        st.caption(
            "**Universidad Nacional de Ingeniería**  \n"
            "Facultad de Ingeniería Económica, Estadística y Ciencias Sociales  \n"
            "Escuela de Ingeniería Estadística"
        )
        st.markdown(
            "- [Victoria La Rosa](https://www.linkedin.com/in/lindsay-la-rosa-126535280/) | [📧](mailto:victoria.larosa.a@uni.pe)\n"
            "- [Valeria Linares](https://www.linkedin.com/in/valeria-linares-rodriguez-1bbba5279/) | [📧](mailto:valeria.linares.r@uni.pe)\n"
            "- [William Ramirez](https://www.linkedin.com/in/wramirezcc/) | [📧](mailto:william.ramirez.c@uni.pe)\n"
            "- [Martin Vargas](https://www.linkedin.com/in/mvargasch) | [📧](mailto:mvargasch@uni.pe)"
        )

    dff = df.copy()
    dep_filt = _columna_filtro_departamento(dff)
    if len(sel_dep) > 0:
        dff = dff[dff[dep_filt].isin(sel_dep)]
    if len(sel_tit) > 0:
        dff = dff[dff["titulado"].isin(sel_tit)]
    if len(sel_sec) > 0:
        dff = dff[dff["sector_laboral"].isin(sel_sec)]
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
        _fmt_soles(m_m) if len(hom) else "—",
        help=_var_help("ingreso_mensual") + " Promedio en submuestra masculina.",
    )
    c2.metric(
        "Media ingreso — Femenino",
        _fmt_soles(m_f) if len(muj) else "—",
        help=_var_help("ingreso_mensual") + " Promedio en submuestra femenina.",
    )
    if gap is not None:
        c3.metric(
            "Brecha (media)",
            f"{round(gap, ND):.{ND}f} %",
            help="(media hombres − media mujeres) / media hombres × 100, sobre ingreso mensual.",
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
            f'de mujeres es aproximadamente <strong style="color:{C["brecha"]}">{round(gap, ND):.{ND}f} %</strong> menor '
            f"que el de hombres (brecha relativa).</div>",
            unsafe_allow_html=True,
        )
    elif gap is not None and gap < 0:
        st.markdown(
            f'<div class="info-box">La media femenina supera a la masculina en magnitud relativa '
            f'(<strong style="color:{C["brecha"]}">{round(gap, ND):.{ND}f} %</strong>; interpretar con precaución).</div>',
            unsafe_allow_html=True,
        )

    tab1, tab2, tab3, tab4 = st.tabs([
        "Visión General y Narrativa",
        "Distribución y comparación",
        "Brecha por dimensiones",
        "Predicción (ML)",
    ])

    with tab1:
        st.subheader("Análisis Descriptivo Dinámico")
        st.markdown(
            "El dashboard interpreta matemáticamente la submuestra elegida para extraer hallazgos "
            "estadísticos en lenguaje claro."
        )
        
        if gap is not None:
            v_gap = abs(round(gap, ND))
            if gap > 0:
                texto_brecha = f"Se evidencia una **brecha salarial del {v_gap}%** a favor de los egresados hombres."
            elif gap < 0:
                texto_brecha = f"Se evidencia una **brecha salarial atípica del {v_gap}%** a favor de las egresadas mujeres."
            else:
                texto_brecha = "Prácticamente no existe brecha salarial (0%) general, los ingresos promedios son equitativos."
        else:
            texto_brecha = "No existen suficientes datos para calcular una brecha válida en esta selección."
            
        texto_dep = "a nivel país" if not sel_dep else f"en los departamentos de **{', '.join(map(str, sel_dep))}**"
        
        texto_titulado = ""
        if sel_tit:
            lbls_tit = [MAP_SI_NO.get(k, str(k)) for k in sel_tit]
            texto_titulado = f", considerando titulados: **{', '.join(lbls_tit)}**"
            
        texto_sector = ""
        if sel_sec:
            lbls_sec = [_etiqueta_codigo(map_sec_sidebar, k) for k in sel_sec]
            texto_sector = f", dentro del sector **{', '.join(lbls_sec)}**"

        st.info(
            f"Analizando a los egresados residentes {texto_dep}{texto_titulado}{texto_sector}; interactuamos con "
            f"una muestra de **{len(dff):,}** observaciones representativas con ingresos documentados. \n\n"
            f"Bajo las reglas actuales de selección, **el ingreso medio de la población masculina es {_fmt_soles(m_m)}**, "
            f"mientras que **la población femenina promedia {_fmt_soles(m_f)}**. {texto_brecha}"
        )
        
        if len(dff) > 0 and gap is not None and gap > 0:
            brecha_abs = m_m - m_f
            st.markdown(
                f"> 💡 En términos monetarios (absolutos), esta brecha equivale a que, en promedio, una mujer se priva "
                f"de ganar **__{_fmt_soles(brecha_abs)}__** mensualmente en comparación de un hombre "
                f"con el mismo perfil seleccionado."
            )
            
        if len(dff) < 100 and len(dff) > 0:
            st.warning(
                "⚠️ **Atención:** El tamaño de la muestra es muy pequeño (N < 100). Las medias mostradas "
                "pueden estar fuertemente sesgadas por ingresos anómalos o extremos ('outliers')."
            )
            
        st.divider()
        st.subheader("Acerca de los Datos")
        st.markdown(
            "- **Origen:** La data expuesta procede de cuestionarios codificados de egresados.\n"
            "- **Enfoque Descriptivo:** Esta herramienta busca encontrar áreas de mejora a nivel local e institucional.\n"
            "- **Explorar:** Pasa a la pestaña de _Distribución y comparación_ para visualizar las curvas de ingresos."
        )

    with tab2:
        col_a, col_b = st.columns(2)
        with col_a:
            fig_box = px.box(
                dff,
                x="genero_label",
                y="ingreso_mensual",
                color="genero_label",
                points=False,
                labels={
                    "ingreso_mensual": "Ingreso mensual (S/)",
                    "genero_label": "Género",
                },
                color_discrete_map=COLOR_GENERO,
                category_orders={"genero_label": ["Masculino", "Femenino"]},
            )
            fig_box.update_layout(showlegend=False, height=420)
            fig_box.update_yaxes(tickformat=f".{ND}f")
            _plotly_base_layout(fig_box, "Ingreso mensual por género")
            st.caption("Comparación de ingreso mensual por género.")
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
                    "salario_hora": "Salario por hora (S/)",
                    "genero_label": "Género",
                },
                color_discrete_map=COLOR_GENERO,
                category_orders={"genero_label": ["Masculino", "Femenino"]},
            )
            fig_v.update_layout(showlegend=False, height=420)
            fig_v.update_yaxes(tickformat=f".{ND}f")
            _plotly_base_layout(fig_v, "Salario por hora (S/) — extremos recortados al p99.5")
            st.caption(
                "Salario por hora estimado a partir del ingreso y las horas. "
                "Extremos recortados al percentil 99.5."
            )
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
            xaxis_title="Ingreso mensual (S/)",
            yaxis_title="Frecuencia",
            height=400,
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        )
        _plotly_base_layout(fig_hist, "Ingreso mensual")
        fig_hist.update_xaxes(tickformat=f".{ND}f")
        st.caption("Distribución del ingreso mensual en la submuestra filtrada.")
        st.plotly_chart(fig_hist, use_container_width=True)

        st.caption(
            f"Medianas — Masculino: {_fmt_soles(med_m)} · Femenino: {_fmt_soles(med_f)} "
            "(submuestra filtrada)."
        )

    with tab3:
        st.subheader("Brecha por departamento")
        st.caption("Departamento de residencia.")
        dmap_plot = load_departamento_id_to_nombre()
        if "departamento_id" in dff.columns:
            bd = agregar_brecha_por_grupo(dff, "departamento_id", min_n=25)
            if not bd.empty:
                bd["departamento_nombre"] = bd["departamento_id"].apply(
                    lambda x: _departamento_nombre_from_id(dmap_plot, x)
                )
                y_dep = "departamento_nombre"
            else:
                y_dep = "departamento_id"
        else:
            bd = agregar_brecha_por_grupo(dff, "departamento", min_n=25)
            y_dep = "departamento"
        if bd.empty:
            st.warning("No hay suficientes observaciones por género en los grupos; afloja filtros.")
        else:
            bd_plot = bd.head(20).sort_values("brecha_pct", ascending=True)
            fig_d = px.bar(
                bd_plot,
                x="brecha_pct",
                y=y_dep,
                orientation="h",
                labels={"brecha_pct": "Brecha % (M vs F)", y_dep: "Departamento"},
                color="brecha_pct",
                color_continuous_scale=[
                    [0.0, C["neutro_claro"]],
                    [0.5, "#CBB6E6"],
                    [1.0, C["brecha"]],
                ],
            )
            fig_d.update_layout(height=520, yaxis={"categoryorder": "total ascending"})
            fig_d.update_xaxes(tickformat=f".{ND}f")
            _plotly_base_layout(fig_d, "Top departamentos por brecha de media")
            st.caption(
                "Brecha relativa entre las medias de ingreso mensual de hombres y de mujeres, por departamento."
            )
            st.plotly_chart(fig_d, use_container_width=True)

        c21, c22 = st.columns(2)
        map_sec = load_sector_laboral_map()
        map_tam = load_tamanio_empresa_map()
        with c21:
            st.subheader("Por sector laboral")

            bs = agregar_brecha_por_grupo(dff, "sector_laboral", min_n=40)
            if not bs.empty:
                bs_plot = bs.sort_values("brecha_pct", ascending=False).head(12).copy()
                bs_plot["sector_etiqueta"] = bs_plot["sector_laboral"].apply(
                    lambda v: _etiqueta_codigo(map_sec, v)
                )
                fig_s = px.bar(
                    bs_plot,
                    x="sector_etiqueta",
                    y="brecha_pct",
                    labels={"brecha_pct": "Brecha (%)", "sector_etiqueta": "Sector laboral"},
                    color_discrete_sequence=[C["brecha"]],
                )
                fig_s.update_layout(xaxis_title="Sector laboral", yaxis_title="Brecha (%)")
                fig_s.update_yaxes(tickformat=f".{ND}f")
                _plotly_base_layout(fig_s, "Brecha por sector laboral")
                st.plotly_chart(fig_s, use_container_width=True)
        with c22:
            st.subheader("Por tamaño de empresa")

            bt = agregar_brecha_por_grupo(dff, "tamanio_empresa", min_n=40)
            if not bt.empty:
                bt_plot = bt.sort_values("brecha_pct", ascending=False).copy()
                bt_plot["tamanio_etiqueta"] = bt_plot["tamanio_empresa"].apply(
                    lambda v: _etiqueta_codigo(map_tam, v)
                )
                fig_t = px.bar(
                    bt_plot,
                    x="tamanio_etiqueta",
                    y="brecha_pct",
                    labels={"brecha_pct": "Brecha (%)", "tamanio_etiqueta": "Tamaño de empresa"},
                    color_discrete_sequence=[C["brecha"]],
                )
                fig_t.update_layout(xaxis_title="Tamaño de empresa", yaxis_title="Brecha (%)")
                fig_t.update_yaxes(tickformat=f".{ND}f")
                _plotly_base_layout(fig_t, "Brecha por tamaño de empresa")
                st.plotly_chart(fig_t, use_container_width=True)

        st.divider()
        st.subheader("Carreras mejor pagadas por género")
        st.caption("Top 10 de carreras con los mejores ingresos promedio en la muestra actual.")
        map_carreras = load_carrera_map()
        if not dff.empty and "carrera_id" in dff.columns:
            c_counts = dff["carrera_id"].value_counts()
            valid_carreras = c_counts[c_counts >= 10].index
            if len(valid_carreras) < 3:
                valid_carreras = c_counts[c_counts >= 3].index
                
            df_carreras = dff[dff["carrera_id"].isin(valid_carreras)]
            if not df_carreras.empty:
                def _fmt_carrera(x):
                    return str(map_carreras.get(int(x), f"Cód {int(x)}")).title()

                top_carreras = df_carreras.groupby("carrera_id")["ingreso_mensual"].mean().nlargest(10).index
                df_top = df_carreras[df_carreras["carrera_id"].isin(top_carreras)].copy()
                
                df_top_group = df_top.groupby(["carrera_id", "genero_label"])["ingreso_mensual"].mean().reset_index()
                mean_order = df_top.groupby("carrera_id")["ingreso_mensual"].mean().sort_values(ascending=False).index
                    
                df_top_group["carrera_str"] = df_top_group["carrera_id"].apply(_fmt_carrera)
                
                fig_c = px.bar(
                    df_top_group,
                    x="carrera_str",
                    y="ingreso_mensual",
                    color="genero_label",
                    barmode="group",
                    labels={
                        "ingreso_mensual": "Ingreso promedio (S/)", 
                        "carrera_str": "Carrera",
                        "genero_label": "Género"
                    },
                    color_discrete_map=COLOR_GENERO,
                    category_orders={"carrera_str": [_fmt_carrera(x) for x in mean_order]}
                )
                fig_c.update_layout(xaxis_title="Carrera", yaxis_title="Ingreso Mensual Promedio (S/)")
                fig_c.update_yaxes(tickformat=f".{ND}f")
                _plotly_base_layout(fig_c, "Top 10 Carreras Mejor Pagadas")
                st.plotly_chart(fig_c, use_container_width=True)

                st.divider()
                st.subheader("Carreras peor pagadas por género")
                st.caption("Top 10 de carreras con los menores ingresos promedio en la muestra actual.")
                
                bottom_carreras = df_carreras.groupby("carrera_id")["ingreso_mensual"].mean().nsmallest(10).index
                df_bottom = df_carreras[df_carreras["carrera_id"].isin(bottom_carreras)].copy()
                
                df_bottom_group = df_bottom.groupby(["carrera_id", "genero_label"])["ingreso_mensual"].mean().reset_index()
                mean_order_b = df_bottom.groupby("carrera_id")["ingreso_mensual"].mean().sort_values(ascending=True).index
                    
                df_bottom_group["carrera_str"] = df_bottom_group["carrera_id"].apply(_fmt_carrera)
                
                fig_b = px.bar(
                    df_bottom_group,
                    x="carrera_str",
                    y="ingreso_mensual",
                    color="genero_label",
                    barmode="group",
                    labels={
                        "ingreso_mensual": "Ingreso promedio (S/)", 
                        "carrera_str": "Carrera",
                        "genero_label": "Género"
                    },
                    color_discrete_map=COLOR_GENERO,
                    category_orders={"carrera_str": [_fmt_carrera(x) for x in mean_order_b]}
                )
                fig_b.update_layout(xaxis_title="Carrera", yaxis_title="Ingreso Mensual Promedio (S/)")
                fig_b.update_yaxes(tickformat=f".{ND}f")
                _plotly_base_layout(fig_b, "Top 10 Carreras Peor Pagadas")
                st.plotly_chart(fig_b, use_container_width=True)
            else:
                st.warning("No hay suficientes datos por carrera para calcular el top de ingresos.")

    with tab4:
        st.markdown(
            f'<p style="color:{C["texto"]};">Predicción con <strong>CatBoost</strong> ya entrenado '
            "(carpeta <strong>artifacts</strong> del repositorio). El modelo estima el "
            "<strong>ingreso mensual</strong> (entrenado en escala logarítmica del ingreso, según "
            "los metadatos guardados).</p>",
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
                _fmt_soles(metrics["mae"]) if metrics.get("mae") is not None else "—",
            )
            m2.metric(
                "R² (validación)",
                _fmt_round(metrics["r2"]) if metrics.get("r2") is not None else "—",
            )


            st.subheader("Estimar ingreso con tus valores")
            st.caption(
                "Completa los campos con las categorías de la encuesta (ver glosario arriba). "
                "Los números son códigos de categoría, no montos monetarios salvo donde se indique."
            )
            dep_col_model = "departamento" if "departamento" in df.columns else "departamento_id"
            dmap_ui = load_departamento_id_to_nombre()
            map_sec_ui = load_sector_laboral_map()
            map_tam_ui = load_tamanio_empresa_map()
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
                if dep_col_model == "departamento_id" and dmap_ui:
                    nombres_ord = sorted(dmap_ui.values())
                    nombre_a_id = {v: k for k, v in dmap_ui.items()}
                    def_nm = _departamento_nombre_from_id(
                        dmap_ui, _df_int_default(df, "departamento_id", 15)
                    )
                    dep_idx = nombres_ord.index(def_nm) if def_nm in nombres_ord else 0
                    dep_nombre = st.selectbox(
                        "Departamento",
                        nombres_ord,
                        index=dep_idx,
                        help=_var_help("departamento_nombre"),
                    )
                    dep_sel = float(nombre_a_id[dep_nombre])
                else:
                    deps_all = sorted(df[dep_col_model].dropna().unique().tolist())
                    dep_sel = st.selectbox(
                        "Departamento",
                        deps_all,
                        index=0,
                        help=_var_help(dep_col_model),
                    )
            with gc2:
                opts_culm = _get_opciones(df, "culmino_estudio")
                def_culm = float(_df_int_default(df, "culmino_estudio", 1))
                culm = st.selectbox(
                    "¿Culminó estudios?",
                    options=opts_culm,
                    index=opts_culm.index(def_culm) if def_culm in opts_culm else 0,
                    format_func=lambda k: MAP_SI_NO.get(k, f"Opción {int(k)}"),
                    help=_var_help("culmino_estudio"),
                )
                def_ing_pe = _df_int_default(df, "ingreso_primer_empleo", 2)
                ing_pe_str = st.text_input(
                    "Ingreso primer empleo",
                    value=str(def_ing_pe),
                    help=_var_help("ingreso_primer_empleo") + " (Campo de texto)",
                )
                ing_pe = _safe_float(ing_pe_str)
                
                opts_cat_oc = _get_opciones(df, "categoria_ocupacional")
                def_cat_oc = float(_df_int_default(df, "categoria_ocupacional", 3))
                cat_oc = st.selectbox(
                    "Categoría ocupacional",
                    options=opts_cat_oc,
                    index=opts_cat_oc.index(def_cat_oc) if def_cat_oc in opts_cat_oc else 0,
                    format_func=lambda k: MAP_CAT_OCUPACIONAL.get(k, f"Opción {int(k)}"),
                    help=_var_help("categoria_ocupacional"),
                )
            gc3, gc4 = st.columns(2)
            with gc3:
                opts_t_con = _get_opciones(df, "tipo_contrato")
                def_t_con = float(_df_int_default(df, "tipo_contrato", 2))
                t_con = st.selectbox(
                    "Tipo de contrato",
                    options=opts_t_con,
                    index=opts_t_con.index(def_t_con) if def_t_con in opts_t_con else 0,
                    format_func=lambda k: MAP_TIPO_CONTRATO.get(k, f"Opción {int(k)}"),
                    help=_var_help("tipo_contrato"),
                )
                opts_sec = sorted(map_sec_ui.keys()) if map_sec_ui else [1, 2, 3, 4, 5]
                def_sec = _df_int_default(df, "sector_laboral", 5)
                ix_sec = opts_sec.index(def_sec) if def_sec in opts_sec else 0
                sec_cod = st.selectbox(
                    "Sector laboral",
                    options=opts_sec,
                    index=ix_sec,
                    format_func=lambda k: _etiqueta_codigo(map_sec_ui, k),
                    help=_var_help("sector_laboral")
                    + " El modelo recibe el código numérico (1–5), igual que en el entrenamiento.",
                )
                sec = float(sec_cod)
                opts_tam = sorted(map_tam_ui.keys()) if map_tam_ui else [1, 2, 3, 4, 5]
                def_tam = _df_int_default(df, "tamanio_empresa", 3)
                ix_tam = opts_tam.index(def_tam) if def_tam in opts_tam else 0
                tam_cod = st.selectbox(
                    "Tamaño de empresa",
                    options=opts_tam,
                    index=ix_tam,
                    format_func=lambda k: _etiqueta_codigo(map_tam_ui, k),
                    help=_var_help("tamanio_empresa")
                    + " El modelo recibe el código numérico (1–5), igual que en el entrenamiento.",
                )
                tam = float(tam_cod)
            with gc4:
                opts_tit = _get_opciones(df, "titulado")
                def_tit = float(_df_int_default(df, "titulado", 2))
                tit = st.selectbox(
                    "Titulado",
                    options=opts_tit,
                    index=opts_tit.index(def_tit) if def_tit in opts_tit else 0,
                    format_func=lambda k: MAP_SI_NO.get(k, f"Opción {int(k)}"),
                    help=_var_help("titulado"),
                )
                
                opts_cm = _get_opciones(df, "cuadro_merito")
                def_cm = float(_df_int_default(df, "cuadro_merito", 2))
                cm = st.selectbox(
                    "Cuadro de méritos",
                    options=opts_cm,
                    index=opts_cm.index(def_cm) if def_cm in opts_cm else 0,
                    format_func=lambda k: MAP_SI_NO.get(k, f"Opción {int(k)}"),
                    help=_var_help("cuadro_merito"),
                )
                
                opts_pg = _get_opciones(df, "postgrado")
                def_pg = float(_df_int_default(df, "postgrado", 2))
                pg = st.selectbox(
                    "Postgrado",
                    options=opts_pg,
                    index=opts_pg.index(def_pg) if def_pg in opts_pg else 0,
                    format_func=lambda k: MAP_SI_NO.get(k, f"Opción {int(k)}"),
                    help=_var_help("postgrado"),
                )
            gc5, gc6 = st.columns(2)
            with gc5:
                opts_idi = _get_opciones(df, "idioma_extranjero")
                def_idi = float(_df_int_default(df, "idioma_extranjero", 2))
                idi = st.selectbox(
                    "Idioma extranjero",
                    options=opts_idi,
                    index=opts_idi.index(def_idi) if def_idi in opts_idi else 0,
                    format_func=lambda k: MAP_IDIOMA_EXTRANJERO.get(k, f"Nivel / Opción {int(k)}"),
                    help=_var_help("idioma_extranjero"),
                )
                
                opts_pra = _get_opciones(df, "practicas_preprof")
                def_pra = float(_df_int_default(df, "practicas_preprof", 1))
                pra = st.selectbox(
                    "Prácticas preprofesionales",
                    options=opts_pra,
                    index=opts_pra.index(def_pra) if def_pra in opts_pra else 0,
                    format_func=lambda k: MAP_SI_NO.get(k, f"Opción {int(k)}"),
                    help=_var_help("practicas_preprof"),
                )
            with gc6:
                opts_ges = _get_opciones(df, "gestion_universidad")
                def_ges = float(_df_int_default(df, "gestion_universidad", 1))
                ges = st.selectbox(
                    "Gestión universidad",
                    options=opts_ges,
                    index=opts_ges.index(def_ges) if def_ges in opts_ges else 0,
                    format_func=lambda k: MAP_GEST_UNIV.get(k, f"Opción {int(k)}"),
                    help=_var_help("gestion_universidad"),
                )
                map_oc = load_ocupacion_map()
                opts_oc = _get_opciones(df, "ocupacion_cod")
                def_oc = float(_df_int_default(df, "ocupacion_cod", opts_oc[0] if opts_oc else 0))
                ocup_cod = st.selectbox(
                    "Ocupación principal",
                    options=opts_oc,
                    index=opts_oc.index(def_oc) if def_oc in opts_oc else 0,
                    format_func=lambda k: str(map_oc.get(int(k), f"Cód {int(k)}")).title(),
                    help=_var_help("ocupacion_cod"),
                )
                
                map_act = load_actividad_economica_map()
                opts_act = _get_opciones(df, "actividad_economica_cod")
                def_act = float(_df_int_default(df, "actividad_economica_cod", opts_act[0] if opts_act else 0))
                act_cod = st.selectbox(
                    "Actividad económica",
                    options=opts_act,
                    index=opts_act.index(def_act) if def_act in opts_act else 0,
                    format_func=lambda k: str(map_act.get(int(k), f"Cód {int(k)}")).title(),
                    help=_var_help("actividad_economica_cod"),
                )
                
            map_carr = load_carrera_map()
            opts_car = _get_opciones(df, "carrera_id")
            def_car = float(_df_int_default(df, "carrera_id", opts_car[0] if opts_car else 0))
            carrera_id_val = st.selectbox(
                "Carrera",
                options=opts_car,
                index=opts_car.index(def_car) if def_car in opts_car else 0,
                format_func=lambda k: str(map_carr.get(int(k), f"Cód {int(k)}")).title(),
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
                st.success(f"**Ingreso mensual estimado:** {_fmt_soles(pred)}")

            st.divider()
            st.subheader("Comparar predicción cambiando solo el género")
            row_m = row.copy()
            row_f = row.copy()
            row_m["genero"] = 1.0
            row_f["genero"] = 2.0
            Xm = _prepare_features_for_catboost(row_m, feat_cols)
            Xf = _prepare_features_for_catboost(row_f, feat_cols)
            pm, pf = _decode_pred(float(pipe.predict(Xm)[0])), _decode_pred(float(pipe.predict(Xf)[0]))
            st.write(f"Misma ficha con **Masculino**: {_fmt_soles(pm)}")
            st.write(f"Misma ficha con **Femenino**: {_fmt_soles(pf)}")
            if pm > 0:
                diff = (pm - pf) / pm * 100.0
                st.info(
                    f"Diferencia relativa del modelo entre ambas filas: **{round(diff, ND):.{ND}f} %** "
                    f"(asociación aprendida por el modelo, no efecto causal)."
                )


if __name__ == "__main__":
    main()
