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
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

BASE = Path(__file__).resolve().parent
CSV_PATH = BASE / "df_final.csv"

GENERO_MAP = {1.0: "Masculino", 2.0: "Femenino"}
GENERO_INV = {"Masculino": 1.0, "Femenino": 2.0}


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


def build_model_pipe(df: pd.DataFrame):
    """Pipeline para predecir ingreso_mensual."""
    d = df.copy()
    d = d.dropna(subset=["ingreso_mensual", "genero"])
    d = d[d["ingreso_mensual"] > 0]

    num_cols = [
        "culmino_estudio",
        "genero",
        "horas_semanales",
        "ingreso_primer_empleo",
        "categoria_ocupacional",
        "tipo_contrato",
        "sector_laboral",
        "tamanio_empresa",
        "titulado",
        "cuadro_merito",
        "postgrado",
        "idioma_extranjero",
        "practicas_preprof",
        "gestion_universidad",
    ]
    cat_cols = ["departamento"]

    X = d[num_cols + cat_cols].copy()
    y = d["ingreso_mensual"].astype(float)

    num_pipe = Pipeline(
        steps=[("imputer", SimpleImputer(strategy="median"))]
    )
    cat_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False, max_categories=30)),
        ]
    )
    pre = ColumnTransformer(
        transformers=[
            ("num", num_pipe, num_cols),
            ("cat", cat_pipe, cat_cols),
        ],
        remainder="drop",
    )
    model = RandomForestRegressor(
        n_estimators=120,
        max_depth=14,
        min_samples_leaf=4,
        random_state=42,
        n_jobs=-1,
    )
    pipe = Pipeline(steps=[("pre", pre), ("rf", model)])
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    pipe.fit(X_train, y_train)
    pred = pipe.predict(X_test)
    metrics = {
        "mae": float(mean_absolute_error(y_test, pred)),
        "r2": float(r2_score(y_test, pred)),
    }
    return pipe, num_cols, cat_cols, metrics, X.columns.tolist()


def main():
    st.set_page_config(
        page_title="Brecha salarial por género",
        page_icon="📊",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    st.title("Brecha salarial entre géneros")
    st.caption(
        "Panel exploratorio y modelo predictivo de **ingreso mensual** — datos de egresados "
        "(codificación INEI: 1 Masculino, 2 Femenino)."
    )

    df = load_data()

    with st.sidebar:
        st.header("Filtros")
        deps = sorted(df["departamento"].dropna().unique().tolist())
        sel_dep = st.multiselect(
            "Departamento (vacío = todos)",
            deps,
            default=[],
            help="Si no eliges ninguno, se usan todos los departamentos.",
        )
        ing_min, ing_max = float(df["ingreso_mensual"].min()), float(df["ingreso_mensual"].max())
        r_ing = st.slider(
            "Rango ingreso mensual (S/)",
            min_value=0.0,
            max_value=float(max(ing_max, 1)),
            value=(0.0, float(max(ing_max, 1))),
        )
        r_horas = st.slider(
            "Horas semanales (máx.)",
            min_value=1,
            max_value=max(int(df["horas_semanales"].max()), 1),
            value=max(int(df["horas_semanales"].quantile(0.99)), 48),
        )
        st.divider()
        st.markdown("**Dataset**")
        st.text(f"Filas: {len(df):,}")
        st.text(f"Columnas: {df.shape[1]}")

    dff = df.copy()
    if len(sel_dep) > 0:
        dff = dff[dff["departamento"].isin(sel_dep)]
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
    c1.metric("Media ingreso — Masculino", f"S/ {m_m:,.0f}" if len(hom) else "—")
    c2.metric("Media ingreso — Femenino", f"S/ {m_f:,.0f}" if len(muj) else "—")
    if gap is not None:
        c3.metric("Brecha (media)", f"{gap:.1f} %", help="(media M − media F) / media M × 100")
    else:
        c3.metric("Brecha (media)", "—")
    c4.metric("Observaciones (filtradas)", f"{len(dff):,}")

    if gap is not None and gap > 0:
        st.info(
            f"Con los filtros actuales, el ingreso mensual **promedio** de mujeres es "
            f"aproximadamente **{gap:.1f} % menor** que el de hombres (misma métrica de brecha relativa)."
        )
    elif gap is not None and gap < 0:
        st.info(
            f"Con los filtros actuales, la media femenina supera a la masculina en magnitud relativa "
            f"(brecha calculada: **{gap:.1f} %**; interpretar con precaución por composición muestral)."
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
                title="Ingreso mensual por género",
                labels={"ingreso_mensual": "S/ mensual", "genero_label": "Género"},
            )
            fig_box.update_layout(showlegend=False, height=420)
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
                title="Salario por hora (S/) — valores extremos recortados al percentil 99.5",
                labels={"salario_hora": "S/ por hora", "genero_label": "Género"},
            )
            fig_v.update_layout(showlegend=False, height=420)
            st.plotly_chart(fig_v, use_container_width=True)

        st.subheader("Histograma comparativo")
        fig_hist = go.Figure()
        for label, series in [("Masculino", hom), ("Femenino", muj)]:
            fig_hist.add_trace(
                go.Histogram(
                    x=series,
                    name=label,
                    opacity=0.65,
                    nbinsx=45,
                )
            )
        fig_hist.update_layout(
            barmode="overlay",
            title="Ingreso mensual",
            xaxis_title="S/",
            yaxis_title="Frecuencia",
            height=400,
        )
        st.plotly_chart(fig_hist, use_container_width=True)

        st.caption(
            f"Medianas — Masculino: S/ {med_m:,.0f} · Femenino: S/ {med_f:,.0f} "
            f"(en submuestra filtrada)."
        )

    with tab2:
        st.subheader("Brecha por departamento")
        bd = agregar_brecha_por_grupo(dff, "departamento", min_n=25)
        if bd.empty:
            st.warning("No hay suficientes observaciones por género en los grupos; afloja filtros.")
        else:
            bd_plot = bd.head(20).sort_values("brecha_pct", ascending=True)
            fig_d = px.bar(
                bd_plot,
                x="brecha_pct",
                y="departamento",
                orientation="h",
                title="Top departamentos por brecha de media (ingreso mensual)",
                labels={"brecha_pct": "Brecha % (M vs F)", "departamento": "Departamento"},
                color="brecha_pct",
                color_continuous_scale="RdBu_r",
            )
            fig_d.update_layout(height=520, yaxis={"categoryorder": "total ascending"})
            st.plotly_chart(fig_d, use_container_width=True)

        c21, c22 = st.columns(2)
        with c21:
            st.subheader("Por sector laboral (código)")
            bs = agregar_brecha_por_grupo(dff, "sector_laboral", min_n=40)
            if not bs.empty:
                fig_s = px.bar(
                    bs.sort_values("brecha_pct", ascending=False).head(12),
                    x="sector_laboral",
                    y="brecha_pct",
                    title="Brecha media % por sector",
                    labels={"brecha_pct": "Brecha %", "sector_laboral": "Sector (cod.)"},
                )
                st.plotly_chart(fig_s, use_container_width=True)
        with c22:
            st.subheader("Por tamaño de empresa (código)")
            bt = agregar_brecha_por_grupo(dff, "tamanio_empresa", min_n=40)
            if not bt.empty:
                fig_t = px.bar(
                    bt.sort_values("brecha_pct", ascending=False),
                    x="tamanio_empresa",
                    y="brecha_pct",
                    title="Brecha media % por tamaño",
                    labels={"brecha_pct": "Brecha %", "tamanio_empresa": "Tamaño (cod.)"},
                )
                st.plotly_chart(fig_t, use_container_width=True)

    with tab3:
        st.markdown(
            "Modelo **Random Forest** sobre variables numéricas codificadas + **departamento** "
            "(one-hot acotado). Objetivo: **ingreso mensual**."
        )
        with st.spinner("Entrenando modelo…"):
            try:
                pipe, num_cols, cat_cols, metrics, _ = build_model_pipe(df)
            except Exception as e:
                st.error(f"No se pudo entrenar el modelo: {e}")
                return

        m1, m2 = st.columns(2)
        m1.metric("MAE (validación)", f"S/ {metrics['mae']:,.0f}")
        m2.metric("R² (validación)", f"{metrics['r2']:.3f}")

        st.subheader("Estimar ingreso con tus valores")
        deps_all = sorted(df["departamento"].dropna().unique().tolist())
        gc1, gc2 = st.columns(2)
        with gc1:
            gen_sel = st.selectbox("Género", ["Masculino", "Femenino"])
            horas = st.number_input("Horas semanales", min_value=1, max_value=100, value=40)
            dep_sel = st.selectbox("Departamento", deps_all, index=deps_all.index("LIMA") if "LIMA" in deps_all else 0)
        with gc2:
            culm = st.number_input("Culminó estudio (cod.)", min_value=0.0, max_value=5.0, value=1.0)
            ing_pe = st.number_input("Ingreso primer empleo (cod., opcional)", min_value=0.0, max_value=20.0, value=2.0)
            cat_oc = st.number_input("Categoría ocupacional (cod.)", min_value=0.0, max_value=10.0, value=3.0)
        gc3, gc4 = st.columns(2)
        with gc3:
            t_con = st.number_input("Tipo contrato (cod.)", min_value=0.0, max_value=10.0, value=2.0)
            sec = st.number_input("Sector laboral (cod.)", min_value=0.0, max_value=10.0, value=5.0)
            tam = st.number_input("Tamaño empresa (cod.)", min_value=0.0, max_value=10.0, value=3.0)
        with gc4:
            tit = st.number_input("Titulado (cod.)", min_value=0.0, max_value=5.0, value=2.0)
            cm = st.number_input("Cuadro mérito (cod.)", min_value=0.0, max_value=5.0, value=2.0)
            pg = st.number_input("Postgrado (cod.)", min_value=0.0, max_value=5.0, value=2.0)
        gc5, gc6 = st.columns(2)
        with gc5:
            idi = st.number_input("Idioma extranjero (cod.)", min_value=0.0, max_value=10.0, value=2.0)
            pra = st.number_input("Prácticas preprof. (cod.)", min_value=0.0, max_value=10.0, value=1.0)
        with gc6:
            ges = st.number_input("Gestión universidad (cod.)", min_value=0.0, max_value=10.0, value=1.0)

        row = pd.DataFrame(
            [
                {
                    "culmino_estudio": culm,
                    "genero": GENERO_INV[gen_sel],
                    "horas_semanales": float(horas),
                    "ingreso_primer_empleo": ing_pe,
                    "categoria_ocupacional": cat_oc,
                    "tipo_contrato": t_con,
                    "sector_laboral": sec,
                    "tamanio_empresa": tam,
                    "titulado": tit,
                    "cuadro_merito": cm,
                    "postgrado": pg,
                    "idioma_extranjero": idi,
                    "practicas_preprof": pra,
                    "gestion_universidad": ges,
                    "departamento": dep_sel,
                }
            ]
        )

        if st.button("Predecir ingreso mensual", type="primary"):
            pred = float(pipe.predict(row)[0])
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
        pm, pf = float(pipe.predict(row_m)[0]), float(pipe.predict(row_f)[0])
        st.write(f"Misma ficha con **Masculino**: S/ {pm:,.0f}")
        st.write(f"Misma ficha con **Femenino**: S/ {pf:,.0f}")
        if pm > 0:
            diff = (pm - pf) / pm * 100.0
            st.info(
                f"Diferencia relativa del modelo entre ambas filas: **{diff:.1f} %** "
                f"(asociación aprendida por el bosque, no efecto causal)."
            )


if __name__ == "__main__":
    main()
