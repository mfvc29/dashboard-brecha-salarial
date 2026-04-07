# Dashboard — Brecha salarial por género

App Streamlit para el datathon. Datos: `df_final.csv` (misma carpeta que el código).

## Despliegue en Streamlit Community Cloud

1. Sube **esta carpeta** como raíz del repositorio en GitHub (o apunta el *Main file path* a la ruta correcta si el repo incluye más carpetas).
2. Entra a [share.streamlit.io](https://share.streamlit.io) e inicia sesión con GitHub.
3. **New app** → elige el repo y la rama.
4. **Main file path:** `streamlit_app.py`  
   (Alternativa: `app.py` y configúralo igual en la pantalla de la app.)
5. **Deploy.** La primera vez puede tardar unos minutos en instalar dependencias.

### Archivos que usa la nube

| Archivo | Rol |
|--------|-----|
| `streamlit_app.py` | Entrada recomendada para Cloud |
| `app.py` | Lógica del dashboard |
| `requirements.txt` | Dependencias pip |
| `df_final.csv` | Dataset (debe estar en el repo) |

No subas `.streamlit/secrets.toml` con secretos; ya está ignorado en `.gitignore`.

## Ejecutar en local

```bash
pip install -r requirements.txt
streamlit run streamlit_app.py
```
