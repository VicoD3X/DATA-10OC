# streamlit_app.py
# ------------------------------------------------------------
# MyContent — MVP Recommandation
# Client Streamlit consommant une API Azure Functions
# ------------------------------------------------------------

import time
import requests
import pandas as pd
import streamlit as st


# =========================
# Configuration générale
# =========================
API_URL = "https://p10oc-recommender.azurewebsites.net/api/recommend"
DATA_DIR = "data"

st.set_page_config(
    page_title="MyContent — MVP Recommandation",
    layout="centered"
)

st.title("MyContent — MVP de recommandation d’articles")
st.write(
    "Cette application Streamlit consomme une API Azure Functions exposant "
    "un moteur de recommandation d’articles. Aucun calcul de machine learning "
    "n’est effectué côté interface."
)


# =========================
# Chargement des données locales
# (métadonnées et baseline)
# =========================
@st.cache_data
def load_data():
    articles = pd.read_csv(f"{DATA_DIR}/articles_metadata.csv")
    clicks = pd.read_csv(f"{DATA_DIR}/clicks_sample.csv")

    # Liste des utilisateurs disponibles (dataset local)
    user_ids = sorted(
        clicks["user_id"]
        .dropna()
        .astype(int)
        .unique()
        .tolist()
    )

    # Articles les plus populaires (baseline simple)
    top_popular = (
        clicks["click_article_id"]
        .dropna()
        .astype(int)
        .value_counts()
        .head(10)
        .index
        .tolist()
    )

    return articles, clicks, user_ids, top_popular


articles, clicks, user_ids, top_popular = load_data()


# =========================
# Interface utilisateur
# =========================
user_id = st.selectbox("Identifiant utilisateur", user_ids)
k = st.slider("Nombre d’articles recommandés", 1, 10, 5)

show_raw = st.checkbox(
    "Afficher la réponse JSON brute (debug)",
    value=False
)


# =========================
# Appel de l’API Azure
# =========================
if st.button("Générer les recommandations"):
    with st.spinner("Appel du moteur de recommandation Azure…"):
        t0 = time.perf_counter()
        response = requests.get(
            API_URL,
            params={"user_id": int(user_id), "k": int(k)},
            timeout=10
        )
        latency = time.perf_counter() - t0

    if response.status_code != 200:
        st.error(
            f"Erreur lors de l’appel à l’API Azure "
            f"(HTTP {response.status_code})."
        )
        st.stop()

    data = response.json()

    if show_raw:
        st.json(data)

    recs = data.get("recommendations", [])
    mode = data.get("mode", "unknown")

    st.success(
        f"Résultats pour l’utilisateur {user_id} "
        f"(mode : {mode}, latence : {latency:.2f}s)"
    )

    if not recs:
        st.warning("Aucune recommandation retournée.")
        st.stop()

    # Recommandations personnalisées
    df_recs = (
        pd.DataFrame({"article_id": recs})
        .merge(articles, on="article_id", how="left")
    )
    st.dataframe(df_recs, use_container_width=True)

    # Baseline popularité (comparaison)
    st.subheader("Baseline : popularité globale")
    df_base = (
        pd.DataFrame({"article_id": top_popular[:int(k)]})
        .merge(articles, on="article_id", how="left")
    )
    st.dataframe(df_base, use_container_width=True)
