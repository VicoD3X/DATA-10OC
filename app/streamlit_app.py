import sys
from pathlib import Path

# Ajout de la racine du projet au PYTHONPATH pour importer src/
ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

import streamlit as st
import pandas as pd

from src.recommender import load_recommender

# Configuration générale de l'application
st.set_page_config(page_title="MyContent — MVP Recommandation", layout="centered")

# Titre et description fonctionnelle
st.title("MyContent — MVP de recommandation d’articles")
st.write(
    "Cette application permet de générer une liste d’articles recommandés "
    "à partir de l’historique de navigation d’un utilisateur."
)

@st.cache_resource
def load_assets():
    # Chargement du moteur de recommandation, des données de clics et des métadonnées articles
    rec, clicks = load_recommender(data_dir="data")
    articles = pd.read_csv("data/articles_metadata.csv")
    user_ids = sorted(clicks["user_id"].unique().tolist())
    return rec, clicks, articles, user_ids

# Initialisation des ressources
rec, clicks, articles, user_ids = load_assets()

# Paramètres de génération des recommandations
user_id = st.selectbox("Identifiant utilisateur", user_ids)
k = st.slider("Nombre d’articles recommandés", 1, 10, 5)

# Option d’affichage de la recommandation par popularité
show_popularity = st.checkbox(
    "Afficher la recommandation basée sur la popularité globale",
    value=False
)

# Déclenchement de la recommandation
if st.button("Générer les recommandations"):
    # Calcul des recommandations personnalisées
    recs = rec.recommend(user_id, clicks, k=k)

    st.success(f"Résultats pour l’utilisateur {user_id}")

    # Affichage de la liste brute des identifiants recommandés
    st.write("Identifiants des articles recommandés :", recs)

    # Tableau enrichi avec les métadonnées des articles
    df_recs = (
        pd.DataFrame({"article_id": recs})
        .merge(articles, on="article_id", how="left")
    )
    st.dataframe(df_recs, use_container_width=True)

    # Affichage optionnel de la baseline popularité
    if show_popularity:
        base = rec.top_popular[:k]
        st.info("Recommandation de référence basée sur la popularité")
        st.write("Identifiants des articles populaires :", base)
        df_base = (
            pd.DataFrame({"article_id": base})
            .merge(articles, on="article_id", how="left")
        )
        st.dataframe(df_base, use_container_width=True)
