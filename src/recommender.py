from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List

import joblib          # Type: ignore
import numpy as np     # Type: ignore
import pandas as pd    # Type: ignore


@dataclass
class Recommender:
    X: np.ndarray                 # Matrice d'embeddings L2-normalisés (cosine)
    top_popular: List[int]        # Articles les plus populaires (fallback)
    eps: float = 1e-12            # Terme de stabilité numérique

    def recommend(self, user_id: int, clicks: pd.DataFrame, k: int = 5, pool: int = 300) -> List[int]:
        # Récupération de l’historique de clics de l’utilisateur
        hist = clicks.loc[clicks["user_id"] == user_id, "click_article_id"].astype(int).tolist()
        hist = [a for a in hist if 0 <= a < self.X.shape[0]]  # Filtrage des IDs invalides

        # Cas cold-start : aucun historique disponible
        if len(hist) == 0:
            return self.top_popular[:k]

        # Construction du profil utilisateur par moyenne des embeddings
        u = self.X[hist].mean(axis=0)
        u = u / max(np.linalg.norm(u), self.eps)  # Normalisation L2

        # Calcul des similarités cosine avec tous les articles
        scores = self.X @ u

        # Sélection rapide des meilleurs candidats sans tri global
        pool = min(pool, self.X.shape[0] - 1)
        idx = np.argpartition(-scores, pool)[:pool]
        idx = idx[np.argsort(-scores[idx])]  # Tri final des candidats

        # Filtrage des articles déjà cliqués
        hist_set = set(hist)
        recs: List[int] = []
        for i in idx:
            aid = int(i)  # index = article_id
            if aid not in hist_set:
                recs.append(aid)
            if len(recs) == k:
                break

        # Complément avec les articles populaires si nécessaire
        if len(recs) < k:
            for aid in self.top_popular:
                if aid not in hist_set and aid not in recs:
                    recs.append(aid)
                if len(recs) == k:
                    break

        return recs


def _l2_normalize(X: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    norms = np.linalg.norm(X, axis=1, keepdims=True)  # Norme L2 par vecteur
    return X / np.maximum(norms, eps)                # Normalisation sécurisée


def build_top_popular(clicks: pd.DataFrame, n: int = 200) -> List[int]:
    # Construction de la liste des articles les plus cliqués
    return (
        clicks["click_article_id"]
        .value_counts()
        .head(n)
        .index
        .astype(int)
        .tolist()
    )


def load_recommender(
    data_dir: str | Path,
    clicks_filename: str = "clicks_sample.csv",
    embeddings_filename: str = "articles_embeddings_pca50.joblib",
) -> tuple[Recommender, pd.DataFrame]:
    """
    Chargement des ressources nécessaires au recommender :
    - historique de clics
    - embeddings articles
    - popularité globale
    """
    data_dir = Path(data_dir)

    clicks_path = data_dir / clicks_filename
    emb_path = data_dir / embeddings_filename

    clicks = pd.read_csv(clicks_path)  # Données de clics utilisateurs

    X_reduced = joblib.load(emb_path)   # Embeddings PCA des articles
    X = X_reduced.astype("float32")
    X = _l2_normalize(X)                # Normalisation pour cosine similarity

    top_popular = build_top_popular(clicks, n=200)

    rec = Recommender(X=X, top_popular=top_popular)
    return rec, clicks
