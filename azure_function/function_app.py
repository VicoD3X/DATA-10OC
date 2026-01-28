import json
import logging
import os
from io import BytesIO
from typing import List, Tuple, Optional

import azure.functions as func
from azure.storage.blob import BlobServiceClient
import joblib
import numpy as np
import pandas as pd

app = func.FunctionApp(http_auth_level=func.AuthLevel.ANONYMOUS)

# Cache en mémoire (réutilisé tant que l'instance Azure reste active)
_X: Optional[np.ndarray] = None
_CLICKS: Optional[pd.DataFrame] = None
_TOP_POPULAR: Optional[List[int]] = None

_EPS = 1e-12

# Blob config
_CONTAINER = "mycontent-assets"
_EMB_BLOB = "articles_embeddings_pca50.joblib"
_CLICKS_BLOB = "clicks_sample.csv"

# Cache du client blob
_BSC: Optional[BlobServiceClient] = None


def _get_blob_service() -> BlobServiceClient:
    global _BSC
    if _BSC is not None:
        return _BSC

    conn = os.environ.get("RECO_STORAGE")
    if not conn:
        raise RuntimeError("Missing RECO_STORAGE environment variable")

    _BSC = BlobServiceClient.from_connection_string(conn)
    return _BSC


def _load_blob_bytes(container: str, blob_name: str) -> bytes:
    bsc = _get_blob_service()
    bc = bsc.get_container_client(container)
    return bc.download_blob(blob_name).readall()


def _l2_normalize(X: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    return X / np.maximum(norms, eps)


def _build_top_popular(clicks: pd.DataFrame, n: int = 200) -> List[int]:
    return (
        clicks["click_article_id"]
        .value_counts()
        .head(n)
        .index
        .astype(int)
        .tolist()
    )


def _ensure_assets_loaded():
    global _X, _CLICKS, _TOP_POPULAR

    if _X is None:
        logging.info("Loading embeddings from Blob Storage (SDK)")
        emb_bytes = _load_blob_bytes(_CONTAINER, _EMB_BLOB)
        X = joblib.load(BytesIO(emb_bytes))
        X = np.asarray(X, dtype=np.float32)
        X = _l2_normalize(X, eps=_EPS)
        _X = X

    if _CLICKS is None:
        logging.info("Loading clicks from Blob Storage (SDK)")
        clicks_bytes = _load_blob_bytes(_CONTAINER, _CLICKS_BLOB)
        _CLICKS = pd.read_csv(BytesIO(clicks_bytes))

    if _TOP_POPULAR is None:
        _TOP_POPULAR = _build_top_popular(_CLICKS, n=200)

    return _X, _CLICKS, _TOP_POPULAR


def _recommend(
    user_id: int,
    X: np.ndarray,
    clicks: pd.DataFrame,
    top_popular: List[int],
    k: int = 5,
    pool: int = 300,
) -> Tuple[List[int], str]:

    hist = clicks.loc[clicks["user_id"] == user_id, "click_article_id"].astype(int).tolist()
    hist = [a for a in hist if 0 <= a < X.shape[0]]

    if len(hist) == 0:
        return top_popular[:k], "cold_start_popularity"

    u = X[hist].mean(axis=0)
    u = u / max(np.linalg.norm(u), _EPS)

    scores = X @ u
    pool = min(pool, X.shape[0] - 1)

    idx = np.argpartition(-scores, pool)[:pool]
    idx = idx[np.argsort(-scores[idx])]

    hist_set = set(hist)
    recs: List[int] = []

    for i in idx:
        aid = int(i)
        if aid not in hist_set:
            recs.append(aid)
        if len(recs) == k:
            break

    if len(recs) < k:
        for aid in top_popular:
            if aid not in hist_set and aid not in recs:
                recs.append(aid)
            if len(recs) == k:
                break

    return recs, "content_based_pca50"


@app.function_name(name="recommend")
@app.route(route="recommend", methods=["GET", "POST"])
def recommend(req: func.HttpRequest) -> func.HttpResponse:
    try:
        user_id_raw = req.params.get("user_id")
        k_raw = req.params.get("k", "5")

        if user_id_raw is None and req.method == "POST":
            body = req.get_json()
            user_id_raw = body.get("user_id")
            k_raw = body.get("k", 5)

        if user_id_raw is None:
            return func.HttpResponse(
                json.dumps({"error": "Missing parameter: user_id"}),
                status_code=400,
                mimetype="application/json",
            )

        user_id = int(user_id_raw)
        k = max(1, min(int(k_raw), 10))

        X, clicks, top_popular = _ensure_assets_loaded()
        recs, mode = _recommend(user_id, X, clicks, top_popular, k=k)

        return func.HttpResponse(
            json.dumps({
                "user_id": user_id,
                "k": k,
                "mode": mode,
                "recommendations": recs,
            }),
            status_code=200,
            mimetype="application/json",
        )

    except Exception as e:
        logging.exception("Recommendation error")
        return func.HttpResponse(
            json.dumps({"error": str(e)}),
            status_code=500,
            mimetype="application/json",
        )
