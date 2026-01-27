from src.recommender import load_recommender

rec, clicks = load_recommender(data_dir="data")

for uid in [0, 1, 2, 999999]:
    print(uid, "->", rec.recommend(uid, clicks, k=5))
