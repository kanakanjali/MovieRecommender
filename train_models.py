"""
train_models.py
Run this once after download_data.py to train and save all models.
    python train_models.py
"""

import os
import sys
import time
import pickle
import pandas as pd

BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
DATA_DIR   = os.path.join(BASE_DIR, "data")
MODEL_DIR  = os.path.join(BASE_DIR, "models")

sys.path.insert(0, BASE_DIR)

from src.content_based import ContentBasedRecommender
from src.collaborative_filtering import CollaborativeFilteringRecommender
from src.cold_start import ColdStartRecommender


def load_data():
    movies_path  = os.path.join(DATA_DIR, "movies.csv")
    ratings_path = os.path.join(DATA_DIR, "ratings.csv")
    if not os.path.exists(movies_path) or not os.path.exists(ratings_path):
        print("ERROR: data files not found. Run 'python download_data.py' first.")
        sys.exit(1)
    movies  = pd.read_csv(movies_path)
    ratings = pd.read_csv(ratings_path)
    print(f"Loaded {len(movies)} movies and {len(ratings)} ratings.")
    return movies, ratings


def train_content(movies):
    print("\n[1/3] Training Content-Based model (TF-IDF + cosine similarity) …")
    t = time.time()
    cb = ContentBasedRecommender()
    cb.fit(movies)
    elapsed = round(time.time() - t, 1)
    print(f"    Done in {elapsed}s")
    # Sanity check with a known title
    known = movies["title"].iloc[0]
    sample = cb.recommend(known, n=3)
    print(f"    '{known}' → {list(sample['title'])}")
    return cb


def train_collaborative(movies, ratings):
    print("\n[2/3] Training Collaborative Filtering (SVD) …")
    t = time.time()
    cf = CollaborativeFilteringRecommender(n_factors=50)
    metrics = cf.fit(ratings, movies)
    elapsed = round(time.time() - t, 1)
    print(f"    Done in {elapsed}s")
    print(f"    RMSE = {metrics['rmse']}  |  MAE = {metrics['mae']}")
    return cf, metrics


def train_cold_start(movies, ratings):
    print("\n[3/3] Building Cold-Start recommender …")
    cs = ColdStartRecommender()
    cs.fit(movies, ratings)
    top = cs.top_rated(n=3)
    print(f"    Top rated: {list(top['title'])}")
    os.makedirs(MODEL_DIR, exist_ok=True)
    with open(os.path.join(MODEL_DIR, "cold_start.pkl"), "wb") as f:
        pickle.dump(cs, f)
    return cs


def save_metrics(cf_metrics):
    os.makedirs(MODEL_DIR, exist_ok=True)
    metrics = {
        "collaborative": cf_metrics,
        "content": {"rmse": None, "mae": None},
    }
    with open(os.path.join(MODEL_DIR, "metrics.pkl"), "wb") as f:
        pickle.dump(metrics, f)
    print(f"\nMetrics saved → {os.path.join(MODEL_DIR, 'metrics.pkl')}")


if __name__ == "__main__":
    movies, ratings = load_data()
    cb              = train_content(movies)
    cf, cf_metrics  = train_collaborative(movies, ratings)
    cs              = train_cold_start(movies, ratings)
    save_metrics(cf_metrics)
    print("\n✅ All models trained and saved to /models/")
    print("   Now run:  streamlit run app.py")
