"""
src/collaborative_filtering.py
Collaborative Filtering using SVD (scipy) — no external surprise library needed.
Uses scipy.sparse.linalg.svds for matrix factorisation, with proper train/test RMSE & MAE.
"""

import os
import pickle

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import svds
from sklearn.model_selection import KFold

MODEL_DIR = os.path.join(os.path.dirname(__file__), "..", "models")
SVD_PATH  = os.path.join(MODEL_DIR, "svd_model.pkl")


class CollaborativeFilteringRecommender:
    """
    User-based collaborative filtering via truncated SVD.
    Works entirely with scipy/numpy — no external surprise library required.
    """

    def __init__(self, n_factors: int = 50):
        self.n_factors = n_factors
        self.U = None        # user factors
        self.sigma = None    # singular values
        self.Vt = None       # item factors
        self.user_ratings_mean = None
        self.user_index: dict  = {}   # userId -> row index
        self.movie_index: dict = {}   # movieId -> col index
        self.idx_to_user: dict  = {}
        self.idx_to_movie: dict = {}
        self.movies: pd.DataFrame = None
        self.ratings: pd.DataFrame = None
        self.rmse: float = None
        self.mae: float  = None

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------
    def fit(self, ratings: pd.DataFrame, movies: pd.DataFrame) -> dict:
        self.ratings = ratings.copy()
        self.movies  = movies.copy()

        # Build index maps
        users  = sorted(ratings["userId"].unique())
        mids   = sorted(ratings["movieId"].unique())
        self.user_index   = {u: i for i, u in enumerate(users)}
        self.movie_index  = {m: j for j, m in enumerate(mids)}
        self.idx_to_user  = {i: u for u, i in self.user_index.items()}
        self.idx_to_movie = {j: m for m, j in self.movie_index.items()}

        n_users  = len(users)
        n_movies = len(mids)

        # ── 80/20 train/test split ────────────────────────────────────
        from sklearn.model_selection import train_test_split
        train_df, test_df = train_test_split(ratings, test_size=0.20, random_state=42)

        # Build user-item matrix from TRAIN only
        rows = train_df["userId"].map(self.user_index).values
        cols = train_df["movieId"].map(self.movie_index).values
        vals = train_df["rating"].values.astype(float)

        mat = csr_matrix((vals, (rows, cols)), shape=(n_users, n_movies))

        # Mean-centre by user
        mat_dense = mat.toarray()
        self.user_ratings_mean = np.true_divide(
            mat_dense.sum(axis=1),
            (mat_dense != 0).sum(axis=1).clip(min=1),
        )
        mat_demeaned = mat_dense.copy()
        for i in range(n_users):
            mask = mat_dense[i] != 0
            mat_demeaned[i, mask] -= self.user_ratings_mean[i]

        # SVD
        k = min(self.n_factors, min(n_users, n_movies) - 1)
        self.U, self.sigma, self.Vt = svds(
            csr_matrix(mat_demeaned), k=k
        )
        # svds returns singular values in ascending order — reverse for descending
        idx = np.argsort(self.sigma)[::-1]
        self.U     = self.U[:, idx]
        self.sigma = self.sigma[idx]
        self.Vt    = self.Vt[idx, :]

        # Reconstruct predicted ratings
        self._predicted_all = (
            np.dot(np.dot(self.U, np.diag(self.sigma)), self.Vt)
            + self.user_ratings_mean.reshape(-1, 1)
        )
        # Clip to valid range
        self._predicted_all = np.clip(self._predicted_all, 1.0, 5.0)

        # ── Evaluate on test set ──────────────────────────────────────
        errors, abs_errors = [], []
        for _, row in test_df.iterrows():
            uid  = row["userId"]
            mid  = row["movieId"]
            true = row["rating"]
            # Only evaluate if user and movie are in training index
            if uid in self.user_index and mid in self.movie_index:
                pred = self._predicted_all[
                    self.user_index[uid], self.movie_index[mid]
                ]
                errors.append((true - pred) ** 2)
                abs_errors.append(abs(true - pred))

        self.rmse = round(float(np.sqrt(np.mean(errors))), 4) if errors else None
        self.mae  = round(float(np.mean(abs_errors)), 4)      if abs_errors else None

        os.makedirs(MODEL_DIR, exist_ok=True)
        self.save()
        return {"rmse": self.rmse, "mae": self.mae}

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------
    def predict_rating(self, user_id: int, movie_id: int) -> float:
        if self.U is None:
            raise RuntimeError("Call fit() first.")
        if user_id not in self.user_index or movie_id not in self.movie_index:
            # Fall back to global mean
            return round(float(self.ratings["rating"].mean()), 4)
        ui = self.user_index[user_id]
        mi = self.movie_index[movie_id]
        return round(float(self._predicted_all[ui, mi]), 4)

    def recommend(self, user_id: int, n: int = 10) -> pd.DataFrame:
        if self.U is None:
            raise RuntimeError("Call fit() first.")

        rated_ids = set(
            self.ratings[self.ratings["userId"] == user_id]["movieId"].tolist()
        )
        all_movie_ids = self.movies["movieId"].tolist()
        unrated = [mid for mid in all_movie_ids if mid not in rated_ids]

        preds = [(mid, self.predict_rating(user_id, mid)) for mid in unrated]
        preds.sort(key=lambda x: x[1], reverse=True)
        top_n = preds[:n]

        top_ids    = [mid for mid, _ in top_n]
        top_scores = {mid: round(est, 4) for mid, est in top_n}

        result = self.movies[self.movies["movieId"].isin(top_ids)][
            ["movieId", "title", "genres"]
        ].copy()
        result["predicted_rating"] = result["movieId"].map(top_scores)
        result = result.sort_values("predicted_rating", ascending=False)
        return result.reset_index(drop=True)

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------
    def save(self):
        os.makedirs(MODEL_DIR, exist_ok=True)
        payload = {
            "U": self.U, "sigma": self.sigma, "Vt": self.Vt,
            "user_ratings_mean": self.user_ratings_mean,
            "user_index": self.user_index,
            "movie_index": self.movie_index,
            "idx_to_user": self.idx_to_user,
            "idx_to_movie": self.idx_to_movie,
            "predicted_all": self._predicted_all,
            "rmse": self.rmse, "mae": self.mae,
        }
        with open(SVD_PATH, "wb") as f:
            pickle.dump(payload, f)

    def load(self, ratings: pd.DataFrame, movies: pd.DataFrame):
        self.ratings = ratings.copy()
        self.movies  = movies.copy()
        with open(SVD_PATH, "rb") as f:
            p = pickle.load(f)
        self.U                 = p["U"]
        self.sigma             = p["sigma"]
        self.Vt                = p["Vt"]
        self.user_ratings_mean = p["user_ratings_mean"]
        self.user_index        = p["user_index"]
        self.movie_index       = p["movie_index"]
        self.idx_to_user       = p["idx_to_user"]
        self.idx_to_movie      = p["idx_to_movie"]
        self._predicted_all    = p["predicted_all"]
        self.rmse              = p.get("rmse")
        self.mae               = p.get("mae")
