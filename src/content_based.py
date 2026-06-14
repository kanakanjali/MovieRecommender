"""
src/content_based.py
Content-Based Filtering using TF-IDF on combined title + genres.
"""

import os
import pickle

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

MODEL_DIR = os.path.join(os.path.dirname(__file__), "..", "models")
TFIDF_PATH = os.path.join(MODEL_DIR, "tfidf_vectorizer.pkl")
SIM_PATH = os.path.join(MODEL_DIR, "cosine_sim.pkl")


class ContentBasedRecommender:
    """
    Recommends movies similar to a query movie using TF-IDF + cosine similarity
    on a combined 'title + genres' text field.
    """

    def __init__(self):
        self.movies: pd.DataFrame = None
        self.tfidf: TfidfVectorizer = None
        self.cosine_sim = None
        self._title_to_idx: dict = {}

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------
    def fit(self, movies: pd.DataFrame) -> None:
        """
        Parameters
        ----------
        movies : DataFrame with columns ['movieId', 'title', 'genres']
        """
        self.movies = movies.reset_index(drop=True).copy()

        # Replace pipes in genres with spaces so TF-IDF treats each genre
        # as a separate token.
        self.movies["genres_clean"] = (
            self.movies["genres"].fillna("unknown").str.replace("|", " ", regex=False)
        )

        # Combined feature: title (repeated twice for slight title boost) + genres
        self.movies["combined"] = (
            self.movies["title"].fillna("") + " "
            + self.movies["title"].fillna("") + " "
            + self.movies["genres_clean"]
        )

        self.tfidf = TfidfVectorizer(stop_words="english", ngram_range=(1, 2))
        tfidf_matrix = self.tfidf.fit_transform(self.movies["combined"])
        self.cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)

        # Map lowercased title → integer index for fast lookup
        self._title_to_idx = {
            title.lower(): idx
            for idx, title in enumerate(self.movies["title"])
        }

        os.makedirs(MODEL_DIR, exist_ok=True)
        with open(TFIDF_PATH, "wb") as f:
            pickle.dump(self.tfidf, f)
        with open(SIM_PATH, "wb") as f:
            pickle.dump(self.cosine_sim, f)

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------
    def recommend(self, movie_title: str, n: int = 10) -> pd.DataFrame:
        """
        Return top-n movies most similar to *movie_title*.

        Returns
        -------
        DataFrame with columns: movieId, title, genres, similarity_score
        """
        if self.movies is None:
            raise RuntimeError("Call fit() first.")

        key = movie_title.strip().lower()
        if key not in self._title_to_idx:
            # Partial-match fallback
            matches = [t for t in self._title_to_idx if key in t]
            if not matches:
                return pd.DataFrame(columns=["movieId", "title", "genres", "similarity_score"])
            key = matches[0]

        idx = self._title_to_idx[key]
        sim_scores = list(enumerate(self.cosine_sim[idx]))
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
        # Exclude the query movie itself
        sim_scores = [(i, s) for i, s in sim_scores if i != idx][:n]

        indices = [i for i, _ in sim_scores]
        scores = [s for _, s in sim_scores]

        result = self.movies.iloc[indices][["movieId", "title", "genres"]].copy()
        result["similarity_score"] = [round(s, 4) for s in scores]
        return result.reset_index(drop=True)

    def search_titles(self, query: str, limit: int = 10) -> list:
        """Return movie titles containing *query* (case-insensitive)."""
        if self.movies is None:
            return []
        q = query.strip().lower()
        return [
            t for t in self.movies["title"]
            if q in t.lower()
        ][:limit]

    # ------------------------------------------------------------------
    # Persistence helpers
    # ------------------------------------------------------------------
    def save(self):
        if self.movies is None:
            raise RuntimeError("Nothing to save — call fit() first.")
        os.makedirs(MODEL_DIR, exist_ok=True)
        with open(TFIDF_PATH, "wb") as f:
            pickle.dump(self.tfidf, f)
        with open(SIM_PATH, "wb") as f:
            pickle.dump(self.cosine_sim, f)

    def load(self, movies: pd.DataFrame):
        self.movies = movies.reset_index(drop=True).copy()
        self.movies["genres_clean"] = (
            self.movies["genres"].fillna("unknown").str.replace("|", " ", regex=False)
        )
        self._title_to_idx = {
            title.lower(): idx
            for idx, title in enumerate(self.movies["title"])
        }
        with open(TFIDF_PATH, "rb") as f:
            self.tfidf = pickle.load(f)
        with open(SIM_PATH, "rb") as f:
            self.cosine_sim = pickle.load(f)
