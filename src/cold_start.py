"""
src/cold_start.py
Cold-Start handling for new users with no rating history.
"""

import pandas as pd


class ColdStartRecommender:
    """
    For brand-new users who have no ratings yet.
    Surfaces trending, top-rated, or genre-popular movies.
    """

    def __init__(self):
        self.movies: pd.DataFrame = None
        self.ratings: pd.DataFrame = None
        self._enriched: pd.DataFrame = None

    def fit(self, movies: pd.DataFrame, ratings: pd.DataFrame) -> None:
        self.movies = movies.copy()
        self.ratings = ratings.copy()

        # Aggregate statistics per movie
        stats = (
            ratings.groupby("movieId")
            .agg(rating_count=("rating", "count"), avg_rating=("rating", "mean"))
            .reset_index()
        )
        stats["avg_rating"] = stats["avg_rating"].round(2)

        # Bayesian average: C = global mean, m = minimum votes threshold (25th pct)
        C = stats["avg_rating"].mean()
        m = stats["rating_count"].quantile(0.25)
        stats["bayesian_avg"] = (
            (stats["rating_count"] * stats["avg_rating"] + m * C)
            / (stats["rating_count"] + m)
        ).round(4)

        self._enriched = movies.merge(stats, on="movieId", how="left")
        self._enriched["rating_count"] = self._enriched["rating_count"].fillna(0).astype(int)
        self._enriched["avg_rating"] = self._enriched["avg_rating"].fillna(0.0)
        self._enriched["bayesian_avg"] = self._enriched["bayesian_avg"].fillna(0.0)

    # ------------------------------------------------------------------
    def top_rated(self, n: int = 10) -> pd.DataFrame:
        """Return movies with the highest Bayesian-averaged rating."""
        if self._enriched is None:
            raise RuntimeError("Call fit() first.")
        return (
            self._enriched[self._enriched["rating_count"] >= 20]
            .sort_values("bayesian_avg", ascending=False)
            .head(n)[["movieId", "title", "genres", "avg_rating", "rating_count"]]
            .reset_index(drop=True)
        )

    def most_popular(self, n: int = 10) -> pd.DataFrame:
        """Return most-rated (most popular) movies."""
        if self._enriched is None:
            raise RuntimeError("Call fit() first.")
        return (
            self._enriched.sort_values("rating_count", ascending=False)
            .head(n)[["movieId", "title", "genres", "avg_rating", "rating_count"]]
            .reset_index(drop=True)
        )

    def by_genre(self, genre: str, n: int = 10) -> pd.DataFrame:
        """Return top-rated movies that belong to a given genre."""
        if self._enriched is None:
            raise RuntimeError("Call fit() first.")
        mask = self._enriched["genres"].str.contains(genre, case=False, na=False)
        return (
            self._enriched[mask]
            .sort_values("bayesian_avg", ascending=False)
            .head(n)[["movieId", "title", "genres", "avg_rating", "rating_count"]]
            .reset_index(drop=True)
        )

    def all_genres(self) -> list:
        """Return a sorted list of all unique genres in the dataset."""
        if self._enriched is None:
            return []
        genres = set()
        for g_str in self._enriched["genres"].dropna():
            for g in g_str.split("|"):
                genres.add(g.strip())
        return sorted(genres)
