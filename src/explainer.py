"""
src/explainer.py
Explainability layer — generates human-readable reasons for each recommendation.
"""

import pandas as pd


class Explainer:
    """
    Generates natural-language reasons for why a movie was recommended.
    Works alongside both content-based and collaborative recommendations.
    """

    def __init__(self):
        self.movies: pd.DataFrame = None
        self.ratings: pd.DataFrame = None

    def fit(self, movies: pd.DataFrame, ratings: pd.DataFrame) -> None:
        self.movies = movies.copy()
        self.ratings = ratings.copy()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _genres_of(self, movie_id: int) -> set:
        row = self.movies[self.movies["movieId"] == movie_id]
        if row.empty:
            return set()
        return set(row.iloc[0]["genres"].split("|"))

    def _title_of(self, movie_id: int) -> str:
        row = self.movies[self.movies["movieId"] == movie_id]
        if row.empty:
            return f"Movie {movie_id}"
        return row.iloc[0]["title"]

    def _common_genres(self, id_a: int, id_b: int) -> list:
        return sorted(self._genres_of(id_a) & self._genres_of(id_b))

    def _avg_rating(self, movie_id: int) -> float:
        subset = self.ratings[self.ratings["movieId"] == movie_id]["rating"]
        return round(subset.mean(), 2) if not subset.empty else 0.0

    def _rating_count(self, movie_id: int) -> int:
        return int((self.ratings["movieId"] == movie_id).sum())

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def explain_content(self, query_movie_id: int, recommended_movie_id: int) -> list:
        """Reasons for a content-based recommendation."""
        reasons = []
        common = self._common_genres(query_movie_id, recommended_movie_id)
        query_title = self._title_of(query_movie_id)
        if common:
            g_str = ", ".join(common[:3])
            reasons.append(f"Shares {g_str} genre(s) with '{query_title}'")
        avg = self._avg_rating(recommended_movie_id)
        count = self._rating_count(recommended_movie_id)
        if avg >= 3.8 and count >= 30:
            reasons.append(f"Highly rated by the community (avg {avg}/5 from {count} ratings)")
        if not reasons:
            reasons.append(f"Similar title/genre profile to '{query_title}'")
        return reasons

    def explain_collaborative(self, user_id: int, recommended_movie_id: int,
                               ratings: pd.DataFrame) -> list:
        """Reasons for a collaborative filtering recommendation."""
        reasons = []
        avg = self._avg_rating(recommended_movie_id)
        count = self._rating_count(recommended_movie_id)

        # Users who also liked something this user rated highly
        user_highly_rated = ratings[
            (ratings["userId"] == user_id) & (ratings["rating"] >= 4)
        ]["movieId"].tolist()

        # Find other users who also rated the recommended movie well
        rec_raters = set(
            ratings[
                (ratings["movieId"] == recommended_movie_id) & (ratings["rating"] >= 4)
            ]["userId"].tolist()
        )

        overlap_count = 0
        for mid in user_highly_rated[:20]:  # check against first 20 highly-rated
            similar_users = set(
                ratings[
                    (ratings["movieId"] == mid) & (ratings["rating"] >= 4)
                ]["userId"].tolist()
            )
            overlap_count += len(similar_users & rec_raters)

        if overlap_count > 5:
            reasons.append(f"Users with similar taste loved this movie")
        if avg >= 3.5 and count >= 20:
            reasons.append(f"Community average: {avg}/5 across {count} ratings")
        if not reasons:
            reasons.append("Predicted to match your personal rating pattern")
        return reasons

    def explain_hybrid(self, query_movie_id: int, user_id: int,
                        recommended_movie_id: int, content_score: float,
                        collab_score: float, ratings: pd.DataFrame) -> list:
        """Combined reasons for a hybrid recommendation."""
        reasons = []
        if content_score > 0.2:
            common = self._common_genres(query_movie_id, recommended_movie_id)
            if common:
                reasons.append(f"Shares {', '.join(common[:2])} genre(s) with your chosen movie")
        if collab_score > 3.5:
            reasons.append(f"Collaborative score: {collab_score}/5 based on users like you")
        avg = self._avg_rating(recommended_movie_id)
        if avg >= 3.8:
            reasons.append(f"Highly rated overall (avg {avg}/5)")
        if not reasons:
            reasons.append("Strong match across both content and user behaviour signals")
        return reasons
