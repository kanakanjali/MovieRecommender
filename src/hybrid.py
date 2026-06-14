"""
src/hybrid.py
Hybrid Recommender — combines Content-Based and Collaborative Filtering.
Weighted average: 40% content + 60% collaborative (configurable).
"""

import pandas as pd

from src.content_based import ContentBasedRecommender
from src.collaborative_filtering import CollaborativeFilteringRecommender
from src.explainer import Explainer


class HybridRecommender:
    """
    Combines content-based similarity with SVD-predicted ratings into a
    single ranked list.

    Score formula:
        hybrid_score = alpha * norm_content + (1 - alpha) * norm_collab
    where alpha = content_weight (default 0.40).
    """

    def __init__(
        self,
        content_model: ContentBasedRecommender,
        collab_model: CollaborativeFilteringRecommender,
        content_weight: float = 0.40,
    ):
        self.content_model = content_model
        self.collab_model = collab_model
        self.content_weight = content_weight
        self.collab_weight = 1.0 - content_weight
        self.explainer = Explainer()
        self._movies: pd.DataFrame = None

    def fit(self, movies: pd.DataFrame, ratings: pd.DataFrame) -> None:
        self._movies = movies.copy()
        self.explainer.fit(movies, ratings)

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------
    def recommend(
        self,
        user_id: int,
        movie_title: str,
        n: int = 10,
        explain: bool = True,
    ) -> pd.DataFrame:
        """
        Parameters
        ----------
        user_id     : int  — the user for personalisation
        movie_title : str  — seed movie for content signal
        n           : int  — number of results
        explain     : bool — add 'reasons' column

        Returns
        -------
        DataFrame: movieId, title, genres, hybrid_score, reasons (optional)
        """
        if self._movies is None:
            raise RuntimeError("Call fit() first.")

        # 1. Content-based candidates (2x n for a bigger pool)
        content_df = self.content_model.recommend(movie_title, n=n * 2)
        if content_df.empty:
            return pd.DataFrame()

        # 2. For each content candidate, get collaborative predicted rating
        records = []
        for _, row in content_df.iterrows():
            mid = int(row["movieId"])
            c_score = float(row["similarity_score"])
            cf_score = self.collab_model.predict_rating(user_id, mid)
            records.append(
                {
                    "movieId": mid,
                    "title": row["title"],
                    "genres": row["genres"],
                    "content_score": c_score,
                    "collab_score": cf_score,
                }
            )

        df = pd.DataFrame(records)

        # 3. Normalise both scores to [0, 1]
        def _norm(series: pd.Series) -> pd.Series:
            mn, mx = series.min(), series.max()
            if mx == mn:
                return series * 0 + 0.5
            return (series - mn) / (mx - mn)

        df["norm_content"] = _norm(df["content_score"])
        df["norm_collab"] = _norm(df["collab_score"])

        # 4. Weighted hybrid score
        df["hybrid_score"] = (
            self.content_weight * df["norm_content"]
            + self.collab_weight * df["norm_collab"]
        ).round(4)

        df = df.sort_values("hybrid_score", ascending=False).head(n).reset_index(drop=True)

        # 5. Explainability
        if explain:
            query_movie_id = self._get_movie_id(movie_title)
            reasons_list = []
            for _, row in df.iterrows():
                r = self.explainer.explain_hybrid(
                    query_movie_id=query_movie_id,
                    user_id=user_id,
                    recommended_movie_id=int(row["movieId"]),
                    content_score=row["content_score"],
                    collab_score=row["collab_score"],
                    ratings=self.collab_model.ratings,
                )
                reasons_list.append(" · ".join(r))
            df["reasons"] = reasons_list

        return df[
            ["movieId", "title", "genres", "hybrid_score"]
            + (["reasons"] if explain else [])
        ].reset_index(drop=True)

    # ------------------------------------------------------------------
    # Helper
    # ------------------------------------------------------------------
    def _get_movie_id(self, movie_title: str) -> int:
        key = movie_title.strip().lower()
        matches = self._movies[
            self._movies["title"].str.lower().str.contains(key, regex=False)
        ]
        if matches.empty:
            return -1
        return int(matches.iloc[0]["movieId"])
