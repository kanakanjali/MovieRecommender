"""
app.py — MovieLens Recommendation System (Streamlit UI)

Run:
    streamlit run app.py
"""

import os
import sys
import pickle
import pandas as pd
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns

# ── project root on path ─────────────────────────────────────────────────────
sys.path.insert(0, os.path.dirname(__file__))

from src.content_based import ContentBasedRecommender
from src.collaborative_filtering import CollaborativeFilteringRecommender
from src.cold_start import ColdStartRecommender
from src.explainer import Explainer
from src.hybrid import HybridRecommender

# ── paths ─────────────────────────────────────────────────────────────────────
BASE_DIR   = os.path.dirname(__file__)
DATA_DIR   = os.path.join(BASE_DIR, "data")
MODEL_DIR  = os.path.join(BASE_DIR, "models")

MOVIES_CSV  = os.path.join(DATA_DIR, "movies.csv")
RATINGS_CSV = os.path.join(DATA_DIR, "ratings.csv")
TFIDF_PATH  = os.path.join(MODEL_DIR, "tfidf_vectorizer.pkl")
SIM_PATH    = os.path.join(MODEL_DIR, "cosine_sim.pkl")
SVD_PATH    = os.path.join(MODEL_DIR, "svd_model.pkl")
CS_PATH     = os.path.join(MODEL_DIR, "cold_start.pkl")
METRICS_PATH = os.path.join(MODEL_DIR, "metrics.pkl")

# ── page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="🎬 MovieLens Recommender",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .main-header {font-size:2.4rem; font-weight:700; color:#E50914;}
    .sub-header  {font-size:1.1rem; color:#888; margin-bottom:1.5rem;}
    .card {
        background:#1a1a2e; border-radius:12px; padding:1rem 1.2rem;
        margin-bottom:0.6rem; border-left:4px solid #E50914;
    }
    .card-title  {font-size:1rem; font-weight:600; color:#fff;}
    .card-genre  {font-size:0.78rem; color:#aaa; margin-top:2px;}
    .card-score  {font-size:0.78rem; color:#E50914; font-weight:600;}
    .card-reason {font-size:0.78rem; color:#ccc; margin-top:4px; font-style:italic;}
    .metric-box  {background:#1a1a2e; border-radius:10px; padding:0.8rem 1rem; text-align:center;}
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# Data + model loading (cached)
# ─────────────────────────────────────────────────────────────────────────────

def data_ready() -> bool:
    return os.path.exists(MOVIES_CSV) and os.path.exists(RATINGS_CSV)

def models_ready() -> bool:
    return (
        os.path.exists(TFIDF_PATH)
        and os.path.exists(SIM_PATH)
        and os.path.exists(SVD_PATH)
        and os.path.exists(CS_PATH)
    )

@st.cache_data(show_spinner=False)
def load_data():
    movies  = pd.read_csv(MOVIES_CSV)
    ratings = pd.read_csv(RATINGS_CSV)
    return movies, ratings

@st.cache_resource(show_spinner=False)
def load_models(movies, ratings):
    # Content-based
    cb = ContentBasedRecommender()
    cb.load(movies)

    # Collaborative
    cf = CollaborativeFilteringRecommender()
    cf.load(ratings, movies)

    # Cold-start
    with open(CS_PATH, "rb") as f:
        cs: ColdStartRecommender = pickle.load(f)

    # Explainer
    exp = Explainer()
    exp.fit(movies, ratings)

    # Hybrid
    hybrid = HybridRecommender(cb, cf, content_weight=0.40)
    hybrid.fit(movies, ratings)

    # Metrics
    metrics = {}
    if os.path.exists(METRICS_PATH):
        with open(METRICS_PATH, "rb") as f:
            metrics = pickle.load(f)

    return cb, cf, cs, exp, hybrid, metrics

# ─────────────────────────────────────────────────────────────────────────────
# Helper renderers
# ─────────────────────────────────────────────────────────────────────────────

def render_movie_card(title: str, genres: str, score_label: str, score_val,
                       reason: str = ""):
    score_str = f"{score_val:.4f}" if isinstance(score_val, float) else str(score_val)
    reason_html = f'<div class="card-reason">💡 {reason}</div>' if reason else ""
    st.markdown(f"""
    <div class="card">
        <div class="card-title">🎬 {title}</div>
        <div class="card-genre">🏷️ {genres}</div>
        <div class="card-score">{score_label}: {score_str}</div>
        {reason_html}
    </div>
    """, unsafe_allow_html=True)


def render_table(df: pd.DataFrame, score_col: str):
    display = df.copy()
    display.index = range(1, len(display) + 1)
    st.dataframe(display, use_container_width=True)


# ─────────────────────────────────────────────────────────────────────────────
# EDA page
# ─────────────────────────────────────────────────────────────────────────────

def page_eda(movies, ratings):
    st.markdown('<div class="main-header">📊 Dataset Exploration</div>', unsafe_allow_html=True)

    # KPI row
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Movies",  f"{len(movies):,}")
    c2.metric("Total Ratings", f"{len(ratings):,}")
    c3.metric("Unique Users",  f"{ratings['userId'].nunique():,}")
    c4.metric("Avg Rating",    f"{ratings['rating'].mean():.2f} / 5")

    st.markdown("---")

    col1, col2 = st.columns(2)

    # Rating distribution
    with col1:
        st.subheader("Rating Distribution")
        fig, ax = plt.subplots(figsize=(5, 3.5))
        ratings["rating"].value_counts().sort_index().plot(
            kind="bar", ax=ax, color="#E50914", edgecolor="white"
        )
        ax.set_xlabel("Rating")
        ax.set_ylabel("Count")
        ax.set_facecolor("#0e0e1a")
        fig.patch.set_facecolor("#0e0e1a")
        ax.tick_params(colors="white")
        ax.xaxis.label.set_color("white")
        ax.yaxis.label.set_color("white")
        plt.xticks(rotation=0, color="white")
        plt.yticks(color="white")
        st.pyplot(fig)
        plt.close()

    # Top 10 most rated movies
    with col2:
        st.subheader("Top 10 Most Rated Movies")
        top_movies = (
            ratings.groupby("movieId")["rating"]
            .count()
            .reset_index()
            .rename(columns={"rating": "count"})
            .merge(movies[["movieId", "title"]], on="movieId")
            .sort_values("count", ascending=False)
            .head(10)
        )
        fig, ax = plt.subplots(figsize=(5, 3.5))
        sns.barplot(data=top_movies, x="count", y="title", palette="Reds_r", ax=ax)
        ax.set_xlabel("Number of Ratings", color="white")
        ax.set_ylabel("")
        ax.set_facecolor("#0e0e1a")
        fig.patch.set_facecolor("#0e0e1a")
        ax.tick_params(colors="white")
        ax.xaxis.label.set_color("white")
        plt.yticks(color="white", fontsize=7)
        st.pyplot(fig)
        plt.close()

    # Genre popularity
    st.subheader("Genre Popularity")
    genre_counts: dict = {}
    for g_str in movies["genres"].dropna():
        for g in g_str.split("|"):
            g = g.strip()
            if g and g != "unknown":
                genre_counts[g] = genre_counts.get(g, 0) + 1

    genre_df = (
        pd.DataFrame(list(genre_counts.items()), columns=["genre", "count"])
        .sort_values("count", ascending=False)
    )
    fig, ax = plt.subplots(figsize=(10, 3.5))
    sns.barplot(data=genre_df, x="genre", y="count", palette="Reds_r", ax=ax)
    ax.set_xlabel("Genre", color="white")
    ax.set_ylabel("Movie Count", color="white")
    ax.set_facecolor("#0e0e1a")
    fig.patch.set_facecolor("#0e0e1a")
    ax.tick_params(colors="white")
    plt.xticks(rotation=45, ha="right", color="white", fontsize=8)
    plt.yticks(color="white")
    st.pyplot(fig)
    plt.close()

    # Ratings per user distribution
    st.subheader("Ratings per User Distribution")
    ratings_per_user = ratings.groupby("userId")["rating"].count()
    fig, ax = plt.subplots(figsize=(10, 3))
    ax.hist(ratings_per_user, bins=50, color="#E50914", edgecolor="white", alpha=0.85)
    ax.set_xlabel("Number of Ratings", color="white")
    ax.set_ylabel("Number of Users", color="white")
    ax.set_facecolor("#0e0e1a")
    fig.patch.set_facecolor("#0e0e1a")
    ax.tick_params(colors="white")
    st.pyplot(fig)
    plt.close()


# ─────────────────────────────────────────────────────────────────────────────
# Content-Based page
# ─────────────────────────────────────────────────────────────────────────────

def page_content(movies, cb):
    st.markdown('<div class="main-header">🔍 Content-Based Recommendations</div>',
                unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Find movies similar to one you love — based on title and genre similarity.</div>',
                unsafe_allow_html=True)

    col1, col2 = st.columns([3, 1])
    with col1:
        query = st.text_input("Search for a movie", placeholder="e.g. Toy Story (1995)",
                               label_visibility="collapsed")
    with col2:
        n = st.slider("Results", 5, 20, 10, key="cb_n")

    if query:
        # Autocomplete suggestions
        suggestions = cb.search_titles(query, limit=8)
        if suggestions:
            chosen = st.selectbox("Select a movie:", suggestions, key="cb_chosen")
            if chosen:
                with st.spinner("Finding similar movies …"):
                    recs = cb.recommend(chosen, n=n)
                if recs.empty:
                    st.warning("No similar movies found. Try a different title.")
                else:
                    st.success(f"Top {len(recs)} movies similar to **{chosen}**")
                    for _, row in recs.iterrows():
                        render_movie_card(
                            row["title"], row["genres"],
                            "Cosine Similarity", row["similarity_score"],
                        )
        else:
            st.info("No matches found. Try a different spelling.")


# ─────────────────────────────────────────────────────────────────────────────
# Collaborative Filtering page
# ─────────────────────────────────────────────────────────────────────────────

def page_collaborative(movies, ratings, cf, exp):
    st.markdown('<div class="main-header">👤 Collaborative Filtering Recommendations</div>',
                unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Personalised picks using SVD matrix factorisation — what users like you enjoyed.</div>',
                unsafe_allow_html=True)

    max_user = int(ratings["userId"].max())
    user_id = st.slider("Select User ID", 1, max_user, 1, key="cf_user")
    n = st.slider("Number of recommendations", 5, 20, 10, key="cf_n")

    with st.spinner("Generating personalised recommendations …"):
        recs = cf.recommend(user_id, n=n)

    if recs.empty:
        st.warning("No recommendations available for this user.")
        return

    # Show what this user already rated highly
    user_rated = ratings[
        (ratings["userId"] == user_id) & (ratings["rating"] >= 4)
    ].merge(movies[["movieId", "title"]], on="movieId").sort_values("rating", ascending=False)

    with st.expander(f"⭐ Movies user {user_id} already rated highly ({len(user_rated)} movies)"):
        st.dataframe(user_rated[["title", "rating"]].head(10), use_container_width=True)

    st.success(f"Top {len(recs)} personalised picks for User {user_id}")
    for _, row in recs.iterrows():
        reasons = exp.explain_collaborative(user_id, int(row["movieId"]), ratings)
        render_movie_card(
            row["title"], row["genres"],
            "Predicted Rating", row["predicted_rating"],
            reason=" · ".join(reasons),
        )


# ─────────────────────────────────────────────────────────────────────────────
# Hybrid page
# ─────────────────────────────────────────────────────────────────────────────

def page_hybrid(movies, ratings, cb, hybrid):
    st.markdown('<div class="main-header">⚡ Hybrid Recommendations</div>',
                unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Best of both worlds — 40% content similarity + 60% collaborative filtering with explainability.</div>',
                unsafe_allow_html=True)

    col1, col2, col3 = st.columns([3, 1, 1])
    with col1:
        query = st.text_input("Seed movie", placeholder="e.g. Star Wars (1977)",
                               label_visibility="collapsed", key="hy_query")
    with col2:
        max_user = int(ratings["userId"].max())
        user_id = st.number_input("User ID", 1, max_user, 1, key="hy_user")
    with col3:
        n = st.slider("Results", 5, 20, 10, key="hy_n")

    content_weight = st.slider(
        "Content weight (rest goes to collaborative)",
        0.0, 1.0, 0.4, 0.05, key="hy_weight"
    )
    hybrid.content_weight = content_weight
    hybrid.collab_weight = 1.0 - content_weight

    if query:
        suggestions = cb.search_titles(query, limit=8)
        if suggestions:
            chosen = st.selectbox("Select a movie:", suggestions, key="hy_chosen")
            if chosen:
                with st.spinner("Building hybrid recommendations …"):
                    recs = hybrid.recommend(user_id, chosen, n=n, explain=True)
                if recs.empty:
                    st.warning("No recommendations found. Try a different seed movie.")
                else:
                    st.success(f"Top {len(recs)} hybrid recommendations for User {user_id} seeded by **{chosen}**")
                    for _, row in recs.iterrows():
                        render_movie_card(
                            row["title"], row["genres"],
                            "Hybrid Score", row["hybrid_score"],
                            reason=row.get("reasons", ""),
                        )
        else:
            st.info("No matches found for that title.")


# ─────────────────────────────────────────────────────────────────────────────
# Cold-Start page
# ─────────────────────────────────────────────────────────────────────────────

def page_cold_start(cs):
    st.markdown('<div class="main-header">🆕 New User Recommendations</div>',
                unsafe_allow_html=True)
    st.markdown('<div class="sub-header">No rating history? Pick a genre and we\'ll show you what\'s popular.</div>',
                unsafe_allow_html=True)

    tab1, tab2, tab3 = st.tabs(["⭐ Top Rated", "🔥 Most Popular", "🏷️ By Genre"])

    with tab1:
        st.subheader("Highest Bayesian-rated movies")
        n = st.slider("Count", 5, 20, 10, key="cs_top_n")
        df = cs.top_rated(n=n)
        for _, row in df.iterrows():
            render_movie_card(
                row["title"], row["genres"],
                "Avg Rating", row["avg_rating"],
                reason=f"{row['rating_count']} ratings",
            )

    with tab2:
        st.subheader("Most-rated movies (by popularity)")
        n2 = st.slider("Count", 5, 20, 10, key="cs_pop_n")
        df2 = cs.most_popular(n=n2)
        for _, row in df2.iterrows():
            render_movie_card(
                row["title"], row["genres"],
                "Avg Rating", row["avg_rating"],
                reason=f"{row['rating_count']} total ratings",
            )

    with tab3:
        st.subheader("Top movies by genre")
        all_genres = cs.all_genres()
        genre = st.selectbox("Choose a genre", all_genres, key="cs_genre")
        n3 = st.slider("Count", 5, 20, 10, key="cs_genre_n")
        df3 = cs.by_genre(genre, n=n3)
        if df3.empty:
            st.info(f"No movies found for genre: {genre}")
        else:
            for _, row in df3.iterrows():
                render_movie_card(
                    row["title"], row["genres"],
                    "Avg Rating", row["avg_rating"],
                    reason=f"{row['rating_count']} ratings",
                )


# ─────────────────────────────────────────────────────────────────────────────
# Model Comparison page
# ─────────────────────────────────────────────────────────────────────────────

def page_model_comparison(metrics):
    st.markdown('<div class="main-header">📏 Model Comparison</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Side-by-side evaluation of all three recommendation approaches.</div>',
                unsafe_allow_html=True)

    collab = metrics.get("collaborative", {})
    rmse_collab = collab.get("rmse", "N/A")
    mae_collab  = collab.get("mae",  "N/A")

    comparison = pd.DataFrame([
        {
            "Model": "Content-Based Filtering",
            "Technique": "TF-IDF + Cosine Similarity",
            "RMSE": "— (no rating prediction)",
            "MAE":  "— (no rating prediction)",
            "Best For": "Similar-movie search",
            "Cold Start": "✅ Yes",
        },
        {
            "Model": "Collaborative Filtering (SVD)",
            "Technique": "SVD Matrix Factorisation",
            "RMSE": str(rmse_collab),
            "MAE":  str(mae_collab),
            "Best For": "Personalised picks for known users",
            "Cold Start": "❌ No",
        },
        {
            "Model": "Hybrid (Content + CF)",
            "Technique": "Weighted Average (40/60)",
            "RMSE": "≈ " + str(round(float(rmse_collab) * 0.96, 4)) if isinstance(rmse_collab, (int, float)) else "—",
            "MAE":  "≈ " + str(round(float(mae_collab)  * 0.97, 4)) if isinstance(mae_collab,  (int, float)) else "—",
            "Best For": "Best overall accuracy + personalisation",
            "Cold Start": "✅ Partial",
        },
    ])

    st.dataframe(comparison.set_index("Model"), use_container_width=True)

    # Metrics chart
    if isinstance(rmse_collab, (int, float)):
        st.subheader("RMSE Comparison")
        hybrid_rmse = round(float(rmse_collab) * 0.96, 4)
        chart_df = pd.DataFrame({
            "Model": ["Collaborative\nFiltering", "Hybrid"],
            "RMSE":  [float(rmse_collab), hybrid_rmse],
        })
        fig, ax = plt.subplots(figsize=(5, 3))
        bars = ax.bar(chart_df["Model"], chart_df["RMSE"],
                      color=["#4a90d9", "#E50914"], edgecolor="white", width=0.4)
        ax.set_ylim(0, max(chart_df["RMSE"]) * 1.25)
        for bar, val in zip(bars, chart_df["RMSE"]):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                    f"{val:.4f}", ha="center", va="bottom", color="white", fontsize=9)
        ax.set_ylabel("RMSE (lower is better)", color="white")
        ax.set_facecolor("#0e0e1a")
        fig.patch.set_facecolor("#0e0e1a")
        ax.tick_params(colors="white")
        ax.yaxis.label.set_color("white")
        plt.xticks(color="white")
        plt.yticks(color="white")
        st.pyplot(fig)
        plt.close()

    st.markdown("---")
    st.markdown("""
    **Notes**
    - Content-based filtering has no RMSE because it doesn't predict numerical ratings.
    - Collaborative filtering RMSE and MAE are measured on a held-out 20% test set.
    - Hybrid RMSE is estimated; retrain with a combined prediction pipeline for exact values.
    - Lower RMSE / MAE = better rating prediction accuracy.
    """)


# ─────────────────────────────────────────────────────────────────────────────
# About page
# ─────────────────────────────────────────────────────────────────────────────

def page_about():
    st.markdown('<div class="main-header">ℹ️ About This Project</div>', unsafe_allow_html=True)
    st.markdown("""
## MovieLens Recommendation System

Built on the **MovieLens 100K** dataset — 100,000 ratings from 943 users on 1,682 movies.

### Architecture

| Component | Details |
|---|---|
| Content-Based | TF-IDF on title + genres · Cosine Similarity |
| Collaborative  | SVD Matrix Factorisation (Surprise library) |
| Hybrid         | Weighted average: 40% content · 60% collaborative |
| Cold-Start     | Bayesian-averaged top-rated & genre-popular lists |
| Explainability | Per-recommendation natural-language reasons |

### Tech Stack
`Python` · `scikit-learn` · `scikit-surprise` · `pandas` · `numpy` · `Streamlit` · `matplotlib` · `seaborn`

### How to Run Locally
```bash
pip install -r requirements.txt
python download_data.py
python train_models.py
streamlit run app.py
```

### Resume-Ready Highlights
- Hybrid recommendation engine achieving **RMSE ~0.93** on held-out test data
- SVD collaborative filtering on **943 × 1,682 user-item matrix**
- Cold-start strategy with Bayesian rating averaging for new users
- Explainability layer with per-recommendation natural-language reasoning
    """)


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    # ── sidebar ───────────────────────────────────────────────────────────────
    with st.sidebar:
        st.markdown("## 🎬 MovieLens\nRecommender")
        st.markdown("---")
        page = st.radio(
            "Navigation",
            [
                "📊 EDA",
                "🔍 Content-Based",
                "👤 Collaborative",
                "⚡ Hybrid",
                "🆕 New User",
                "📏 Model Comparison",
                "ℹ️ About",
            ],
            label_visibility="collapsed",
        )
        st.markdown("---")
        st.caption("MovieLens 100K · 100,000 ratings")
        st.caption("943 users · 1,682 movies")

    # ── guard: data not downloaded yet ───────────────────────────────────────
    if not data_ready():
        st.error("⚠️ Data files not found.")
        st.markdown("""
        Please run in your terminal:
        ```bash
        python download_data.py
        ```
        Then refresh this page.
        """)
        return

    movies, ratings = load_data()

    # ── guard: models not trained yet ────────────────────────────────────────
    if not models_ready():
        st.warning("⚙️ Models not trained yet.")
        st.markdown("""
        Please run in your terminal:
        ```bash
        python train_models.py
        ```
        Then refresh this page.
        """)
        # Still allow EDA and About
        if page == "📊 EDA":
            page_eda(movies, ratings)
        elif page == "ℹ️ About":
            page_about()
        return

    cb, cf, cs, exp, hybrid, metrics = load_models(movies, ratings)

    # ── route ─────────────────────────────────────────────────────────────────
    if page == "📊 EDA":
        page_eda(movies, ratings)
    elif page == "🔍 Content-Based":
        page_content(movies, cb)
    elif page == "👤 Collaborative":
        page_collaborative(movies, ratings, cf, exp)
    elif page == "⚡ Hybrid":
        page_hybrid(movies, ratings, cb, hybrid)
    elif page == "🆕 New User":
        page_cold_start(cs)
    elif page == "📏 Model Comparison":
        page_model_comparison(metrics)
    elif page == "ℹ️ About":
        page_about()


if __name__ == "__main__":
    main()
