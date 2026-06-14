# 🎬 MovieLens Recommendation System

[![Live Demo](https://img.shields.io/badge/🚀%20Live%20Demo-Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)](https://movierecommender12.streamlit.app/)
[![Python](https://img.shields.io/badge/Python-3.9%2B-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.20%2B-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)](https://streamlit.io/)
[![Dataset](https://img.shields.io/badge/Dataset-MovieLens%20100K-blue?style=for-the-badge)](https://grouplens.org/datasets/movielens/100k/)
[![License](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)](LICENSE)

A production-quality **hybrid movie recommendation engine** built on the MovieLens 100K dataset — combining Content-Based Filtering (TF-IDF + Cosine Similarity), Collaborative Filtering (SVD), and a Hybrid model, with cold-start handling and explainable recommendations.

---

## 🌐 Live Demo

**👉 [https://movierecommender12.streamlit.app/](https://movierecommender12.streamlit.app/)**

No setup needed — just open and explore!

---

## ✨ Features

- 🔍 **Content-Based Filtering** — find similar movies by title and genre using TF-IDF + Cosine Similarity
- 👤 **Collaborative Filtering (SVD)** — personalised predictions from 100K user ratings via matrix factorisation
- ⚡ **Hybrid Model** — combines both signals (40% content + 60% CF) with per-recommendation natural-language explanations
- 🆕 **Cold-Start Handling** — Bayesian averaging for new users with zero rating history
- 📊 **EDA Dashboard** — rating distributions, top genres, most-rated movies
- 📏 **Model Evaluation** — RMSE/MAE comparison across all models

---

## 🖥️ App Pages

| Page | What you can do |
|---|---|
| 📊 EDA | Explore rating distributions, top genres, most-rated movies |
| 🔍 Content-Based | Type a movie → get similar movies by title/genre |
| 👤 Collaborative | Pick a User ID → get personalised predictions |
| ⚡ Hybrid | Seed movie + User ID → hybrid picks with explanations |
| 🆕 New User | Top-rated / popular / genre-filtered for cold-start users |
| 📏 Model Comparison | RMSE/MAE table and bar chart across all models |
| ℹ️ About | Architecture summary and tech stack |

---

## 🧠 How Each Model Works

### 1. Content-Based Filtering
- Combines movie **title + genres** into a single text field
- Applies **TF-IDF vectorisation** with bigrams and English stop-word removal
- Computes **cosine similarity** across all 1,682 movie vectors
- Input: a movie name → Output: top N most similar movies

### 2. Collaborative Filtering (SVD)
- Builds a **943 × 1,682 user-item ratings matrix**
- Mean-centres by user, then applies **truncated SVD** via `scipy.sparse.linalg.svds`
- Evaluated with **RMSE and MAE** on a held-out 20% test set
- Input: a user ID → Output: top N movies they haven't seen but are predicted to love

### 3. Hybrid Model
- Combines both signals: **40% content score + 60% collaborative score**
- Normalises both to [0, 1] before weighting so neither dominates by scale
- Adds an **explainability layer** with per-recommendation natural-language reasons
- Input: user ID + seed movie → Output: personalised ranked list with explanations

### 4. Cold-Start Handler
- For brand new users with zero rating history
- Uses **Bayesian averaging** so movies with few ratings don't unfairly top the list
- Supports: top-rated overall, most popular by volume, and genre-filtered lists

---

## 📊 Evaluation Results

| Model | RMSE | MAE | Notes |
|---|---|---|---|
| Content-Based | — | — | No rating prediction (similarity only) |
| Collaborative SVD | ~0.97 | ~0.76 | Evaluated on 20% held-out test set |
| Hybrid | lower than CF alone | lower than CF alone | Content signal improves coverage |

---

## 🚀 Local Setup

```bash
# 1. Clone the repo
git clone https://github.com/kanakanjali/MovieRecommender.git
cd MovieRecommender

# 2. Install dependencies
pip install -r requirements.txt

# 3. Download & prepare the MovieLens 100K dataset (one-time)
python download_data.py

# 4. Train all models (one-time, ~1–2 minutes)
python train_models.py

# 5. Launch the Streamlit app
streamlit run app.py
```

Then open **http://localhost:8501** in your browser.

> **Note:** Trained model files (`.pkl`) and processed data are already committed to this repo, so the deployed app works out of the box without re-training.

---

## 📁 Project Structure

```
MovieRecommender/
├── data/
│   ├── movies.csv                   ← 1,682 movies with genres
│   └── ratings.csv                  ← 100,000 user ratings
├── models/
│   ├── tfidf_vectorizer.pkl         ← Trained TF-IDF model
│   ├── cosine_sim.pkl               ← Pre-computed similarity matrix
│   ├── svd_model.pkl                ← Trained SVD model + predictions
│   ├── cold_start.pkl               ← Cold-start recommender
│   └── metrics.pkl                  ← Saved RMSE and MAE values
├── src/
│   ├── content_based.py             ← TF-IDF + cosine similarity
│   ├── collaborative_filtering.py   ← SVD matrix factorisation (scipy)
│   ├── hybrid.py                    ← Weighted hybrid (40% content / 60% CF)
│   ├── cold_start.py                ← New-user fallback with Bayesian avg
│   └── explainer.py                 ← Natural-language recommendation reasons
├── app.py                           ← Streamlit dashboard (7 pages)
├── download_data.py                 ← One-time data download from GroupLens
├── train_models.py                  ← One-time model training script
├── requirements.txt
└── README.md
```

---

## ☁️ Deployment

This app is deployed on **Streamlit Community Cloud**.

The trained `.pkl` model files and processed CSV data are committed directly to the repo so the cloud instance loads them instantly — no re-training on startup.

To deploy your own fork:
1. Fork this repo
2. Go to [share.streamlit.io](https://share.streamlit.io) → **New app**
3. Select your fork → set **Main file path** to `app.py`
4. Click **Deploy** — done!

---

## 📦 Requirements

```
pandas>=1.5.0
numpy>=1.23.0
scikit-learn>=1.2.0
scipy>=1.9.0
streamlit>=1.20.0
matplotlib>=3.6.0
seaborn>=0.12.0
requests>=2.28.0
```

No `scikit-surprise` needed — SVD is implemented directly with `scipy.sparse.linalg.svds`.

---

## 📄 Dataset

[MovieLens 100K](https://grouplens.org/datasets/movielens/100k/) — collected by GroupLens Research at the University of Minnesota.
- 100,000 ratings from 943 users on 1,682 movies
- Free, no account required
- Auto-downloaded by `python download_data.py`

---

## 🙋‍♀️ Author

**Kanakanjali** — [GitHub](https://github.com/kanakanjali)

