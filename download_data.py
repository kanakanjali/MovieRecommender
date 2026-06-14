"""
download_data.py — Downloads and prepares MovieLens 100K dataset.
Run this once before anything else:  python download_data.py

If the automatic download fails (network restrictions), manually:
  1. Go to https://grouplens.org/datasets/movielens/
  2. Download ml-100k.zip
  3. Place it in the data/ folder
  4. Re-run: python download_data.py
"""

import os
import zipfile
import urllib.request
import pandas as pd

BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
DATA_DIR   = os.path.join(BASE_DIR, "data")
ZIP_PATH   = os.path.join(DATA_DIR, "ml-100k.zip")
EXTRACT_DIR = os.path.join(DATA_DIR, "ml-100k")
MOVIES_OUT = os.path.join(DATA_DIR, "movies.csv")
RATINGS_OUT = os.path.join(DATA_DIR, "ratings.csv")

URL = "https://files.grouplens.org/datasets/movielens/ml-100k.zip"

GENRE_NAMES = [
    "unknown", "Action", "Adventure", "Animation", "Children",
    "Comedy", "Crime", "Documentary", "Drama", "Fantasy",
    "Film-Noir", "Horror", "Musical", "Mystery", "Romance",
    "Sci-Fi", "Thriller", "War", "Western",
]


def download():
    os.makedirs(DATA_DIR, exist_ok=True)
    if os.path.exists(ZIP_PATH):
        print("Zip already present, skipping download.")
        return True
    print(f"Downloading MovieLens 100K from {URL} …")
    try:
        req = urllib.request.Request(URL, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=60) as r:
            data = r.read()
        with open(ZIP_PATH, "wb") as f:
            f.write(data)
        print("Download complete.")
        return True
    except Exception as e:
        print(f"Download failed: {e}")
        print("\nManual steps:")
        print("  1. Visit https://grouplens.org/datasets/movielens/")
        print(f"  2. Download ml-100k.zip into: {DATA_DIR}")
        print("  3. Re-run: python download_data.py")
        return False


def extract():
    if not os.path.exists(ZIP_PATH):
        return False
    if os.path.exists(EXTRACT_DIR):
        print("Already extracted.")
        return True
    with zipfile.ZipFile(ZIP_PATH, "r") as z:
        z.extractall(DATA_DIR)
    print("Extracted ml-100k.zip.")
    return True


def build_movies() -> pd.DataFrame:
    cols = ["movieId", "title", "release_date", "video_date", "imdb_url"] + GENRE_NAMES
    items_path = os.path.join(EXTRACT_DIR, "u.item")
    df = pd.read_csv(items_path, sep="|", names=cols, encoding="latin-1", index_col=False)
    df["genres"] = df[GENRE_NAMES].apply(
        lambda row: "|".join([g for g, v in zip(GENRE_NAMES, row) if v == 1]), axis=1
    )
    df["genres"] = df["genres"].replace("", "unknown")
    movies = df[["movieId", "title", "release_date", "genres"]].copy()
    movies.to_csv(MOVIES_OUT, index=False)
    print(f"Saved {len(movies)} movies → {MOVIES_OUT}")
    return movies


def build_ratings() -> pd.DataFrame:
    ratings_path = os.path.join(EXTRACT_DIR, "u.data")
    df = pd.read_csv(
        ratings_path, sep="\t",
        names=["userId", "movieId", "rating", "timestamp"],
    )
    df.drop(columns=["timestamp"], inplace=True)
    df.to_csv(RATINGS_OUT, index=False)
    print(f"Saved {len(df)} ratings → {RATINGS_OUT}")
    return df


if __name__ == "__main__":
    if os.path.exists(MOVIES_OUT) and os.path.exists(RATINGS_OUT):
        print("Data files already exist. Delete them to re-download.")
    else:
        ok = download()
        if ok and extract():
            build_movies()
            build_ratings()
            print("\nDone! Next: python train_models.py")
        else:
            print("\nSetup incomplete — resolve download issue and retry.")
