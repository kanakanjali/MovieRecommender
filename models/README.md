# Models Folder

This folder is empty by default. Trained model .pkl files are not tracked by git (they are large binary files).

To generate them, run from the project root:

```bash
python train_models.py
```

This will create:
- tfidf_vectorizer.pkl  (TF-IDF model)
- cosine_sim.pkl        (pre-computed similarity matrix)
- svd_model.pkl         (trained SVD collaborative filter)
- cold_start.pkl        (cold-start recommender)
- metrics.pkl           (RMSE and MAE scores)

Note: Run `python download_data.py` first before training.
