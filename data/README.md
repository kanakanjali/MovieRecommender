# Data Folder

This folder is empty by default. The CSV files are not tracked by git.

To generate them, run from the project root:

```bash
python download_data.py
```

This will automatically download MovieLens 100K from https://grouplens.org/datasets/movielens/
and create:
- movies.csv  (1,682 movies with genres)
- ratings.csv (100,000 user ratings)
