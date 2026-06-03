# Anime Recommender — Docker + PySpark + Jupyter

## Folder structure expected on your machine

```
anime-spark/
├── Dockerfile
├── docker-compose.yml
├── .dockerignore
├── notebooks/
|    ├── collaboratif_benchmark.ipynb   -> Becnhmark of collaborative filtering
|    ├── collaboratif_plot.ipynb        -> Plot of collaborative filtering becnhmark 
|    ├── collaboratif_final.ipynb       -> Training of user-item recommender system
|    ├── Draft_collaborative.ipynb      -> Testing of collaborative filtering
|    ├── sparsed_data.ipynb             -> Creation of dataset with sparsed data for benchmark purposes
|    ├── start_up.ipynb                 -> File to start spark session with read of csv data
|    ├── test_content.ipynb             -> File to start spark session with read of csv data
|    ├── recommenders.py                -> Functions to create content-based and hybrid recommeder system
|    ├── coverage.py                    -> Functions to test precision & diversity in recommender systems
|    └── plotting_rating_numbers.ipynb  -> Benchmark & plotting of collaborative filtering (user-item), content-based (iter-item) and hybrid
└── data/
    └── preprocessed/
        ├── reviews.csv     -> Data of review after preprocessing
        └── animes.csv      -> Data about anime after preprocessing
```

## First-time setup

```bash
# 1. Drop your CSV files into data/preprocessed/

# 2. Build and start (takes a few minutes the first time)
docker compose up --build

# 3. Open Jupyter in your browser
#    http://localhost:8888
#    Token: anime123
```

## Day-to-day use (after first build)

```bash
docker compose up        # start
docker compose down      # stop (notebooks and data are preserved)
```

## URLs

| Service    | URL                        |
|------------|----------------------------|
| Jupyter Lab | http://localhost:8888?token=anime123 |
| Spark UI   | http://localhost:4040       |
