# Anime Recommender — Docker + PySpark + Jupyter

## Folder structure expected on your machine

```
anime-spark/
├── Dockerfile
├── docker-compose.yml
├── .dockerignore
├── notebooks/
│   └── anime_recommender.ipynb   ← starter notebook
└── data/
    └── preprocessed/
        ├── reviews.csv
        └── animes.csv            ← put your CSV files here
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

## Notes

- Java 17 is used inside the container — no `--add-opens` flags needed.
- Any notebooks you save inside `notebooks/` persist on your local machine.
- Your CSV data is mounted read-only; Spark reads it directly from `data/preprocessed/`.
- To change the token, edit `JUPYTER_TOKEN` and `--NotebookApp.token` in `docker-compose.yml`.


# Benchmarks
- Run collaborative_benchmark in the docker to get the results of the collaboraive filtering