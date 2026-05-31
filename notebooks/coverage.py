from recommenders import hybridV1
import pandas as pd
import numpy as np
from pyspark.sql.functions import  col, explode

def precision_recall_at_k(user_id, recommendations, dataset, k=10, threshold=7.0):
    """
    recommendations : pandas DataFrame with anime_id column (your hybrid output)
    threshold       : minimum score to consider an anime 'relevant'
    """
    # Ground truth — anime the user rated highly (held out)
    relevant = set(
        dataset.filter(
            (col('user_id') == user_id) & (col('score') >= threshold)
        ).select('anime_id').toPandas()['anime_id'].tolist()
    )

    recommended = set(recommendations['anime_id'].head(k).tolist())

    hits = recommended & relevant

    precision = len(hits) / k
    recall    = len(hits) / len(relevant) if relevant else 0.0

    return {
        'precision@k': precision,
        'recall@k':    recall,
        'hits':        len(hits),
        'k':           k,
        'n_relevant':  len(relevant)
    }


def catalog_coverage(recommendations_list, dataset, sample_n=100):
    """
    recommendations_list : list of pandas DataFrames, one per user
    sample_n             : how many users to sample (full run is expensive)
    """
    all_recommended = set()
    total_anime = dataset.select('anime_id').distinct().count()

    for recs in recommendations_list:
        if recs is not None:
            all_recommended.update(recs['anime_id'].tolist())

    coverage = len(all_recommended) / total_anime
    print(f'Catalog coverage: {coverage:.2%} ({len(all_recommended)} / {total_anime} anime)')
    return coverage


def diversity_score(recommendations, anime_data):
    """
    Measures average pairwise genre dissimilarity within a recommendation list.
    Score of 1.0 = all different genres, 0.0 = all identical genres.
    """
    from itertools import combinations

    recs_with_genre = recommendations.merge(
        anime_data.select('anime_id', 'genre').toPandas(),
        on='anime_id', how='left'
    ).dropna(subset=['genre'])

    if len(recs_with_genre) < 2:
        return 0.0

    genre_sets = [
        set(g.strip("[]'").replace("'", "").split(', '))
        for g in recs_with_genre['genre']
    ]

    dissimilarities = []
    for a, b in combinations(genre_sets, 2):
        if not a or not b:
            continue
        jaccard_sim = len(a & b) / len(a | b)
        dissimilarities.append(1.0 - jaccard_sim)  # dissimilarity

    return float(np.mean(dissimilarities)) if dissimilarities else 0.0


def novelty_score(recommendations, dataset):
    """
    Lower popularity rank = more novel.
    Based on the idea that recommending obvious popular anime is less useful.
    """
    popularity = dataset.groupBy('anime_id') \
        .count().toPandas().set_index('anime_id')['count']

    total_users = dataset.select('user_id').distinct().count()

    scores = []
    for aid in recommendations['anime_id']:
        pop = popularity.get(aid, 1)
        # Self-information: rare items have higher novelty
        novelty = -np.log2(pop / total_users + 1e-9)
        scores.append(novelty)

    return float(np.mean(scores))

def evaluate_all(user_id, ui_model, ui_model_better, ii_model, final_data, 
                 score_history, anime_data, spark, k=10):

    results = {}

    # --- ALS only ---
    target_user = spark.createDataFrame([(user_id,)], ['user_id'])
    als_recs = ui_model.recommendForUserSubset(target_user, k) \
        .select('user_id', explode('recommendations').alias('rec')) \
        .select(col('rec.anime_id').alias('anime_id')) \
        .toPandas()
    
    als_recs_better = ui_model_better.recommendForUserSubset(target_user, k) \
        .select('user_id', explode('recommendations').alias('rec')) \
        .select(col('rec.anime_id').alias('anime_id')) \
        .toPandas()

    # --- Content-based only ---
    ii_recs = ii_model(user_id, k)

    # --- Hybrid ---
    hybrid_recs = hybridV1(
        user_id, ui_model, ii_model, final_data, score_history, spark, n=k
    )
    hybrid_pd = hybrid_recs if hybrid_recs is not None else pd.DataFrame()

    for name, recs in [('ALS', als_recs),('ALS_better', als_recs_better), ('Content', ii_recs), ('Hybrid', hybrid_pd)]:
        if recs is None or recs.empty:
            print(f'{name}: no recommendations')
            continue

        pr    = precision_recall_at_k(user_id, recs, final_data, k=k)
        div   = diversity_score(recs, anime_data)
        nov   = novelty_score(recs, final_data)

        results[name] = {
            'precision@k': round(pr['precision@k'], 4),
            'recall@k':    round(pr['recall@k'],    4),
            'diversity':   round(div,                4),
            'novelty':     round(nov,                4),
        }

    return pd.DataFrame(results).T
