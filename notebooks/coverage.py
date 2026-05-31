from recommenders import hybridV1, calculate_ui_weights
import pandas as pd
import numpy as np
from pyspark.sql.functions import  col, explode
from itertools import combinations

def precision_recall_at_k(user_id, recommendations, dataset, k=10, threshold=7.0):
    relevant = set(
        dataset.filter(
            (col('user_id') == user_id) & (col('score') >= threshold)
        ).select('anime_id').toPandas()['anime_id'].tolist()
    )

    recommended = set(recommendations['anime_id'].head(k).tolist())

    hits = recommended & relevant

    precision = len(hits) / k

    return {
        'precision@k': precision,
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
    popularity = dataset.groupBy('anime_id') \
        .count().toPandas().set_index('anime_id')['count']

    total_users = dataset.select('user_id').distinct().count()

    scores = []
    for aid in recommendations['anime_id']:
        pop = popularity.get(aid, 1)
        novelty = -np.log2(pop / total_users + 1e-9)
        scores.append(novelty)

    return float(np.mean(scores))

def evaluate_all(user_id, ui_model, ui_model_better, ii_model, final_data, 
                 score_history, anime_data, spark, k=10):

    results = {}

    # Collaborative-filtering
    target_user = spark.createDataFrame([(user_id,)], ['user_id'])
    als_recs = ui_model.recommendForUserSubset(target_user, k) \
        .select('user_id', explode('recommendations').alias('rec')) \
        .select(col('rec.anime_id').alias('anime_id')) \
        .toPandas()
    
    als_recs_better = ui_model_better.recommendForUserSubset(target_user, k) \
        .select('user_id', explode('recommendations').alias('rec')) \
        .select(col('rec.anime_id').alias('anime_id')) \
        .toPandas()

    # Content-based
    ii_recs = ii_model(user_id, k)

    # Hybrid
    hybrid_recs = hybridV1(
        user_id, ui_model, ii_model, final_data, score_history, spark, n=k
    )

    hybrid_recs_better = hybridV1(
        user_id, ui_model_better, ii_model, final_data, score_history, spark, n=k
    )

    hybrid_recs_3 = hybridV1(
        user_id, ui_model, ii_model, final_data, score_history, spark, n=k, ui_weight=0.7,ii_weight=0.3
    )

    hybrid_recs_better_3 = hybridV1(
        user_id, ui_model_better, ii_model, final_data, score_history, spark, n=k, ui_weight=0.7,ii_weight=0.3
    )

    hybrid_recs_7 = hybridV1(
        user_id, ui_model, ii_model, final_data, score_history, spark, n=k, ui_weight=0.3,ii_weight=0.7
    )

    hybrid_recs_better_7 = hybridV1(
        user_id, ui_model_better, ii_model, final_data, score_history, spark, n=k, ui_weight=0.3,ii_weight=0.7
    )
    dynamic_w = calculate_ui_weights(final_data, user_id)
    print(f"Dynamic weights: {round(dynamic_w, 4)}")
    hybrid_recs_d = hybridV1(
        user_id, ui_model, ii_model, final_data, score_history, spark, n=k, ui_weight=dynamic_w,ii_weight=(1 - dynamic_w)
    )

    hybrid_recs_better_d = hybridV1(
        user_id, ui_model_better, ii_model, final_data, score_history, spark, n=k, ui_weight=dynamic_w,ii_weight=(1 - dynamic_w)
    )

    for name, recs in [ ('ALS', als_recs),('ALS_better', als_recs_better), ('Content', ii_recs), ('Hybrid', hybrid_recs), ('Hybrid (better ALS)', hybrid_recs_better),
                        ('Hybrid (w=3 ii)', hybrid_recs_3), ('Hybrid (better ALS) (w=3 ii)', hybrid_recs_better_3),('Hybrid (w=7 ii)', hybrid_recs_7), ('Hybrid (better ALS) (w=7 ii)', hybrid_recs_better_7),
                        ('Hybrid (dynamic)', hybrid_recs_d), ('Hybrid (better ALS) (dynamic)', hybrid_recs_better_d)]:

        pr = precision_recall_at_k(user_id, recs, final_data, k=k)
        div = diversity_score(recs, anime_data)
        nov = novelty_score(recs, final_data)

        results[name] = {
            'precision@k': round(pr['precision@k'], 4),
            'diversity': round(div, 4),
            'novelty': round(nov, 4),
        }
    print(f"Number of ratings: {final_data.filter(col('user_id') == user_id).count()}")
    return pd.DataFrame(results).transpose
