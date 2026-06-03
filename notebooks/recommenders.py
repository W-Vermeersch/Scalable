import numpy as np
from pyspark.ml.feature import CountVectorizer
from pyspark.sql.functions import split, lower, trim, col, explode
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

import numpy as np

def jaccard_matrix(matrix):
    # intersection = dot product of binary vectors
    intersection = matrix @ matrix.T
    # union = |A| + |B| - |A∩B|
    row_sums = matrix.sum(axis=1)
    union = row_sums[:, None] + row_sums[None, :] - intersection
    return intersection / (union + 1e-9)

def train_item_item(anime_data, matrix_calc = cosine_similarity):
    anime_clean = anime_data.withColumn('genres_array',
            split(lower(trim(col('genre'))), ',\\s*')
        ).dropna(subset=['genre'])
    
    # vectorize anime genres
    cv = CountVectorizer(inputCol='genres_array', outputCol='tfidf', binary=True)
    anime_vectorized = cv.fit(anime_clean).transform(anime_clean)

    anime_pd = anime_vectorized.select('anime_id', 'tfidf').toPandas()
    anime_pd['vec'] = anime_pd['tfidf'].apply(lambda v: v.toArray())

    # Build matrix and compute all pairwise similarities at once
    matrix = np.vstack(anime_pd['vec'].values)
    sim_matrix = matrix_calc(matrix)  # shape: n_anime x n_anime

    # Map index → anime_id
    uid_to_aid = anime_pd['anime_id'].to_dict()
    aid_to_uid = {v: k for k, v in uid_to_aid.items()}


    def content_recommend(anime_id, n=10):
        if anime_id not in aid_to_uid:
            print(f'anime_id {anime_id} not found')
            return

        idx = aid_to_uid[anime_id]
        scores = sim_matrix[idx]

        # Get top-n most similar (excluding itself)
        top_indices = np.argsort(scores)[::-1][1:n+1]
        
        results = pd.DataFrame({
            'anime_id':  [uid_to_aid[i] for i in top_indices],
            'similarity': [scores[i]     for i in top_indices]
        })

        return results
    
    def score_base_history(anime_id, history_catalog):
        if anime_id not in aid_to_uid:
            return 0.0

        c_idx = aid_to_uid[anime_id]

        weighted_scores = [
            sim_matrix[aid_to_uid[watched_id]][c_idx] * ((user_rating - 5.0) if user_rating != 0.0 else 0.0)
            for watched_id, user_rating in history_catalog
            if watched_id in aid_to_uid
            and watched_id != anime_id
        ]

        if not weighted_scores:
            return 0.0

        return float(np.sum(weighted_scores))
    
    return content_recommend, score_base_history

def get_similar_items_for_user(user_id, dataset, item_model, score_history,n=10):
    history_df = dataset.filter(col('user_id') == user_id) \
        .select('anime_id', 'score').orderBy('score', ascending=False).toPandas()
    

    if history_df.empty:
        print(f'User {user_id} has no watch history')
        return None

    user_history_ids  = history_df['anime_id'].tolist()
    full_history_catalog= list(zip(history_df['anime_id'], history_df['score']))

    candidate_scores = {}   # anime_id → list of weighted similarity scores

    for watched_id in user_history_ids: # Test by swithcing between sliced and not sliced history (better ranks or better performance)

        sim_set = item_model(watched_id, n)

        if sim_set is None or sim_set.empty:
            continue


        for w_idx in sim_set['anime_id']:

            # Skip anime the user already watched
            if w_idx in user_history_ids:
                continue

            if w_idx not in candidate_scores:
                candidate_scores[w_idx] = score_history(w_idx, full_history_catalog) # Better scoring than just rating and similarity


    if not candidate_scores:
        print(f'No candidates found for user {user_id}')
        return None

    results = pd.DataFrame([
        {
            'anime_id':     aid,
            'avg_sim':      scores,
        }
        for aid, scores in candidate_scores.items()
    ])

    # Normalize avg_sim
    s_min, s_max = results['avg_sim'].min(), results['avg_sim'].max()
    results['final_score'] = (results['avg_sim'] - s_min) / (s_max - s_min + 1e-9)

    top_n = results.sort_values('final_score', ascending=False).head(n)


    return top_n[['anime_id', 'avg_sim','final_score']]
    

def hybridV1(user_id, user_item, item_item, dataset, score_history, spark, n=10, ui_weight=0.5, ii_weight=0.5):
    
    # Get item-item candidates
    II_rec = item_item(user_id, n * 5)

    # Build history catalog
    history_catalog = list(zip(
        dataset.filter(col('user_id') == user_id).select('anime_id', 'score').toPandas()['anime_id'],
        dataset.filter(col('user_id') == user_id).select('anime_id', 'score').toPandas()['score']
    ))

    # Get user-item candidates (ALS recommendations)
    target_user = spark.createDataFrame([(user_id,)], ['user_id'])
    UI_rec = user_item.recommendForUserSubset(
        target_user, n * 5
        ).select('user_id', 
                explode('recommendations').alias('rec')
            ).select(
            col('rec.anime_id').alias('anime_id'),
            col('rec.rating').alias('ui_score')
        ).toPandas()

    # Compute ii_score for user-item
    UI_rec['ii_score'] = UI_rec['anime_id'].apply(
        lambda aid: score_history(aid, history_catalog)
    )
    UI_rec_with_II = UI_rec

    # UI_Score over item-item candidates
    pairs = spark.createDataFrame(
        [(user_id, int(a_id), score) for a_id, score in zip(II_rec['anime_id'], II_rec['final_score'])],
        ['user_id', 'anime_id', "ii_score"]
    )
    II_rec_with_UI = user_item.transform(pairs) \
        .select('anime_id', col('prediction').alias('ui_score'), 'ii_score') \
        .dropna(subset=['ui_score']) \
        .toPandas()

    # Merge recommends from item-item and user-item
    merged = pd.concat([
        UI_rec_with_II[['anime_id', 'ui_score', 'ii_score']],
        II_rec_with_UI[['anime_id', 'ui_score', 'ii_score']]
    ]).groupby('anime_id', as_index=False).mean()

    # Normalize ui- & ii-score 
    for c in ['ui_score', 'ii_score']:
        mn, mx = merged[c].min(), merged[c].max()
        merged[f'{c}_norm'] = (merged[c] - mn) / (mx - mn + 1e-9)

    merged['hybrid_score'] = (
        ui_weight * merged['ui_score_norm'] +
        ii_weight * merged['ii_score_norm']
    )

    return merged.sort_values('hybrid_score', ascending=False).head(n)


def calculate_ui_weights(dataset, user_id):
    n = dataset.filter(col('user_id') == user_id).count()

    return calculate_ui_weights_size(n)

def calculate_ui_weights_size(size, scaler=1.0):
    history_factor = np.clip(size / (10.0 * scaler), 1.0, 10.0)
    weight = 0.2 * history_factor

    return np.clip(weight, 0.1, 0.9)