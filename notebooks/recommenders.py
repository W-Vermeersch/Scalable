from pyspark.ml.recommendation import ALS
import numpy as np
from pyspark.ml.feature import CountVectorizer
from pyspark.sql.functions import split, lower, trim, col, regexp_replace
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
import time

def train_user_item(train):
    als = ALS(
    maxIter=10,
    regParam=0.1,
    rank=10,
    userCol='user_id',
    itemCol='anime_uid',
    ratingCol='score',
    coldStartStrategy='drop'  # safe now — all test users exist in train
    )

    return als.fit(train)

def train_item_item(anime_data):
    anime_clean = anime_data \
        .withColumn('genre_stripped',
            regexp_replace(col('genre'), r"[\[\]']", '')  # remove [ ] and '
        ) \
        .withColumn('genres_array',
            split(lower(trim(col('genre_stripped'))), ',\\s*')
        ) \
        .drop('genre_stripped') \
        .dropna(subset=['genre'])
    
    cv = CountVectorizer(inputCol='genres_array', outputCol='tfidf', binary=True)
    anime_vectorized = cv.fit(anime_clean).transform(anime_clean)

    # Pull vectors to driver — fine for a few thousand anime
    anime_pd = anime_vectorized.select('anime_id', 'tfidf').toPandas()
    anime_pd['vec'] = anime_pd['tfidf'].apply(lambda v: v.toArray())

    # Build matrix and compute all pairwise similarities at once
    matrix = np.vstack(anime_pd['vec'].values)
    sim_matrix = cosine_similarity(matrix)  # shape: (n_anime, n_anime)

    # Map index → anime_id
    idx_to_uid = anime_pd['anime_id'].to_dict()
    uid_to_idx = {v: k for k, v in idx_to_uid.items()}

    def content_recommend(anime_id, n=10):
        if anime_id not in uid_to_idx:
            print(f'anime_id {anime_id} not found')
            return

        idx = uid_to_idx[anime_id]
        scores = sim_matrix[idx]

        # Get top-n most similar (excluding itself)
        top_indices = np.argsort(scores)[::-1][1:n+1]
        
        results = pd.DataFrame({
            'anime_id':  [idx_to_uid[i] for i in top_indices],
            'similarity': [scores[i]     for i in top_indices]
        })
        
        # Join genre info back
        genre_lookup = anime_clean.select('anime_id', 'genre').toPandas() \
                                .set_index('anime_id')
        results['genre'] = results['anime_id'].map(genre_lookup['genre'])
        
        return results
    
    return content_recommend

def get_similar_items_for_user(user_id, dataset, item_model, 
                                n=10, popularity_bias=0.0, diversity_bias=0.0):
    base_time = time.time()
    history_df = dataset.filter(col('user_id') == user_id) \
        .select('anime_id', 'score').orderBy('score', ascending=False).toPandas().head(n)

    if history_df.empty:
        print(f'User {user_id} has no watch history')
        return None

    user_history_ids  = set(history_df['anime_id'].tolist())
    user_score_map    = dict(zip(history_df['anime_id'], history_df['score']))

    print(f"Get history: {time.time() - base_time} secs!")
    print(f'Size of history: {len(user_history_ids)}')
    base_time = time.time()
    loop_time = base_time

    candidate_scores = {}   # anime_id → list of weighted similarity scores

    for watched_id in user_history_ids:

       
        user_rating  = user_score_map.get(watched_id, 1.0)  # how much user liked it
        sim_set      = item_model(watched_id, n)                     # similarities to all anime


        for _, row in sim_set.iterrows():

            w_idx = row['anime_id']

            # Skip anime the user already watched
            if w_idx in user_history_ids:
                continue

            sim_score = float(row['similarity'])

            # Weight similarity by how much the user liked the source anime
            # e.g. similarity to a 9/10 anime counts more than to a 5/10 anime
            weighted_sim = sim_score * (user_rating / 10.0)

            if w_idx not in candidate_scores:
                candidate_scores[w_idx] = []
            candidate_scores[w_idx].append(weighted_sim)

        # print(f"Get history: {time.time() - loop_time} secs!")
        # loop_time = time.time()


    print(f"Get candidates: {time.time() - base_time} secs!")
    base_time = time.time()

    if not candidate_scores:
        print(f'No candidates found for user {user_id}')
        return None

    # ------------------------------------------------------------------ #
    # Step 3 — Aggregate: average weighted similarity across all sources
    # ------------------------------------------------------------------ #
    results = pd.DataFrame([
        {
            'anime_id':     aid,
            'avg_sim':      float(np.mean(scores)),
            'source_count': len(scores),   # how many history items pointed to this
        }
        for aid, scores in candidate_scores.items()
    ])

    # # ------------------------------------------------------------------ #
    # # Step 4 — Popularity bias
    # # How often each candidate appears in the full dataset = proxy for popularity
    # # ------------------------------------------------------------------ #
    # if popularity_bias != 0.0:
    #     popularity_map = dataset.groupBy('anime_id') \
    #         .count().toPandas().set_index('anime_id')['count']

    #     results['popularity'] = results['anime_id'].map(popularity_map).fillna(0)

    #     # Normalize popularity to [0, 1]
    #     p_min, p_max = results['popularity'].min(), results['popularity'].max()
    #     results['popularity_norm'] = (results['popularity'] - p_min) / \
    #                                  (p_max - p_min + 1e-9)
    # else:
    #     results['popularity_norm'] = 0.0

    # # ------------------------------------------------------------------ #
    # # Step 5 — Diversity bias
    # # Penalise candidates whose genres heavily overlap with already-seen genres
    # # ------------------------------------------------------------------ #
    # if diversity_bias != 0.0:
    #     # Collect all genres the user has already seen
    #     genre_lookup = anime_clean.select('anime_id', 'genres_array') \
    #         .toPandas().set_index('anime_id')

    #     seen_genres = set()
    #     for aid in user_history_ids:
    #         if aid in genre_lookup.index:
    #             seen_genres.update(genre_lookup.loc[aid, 'genres_array'])

    #     def novelty_score(anime_id):
    #         """1.0 = completely new genres, 0.0 = all genres already seen"""
    #         if anime_id not in genre_lookup.index:
    #             return 0.5
    #         candidate_genres = set(genre_lookup.loc[anime_id, 'genres_array'])
    #         if not candidate_genres:
    #             return 0.5
    #         overlap = len(candidate_genres & seen_genres) / len(candidate_genres)
    #         return 1.0 - overlap  # higher = more novel genres

    #     results['novelty'] = results['anime_id'].map(novelty_score)
    # else:
    #     results['novelty'] = 0.0

    # ------------------------------------------------------------------ #
    # Step 6 — Final score combining all signals
    # ------------------------------------------------------------------ #
    # Normalize avg_sim to [0, 1] first
    s_min, s_max = results['avg_sim'].min(), results['avg_sim'].max()
    results['avg_sim_norm'] = (results['avg_sim'] - s_min) / (s_max - s_min + 1e-9)

    # Remaining weight after biases goes to similarity
    remaining = 1.0 #- popularity_bias - diversity_bias
    assert remaining >= 0, 'popularity_bias + diversity_bias must not exceed 1.0'

    results['final_score'] = (
        remaining        * results['avg_sim_norm']  # +
        # popularity_bias  * results['popularity_norm'] +
        # diversity_bias   * results['novelty']
    )

    # ------------------------------------------------------------------ #
    # Step 7 — Return top n with metadata
    # ------------------------------------------------------------------ #
    top_n = results.sort_values('final_score', ascending=False).head(n)

    print(f"Process results: {time.time() - base_time} secs!")
    base_time = time.time()


    return top_n[['anime_id', 'avg_sim', 'source_count', 
                  'final_score'#, 'popularity_norm', 'novelty'
                  ]]
    