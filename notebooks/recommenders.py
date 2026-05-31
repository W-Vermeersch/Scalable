from pyspark.ml.recommendation import ALS
import numpy as np
from pyspark.ml.feature import CountVectorizer
from pyspark.sql.functions import split, lower, trim, col, regexp_replace, explode, min, max, lit, udf, mean
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from pyspark.sql.types import FloatType
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

    genre_lookup = anime_clean.select('anime_id', 'genre').toPandas().set_index('anime_id')

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
        results['genre'] = results['anime_id'].map(genre_lookup['genre'])
        
        return results
    
    def score_base_history(anime_id, history_catalog):
        if anime_id not in uid_to_idx:
            return 0.0

        c_idx = uid_to_idx[anime_id]

        weighted_scores = [
            sim_matrix[uid_to_idx[watched_id]][c_idx] * ((user_rating - 5.0) if user_rating != 0.0 else 0.0)# (user_rating / 10.0)
            for watched_id, user_rating in history_catalog
            if watched_id in uid_to_idx        # skip history items not in model
            and watched_id != anime_id         # skip if candidate is in history
        ]

        if not weighted_scores:
            return 0.0

        return float(np.mean(weighted_scores))
    
    return content_recommend, score_base_history

def get_similar_items_for_user(user_id, dataset, item_model, score_history,n=10):
    base_time = time.time()
    total_time = time.time()
    history_df = dataset.filter(col('user_id') == user_id) \
        .select('anime_id', 'score').orderBy('score', ascending=False).toPandas()
    
    slice_history = history_df.head(n)

    if history_df.empty:
        print(f'User {user_id} has no watch history')
        return None

    user_history_ids  = history_df['anime_id'].tolist()
    # user_history_catalog= list(zip(slice_history['anime_id'], slice_history['score']))
    full_history_catalog= list(zip(history_df['anime_id'], history_df['score']))

    print(f"Get history: {time.time() - base_time} secs!")
    print(f'Size of history: {len(user_history_ids)}')
    base_time = time.time()
    loop_time = base_time

    candidate_scores = {}   # anime_id → list of weighted similarity scores

    for watched_id in history_df['anime_id']: # Test by swithcing between sliced and not sliced history (better ranks or better performance)

        sim_set = item_model(watched_id, n)


        for w_idx in sim_set['anime_id']:

            # w_idx = row['anime_id']

            # Skip anime the user already watched
            if w_idx in user_history_ids:
                continue

            if w_idx not in candidate_scores:
                candidate_scores[w_idx] = score_history(w_idx, full_history_catalog) # Better scoring than just rating and similarity


    print(f"Get candidates: {time.time() - base_time} secs!")
    base_time = time.time()

    if not candidate_scores:
        print(f'No candidates found for user {user_id}')
        return None

    results = pd.DataFrame([
        {
            'anime_id':     aid,
            # 'avg_sim':      float(np.mean(scores)),
            'avg_sim':      scores,
        }
        for aid, scores in candidate_scores.items()
    ])

    # Normalize avg_sim to [0, 1] first
    s_min, s_max = results['avg_sim'].min(), results['avg_sim'].max()
    results['final_score'] = (results['avg_sim'] - s_min) / (s_max - s_min + 1e-9)

    top_n = results.sort_values('final_score', ascending=False).head(n)

    print(f"Process results: {time.time() - base_time} secs!")
    print(f"Total time: {time.time() - total_time} secs!")


    return top_n[['anime_id', 'avg_sim','final_score']]
    


def hybridV1(user_id, user_item, item_item, dataset, score_history, spark, n=10, ui_weight=0.5, ii_weight=0.5):
    # UI_rec = user_item(user_id, n)
    II_rec = item_item(user_id, n)

    history_df = dataset.filter(col('user_id') == user_id) \
        .select('anime_id', 'score').toPandas()


    target_user = spark.createDataFrame([(user_id,)], ['user_id'])
    UI_rec = user_item.recommendForUserSubset(target_user, 10)

    # Flatten the recommendations array
    UI_rec_flat = UI_rec.select(
        'user_id',
        explode('recommendations').alias('rec')
    ).select(
        'user_id',
        col('rec.anime_id').alias('anime_id'),
        col('rec.rating').alias('ui_score')
    )

    

    pairs = spark.createDataFrame(
        [(user_id, int(aid)) for aid in II_rec['anime_id']],
        ['user_id', 'anime_id']
    )

    # ALS transform scores every (user, item) pair it recognises
    II_after_ALS = user_item.transform(pairs).select(
            'anime_id',
            col('prediction').alias('ui_score')
        ).dropna(
            subset=['ui_score']
        )

    # II_rec_flat = predictions.select(
    #     'user_id',
    #     'anime_id',
    #     col('prediction').alias('ui_score')
    # ).dropna(subset=['ui_score'] # drop cold-start entries ALS couldn't score
    # )

    II_rec_spark = spark.createDataFrame(II_rec[['anime_id', 'avg_sim', 'final_score']].rename(
            columns={'final_score': 'ii_score'}
        ))

    II_rec_flat = II_after_ALS.join(II_rec_spark, on='anime_id', how='inner')

    # UI_rec_flat.show(truncate=False)
    # II_rec_flat.show(truncate=False)

def hybridV1(user_id, user_item, item_item, dataset, score_history, spark, n=10, ui_weight=0.5, ii_weight=0.5):
    

    # Prepare history for item-item scoring of user-item recommendations 
    history_df = dataset.filter(col('user_id') == user_id) \
        .select('anime_id', 'score').toPandas()
    history_catalog = list(zip(history_df['anime_id'], history_df['score']))

    def score_history_udf_fn(anime_id):
        return float(score_history(anime_id, history_catalog))

    score_history_udf = udf(score_history_udf_fn, FloatType())


    # User-item recommendations for user
    target_user = spark.createDataFrame([(user_id,)], ['user_id'])
    UI_rec = user_item.recommendForUserSubset(target_user, 10)

    # Flatten the recommendations from user-item
    UI_rec_flat = UI_rec.select(
        'user_id',
        explode('recommendations').alias('rec')
    ).select(
        'user_id',
        col('rec.anime_id').alias('anime_id'),
        col('rec.rating').alias('ui_score')
    )

    
    # Item-item recommendations for user
    II_rec = item_item(user_id, n)

    # ALS transform add a user-item score to the anime recommended by item-item
    pairs = spark.createDataFrame(
        [(user_id, int(aid)) for aid in II_rec['anime_id']],
        ['user_id', 'anime_id']
    )
    II_after_ALS = user_item.transform(pairs).select(
            'anime_id',
            col('prediction').alias('ui_score')
        ).dropna(
            subset=['ui_score']
        )

    # For ite-item pairs, combine ii_score & ui_score
    II_rec_spark = spark.createDataFrame(II_rec[['anime_id', 'avg_sim', 'final_score']].rename(
            columns={'final_score': 'ii_score'}
        ))
    II_rec_flat = II_after_ALS.join(II_rec_spark, on='anime_id', how='inner')


    # Add ite-item score to result from ALS
    UI_rec_flat = UI_rec_flat.withColumn('ii_score', score_history_udf(col('anime_id')))


    # Merge the recommendations from user-item & item-item
    merged = UI_rec_flat.union(II_rec_flat) \
        .groupBy('anime_id') \
        .agg(
            mean('ui_score').alias('ui_score'),
            mean('ii_score').alias('ii_score')
        )

    # Prepare min and max for normalization
    ui_min, ui_max = merged.agg(
        min('ui_score'), max('ui_score')
    ).first()

    ii_min, ii_max = merged.agg(
        min('ii_score'), max('ii_score')
    ).first()

    # Normalize the scores
    merged = merged.withColumn('ui_score_norm',
            (col('ui_score') - lit(ui_min)) / lit(ui_max - ui_min + 1e-9)
        ).withColumn('ii_score_norm',
            (col('ii_score') - lit(ii_min)) / lit(ii_max - ii_min + 1e-9)
        )

    # Calculate hybrid score with weights
    merged = merged.withColumn(
        'hybrid_score',
        lit(ui_weight) * col('ui_score_norm') +
        lit(ii_weight) * col('ii_score_norm')
    )

    result = merged.orderBy('hybrid_score', ascending=False).limit(n*2)

    result.select('anime_id', 'ui_score', 'ii_score', 'hybrid_score').show(n*2, truncate=False)
    return result


# def hybridV2(user_id, dataset, user_item, item_item, n=10):
#     return