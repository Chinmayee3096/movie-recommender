# recommender.py

import pandas as pd
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer

# --------------------------
# 1️⃣ Load datasets
# --------------------------
tmdb = pd.read_csv("data/tmdb_5000_movies.csv")
indian = pd.read_csv("data/indian_movies.csv")

# --------------------------
# 2️⃣ Preprocess TMDB dataset
# --------------------------
tmdb = tmdb[['title', 'overview', 'genres', 'original_language', 'vote_average']]
tmdb['overview'] = tmdb['overview'].fillna('')
tmdb['genres'] = tmdb['genres'].astype(str)

tmdb.rename(columns={
    'title': 'movie_name',
    'original_language': 'language',
    'vote_average': 'rating'
}, inplace=True)

# Combine genres + overview for TF-IDF
tmdb['tags'] = tmdb['overview'] + " " + tmdb['genres']

# --------------------------
# 3️⃣ Preprocess Indian movies dataset
# --------------------------
indian = indian[['Movie Name', 'Genre', 'Language', 'Rating(10)']]
indian.rename(columns={
    'Movie Name': 'movie_name',
    'Genre': 'genres',
    'Language': 'language',
    'Rating(10)': 'rating'
}, inplace=True)

indian['genres'] = indian['genres'].astype(str)
indian['tags'] = indian['genres']  # Only genres for TF-IDF

# --------------------------
# 4️⃣ Merge datasets
# --------------------------
movies = pd.concat([
    tmdb[['movie_name', 'language', 'rating', 'genres', 'tags']],
    indian[['movie_name', 'language', 'rating', 'genres', 'tags']]
], ignore_index=True)

movies.drop_duplicates(subset=['movie_name'], inplace=True)
movies.reset_index(drop=True, inplace=True)

# --------------------------
# 5️⃣ TF-IDF Vectorization
# --------------------------
vectorizer = TfidfVectorizer(stop_words='english', max_features=5000)
vectors = vectorizer.fit_transform(movies['tags'])

# --------------------------
# 6️⃣ Save necessary files
# --------------------------
pickle.dump(movies, open('movies.pkl', 'wb'))
pickle.dump(vectors, open('vectors.pkl', 'wb'))
pickle.dump(vectorizer, open('vectorizer.pkl', 'wb'))

print("✅ Files saved successfully")
print("🎬 Total movies:", movies.shape[0])