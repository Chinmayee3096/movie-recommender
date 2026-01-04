# app.py

import streamlit as st
import pickle
import requests
from sklearn.metrics.pairwise import cosine_similarity

# ---------------- TMDB API KEY ----------------
TMDB_API_KEY = "9835299d0f83bb7606de326270693bed"  # Replace with your TMDB API key

# ---------------- PAGE CONFIG & STYLE ----------------
st.set_page_config(page_title="Movie Recommender", page_icon="🎬", layout="wide")

st.markdown("""
<style>
/* Dark gradient background for the app */
.stApp {
    background: linear-gradient(135deg, #0f2027, #203a43, #2c5364);
}

/* Headings / labels color */
h1, h2, h3, h4, h5, h6, label {
    color: #ffffff !important;
}
</style>
""", unsafe_allow_html=True)
# ---------------- LOAD DATA ----------------
movies = pickle.load(open("movies.pkl", "rb"))
vectors = pickle.load(open("vectors.pkl", "rb"))

# ---------------- POSTER FETCH FUNCTION ----------------
def fetch_poster(movie_name):
    url = "https://api.themoviedb.org/3/search/movie"
    params = {"api_key": TMDB_API_KEY, "query": movie_name}
    response = requests.get(url, params=params)
    data = response.json()

    if data.get("results"):
        poster_path = data["results"][0].get("poster_path")
        if poster_path:
            return "https://image.tmdb.org/t/p/w500" + poster_path
    return "https://via.placeholder.com/300x450?text=No+Poster"

# ---------------- RECOMMENDER FUNCTION ----------------
def recommend(movie, language, genre, top_n=5):
    filtered = movies[
        (movies['language'].str.lower() == language.lower()) &
        (movies['genres'].str.contains(genre, case=False, na=False))
    ].reset_index(drop=True)

    if movie not in filtered['movie_name'].values:
        return []

    idx = filtered[filtered['movie_name'] == movie].index[0]
    vectors_filtered = vectors[filtered.index]

    similarity_scores = cosine_similarity(vectors_filtered[idx], vectors_filtered).flatten()
    similar_indices = similarity_scores.argsort()[-(top_n+1):-1][::-1]

    return filtered.iloc[similar_indices]

# ---------------- HEADER ----------------
st.title("🎬 Movie Recommendation System")
st.markdown("**Discover movies you’ll love — Indian & International**")

# ---------------- SEARCH BAR ----------------
search_query = st.text_input("🔍 Search for a movie")

# ---------------- LANGUAGE FILTER ----------------
languages = sorted(movies['language'].dropna().unique())
selected_language = st.selectbox("🌍 Select Language", languages)

# ---------------- GENRE FILTER ----------------
filtered_lang = movies[movies['language'].str.lower() == selected_language.lower()]
genres = set()
for g in filtered_lang['genres']:
    if isinstance(g, str):
        for word in g.replace(",", " ").split():
            genres.add(word)
selected_genre = st.selectbox("🎭 Select Genre", sorted(genres))

# ---------------- MOVIE SELECTION ----------------
filtered_movies = filtered_lang[
    filtered_lang['genres'].str.contains(selected_genre, case=False, na=False)
]

if search_query:
    filtered_movies = filtered_movies[
        filtered_movies['movie_name'].str.contains(search_query, case=False)
    ]

movie_list = filtered_movies['movie_name'].values
if len(movie_list) == 0:
    st.warning("No movies found for this search/filters.")
    st.stop()

selected_movie = st.selectbox("🎞️ Select Movie", movie_list)

# ---------------- RECOMMEND BUTTON ----------------
if st.button("✨ Recommend"):
    with st.spinner("Finding similar movies you may like..."):
        results = recommend(selected_movie, selected_language, selected_genre)

    if len(results) == 0:
        st.warning("No recommendations found for this selection.")
    else:
        st.subheader("🍿 Similar Movies You May Like")
        cols = st.columns(5)
        for col, (_, row) in zip(cols, results.iterrows()):
            with col:
                st.image(fetch_poster(row['movie_name']), use_container_width=True)
                st.markdown(f"**{row['movie_name']}**")
                st.markdown(f"⭐ {row['rating']}")