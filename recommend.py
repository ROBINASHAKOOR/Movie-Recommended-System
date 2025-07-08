# import libraris
import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

# Load movies and credits CSV
movies_df = pd.read_csv("movies.csv")
credits_df = pd.read_csv("credits.csv")

# Merge datasets on 'id'
credits_df.columns = ['id', 'title_temp', 'cast', 'crew']  # Rename to avoid column conflict
df = movies_df.merge(credits_df, on='id')

# Clean and fill missing values
df['overview'] = df['overview'].fillna('')

# TF-IDF vectorizer
tfidf = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf.fit_transform(df['overview'])

# Cosine similarity matrix
cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)

# Reverse map of indices and movie titles
indices = pd.Series(df.index, index=df['title']).drop_duplicates()

# Recommendation function
def get_recommendations(title, cosine_sim=cosine_sim):
    if title not in indices:
        return ["Movie not found. Please try another."]

    idx = indices[title]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:11]  # Top 10 similar
    movie_indices = [i[0] for i in sim_scores]
    return df['title'].iloc[movie_indices]

#Streamlit UI

st.title(" Movie Recommendation System")
st.write("Enter a movie title and get similar movie recommendations!")

movie_input = st.text_input("Enter movie title:")

if st.button("Get Recommendations"):
    if movie_input:
        recommendations = get_recommendations(movie_input)
        st.write("### Top 10 Recommended Movies:")
        for i, rec in enumerate(recommendations, 1):
            st.write(f"{i}. {rec}")
    else:
        st.warning("Please enter a movie title.")
