import streamlit as st
import numpy as np
import pandas as pd
import joblib
import requests
from thefuzz import process
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import ast
import os
from dotenv import load_dotenv


# Set Streamlit page configuration
st.set_page_config(page_title="Movie Recommender", layout="wide")

# Load similarity matrix and dataset
file_path = "similarity_matrix.joblib"
similarity_matrix = joblib.load(file_path)
df_4 = pd.read_csv("df_4.csv")  # Load your dataset



# Get the API key from the env file throw load_dotenv to hide the personal key
load_dotenv("api.env")
API_KEY_ = os.getenv("API_KEY_")

# Get movie info throw the API
def get_movie_info(movie_id):
    url = f"https://api.themoviedb.org/3/movie/{movie_id}?api_key={API_KEY_}"
    try:
        response = requests.get(url)
        response.raise_for_status()
        return response.json()

    except requests.exceptions.RequestException as e:
        st.error(f"Error fetching movie information: {e}")
        return None


def get_recommendations_by_movie_title(movie_title, n=10):
    # use of Thefuzz.process for getting the closest to the input
    best_match = process.extractOne(movie_title, df_4['title'].values, score_cutoff=60) # using the Thefuzz library, extractone method to get the closest result to the movie name.
    if best_match:
        movie_index = df_4[df_4['title'] == best_match[0]].index[0] # getting the best match id
    else:
        st.error("Movie not found!")
        return []

    similarity_scores = similarity_matrix[movie_index] # applying similarity matrix
    similar_movies_indexes = np.argsort(similarity_scores)[::-1][1:n+1] #getting best n matches, default 10
    return [int(df_4.iloc[i]['id']) for i in similar_movies_indexes] # getting list with indexes

def get_keywords_recommendations(keywords, n=10):
    keywords = " ".join(keywords.split())
    tfidf = TfidfVectorizer(stop_words='english') # using TfidfVectorizer to convert word into vectors
    movie_data = df_4['director'].astype(str) + df_4['cast_3'].astype(str) + df_4['title'].astype(str) #obtaining movie data from df
    all_text = [keywords] + movie_data.tolist() # joining both list of words
    tfidf_matrix = tfidf.fit_transform(all_text) # obtaining the vectors of those tokens
    key_tfidf = tfidf_matrix[0] # Obtaining the first row, vectors for the keywords
    movie_tfidf = tfidf_matrix[1:] # obtaining the vectors of the movies (the remaining ones)
    result = cosine_similarity(key_tfidf, movie_tfidf) # compute cosine similarity between keywords and all the movies data (cast, director, titles)
    similar_key_movies = sorted(enumerate(result[0]), key=lambda x: x[1], reverse=True) # Get top n most similar movies (excluding itself), using enumerated to create an index
    return [int(df_4.iloc[i[0]].id) for i in similar_key_movies[:n]] #Extract top recommendations ids list

def recommend_movies_by_genre(genre, sorting, top_n=50):

    # Filter the DataFrame
    df_4['genres'] = df_4['genres'].apply(ast.literal_eval) # turning column dataframe strings into a list
    # getting the first appearence of each list to make the search with apply(), then making the filtering (x == to the genre chosen, is a list a len(list) > 0)
    filtered_df = df_4[df_4['genres'].apply(lambda x: x[0] == genre if isinstance(x, list) and len(x) > 0 else False)]
    # Randomly sample with pandas.sample() top_n is number of rows to be selected, len(filtered_df) is the max of rows to be selected
    recommendations = filtered_df.sample(n=min(top_n, len(filtered_df)))
    #sorting by the parameter introduced
    recommendations = recommendations.sort_values(by=[sorting], ascending=False)

    return recommendations['id'].tolist()

# for each movie, display the following code, getting the info from the API when possible, otherwsise from the DF
def display_movie_info(movie_ids):
    for movie_id in movie_ids:
        movie_row = df_4[df_4['id'] == movie_id] # select each movie by their id's in the DATAFRAME
        movie_data = get_movie_info(movie_id) # we get the API data by it's id
        if not movie_row.empty and movie_data: # We get the following data from the API,
            title = movie_data.get("title", "Unknown Title")
            release_date = movie_data.get("release_date", "Unknown Year")
            release_year = release_date.split("-")[0] if release_date else "Unknown Year" # getting the first 4 numbers of the release_date, the year

            # Ensure director and cast are properly formatted, getting the info from the dataframe, because the API doesn't provide it.
            director = movie_row.iloc[0]['director']
            if isinstance(director, str):
                director = director.strip("[]").replace("'", "")

            cast = movie_row.iloc[0]['cast_3']
            if isinstance(cast, str):
                cast = cast.strip("[]").replace("'", "").split(", ")[:3]

            st.subheader(f"{title} ({release_year})") #getting the title and the release year
            st.markdown(f"<p style='font-size:14px;'>Director: {director}     ---     Cast: {', '.join(cast)}</p>", unsafe_allow_html=True)
            st.image(f"https://image.tmdb.org/t/p/w500{movie_row.iloc[0]['poster_path']}", width=200) #setting the poster, getting the path from he df
            st.write(f"**Rating:** {movie_data.get('vote_average', 'N/A')}") # getting the vote_average from the API
            st.write(f"**Overview:** {movie_data.get('overview', 'No overview available.')}") # getting the overview data from the API
            st.markdown("---")


def main():
    st.title("🎬 Movie Recommendation System")
    option = st.sidebar.selectbox("Choose a recommendation type:", ["By Movie Title", "By Keywords", "By Genre"]) #setting the sidebar

    if option == "By Movie Title": #first option
        movie_title = st.text_input("Enter a movie title:") #test input for the movie title
        if st.button("Get Recommendations"):
            movie_ids = get_recommendations_by_movie_title(movie_title)
            display_movie_info(movie_ids)

    elif option == "By Keywords": # second option
        keywords = st.text_input("Enter keywords (e.g., director, actor, movie title):") # text input
        if st.button("Get Recommendations"):
            movie_ids = get_keywords_recommendations(keywords)
            display_movie_info(movie_ids)

    elif option == "By Genre": # third option
        genre = st.selectbox("Select a genre:", ('Drama', 'Comedy', 'Thriller', 'Action', 'Romance', 'Adventure', 'Crime', 'Family', 'Science Fiction', 'Fantasy', 'History', 'Mystery', 'Animation', 'Horror', 'War', 'Music', 'Western', 'Documentary', 'TV Movie'))
        sorting = st.selectbox("Sort by:", ["vote_average", "release_date"])
        if st.button("Get Recommendations"):
            movie_ids = recommend_movies_by_genre(genre, sorting)
            display_movie_info(movie_ids)



if __name__ == "__main__":
    main()




#streamlit run /workspaces/19839496/project/project.py


