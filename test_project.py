import pytest
import pandas as pd
import joblib
from project import get_recommendations_by_movie_title
from project import get_keywords_recommendations
from project import recommend_movies_by_genre


# Load test data
file_path = "similarity_matrix.joblib"
similarity_matrix = joblib.load(file_path)
df_4 = pd.read_csv("df_4.csv")

def test_get_recommendations_by_movie_title():
    movie_title = "Inception"
    recommendations = get_recommendations_by_movie_title(movie_title)
    assert isinstance(recommendations, list)
    assert len(recommendations) > 0
    assert all(isinstance(movie_id, int) for movie_id in recommendations)

def test_get_keywords_recommendations():
    keywords = "action adventure"
    recommendations = get_keywords_recommendations(keywords)
    assert isinstance(recommendations, list)
    assert len(recommendations) > 0
    assert all(isinstance(movie_id, int) for movie_id in recommendations)

def test_recommend_movies_by_genre():
    genre = "Action"
    sorting = "vote_average"
    recommendations = recommend_movies_by_genre(genre, sorting)
    assert isinstance(recommendations, list)
    assert len(recommendations) > 0
    assert all(isinstance(movie_id, int) for movie_id in recommendations)
