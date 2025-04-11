# TMDB_movie_recommendator
My first project with streamlit using data analysis and jupyter noteboks. 

This app has the aim to create three types of movie recommendations using cosine similarity as a measure of similarity between two non-zero vectors (movies) to establish which movies are closest through a similarity matrix.

    - One from another movie title,
    - Another from keywords, like director or actor names or other movie titles.
    - Last one, don't use cosine similarity, it's just a random selector, selecting the genre and then the sorting selection.

    I used two datasets, one from the TMDB kaggle dataset 1 Million movies, which has most of the data, and then another kaggle
    5000 movies dataset which contains cast and crew info, which I used to make the keyword recommendation system. I filtered based on popularity and rate, and then merging both datasets together, I got a final dataset of 2000 movies. I did not include both datasets in the gibhub because it would exceed 100 mb, but links are here:
    https://www.kaggle.com/datasets/asaniczka/tmdb-movies-dataset-2023-930k-movies
    https://www.kaggle.com/datasets/tmdb/tmdb-movie-metadata

    I did an Exploratory data Analysis (all explained in the .ipynb file) of all the data, worrying about missing data, selecting
    only relevant data, converting JSON and String data into python lists.

    then it was the turn to convert all this data into vectors, for that I used some data preprocessing tools, first these two
    for my categorical data:

    - MultiLabelBinarizer to convert my lists of labels (genres, spoken languages, cast,keywords, etc) (multi-label data) into a
    binary format (one-hot encoding)

    - TfidfVectorizer to convert text into numerical vectors based on word importance in a document. It stands for Term
    Frequency-Inverse Document Frequency (TF-IDF).

    Then to preprocess my numerical data, depending of the distribution of these data I used three different tools, the
    standscaller for skewed values, the Robustscaler where I got a lot of outliers, then the MinMaxScaler for the small range values.

    Then I apply the cosine similarity algorithm for each of my features, and I apply weights for the algorithms for the movie
    title name research is applied, getting a "similarity_combined" matrix, then I will export it, then importing to my project through the library "joblib" And that's it for the EDA and data preprocessing.



    Now in my "project.py" file I will need:

    - the "similarity_matrix" imported through joblib and I need the processed dataframe (.csv)

    The six main functions in the application are:

    1. A request (get_movie_info()) to the "The movie Database" API, so I will need my API key, to keep it secret, I will load it
    from an .env file. I could have gotten the API info (title, release year and the poster path) from the dataframe, but I prefered practice this way.


    2. In the get_recommendations_by_movie_title(movie_title, n=10) method, I will use the "TheFuzz" library to get the best
    match out of the search by the user, then I will use the similarity_matrix to get the best 10 scores in the matrix, then sorted would be my recommendation movies indexes.

    3. In the get_keywords_recommendations(keywords, n=10) method, I will combine the words from the user search and all the data
    from the actors, directors and title features, then using the td-idf algorithm (term frequency–inverse document frequency) to convert this combined text to vectors, then using cosine similarity to obtain a list of similarity scores (result) between the keywords vector and the the movies then sorting them in reverse order to get the top similar indexes.

    4. In the recommend_movies_by_genre(genre, sorting, top_n=50) method, I will filter all the movies in the dataframe given a
    certain criteria, the options of the user (genre), then sorted by rating or release_date. Obtaining again this time the 50 top movies indexes.

    5. In the display_movie_info(movie_ids) method, I will specify what I want to be shown for each movie. getting the movie
    title, poster path, overview and the release date from the API, and the other data which is not available from the API, actors cast and  director from the dataframe.

    6. the main function where I develop the streamlit app, as far as there are three recommendations systems, I chose a st.
    sidebar.selecbox(); then for each option is pretty much the same, just apply the method according the option selected, getting the movies id by the method then use the result to use the display_movie_info() to load the movies list.




