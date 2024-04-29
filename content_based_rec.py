import numpy as np
import pandas as pd
import math
import string
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from nltk import pos_tag
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

'''
nltk.download('stopwords')
nltk.download('wordnet')
'''

def show(anime_list):
    nbl, nbc = anime_list.shape
    print("\nNombre de lignes :", nbl)
    print("\nNombre de colonnes :", nbc)
    print("\nInfos\n")
    print(anime_list.info())
    print("\nDescribe\n")
    print(anime_list.describe())
    print("\nHead\n")
    print(anime_list.head(40))

def extract_keywords(anime_ids, anime_list):
    # récupérer tous les synopsis des animes favoris
    fav_anime_synopsis = anime_list.loc[anime_list['anime_id'].isin(anime_ids), 'Synopsis'].tolist()
    # concaténer l'ensemble de ces synopsis
    fav_anime_synopsis = ' '.join(fav_anime_synopsis)
    # récupérer les mots clés
    fav_anime_keywords = fav_anime_synopsis.split()
    fav_anime_keywords = [word.translate(str.maketrans('', '', string.punctuation)).lower() for word in fav_anime_keywords]
    stop_words = set(stopwords.words('english'))
    fav_anime_keywords = [word for word in fav_anime_keywords if word not in stop_words]
    lemmatizer = WordNetLemmatizer()
    fav_anime_keywords = [lemmatizer.lemmatize(word) for word in fav_anime_keywords]
    ## add here more advanced analysis ?
    return ' '.join(fav_anime_keywords)

def preprocess(anime_list): #TODO : Gérer les cas limites 
    anime_list = anime_list.copy()

    ## Dropping columns
    columns_to_keep = ['anime_id', 'Name', 'Genres', 'Synopsis', 'Episodes', 'Aired', 'Studios', 'Duration', 'Rating']
    anime_list = anime_list[columns_to_keep]

    ## Dealing with Genres : use one-hot encoding
    all_genres = set()
    for genres in anime_list['Genres']:
        all_genres.update(genres.split(', '))
    for genre in all_genres:
        anime_list[genre] = anime_list['Genres'].apply(lambda x: 1 if genre in x.split(', ') else 0)
    anime_list.drop(columns=['Genres'], inplace=True)

    ## Dealing with Episodes and Duration : calculate total length
    anime_list['Episodes'] = pd.to_numeric(anime_list['Episodes'], errors='coerce').fillna(0) #case to deal with
    hours = anime_list['Duration'].str.extract(r'(\d+) hr', expand=False).astype(float)
    minutes = anime_list['Duration'].str.extract(r'(\d+) min', expand=False).astype(float)
    hours.fillna(0, inplace=True)
    minutes.fillna(0, inplace=True)
    anime_list['Duration'] = hours * 60 + minutes
    anime_list['Total_Duration'] = anime_list['Duration'] * anime_list['Episodes']
    anime_list.drop(columns=['Episodes'], inplace=True)
    anime_list.drop(columns=['Duration'], inplace=True)

    ## Dealing with Aired => get starting date
    anime_list['Start_Date'] = pd.to_datetime(anime_list['Aired'].str.split(' to ').str[0], errors='coerce')
    anime_list.drop(columns=['Aired'], inplace=True)

    ## Dealing with Studios => keep the first (for now)
    anime_list['Studios'] = anime_list['Studios'].str.split(',', expand=True)[0]
    anime_list['Studios'] = anime_list['Studios'].str.strip()

    ## Dealing with Rating => simplify
    anime_list.loc[anime_list['Rating'].str.contains('G'), 'Rating'] = 'All Ages'
    anime_list.loc[anime_list['Rating'].str.contains('PG-13'), 'Rating'] = 'Teen'
    anime_list.loc[anime_list['Rating'].str.contains('PG'), 'Rating'] = 'Children'
    anime_list.loc[anime_list['Rating'].str.contains('Rx'), 'Rating'] = 'Hentai'
    anime_list.loc[anime_list['Rating'].str.contains(r'R\+'), 'Rating'] = 'Adult'
    anime_list.loc[anime_list['Rating'].str.contains('R'), 'Rating'] = 'Young Adult'
    anime_list.loc[~anime_list['Rating'].isin(['All Ages', 'Children', 'Teen', 'Young Adult', 'Hentai', 'Adult']), 'Rating'] = 'Other'

    ## Dealing with synopsis
    anime_list['Synopsis'] = anime_list['Synopsis'].apply(lambda x: x.translate(str.maketrans('', '', string.punctuation)).lower())
    stop_words = set(stopwords.words('english'))
    anime_list['Synopsis'] = anime_list['Synopsis'].apply(lambda x  : ' '.join([word for word in x.split() if word not in stop_words]))
    lemmatizer = WordNetLemmatizer()
    anime_list['Synopsis'] = anime_list['Synopsis'].apply(lambda x: ' '.join([lemmatizer.lemmatize(word) for word in x.split()]))
    #anime_list['Synopsis'] = anime_list['Synopsis'].apply(lambda x: ' '.join([word for word, pos in pos_tag(word_tokenize(x)) if pos.startswith(('JJ', 'NN', 'VB', 'RB'))]))

    return anime_list

def recommendation_synopsis_based(fav_anime_list, anime_list):
    # Extraction de mots-clés des synopsis des animes favoris
    fav_anime_keywords = extract_keywords(fav_anime_list, anime_list)

    anime_ids = anime_list.loc[~anime_list['anime_id'].isin(fav_anime_list), 'anime_id'].values

    # Calcul de la similarité cosinus entre les mots-clés générés des animes favoris et les synopsis de tous les autres animes
    tfidf_vectorizer = TfidfVectorizer()
    tfidf_matrix_other_anime = tfidf_vectorizer.fit_transform(anime_list.loc[~anime_list['anime_id'].isin(fav_anime_list), 'Synopsis'])
    tfidf_matrix_fav_anime = tfidf_vectorizer.transform([fav_anime_keywords]) 
    cosine_similarities = cosine_similarity(tfidf_matrix_other_anime, tfidf_matrix_fav_anime)

    return pd.DataFrame({'anime_id': anime_ids, 'similarity': cosine_similarities.flatten()})

def recommendation_genre_based(fav_anime_list, anime_list):
    anime_ids = anime_list.loc[~anime_list['anime_id'].isin(fav_anime_list), 'anime_id'].values
    similarities = []

    fav_genres = anime_list.loc[anime_list['anime_id'].isin(fav_anime_list), ~anime_list.columns.isin(['anime_id', 'Name', 'Total_Duration', 'Start_Date', 'Studios', 'Rating', 'Synopsis'])].sum()
    fav_genres_prop = fav_genres / fav_genres.sum()

    other_anime_genres = anime_list.loc[~anime_list['anime_id'].isin(fav_anime_list), ~anime_list.columns.isin(['Name', 'Total_Duration', 'Start_Date', 'Studios', 'Rating', 'Synopsis'])]
    for _, row in other_anime_genres.drop(columns=['anime_id']).iterrows():
        genre_similarity = sum(row[genre] * fav_genres_prop[genre] for genre in fav_genres_prop.index)
        similarities.append(genre_similarity)
       
    return pd.DataFrame({'anime_id': anime_ids, 'similarity': similarities})

def recommendation_duration_based(fav_anime_list, anime_list):
    anime_ids = anime_list.loc[~anime_list['anime_id'].isin(fav_anime_list), 'anime_id'].values
    similarities = []

    avg_fav_duration = anime_list.loc[anime_list['anime_id'].isin(fav_anime_list), 'Total_Duration'].mean()
    print(avg_fav_duration)

    other_anime_durations = anime_list.loc[~anime_list['anime_id'].isin(fav_anime_list), 'Total_Duration']

    for duration in other_anime_durations:
        relative_difference = abs(duration - avg_fav_duration) / max(duration, avg_fav_duration)
        duration_similarity = 1 - relative_difference
        similarities.append(duration_similarity)

    return pd.DataFrame({'anime_id': anime_ids, 'similarity': similarities})

def recommendation_studios_based(fav_anime_list, anime_list):
    anime_ids = anime_list.loc[~anime_list['anime_id'].isin(fav_anime_list), 'anime_id'].values
    fav_studio_counts = anime_list.loc[anime_list['anime_id'].isin(fav_anime_list), 'Studios'].value_counts()
    fav_studio_prop = fav_studio_counts / fav_studio_counts.sum()
    print(fav_studio_prop)

    return pd.DataFrame({'anime_id': anime_ids, 'similarity': anime_list})

def recommend_anime(similarities_tab):
    sorted_df = similarities_tab.sort_values(by='total_similarity', ascending=False)
    #sorted_df = similarities_tab.sort_values(by='similarity', ascending=False)
    top_anime_ids = sorted_df.head(30)['anime_id'].tolist()
    recommended_animes = []
    for anime_id in top_anime_ids:
        anime_name = anime_list.loc[anime_list['anime_id'] == anime_id, 'Name'].iloc[0]
        recommended_animes.append({'anime_id': anime_id, 'Name': anime_name})
    return pd.DataFrame(recommended_animes)


if __name__ == "__main__":
    
    fav_anime_list = [21]
    anime_list = pd.read_parquet('anime/anime.parquet')
    anime_list = preprocess(anime_list)
    show(anime_list)
    
    genre_cosine_similarities_tab = recommendation_genre_based(fav_anime_list, anime_list)
    duration_cosine_similarities_tab = recommendation_duration_based(fav_anime_list, anime_list)
    synopsis_cosine_similarities_tab = recommendation_synopsis_based(fav_anime_list, anime_list)
    #studios_cosine_similarities_tab = recommendation_studios_based(fav_anime_list, anime_list)

    print(genre_cosine_similarities_tab)
    print(duration_cosine_similarities_tab)
    print(synopsis_cosine_similarities_tab)

    combined_tab = pd.merge(genre_cosine_similarities_tab, duration_cosine_similarities_tab, on='anime_id', suffixes=('_genre', '_duration'))
    combined_tab = pd.merge(combined_tab, synopsis_cosine_similarities_tab, on='anime_id', suffixes=('', '_synopsis'))
    combined_tab['total_similarity'] = 0.2*combined_tab['similarity_genre'] + 0.2*combined_tab['similarity_duration'] + 0.6*combined_tab['similarity']
    recommended_animes = recommend_anime(combined_tab)
    
    #recommended_animes = recommend_anime(synopsis_cosine_similarities_tab)

    print(recommended_animes[['anime_id', 'Name']])