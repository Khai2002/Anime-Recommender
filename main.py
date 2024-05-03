from collab_based_rec import *
from content_based_rec import *

fav_anime_list = [(21,10),(1,9)]
filter_name = 0

def filter_anime_name(fav_anime_list, recommended_animes) :
    fav_anime_ids = [anime_id for anime_id, _ in fav_anime_list]
    fav_anime_list_names  = anime_list[anime_list['anime_id'].isin(fav_anime_ids)]['Name']
    recommended_animes_names = recommended_animes['Name']
    vectorizer = TfidfVectorizer(stop_words='english').fit(anime_list['Name'])  # Entraînement sur tous les noms d'anime
    fav_vectors = vectorizer.transform(fav_anime_list_names.values)
    rec_vectors = vectorizer.transform(recommended_animes_names.values)
    similarity_matrix = linear_kernel(rec_vectors, fav_vectors)
    threshold = 0.2
    similar_indices = (similarity_matrix > threshold).any(axis=1)
    filtered_recommendation = recommended_animes[~similar_indices]
    filtered_recommendation = filtered_recommendation[['anime_id']]
    return filtered_recommendation.head(30)['anime_id'].tolist()


def show_names(anime_ids, anime_list):
    animes = []
    for anime_id in anime_ids:
        anime_name = anime_list.loc[anime_list['anime_id'] == anime_id, 'Name'].iloc[0]
        animes.append({'anime_id': anime_id, 'Name': anime_name})

    print(pd.DataFrame(animes))

def merge_score(collab_tab, content_tab):
    final_tab = pd.merge(content_tab, collab_tab, on='anime_id', suffixes=('_content', '_collab'), how='left')
    final_tab['recommend_score_collab'] = final_tab['recommend_score_collab'].fillna(0)
    final_tab['recommend_score_content'] = (final_tab['recommend_score_content'] - final_tab['recommend_score_content'].min()) / (final_tab['recommend_score_content'].max() - final_tab['recommend_score_content'].min())
    final_tab['recommend_score_collab'] = (final_tab['recommend_score_collab'] - final_tab['recommend_score_collab'].min()) / (final_tab['recommend_score_collab'].max() - final_tab['recommend_score_collab'].min())
    final_tab['total_score'] = final_tab['recommend_score_content'] + final_tab['recommend_score_collab']
    return final_tab

def recommendation_anime(fav_anime_list, filter_name=0):
    collab_tab = get_recommandation_collab_tab(fav_anime_list)
    content_tab = get_recommandation_content_tab(fav_anime_list)

    similarities_tab = merge_score(collab_tab, content_tab)

    sorted_df = similarities_tab.sort_values(by='total_score', ascending=False)
    top_anime_ids = sorted_df.head(200)['anime_id'].tolist()

    if filter_name == 1:
        recommended_animes = []
        for anime_id in top_anime_ids:
            anime_name = anime_list.loc[anime_list['anime_id'] == anime_id, 'Name'].iloc[0]
            recommended_animes.append({'anime_id': anime_id, 'Name': anime_name})

        anime_names = pd.DataFrame(recommended_animes)
        top_anime_ids = filter_anime_name(fav_anime_list, anime_names)
        
    return top_anime_ids[:30]

def recommandation_anime_content_based(fav_anime_list, filter_name=0) :
    content_tab = get_recommandation_content_tab(fav_anime_list)
    #print(content_tab)
    sorted_df = content_tab.sort_values(by='recommend_score', ascending=False)
    top_anime_ids = sorted_df.head(200)['anime_id'].tolist()
    if filter_name == 1:
        recommended_animes = []
        for anime_id in top_anime_ids:
            anime_name = anime_list.loc[anime_list['anime_id'] == anime_id, 'Name'].iloc[0]
            recommended_animes.append({'anime_id': anime_id, 'Name': anime_name})

        anime_names = pd.DataFrame(recommended_animes)
        top_anime_ids = filter_anime_name(fav_anime_list, anime_names)
    
    return top_anime_ids[:30]








if __name__ == "__main__":
    anime_list = pd.read_parquet('anime/anime.parquet')
    #recommandation_anime_content_based(fav_anime_list)

    #show_names(recommendation_anime(fav_anime_list, 1), anime_list)

    show_names(recommandation_anime_content_based(fav_anime_list, 1), anime_list)



