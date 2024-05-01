from collab_based_rec import *
from content_based_rec import *

fav_anime_list = [21]

def show_names(anime_ids, anime_list):
    animes = []
    for anime_id in anime_ids:
        anime_name = anime_list.loc[anime_list['anime_id'] == anime_id, 'Name'].iloc[0]
        animes.append({'anime_id': anime_id, 'Name': anime_name})

    print(pd.DataFrame(animes))

def merge_score(collab_tab, content_tab):
    final_tab = pd.merge(content_tab, collab_tab, on='anime_id', suffixes=('_content', '_collab'), how='left')
    final_tab['recommend_score_collab'] = final_tab['recommend_score_collab'].fillna(0)
    final_tab['total_score'] = final_tab['recommend_score_content'] + 100*final_tab['recommend_score_collab']
    return final_tab

def recommendation_anime(fav_anime_list):
    collab_tab = get_recommandation_collab_tab(fav_anime_list)
    content_tab = get_recommandation_content_tab(fav_anime_list)
    similarities_tab = merge_score(collab_tab, content_tab)

    sorted_df = similarities_tab.sort_values(by='total_score', ascending=False)
    top_anime_ids = sorted_df.head(30)['anime_id'].tolist()
    return top_anime_ids

if __name__ == "__main__":
    anime_list = pd.read_parquet('anime/anime.parquet')
    show_names(recommendation_anime(fav_anime_list), anime_list)


