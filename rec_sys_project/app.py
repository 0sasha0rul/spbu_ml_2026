import streamlit as st
import pandas as pd
import numpy as np
import scipy.sparse as sparse
from spbu_ml_2026.rec_sys_project.utils import load_model, load_data, render_game_card

st.set_page_config(page_title="Game RecSys", layout="wide")

model = load_model()
# ДОБАВЛЕН als_data
games_info, item_features, test_data, fallback, als_data = load_data() 

if 'liked_games' not in st.session_state:
    st.session_state['liked_games'] = []

st.title("🎮 Игровая рекомендательная система")
tab1, tab2, tab3 = st.tabs(["🔍 Поиск и Магазин", "👤 Мой профиль (Live Recs)", "🧪 Тест существующих юзеров"])

with tab1:
    search_query = st.text_input("Найти игру по названию (например, 'Counter-Strike'):")
    if search_query:
        results = games_info[games_info['name'].str.contains(search_query, case=False, na=False)].head(8)
        cols = st.columns(4)
        for i, (_, row) in enumerate(results.iterrows()):
            with cols[i % 4]:
                render_game_card(row)
                if st.button("Добавить в профиль", key=f"add_{row['appid']}"):
                    if row['appid'] not in st.session_state['liked_games']:
                        st.session_state['liked_games'].append(row['appid'])
                        st.success("Добавлено!")


with tab2:
    st.subheader("Настройки профиля")
    country = st.selectbox("Ваша страна:", ["RU", "US", "CN", "DE", "PL", "GB", "UNKNOWN"])
    
    st.write(f"**Выбрано игр:** {len(st.session_state['liked_games'])}")
    if st.button("Очистить профиль"):
        st.session_state['liked_games'] = []
        st.rerun()

    if len(st.session_state['liked_games']) < 3:
        if country in fallback and country != 'UNKNOWN':
            st.warning(f"Выбрано мало игр (Холодный старт). Показываем Топ-10 популярных в регионе: {country}")
            fallback_list = fallback[country]
        else:
            st.warning("Выбрано мало игр (Холодный старт). Показываем Глобальный Топ-10:")
            fallback_list = fallback.get('global', fallback.get('general_fallback', []))
            
        fallback_df = pd.DataFrame({'appid': fallback_list})
        valid_fallback = fallback_df[fallback_df['appid'].isin(games_info['appid'])].head(10)
        recs = valid_fallback.merge(games_info, on='appid', how='left')
        
    else:
        st.success("ALS генерирует 300 кандидатов -> CatBoost выбирает Топ-10...")
        
        # 1. Готовим данные нового пользователя для ALS
        als_model = als_data['model']
        item2idx = als_data['item2idx']
        idx2item = als_data['idx2item']
        
        user_items_indices = [item2idx[appid] for appid in st.session_state['liked_games'] if appid in item2idx]
        
        # Если ни одной игры нет в истории обучения ALS (редкий кейс), берем фолбэк
        if not user_items_indices:
            st.error("Выбранные игры слишком новые или редкие. Выберите другие.")
            st.stop()
            
        # Создаем вектор юзера (1 строка, N колонок-игр)
        user_sparse = sparse.csr_matrix(
            ([1.0] * len(user_items_indices), ([0] * len(user_items_indices), user_items_indices)), 
            shape=(1, len(item2idx))
        )
        
        # 2. ALS "на лету" находит 300 кандидатов! (THE RIGHT WAY)
        ids, scores = als_model.recommend(
            userid=0, 
            user_items=user_sparse, 
            N=300, 
            filter_already_liked_items=True,
            recalculate_user=True # Ключевой параметр для новых юзеров
        )
        
        candidate_appids = [idx2item[idx] for idx in ids]
        
        # 3. Собираем фичи для CatBoost
        candidates = item_features[item_features['appid'].isin(candidate_appids)].copy()
        
        # Джоиним реальные скоры и ранги от ALS
        als_results = pd.DataFrame({'appid': candidate_appids, 'als_score': scores, 'als_rank': np.arange(1, len(ids) + 1)})
        candidates = candidates.merge(als_results, on='appid', how='inner')
        
        # Статические фичи юзера
        candidates['loccountrycode'] = country
        candidates['user_total_games_played'] = len(st.session_state['liked_games'])
        candidates['user_total_playtime'] = 1000 
        candidates['user_avg_playtime'] = 100
        candidates['account_age_days'] = 30
        
        # Динамические фичи пересечения (User Affinity)
        liked_df = item_features[item_features['appid'].isin(st.session_state['liked_games'])]
        mh_cols = [c for c in item_features.columns if c.startswith('genres_') or c.startswith('categories_')]
        user_affinity = liked_df[mh_cols].mean().to_dict()
        
        for k, v in user_affinity.items():
            candidates[f'user_affinity_{k}'] = v
            
        # 4. Финальное ранжирование CatBoost
        X_pred = candidates[model.feature_names_]
        candidates['cb_score'] = model.predict(X_pred)
        
        valid_candidates = candidates[candidates['appid'].isin(games_info['appid'])]
        top_appids = valid_candidates.sort_values('cb_score', ascending=False).head(10)[['appid']]
        recs = top_appids.merge(games_info, on='appid', how='left')

    cols = st.columns(5)
    for i, (_, row) in enumerate(recs.iterrows()):
        with cols[i % 5]:
            render_game_card(row)


with tab3:
    st.subheader("Проверка качества переранжирования на тестовой выборке")
    test_users = test_data['steamid'].unique()[:50]
    selected_user = st.selectbox("Выберите ID тестового пользователя:", test_users)
    
    user_data = test_data[test_data['steamid'] == selected_user].copy()
    
    X_pred = user_data[model.feature_names_]
    user_data['cb_score'] = model.predict(X_pred)
    
    st.info(f"Для пользователя {selected_user} первый этап (ALS) отобрал 300 кандидатов. Выведен Топ-10 после финального ранжирования CatBoost:")
    
    # ИСПРАВЛЕНИЕ ЗДЕСЬ ТОЖЕ
    valid_user_data = user_data[user_data['appid'].isin(games_info['appid'])]
    cb_top = valid_user_data.sort_values('cb_score', ascending=False).head(10)[['appid']]
    cb_games = cb_top.merge(games_info, on='appid', how='left')
    
    cols = st.columns(5)
    for i, (_, row) in enumerate(cb_games.iterrows()):
        with cols[i % 5]:
            render_game_card(row)