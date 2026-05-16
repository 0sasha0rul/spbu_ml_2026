import streamlit as st
import pandas as pd
import pickle
from catboost import CatBoostRanker

@st.cache_resource
def load_model():
    model = CatBoostRanker()
    model.load_model('artifacts/catboost_ranker_tuned.cbm')
    return model

@st.cache_data
def load_data():
    games_info = pd.read_csv('data/game_details.csv')
    item_features = pd.read_parquet('artifacts/item_features_multihot.parquet')
    test_data = pd.read_parquet('artifacts/test_final.parquet')
    
    with open('artifacts/fallback_recommendations.pkl', 'rb') as f:
        fallback = pickle.load(f)
        
    with open('artifacts/als_model_data.pkl', 'rb') as f:  # ВАЖНО: имя файла здесь!
        als_data = pickle.load(f)
        
    return games_info, item_features, test_data, fallback, als_data
        
def render_game_card(game_row):
    price_tag = "Free" if game_row['is_free'] else "Paid"
    html = f"""
    <div style="background-color: #2a475e; border-radius: 10px; padding: 10px; margin-bottom: 15px; box-shadow: 2px 2px 5px rgba(0,0,0,0.5);">
        <img src="{game_row['header_image']}" style="width: 100%; border-radius: 5px;">
        <h4 style="color: #ffffff; margin-top: 10px; margin-bottom: 5px; font-size: 16px; white-space: nowrap; overflow: hidden; text-overflow: ellipsis;">{game_row['name']}</h4>
        <p style="color: #66c0f4; font-size: 12px; margin: 0;">{price_tag}</p>
    </div>
    """
    st.markdown(html, unsafe_allow_html=True)