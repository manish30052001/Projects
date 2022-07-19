# -*- coding: utf-8 -*-
"""
Created on Thu Jul 14 23:33:38 2022

@author: HP
"""
import streamlit as st
import pickle
import pandas as pd
import requests

def fetch_poster(movie_id):
    response = requests.get('https://api.themoviedb.org/3/movie/{}?api_key=7e0431bf5eacaaa883e23b888bc2d464&language=en-US'.format(movie_id))
    data = response.json()
    return ('https://image.tmdb.org/t/p/w500/' + data['poster_path'])

new_df = pd.read_csv('New_df_movies.csv')
movies_list = new_df['title'].values
similarity = pickle.load(open('similarity.pkl', 'rb'))
st.title("Movie Recommender System")

selected_movie_name = st.selectbox(
     'How would you like to be contacted?',
     movies_list)

st.write('You selected:', selected_movie_name)

def recommend(movie):
    movie_index = new_df[new_df['title'] == movie].index[0]
    distances = similarity[movie_index]
    movies_list = sorted(list(enumerate(distances)), reverse=True, key = lambda x:x[1])[1:16]
    
    recommended_movies = []
    recommended_movies_posters = []
    for i in movies_list:
        movie_id = new_df.iloc[i[0]].movie_id
        recommended_movies.append(new_df.iloc[i[0]].title)
        recommended_movies_posters.append(fetch_poster(movie_id))
    return recommended_movies, recommended_movies_posters

if st.button('Recommend'):
    names, poster = recommend(selected_movie_name)
    i = 0
    for j in range(0,3):
        col_list = st.columns(5)
        for col in col_list:
            with col:
                st.markdown(names[i])
                st.image(poster[i])
                i += 1
    
    