#!/usr/bin/env python
# coding: utf-8

# # Kave Ar-Ge Odaklı Yapay Zeka Eğitimi Sınav Sorusu 
# Bu çalışma kapsamında sizden öncelikle movielens verisetini kullanarak filmler arasındaki benzerlikleri bulmanız. Ardından bu benzerlikleri kullanarak kişilere film önerisi yapmanızdır. Sonrasında ise bu öneri yapan fonksiyonu Streamlit ile bir uygulama haline getirip kodlarını bizimle paylaşmanızı bekliyoruz.
# 
# # Önemli Not: Başvuru kabulü için size sorulan soruyu çözmenizden çok, o soruyu çözmek için ne kadar uğraştığınız önemlidir. Motivasyonu yüksek gençlerle çalışmak çok farklı, bunu biliyoruz, sizi önemsiyoruz ve bekliyoruz.
# 
# 
# 
# # Soru İçeriği
# 
# #### 1. MovieLens verisetini kullanarak film önerisi yapan bir algoritmanın yazılması
# #### 2. Kişiden film ismi alınınca ona benzer filmleri önerebilen fonksiyonun yazılması
# #### 3. Çözümün Streamlit ile bu kullanıcının kullanabileceği bir uygulama haline getirilmesi

# # 1. MovieLens verisetini kullanarak film önerisi yapan bir algoritmanın yazılması
# 
# Bu bölüm kapsamında sizden ekte sunduğumuz verisetinden filmlerin arasındaki benzerliği bulabileceğiniz ve bu benzerlikler üzerinden kullanıcılara film önerebileceğiniz bir algoritma geliştirmenizi bekliyoruz. 
# 
# Bu bölümde yardım alabileceğiniz kaynaklar
# - [How To Build Your First Recommender System Using Python & MovieLens Dataset](https://analyticsindiamag.com/how-to-build-your-first-recommender-system-using-python-movielens-dataset/)
# - [Build Recommender Systems with Movielens Dataset in Python](https://www.codespeedy.com/build-recommender-systems-with-movielens-dataset-in-python/)
# - [Collaborative Filtering for Movie Recommendations](https://www.kaggle.com/code/faressayah/collaborative-filtering-for-movie-recommendations)

# In[1]:


import pandas as pd
import numpy as np
import os


# In[2]:


os.listdir('datasets')


# In[3]:


movies = pd.read_csv('datasets/movies.csv')
movies.head(10)


# In[4]:


ratings = pd.read_csv('datasets/ratings.csv')
ratings.head(10)


# ### Verisetlerini bir araya getirelim. 

# In[5]:


# MovieID üzerinden kişilerin yorumlarına film isimlerini ve genrelerini ekliyoruz. 
df = pd.merge(ratings, movies, how='left', on='movieId')

df.head(10)


# # Feature Engineering

# <h4>Average Rating</h4>

# In[6]:


average_ratings = pd.DataFrame(df.groupby('title')['rating'].mean())

average_ratings.head(10)


# <h4>Total Number of Ratings</h4>

# In[7]:


average_ratings['total ratings'] = pd.DataFrame(df.groupby('title')['rating'].count())

average_ratings.head(10)


# <h4>Calculating the Correlation</h4>

# In[8]:


movie_user = df.pivot_table(index='userId', columns='title', values='rating')

movie_user.head(10)


# In[10]:


correlations_test = movie_user.corrwith(movie_user['Toy Story (1995)'])

correlations_test.head(10)


# In[11]:


recommendation_test = pd.DataFrame(correlations_test, columns=['correlation'])
recommendation_test.dropna(inplace=True)
recommendation_test = recommendation_test.join(average_ratings['total ratings'])

recommendation_test.head()


# <h4>Testing the Recommendation System</h4>

# In[12]:


recc_test = recommendation_test[recommendation_test['total ratings']>100].sort_values('correlation', ascending=False).reset_index()
recc_test = recc_test.merge(movies, on='title', how='left')
recc_test.head(10)


# ## 2. Kişiden film ismi alınınca ona benzer filmleri önerebilen fonksiyonun yazılması
# 
# Bundan sonrasında verisetini kullanıp çeşitli ön işlemelerden ve geliştirmelerden sonra alttaki gibi bir fonksiyon oluşturmanızı bekliyoruz. 

# In[13]:


def film_oner(movie_id):
    for i in range(len(df.title)):
        if (df.movieId[i] == movie_id):
            movie_name = df.title[i]
            break
    correlations = movie_user.corrwith(movie_user[movie_name])
    
    recommendation = pd.DataFrame(correlations, columns=['Correlation'])
    recommendation.dropna(inplace=True)
    recommendation = recommendation.join(average_ratings['total ratings'])
    
    recc = recommendation[recommendation['total ratings']>100].sort_values('Correlation', ascending=False).reset_index()
    recc = recc.merge(movies, on='title', how='left')
    
    recommended_movies = []
    for i in range(1, 6):
        recommended_movies.append(recc['title'][i])
    
    return recommended_movies


# In[15]:


my_movies = film_oner(1)
for i in range(len(my_movies)):
    print(my_movies[i])


# In[16]:


def isimle_film_oner(movie_name):
    correlations = movie_user.corrwith(movie_user[movie_name])
    
    recommendation = pd.DataFrame(correlations, columns=['Correlation'])
    recommendation.dropna(inplace=True)
    recommendation = recommendation.join(average_ratings['total ratings'])
    
    recc = recommendation[recommendation['total ratings']>100].sort_values('Correlation', ascending=False).reset_index()
    recc = recc.merge(movies, on='title', how='left')
    
    recommended_movies = []
    for i in range(1, 6):
        recommended_movies.append(recc['title'][i])
    
    return recommended_movies


# In[17]:


my_movie_name = "Twilight (2008)"
my_movies = isimle_film_oner(my_movie_name)
for i in range(len(my_movies)):
    print(my_movies[i])


# ## 3. Çözümün Streamlit ile bu kullanıcının kullanabileceği bir uygulama haline getirilmesi
# 
# Bu kısımda ise oluşturduğunuz fonksiyonu ektekine benzer bir arayüzde çalıştırmanızı bekliyoruz. 
# 
# ![alt text](streamlit-example.png "Örnek")
# 
# Yararlanabileceğiniz kaynaklar;
# - [How to Collect user inputs with Streamlit](https://www.youtube.com/watch?v=RHzjE-WBaSk)
# - [8 Best Streamlit Machine Learning Web App Examples in 2022](https://omdena.com/blog/streamlit-web-app-examples/)

# In[18]:


import streamlit as st


# In[ ]:




