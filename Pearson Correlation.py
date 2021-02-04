#!/usr/bin/env python
# coding: utf-8

# In[1]:


from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn import svm
import pandas as pd


# In[2]:


userInput1 = "The Big Sick (2017)"


# In[3]:


titles = pd.read_csv("movies.csv")
data = pd.read_csv("ratings.csv")


# In[4]:


data = pd.merge(titles, data, on="movieId")
reviews = data.groupby("title")["rating"].agg(["count"]).reset_index().round(1)
movies = pd.crosstab(data["userId"], data["title"], values=data["rating"], aggfunc="sum")


# In[5]:


similarity = movies.corrwith(movies[userInput1], method="pearson")


# In[6]:


correlatedMovies = pd.DataFrame(similarity, columns=["correlation"])
correlatedMovies = pd.merge(correlatedMovies, reviews, on="title")
correlatedMovies = pd.merge(correlatedMovies, titles, on="title")


# In[7]:


output = correlatedMovies[ (correlatedMovies["count"] >= 100)].sort_values("correlation",ascending=False)


# In[8]:


output = output[((output.title != userInput1))]


# In[9]:


print(output)


# In[10]:


output=output.rename(columns={"title": ("Movies Suggestions based on " + userInput1 ), "genres": "Genres", "correlation": "Correlation"}).head(25)


# In[11]:


output.head(10)


# In[12]:


reviews.head()


# In[13]:


movies.head()


# In[14]:


data.head()


# In[15]:


data.shape


# In[16]:


new_rat=data.groupby(['userId','rating'])


# In[17]:


new_rat.first()


# In[18]:


movies.head()


# In[19]:


a=titles.groupby('genres')


# In[20]:


print(a.get_group('Comedy|Romance'))


# In[ ]:




