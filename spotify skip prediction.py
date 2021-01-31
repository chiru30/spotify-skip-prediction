#!/usr/bin/env python
# coding: utf-8

# ## SPOTIFY SKIP PREDICTION
# ### NAME : CHIRANTHANA R R
# 

# ### IMPORTING LIBRARIES

# In[119]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')


# ### LOADING THE DATASET

# In[120]:


tracks=pd.read_csv("track_feats.csv")


# ### DATA INSIGHTS

# In[121]:


tracks.shape


# In[122]:


tracks.info()


# In[123]:


tracks.columns


# In[124]:


tracks.head()


# In[125]:


tracks.describe()


# In[126]:


plt.figure(figsize=(10,9))
plt.title("Song Trends Over Time")

lines = ["acousticness","danceability","energy", 
         "instrumentalness", "liveness", "valence", "speechiness"]

for line in lines:
    ax = sns.lineplot(x='release_year', y=line, data=tracks)
    
    
plt.ylabel("value")
plt.legend(lines)


# In[127]:


plt.figure(figsize=(20,20))
sns.heatmap(tracks.corr(), fmt= '.2g',annot=True, cmap="Oranges")


# In[128]:


train=pd.read_csv("train_data_20.csv")


# ### DATA INSIGHTS

# In[129]:


train.shape


# In[130]:


train.info()


# In[131]:


train.columns


# In[132]:


train.describe()


# In[133]:


train.head()


# ###  VISUALIZATION

# In[134]:


print(train.session_length.value_counts(normalize=True, sort=False))
train.session_length.hist()


# In[135]:


print(train.skip_2.value_counts(normalize=True, sort=False))
train.skip_2.astype(np.int).hist()


# In[136]:


train.context_switch.hist()


# In[137]:


print(train.premium.value_counts())
train.premium.astype(np.int).hist()


# In[138]:


train.hour_of_day.hist()


# In[139]:


tracks.duration.hist();
print("Total durations are", len(set(tracks.duration)))
print(np.mean(tracks.duration.values))


# In[140]:


plt.figure(figsize=(8,5))
sns.countplot(x=tracks['release_year'])


# In[141]:


tracks.us_popularity_estimate.hist();


# In[142]:


tracks.beat_strength.hist();


# In[143]:


tracks.acoustic_vector_0.hist();


# In[144]:


tracks.time_signature.hist()


# In[145]:


plt.figure(figsize=(20,15))
sns.heatmap(tracks.corr(),annot=True);


# In[148]:


skipped=[]
for i in train["not_skipped"]:
    if (i==0):
        skipped.append(1)
    else:
        skipped.append(0)


# In[147]:


#INPUT FEATURES
X=train.drop(["session_id", "track_id_clean","skipped"], axis=1)

#TARGET FEATURE
y=train["skipped"]


# ## model

# In[28]:


skipped=[]
for i in train["not_skipped"]:
    if (i==0):
        skipped.append(1)
    else:
        skipped.append(0)


# In[29]:


train["skipped"]= pd.DataFrame(data= skipped)


# In[30]:


train=train.drop("not_skipped", axis=1)


# In[31]:


train.head(3)


# In[32]:


train.shape


# In[33]:


X=train.drop(["session_id", "track_id_clean","skipped"], axis=1)

y=train["skipped"]


# In[34]:


from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)


# In[35]:


from sklearn.tree import DecisionTreeClassifier
classifier = DecisionTreeClassifier(random_state=0)


# In[36]:


classifier.fit(x_train, y_train)


# In[37]:


ypred=classifier.predict(x_test)


# In[38]:


classifier.score(x_test, y_test)


# In[39]:


classifier.score(x_test, y_test)


# In[40]:


val=pd.read_csv("val_data_20.csv")


# ## insights

# In[41]:


val.info()


# In[42]:


val.shape


# In[43]:


val.columns


# In[44]:


val.head(10)


# In[45]:


skipped=[]
for i in val["not_skipped"]:
    if (i==0):
        skipped.append(1)
    else:
        skipped.append(0)


# In[46]:


val["skipped"]= pd.DataFrame(data= skipped)


# In[47]:


val=val.drop("not_skipped", axis=1)


# In[48]:


xv=val.drop(["session_id", "track_id_clean","skipped"], axis=1)
yv=val["skipped"]


# In[49]:


from sklearn.model_selection import train_test_split
x_train_val, x_test_val, y_train_val, y_test_val = train_test_split(xv, yv, test_size=0.2, random_state=0)


# In[50]:


classifier.fit(x_train_val, y_train_val)


# In[51]:


ypred2=classifier.predict(x_test_val)


# In[52]:


classifier.score(x_test_val, y_test_val)


# In[53]:


from sklearn.metrics import mean_absolute_error
mean_absolute_error(y_test_val, ypred2)


# In[54]:


test=pd.read_csv("test_data_20.csv")


# ### data insights

# In[55]:


test.info()


# In[56]:


test.shape


# In[57]:


test.columns


# In[58]:


skipped=[]
for i in test["not_skipped"]:
    if (i==0):
        skipped.append(1)
    else:
        skipped.append(0)


# In[59]:


test["skipped"]= pd.DataFrame(data= skipped)


# In[60]:


test=test.drop("not_skipped", axis=1)


# In[61]:


x=test.drop(["session_id", "track_id_clean","skipped"], axis=1)
Y=test["skipped"]


# In[62]:


ypred1=classifier.predict(x)


# In[63]:


classifier.score(x, Y)


# In[64]:


from sklearn.metrics import confusion_matrix
confusion_matrix(Y, ypred1)


# In[65]:


from sklearn.metrics import mean_absolute_error
mean_absolute_error(Y, ypred1)


# ### renaming 

# In[66]:


train=train.rename(columns={"track_id_clean": "track_id"})


# In[67]:


train.shape


# In[68]:


train.columns


# In[69]:


train.head(3)


# In[70]:


train_data= train.copy()


# In[71]:


track_feats= tracks.copy()


# In[72]:


merged= pd.merge(track_feats, train_data, on=["track_id"])
merged.shape


# In[73]:


merged.to_csv("newdata.csv")


# In[74]:


X=merged.drop(["session_id","track_id","skipped"], axis=1)
y=merged["skipped"]


# In[75]:


from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)


# In[76]:


from sklearn.tree import DecisionTreeClassifier
classifier = DecisionTreeClassifier(random_state=0)


# In[77]:


classifier.fit(x_train, y_train)


# In[78]:


ypred_merged= classifier.predict(x_test)


# In[79]:


classifier.score(x_test, y_test)


# In[80]:


from sklearn.metrics import mean_absolute_error
mean_absolute_error(y_test, ypred_merged)


# In[81]:


xm= merged.drop(["session_id","track_id", "skipped"], axis= 1)
ym= merged["skipped"]


# In[82]:


from sklearn.decomposition import PCA
pca_merged= PCA(n_components=3)
principalcomponents_merged= pca_merged.fit_transform(xm)


# In[83]:


prinicpal_merged= pd.DataFrame(data=principalcomponents_merged, columns= ["PC_1", "PC_2", "PC_3"] )


# In[87]:


prinicpal_merged.head(10)


# In[88]:


from sklearn.model_selection import train_test_split
x_train_m, x_test_m, y_train_m, y_test_m = train_test_split(prinicpal_merged, ym, test_size=0.2, random_state=0)


# In[89]:


classifier.fit(x_train_m, y_train_m)


# In[90]:


ypred3=classifier.predict(x_test_m)


# In[91]:


classifier.score(x_test_m, y_test_m)

