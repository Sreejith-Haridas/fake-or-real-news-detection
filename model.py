#!/usr/bin/env python
# coding: utf-8

# In[11]:


import pandas as pd
import numpy as np

fake = pd.read_csv("C:\\Users\\sridhish\\Downloads\\archive\\Fake.csv")
real = pd.read_csv("C:\\Users\\sridhish\\Downloads\\archive\\True.csv")


# In[12]:


fake.head()


# In[15]:


real.head()


# In[17]:


import pandas as pd

fake = pd.read_csv("C:\\Users\\sridhish\\Downloads\\archive\\Fake.csv")
real = pd.read_csv("C:\\Users\\sridhish\\Downloads\\archive\\True.csv")
fake["Label"] = 0
real["Label"] = 1
df = pd.concat([fake,real])
df = df.sample(frac=1).reset_index(drop = True)    # shuffle the data


# In[19]:


import nltk
import re
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

nltk.download("stopwords")
nltk.download("wordnet")
stop_words = set(stopwords.words("english"))
lemmatizer = WordNetLemmatizer()
def clean_text(text):
    text=re.sub(r'[^a-zA-Z]',' ',text.lower())
    words = text.split()
    words = [lemmatizer.lemmatize(word) for word in words if word not in stop_words]
    return ' '.join(words)
df['cleaned_text'] = df['text'].apply(clean_text)


# In[20]:


from sklearn.feature_extraction.text import TfidfVectorizer

tfidf = TfidfVectorizer(max_features=5000)
X = tfidf.fit_transform(df['cleaned_text']).toarray().astype(np.float32)
y = df['Label']


# In[21]:


from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

X_train,X_test,y_train,y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = LogisticRegression()
model.fit(X_train,y_train)
y_pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test,y_pred))
print(classification_report(y_test,y_pred))


# In[22]:


import joblib

joblib.dump(tfidf, 'vectorizer.pkl')
joblib.dump(model, 'model.pkl')


# In[ ]:


pip install streamlit


# In[ ]:




