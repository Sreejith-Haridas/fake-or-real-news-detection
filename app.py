#!/usr/bin/env python
# coding: utf-8

# In[9]:


import pandas as pd

fake = pd.read_csv("C:\\Users\\sridhish\\Downloads\\archive\\Fake.csv")
real = pd.read_csv("C:\\Users\\sridhish\\Downloads\\archive\\True.csv")


# In[11]:


fake.head()


# In[13]:


real.head()


# In[15]:


import pandas as pd

fake = pd.read_csv("C:\\Users\\sridhish\\Downloads\\archive\\Fake.csv")
real = pd.read_csv("C:\\Users\\sridhish\\Downloads\\archive\\True.csv")
fake["Label"] = 0
real["Label"] = 1
df = pd.concat([fake,real])
df = df.sample(frac=1).reset_index(drop = True)    # shuffle the data


# In[17]:


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


# In[18]:


from sklearn.feature_extraction.text import TfidfVectorizer

tfidf = TfidfVectorizer(max_features=5000)
X = tfidf.fit_transform(df['cleaned_text']).toarray()
y = df['Label']


# In[19]:


from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

X_train,X_test,y_train,y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = LogisticRegression()
model.fit(X_train,y_train)
y_pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test,y_pred))
print(classification_report(y_test,y_pred))


# In[20]:


import joblib

joblib.dump(tfidf, 'vectorizer.pkl')
joblib.dump(model, 'model.pkl')


# In[23]:


import streamlit as st
import joblib

model = joblib.load("model.pkl")
vectorizer = joblib.load("vectorizer.pkl")
st.set_page_config(page_title = "Fake News Detector", page_icon = "📰")

st.title("📰 Fake News Detection App")
st.write("Paste any news article content below to check if it's **Fake** or **Real**.")

news_input = st.text_area("Paste news content here: ", height = 300)

if st.button("Check"):
    if news_input.strip() == "":
        st.warning("⚠️ Please enter some news content.")
    else:
        vectorized_text = vectorizer.transform([news_input])
        prediction = model.predict(vectorized_text)[0]

        if prediction == 1:
            st.success("✅ This appears to be **REAL** news.")
        else:
            st.error("🚨 This appears to be **FAKE** news.")


# In[ ]:




