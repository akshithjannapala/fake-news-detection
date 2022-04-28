#!/usr/bin/env python
# coding: utf-8

# In[1]:


pip install nltK


# In[70]:


nltk.download('punkt')


# In[1]:


import nltk 


# In[2]:


import pandas as pd
fake=pd.read_csv("True-211023-185340.csv")
true=pd.read_csv("Fake-211023-185413.csv")


# In[3]:


display(true.info())


# In[4]:


display(fake.info())


# In[75]:


display(true.head(10))


# In[5]:


display(fake.head(10))


# In[6]:


print(fake.subject.value_counts())


# In[7]:


print(true.subject.value_counts())


# In[8]:


fake['target']=0
true['target']=1


# select1=select.reset_index(drop=1) 

# In[80]:


data=pd.concat([fake,true],axis=0)


# In[81]:


data1=data.reset_index(drop=True)


# In[82]:


data=data.drop(['subject','date','title'],axis=1)


# In[83]:


print(data.columns)


# # tokenization

# In[84]:


from nltk.tokenize import word_tokenize


# In[85]:


data['text']=data['text'].apply(word_tokenize)


# In[86]:


print(data.head(10))


# # stemming

# In[88]:


from nltk.stem.snowball import SnowballStemmer
porter=SnowballStemmer("english")


# In[89]:


def stem_it(text):
    return [porter.stem(word) for word in text ]


# In[91]:


data['text']=data['text'].apply(stem_it)


# In[93]:


print(data.head(10))


# # stopword removal

# In[106]:


import nltk
from nltk.corpus import stopwords
nltk.download('stopwords')
print(stopwords.words('english'))


# In[108]:


data['text']=data['text'].apply(''.join)


# # splitting up of data
# 

# In[110]:


from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(data['text'],data['target'],test_size=0.25)
display(x_train.head())
print('\n')
display(y_train.head())


# # vectorization
# 

# In[121]:


from sklearn.feature_extraction.text import TfidfVectorizer 
my_tfidf= TfidfVectorizer(max_df=0.7)

tfidf_train=my_tfidf.fit_transform(x_train)
tfidf_test=my_tfidf.transform(x_test)


# In[122]:


print(tfidf_train)


# # logistic regression

# In[123]:


from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score


# In[124]:


model_1=LogisticRegression(max_iter=900)
model_1.fit(tfidf_train,y_train)
pred_1=model_1.predict(tfidf_test)
cr1=accuracy_score(y_test,pred_1)
print(cr1*100)


# # passive aggressive classifier

# In[126]:


from sklearn.linear_model import PassiveAggressiveClassifier

model=PassiveAggressiveClassifier(max_iter=50)
model.fit(tfidf_train,y_train)


# In[127]:


y_pred=model.predict(tfidf_test)
accscore=accuracy_score(y_test,y_pred)
print('the accuracy of prediction is ',accscore*100)


# In[ ]:





# In[ ]:





# In[ ]:





# In[2]:


import cv2

cap = cv2.VideoCapture(0)

while True:
    success, img = cap.read()
    cv2.imshow('img',img)

    
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


# In[3]:


get_ipython().system('pip install opencv-python')

