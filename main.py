import warnings
warnings.filterwarnings('ignore')

import os
import re
import nltk
import string
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
 
Reviewdata=pd.read_csv('train.csv')

count=Reviewdata.isnull().sum().sort_values(ascending=False)
percentage=((Reviewdata.isnull().sum()/len(Reviewdata)*100)).sort_values(ascending=False)
missing_data=pd.concat([count,percentage],axis=1,
                       keys=['Count','Percentage'])

Reviewdata.drop(columns=['User_ID','Browser_Used','Device_Used'],inplace=True)

def text_clean_1(text):
    text = text.lower()
    text = re.sub('\[.*?\]', '', text)
    text = re.sub('[%s]' % re.escape(string.punctuation), '', text)
    text = re.sub('\w*\d\w*', '', text)
    return text

cleaned1 = lambda x: text_clean_1(x)
Reviewdata['cleaned_description'] = pd.DataFrame(Reviewdata.Description.apply(cleaned1))

Independent_var = Reviewdata.cleaned_description
Dependent_var = Reviewdata.Is_Response

IV_train, IV_test, DV_train, DV_test = train_test_split(Independent_var, Dependent_var, test_size = 0.1, random_state = 225)

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

tvec = TfidfVectorizer()
clf2 = LogisticRegression(solver = "lbfgs")


model = Pipeline([('vectorizer',tvec),('classifier',clf2)])

model.fit(IV_train, DV_train)


predictions = model.predict(IV_test)

confusion_matrix(predictions, DV_test)#array was printed

example = ["I'm satisfied"]
result = model.predict(example)

import pickle
with open("model.pkl","wb") as f:
    pickle.dump(model,f)