import numpy as np 
import pandas as pd
import nltk 
from nltk import word_tokenize
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
nltk.download('punkt')
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

from nltk import word_tokenizer
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
data =pd.read_csv("single_qna.csv")
print(data.head()) 
data.drop(columns ="UnixTime",inplace =True)
data.drop(columns ="AnswerType",inplace =True)
data.drop(columns ="Asin",inplace =True)
df_cleaned =data.dropna()
print(df_cleaned.isnull().sum())





