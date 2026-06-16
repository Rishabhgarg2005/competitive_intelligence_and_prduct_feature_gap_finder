import numpy as np 
import pandas as pd
import nltk 
from nltk import word_tokenize
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
nltk.download('punkt')
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))


from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
data =pd.read_csv("single_qna.csv")
print(data.head()) 
data.drop(columns ="UnixTime",inplace =True)
data.drop(columns ="AnswerType",inplace =True)
data.drop(columns ="Asin",inplace =True)
df_cleaned =data.dropna()
print(df_cleaned.isnull().sum())
def convert(text):
  tokens =word_tokenize(str(text).lower())
  cleaned_tokens =[w for w in tokens if w.isalnum() and w not in stop_words]
  return"".join(cleaned_tokens)

data["cleaned"]=data["Question"].apply(convert)
print(data["cleaned"].str.split())