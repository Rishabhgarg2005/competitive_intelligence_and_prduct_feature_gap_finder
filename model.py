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
data =pd.read_csv("Amazon_reviews_2023.csv",nrows =5000)
data.drop("images",axis =1,inplace =True)
# print(data.info())
# print(data.head())

def convert(text):
    if text =="True":
        return 1
    else:
        return 0

data["verified_purchase"]=data["verified_purchase"].apply(convert)
data["updated_date"]=pd.to_datetime(data["timestamp"],errors="coerce")
# print(data.head())
# print(data.info())
def convert2(num):
    if num ==0:
       return 0
    else:
        return 1
data["helpful_vote"]=data["helpful_vote"].apply(convert2)

def convert3(text):
    if text =="TRUE":
        return 1
    else:
        return 0

data["verified_purchase"]=data["verified_purchase"].apply(convert3)

def convert4(num):
    if num == 1 or num==2:
        return 0
    elif num ==3:
        return 1
    else:
        return 2
data["rating"]=data["rating"].apply(convert4)

tdata =["title","text","asin", "parent_asin","user_id","timestamp",]
for feature in tdata:
  if feature in data.columns:
   data[feature]=data[feature].astype(str).str.lower()

print(data["title"])






