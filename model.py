import numpy as np 
import pandas as pd
import nltk 
from xgboost import XGBClassifier
from nltk import word_tokenize
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from sklearn.metrics import accuracy_score,confusion_matrix,classification_report

from sklearn.feature_extraction.text import TfidfVectorizer
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
stop_words = set(stopwords.words('english'))

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

data =pd.read_csv("Amazon_reviews_2023.csv",nrows =5000)
data.drop(columns ="images",inplace =True)
data["datetime"] =pd.to_datetime(data["timestamp"],format="%Y-%m-%d %H:%M:%S.%f")
data.drop(columns ="timestamp",inplace = True)

def convert2(text):
   if text ==0:
        return 0
   else:
      return 1
   
data["helpful_vote"]=data["helpful_vote"].apply(convert2)

def convert3(text):
   if text =="TRUE":
      return 1
   else:
      return 0
data["verified_purchase"] =data["verified_purchase"].apply(convert3)


def convert4(num):
    if num == 1 or num==2:
        return 0 
    elif num ==3:
        return 1
    else:
        return 2
data["rating"]=data["rating"].apply(convert4)

tdata =["title","text"]
for feature in tdata:
  if feature in data.columns:
   data[feature]=data[feature].astype(str).str.lower()

print(data["title"])
data = data.dropna(subset=tdata)
# print(data.head())
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

def cleaned_text(text):
   tokens = word_tokenize(text.lower())

   cleaned_tokens = [
      lemmatizer.lemmatize(token, pos="v")
      for token in tokens
      if token not in stop_words
   ]
   return " ".join(cleaned_tokens)

for feature in tdata:
   data[feature] = data[feature].apply(cleaned_text)
x =data.drop(columns ="rating",errors ="ignore")
y =data["rating"]
print(data.head())

X_train, X_val, Y_train, Y_val = train_test_split(x, y, test_size=0.2, random_state=42)
tfidf =TfidfVectorizer(max_features =5000,ngram_range=(1,2))
X_train_tfidf =tfidf.fit_transform(X_train["text"])

X_val_tfidf =tfidf.transform(X_val["text"])


model =XGBClassifier(n_estimators=150,
    max_depth=6,
    learning_rate=0.1,
    objective='multi:softprob',
    random_state=42
)


from imblearn.over_sampling import SMOTE
smote =SMOTE(random_state =42)
X_train_imb,Y_train_imb =smote.fit_resample(X_train_tfidf,Y_train)

model.fit(X_train_imb,Y_train_imb,)

y_pred =model.predict(X_val_tfidf)
print(confusion_matrix(y_pred,Y_val))
print(classification_report(Y_val,y_pred))
print(accuracy_score(Y_val,y_pred))

model.save_model("sentiment_classifier.json")



