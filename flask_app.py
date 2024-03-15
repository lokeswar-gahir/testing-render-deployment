
# A very simple Flask Hello World app for you to get started with...

print("importing dependencies...", end="")
from flask import Flask
# import pandas as pd
# from nltk.stem.porter import PorterStemmer
# from sklearn.feature_extraction.text import CountVectorizer
# from sklearn.model_selection import train_test_split
# from sklearn.naive_bayes import BernoulliNB
# from sklearn.metrics import accuracy_score
from pickle import load
print("done")
# ps = PorterStemmer()


with open("similarity_32.pkl","rb") as f:
    similarity = load(f)
# print("importing dataset...")
# df = pd.read_csv("IMDB Dataset.csv")
# print("done")
# df["sentiment"]=df["sentiment"].replace({"positive": 1, "negative":0})
# print("Removing html tags...")
# df["review"]=df["review"].apply(remove_html)
# df["review"]=df["review"].str.lower()
# print("Removing special characters...", end="")
# df["review"]=df["review"].apply(remove_special_ch)
# print("done")
# print("applying stem to the review column(this might take long time)...", end="")
# df["review"]=df["review"].apply(apply_stem) #takes lots of time
# print("done")
# y = df["sentiment"].values
# print("vectorizing...", end="")
# cv = CountVectorizer(max_features=3000, stop_words="english")
# x = cv.fit_transform(df["review"]).toarray()
# print("done")
# print("splitting data...", end="")
# x_train, x_test, y_train, y_test = train_test_split(x,y, test_size=0.1, random_state=0)
# print("done")
# clf3 = BernoulliNB()
# print("Training the model...", end="")
# clf3.fit(x_train, y_train)
# print("done")
# y_pred = clf3.predict(x_test)
# print("Accuracy score of BernoulliNB():",accuracy_score(y_test, y_pred))# 8458

app = Flask(__name__)


@app.route('/')
def hello_world():
    return 'Hello from Flask!'
if __name__=="__main__":
    app.run()
