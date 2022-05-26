# Import_Libraries
import sys

import numpy as np
import matplotlib.pyplot as plt
import pandas as ps
import random

# Import_dataset
data = ps.read_json('intend.json')

# Clean_Data
import re
import nltk

nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from nltk.stem import WordNetLemmatizer

corpus = []
x = []
y = []
for intent in data['intents']:
    for pattern in intent['patterns']:
        tokens = re.sub('[^a-zA-Z]', ' ', pattern)
        tokens = tokens.lower()
        tokens = tokens.split()
        ps = PorterStemmer()
        all_words = stopwords.words('english')
        tokens = [ps.stem(words) for words in tokens if not words in set(all_words)]
        tokens = ' '.join(tokens)
        corpus.append(tokens)
        y.append(intent['tag'])
# Bag_of_word model
from sklearn.feature_extraction.text import CountVectorizer

cv = CountVectorizer(max_features=20)
x = cv.fit_transform(corpus)

# train_test_split
from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)

# apply random_Forest_Classifier model
from sklearn.ensemble import RandomForestClassifier

classifier = RandomForestClassifier(n_estimators=100, criterion="entropy")

classifier.fit(x_train, y_train)
y_pred = classifier.predict(x_test)

# making confusion metrics
from sklearn.metrics import confusion_matrix, accuracy_score

cm = confusion_matrix(y_test, y_pred)
print(accuracy_score(y_test, y_pred))

# Take the user input and run chatbot
def chatbot_response(userText):

        tokens1 = re.sub('[^a-zA-Z]', ' ', userText)

        tokens1 = nltk.word_tokenize(userText)
        tokens1 = [word.lower() for word in tokens1]
        x1 = cv.transform(tokens1)
        thresh = 0.5
        result = classifier.predict_proba(x1)
        y1_pred = classifier.predict(x1)
        max = np.amax(result)
        c = 0
        for i in result:
            if np.max(i) == max:
                break
            else:
                c += 1

        for i in data['intents']:
            if i['tag'] == y1_pred[c]:
                return random.choice(i['responses'])
            else:
                pass

from flask import Flask, render_template, request
app = Flask(__name__)
app.static_folder = 'static'
@app.route("/")
def home():
    return render_template("index.html")

@app.route("/get")
def get_bot_response():
    userText = request.args.get('msg')
    if userText == 'quit':
        sys.exit()
    else:
        return chatbot_response(userText)

if __name__ == "__main__":
    app.run()

