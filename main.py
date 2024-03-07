import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
import sklearn.naive_bayes

data = pd.read_json('data/Sarcasm.json', lines=True)

data = data[['headline', 'is_sarcastic']]
data['is_sarcastic'] = data['is_sarcastic'].map({0: 'Not Sarcasm', 1: 'Sarcasm'})

x = np.array(data['headline'])
y = np.array(data['is_sarcastic'])

cv = CountVectorizer()
x = cv.fit_transform(x)
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.20, random_state=42)

model = sklearn.naive_bayes.BernoulliNB()
model.fit(x_train, y_train)

# Testing
user = input('Enter a Text:')
data = cv.transform([user]).toarray()
output = model.predict(data)
print(output)


