import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression

inputFile = 'MoviesDataset.csv'

# Using panda to import the csv file and separate the summaries from the sentiments
# header=0 skips the first line since it's just the titles
df = pd.read_csv(inputFile, names=['Summary', 'Sentiment'], sep=',',
                 header=0)

print(df)

# Keep the values into lists
summary = df['Summary'].values
sentiment = df['Sentiment'].values

# Using test_train_split from the sklearn library to train the system
# test_size is 1% (=0.01) so train_size becomes 99%
# Also random_state is used (has default value = 'None'), so each run will provide different results
x_reviews_train, x_reviews_test, y_reviews_train, y_reviews_test = train_test_split(summary, sentiment, test_size=0.1)

vectorizer = CountVectorizer()
vectorizer.fit(x_reviews_train)

X_train = vectorizer.transform(x_reviews_train)
X_test = vectorizer.transform(x_reviews_test)

classifier = LogisticRegression()
classifier.fit(X_train, y_reviews_train)

accuracy = classifier.score(X_test, y_reviews_test)
# accuracy number converted to percentage rounded to 2 floating points (and converted to string)
print("Prediction accuracy:", str(round(float(accuracy*100), 2)) + "%\n")

print("Below you can manually input a sentence and it's predicted sentiment will be returned\n")

while 1:

    inp = input("Please type a sentence to (or 'exit' to terminate): ")

    if inp == "exit":
        print("Exiting...")
        break

    X_new = vectorizer.transform([inp])

    if int(classifier.predict(X_new)) == 1:
        print("Sentence has:\tPositive sentiment\n")
    else:
        print("Sentence has:\tNegative sentiment\n")

