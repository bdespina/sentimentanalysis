import pandas as pd
#The module:sklearn.naive_bayes module implements Naive Bayes algorithms.
# These are supervised learning methods based on applying Bayes' theorem
# with strong (naive) feature independence assumptions.
#Imports Gauss classs
from sklearn.naive_bayes import GaussianNB
#The module:sklearn.metrics module includes score functions,
# performance metrics and pairwise metrics and distance computations.
#Compute confusion matrix to evaluate the accuracy of a classification.
from sklearn.metrics import confusion_matrix
from sklearn.feature_extraction.text import CountVectorizer
#Split arrays or matrices into random train and test subsets
from sklearn.model_selection import train_test_split
#Naive Bayes classifier for polynomial models
from sklearn import datasets


inputFile = "MoviesDataset.csv"

# Using panda to import the csv file and3 separate the summaries from the sentiments
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
x_train, x_test, y_train, y_test = train_test_split(summary, sentiment, test_size=0.1)

vectorizer = CountVectorizer()
vectorizer.fit(x_train)

X_train = vectorizer.transform(x_train)
X_test = vectorizer.transform(x_test)


from sklearn.naive_bayes import MultinomialNB

MNB = MultinomialNB()
MNB.fit(X_train , y_train)


from sklearn import metrics
predicted = MNB.predict(X_test)
accuracy_score = metrics.accuracy_score(predicted, y_test)

print("The accuracy is" , str('{:04.2f}'.format(accuracy_score*100))+'%')

print("Below you can manually input a sentence and it's predicted sentiment will be returned\n")

while 1:

    inp = input("Please type a sentence to (or 'exit' to terminate): ")

    if inp == "exit":
        print("Exiting...")
        break

    X_new = vectorizer.transform([inp])

    if int(MNB.predict(X_new)) == 1:
        print("Sentence has:\tPositive sentiment\n")
    else:
        print("Sentence has:\tNegative sentiment\n")
