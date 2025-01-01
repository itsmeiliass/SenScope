# ----------------------------------------------------------------------------------------------------------------------#
# Import required libraries and tools for Natural Language Processing, data processing, visualization, and modeling

import nltk
nltk.download("stopwords")  # Download stopwords list for text preprocessing
nltk.download('punkt')      # Download punkt tokenizer

# Libraries for data manipulation and visualization
import pandas as pd
import numpy as np
import re
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib import style
style.use('ggplot')  # Set style for visualizations

# Libraries for text processing and NLP
from textblob import TextBlob
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords
stop_words = set(stopwords.words('english'))  # Set of English stopwords

# Libraries for machine learning models and evaluation
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay
# ----------------------------------------------------------------------------------------------------------------------#

# Load and clean dataset

# Import the dataset
df = pd.read_csv('twitter_training.csv')
print(df.describe())  # Print statistical summary of the dataset
print(df.dtypes)      # Print data types of each column
print(df.info())      # Display dataset information, including null counts

# Display column names, shape, and unique sentiment labels
print(df.columns)
print('Shape of data is:', df.shape)
print('Unique sentiment labels:', df['sentiment'].unique())

# Drop duplicate rows and null values to clean the dataset
df.drop_duplicates(inplace=True)
df.dropna(inplace=True)

# Verify data cleaning
print(df.head())  # Display first 5 rows
print(df.tail())  # Display last 5 rows
print("NaN values:\n", df.isnull().sum())  # Check for remaining NaN values

# Verify dataset statistics after cleaning
print(df.describe())
print(df.dtypes)
print(df.info())

# ----------------------------------------------------------------------------------------------------------------------#

# Preprocess tweet content by removing unnecessary columns and cleaning text
text_df = df.drop(['entity', 'sentiment'], axis=1)  # Drop irrelevant columns

# Define a function for cleaning text
def data_processing(text):
    text = text.lower()  # Convert text to lowercase
    text = re.sub(r"https\S+|www\S+https\S+", '', text, flags=re.MULTILINE)  # Remove URLs
    text = re.sub(r'\@w+|\#', '', text)  # Remove mentions and hashtags
    text = re.sub(r'[^\w\s]', '', text)  # Remove special characters
    text_tokens = word_tokenize(text)  # Tokenize text
    filtered_text = [w for w in text_tokens if w not in stop_words]  # Remove stopwords
    return " ".join(filtered_text)

text_df['text'] = text_df['Tweet_content,'].apply(data_processing)  # Apply text cleaning function

# Define a function for stemming
stemmer = PorterStemmer()
def stemming(data):
    return [stemmer.stem(word) for word in data]

text_df['Tweet_content,'] = text_df['Tweet_content,'].apply(stemming)  # Apply stemming

# ----------------------------------------------------------------------------------------------------------------------#

# Sentiment polarity analysis using TextBlob

# Define functions to calculate polarity and sentiment category
def polarity(text):
    return TextBlob(text).sentiment.polarity

def sentiment(label):
    if label < 0:
        return "Negative"
    elif label == 0:
        return "Neutral"
    else:
        return "Positive"

# Apply polarity and sentiment functions
text_df['polarity'] = text_df['text'].apply(polarity)
text_df['sentiment'] = text_df['polarity'].apply(sentiment)
print(text_df.tail(20))  # Display final dataset with added sentiment labels

# ----------------------------------------------------------------------------------------------------------------------#

# Visualize sentiment distribution

# Original sentiment distribution
fig = plt.figure(figsize=(5, 5))
sns.countplot(x='sentiment', data=df)
plt.savefig("mygraph.png")

# Sentiment distribution after applying sentiment functions
fig = plt.figure(figsize=(5, 5))
sns.countplot(x='sentiment', data=text_df)
plt.savefig("mygraph2.png")

# ----------------------------------------------------------------------------------------------------------------------#

# Build and train sentiment classification models

# Convert text data to feature vectors using CountVectorizer with n-grams
vect = CountVectorizer(ngram_range=(1, 2)).fit(text_df['text'])

# Define feature matrix X and target variable Y
X = text_df['text']
Y = text_df['sentiment']
X = vect.transform(X)  # Transform text data into numeric format

# Split data into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

print("Training set size:", x_train.shape)
print("Test set size:", x_test.shape)

# Train a Logistic Regression model
logreg = LogisticRegression()
logreg.fit(x_train, y_train)
logreg_pred = logreg.predict(x_test)
logreg_acc = accuracy_score(logreg_pred, y_test)

# Evaluate the Logistic Regression model
print("Test accuracy: {:.2f}%".format(logreg_acc * 100))
print(confusion_matrix(y_test, logreg_pred))
print(classification_report(y_test, logreg_pred))

# Visualize the confusion matrix
style.use('classic')
cm = confusion_matrix(y_test, logreg_pred, labels=logreg.classes_)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=logreg.classes_)
disp.plot()
plt.savefig('mygraph3.png')

# ----------------------------------------------------------------------------------------------------------------------#

# Hyperparameter tuning for Logistic Regression using GridSearchCV
param_grid = {'C': [0.001, 0.01, 0.1, 1, 10]}
grid = GridSearchCV(LogisticRegression(), param_grid)
grid.fit(x_train, y_train)

# Evaluate the best Logistic Regression model
print("Best parameters:", grid.best_params_)
y_pred = grid.predict(x_test)
logreg_acc = accuracy_score(y_pred, y_test)
print("Improved accuracy: {:.2f}%".format(logreg_acc * 100))
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))

# ----------------------------------------------------------------------------------------------------------------------#

# Train and evaluate Linear Support Vector Machine (SVM) model
from sklearn.svm import LinearSVC
SVCmodel = LinearSVC()
SVCmodel.fit(x_train, y_train)

svc_pred = SVCmodel.predict(x_test)
svc_acc = accuracy_score(svc_pred, y_test)

print("SVM test accuracy: {:.2f}%".format(svc_acc * 100))
print(confusion_matrix(y_test, svc_pred))
print(classification_report(y_test, svc_pred))

# Hyperparameter tuning for SVM using GridSearchCV
from sklearn.model_selection import GridSearchCV
param_grid = {
    'C': [0.01, 0.1, 1, 10],
    'kernel': ["linear", "poly", "rbf", "sigmoid"],
    'degree': [1, 3, 5, 7],
    'gamma': [0.01, 1]
}
grid = GridSearchCV(SVCmodel, param_grid)
grid.fit(x_train, y_train)

# Evaluate the best SVM model
print("Best parameters:", grid.best_params_)
y_pred = grid.predict(x_test)
logreg_acc = accuracy_score(y_pred, y_test)
print("Improved SVM accuracy: {:.2f}%".format(logreg_acc * 100))
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))
