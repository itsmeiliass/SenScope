#----------------------------------------------------------------------------------------------------------------------------------#
# Import necessary libraries
import pandas as pd
import numpy as np
import re
import seaborn as sns

import matplotlib.pyplot as plt
from matplotlib import style
style.use('ggplot')  # Set default plot style to 'ggplot'
from textblob import TextBlob  # For sentiment analysis

#nltk "natural language tool-kil"
import nltk
nltk.download("stopwords")  # Download stopwords data from NLTK
from nltk.tokenize import RegexpTokenizer  # Tokenizer that splits based on word characters
from nltk.stem import PorterStemmer  # Stemming to reduce words to their root form
from nltk.corpus import stopwords  # For stopword removal
stop_words = set(stopwords.words('english'))  # Define stopwords in English
#sklearn for model
from sklearn.feature_extraction.text import CountVectorizer  # For converting text to feature vectors
from sklearn.model_selection import train_test_split  # For splitting data into training and test sets
from sklearn.linear_model import LogisticRegression  # Logistic regression model for classification
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay  # For evaluation

#----------------------------------------------------------------------------------------------------------------------------------#
# Data Preparation

# Import dataset from a CSV file
df = pd.read_csv('twitter_training.csv')

# Display basic statistics and information about the dataset
print(df.describe())
print(df.dtypes)
print(df.info())
print(df.columns)
print('Shape of data:', df.shape)
print(df['sentiment'].unique())

# Remove duplicate rows and rows with missing values (NaN)
df.drop_duplicates(inplace=True)
df.dropna(inplace=True)

# Display the cleaned dataset
print(df.head())
print(df.tail())

# Check for missing values
nan_values = df.isnull().sum()
print("Missing values:\n", nan_values)

# Display summary statistics and data types
print(df.describe())
print(df.dtypes)
print(df.info())
print(df.columns)
print('Shape of data:', df.shape)
print(df['sentiment'].unique())

# Data processing: Clean tweet content and extract relevant features
text_df = df.drop(['entity', 'sentiment'], axis=1)  # Drop irrelevant columns

# Create a tokenizer that splits on non-word characters
tokenizer = RegexpTokenizer(r'\w+')

# Function to clean and process the tweet content
def data_processing(text):
    text = text.lower()  # Convert text to lowercase
    text = re.sub(r"https\S+|www\S+https\S+", '', text, flags=re.MULTILINE)  # Remove URLs
    text = re.sub(r'\@w+|\#', '', text)  # Remove mentions and hashtags
    text = re.sub(r'[^\w\s]', '', text)  # Remove special characters
    text_tokens = tokenizer.tokenize(text)  # Tokenize text using the tokenizer
    filtered_text = [w for w in text_tokens if not w in stop_words]  # Remove stopwords
    return " ".join(filtered_text)  # Join tokens back into a single string

# Apply text processing to the dataset
text_df['text'] = text_df['Tweet_content,'].apply(data_processing)

# Initialize Porter Stemmer to perform stemming on words
stemmer = PorterStemmer()

# Function to apply stemming to the processed text
def stemming(data):
    text = [stemmer.stem(word) for word in data]  # Stem each word
    return data

# Apply stemming to the tweet content
text_df['Tweet_content,'] = text_df['Tweet_content,'].apply(lambda x: stemming(x))

#----------------------------------------------------------------------------------------------------------------#
# Sentiment Analysis

# Function to calculate polarity of the text using TextBlob
def polarity(text):
    return TextBlob(text).sentiment.polarity

# Apply polarity function to the dataset
text_df['polarity'] = text_df['text'].apply(polarity)

# Function to assign sentiment labels based on polarity
def sentiment(label):
    if label < 0:
        return "Negative"
    elif label == 0:
        return "Neutral"
    elif label > 0:
        return "Positive"

# Apply sentiment function based on polarity values
text_df['sentiment'] = text_df['polarity'].apply(sentiment)

# Display the last few rows of the dataset with sentiment labels
print(text_df.tail(20))

#----------------------------------------------------------------------------------------------------------------#
# Data Visualization

# Visualize the distribution of sentiment in the original dataset
fig = plt.figure(figsize=(5, 5))
sns.countplot(x='sentiment', data=df)
plt.savefig("mygraph.png")

# Visualize the distribution of sentiment in the processed dataset
fig = plt.figure(figsize=(5, 5))
sns.countplot(x='sentiment', data=text_df)
plt.savefig("mygraph2.png")

#----------------------------------------------------------------------------------------------------------------#
# Model Training

# Convert text data into feature vectors using CountVectorizer
vect = CountVectorizer(ngram_range=(1, 2)).fit(text_df['text'])
X = text_df['text']
Y = text_df['sentiment']
X = vect.transform(X)

# Split the dataset into training and test sets (80% training, 20% testing)
x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

# Print the sizes of the train and test sets
print("Size of x_train:", x_train.shape)
print("Size of y_train:", y_train.shape)
print("Size of x_test:", x_test.shape)
print("Size of y_test:", y_test.shape)

# Ignore warnings related to model fitting
import warnings
warnings.filterwarnings('ignore')

# Initialize Logistic Regression model
logreg = LogisticRegression()
logreg.fit(x_train, y_train)

# Predict the sentiment using the trained model
logreg_pred = logreg.predict(x_test)

# Calculate accuracy and display evaluation metrics
logreg_acc = accuracy_score(logreg_pred, y_test)
print("Test accuracy: {:.2f}%".format(logreg_acc * 100))
print(confusion_matrix(y_test, logreg_pred))
print("\n")
print(classification_report(y_test, logreg_pred))

# Display confusion matrix for the logistic regression model
style.use('classic')
cm = confusion_matrix(y_test, logreg_pred, labels=logreg.classes_)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=logreg.classes_)
disp.plot()
plt.savefig('mygraph3.png')

#----------------------------------------------------------------------------------------------------------------#
# Model Improvement 1: Hyperparameter Tuning with GridSearchCV

from sklearn.model_selection import GridSearchCV
print("GridSearchCV for Logistic Regression")

# Define parameter grid for logistic regression
param_grid = {'C': [0.001, 0.01, 0.1, 1, 10]}
grid = GridSearchCV(LogisticRegression(), param_grid)
grid.fit(x_train, y_train)

# Print best parameters from GridSearchCV
print("Best parameters:", grid.best_params_)

# Predict sentiment using the best model from GridSearchCV
y_pred = grid.predict(x_test)

# Calculate accuracy and display evaluation metrics
logreg_acc = accuracy_score(y_pred, y_test)
print("Test accuracy after tuning: {:.2f}%".format(logreg_acc * 100))
print(confusion_matrix(y_test, y_pred))
print("\n")
print(classification_report(y_test, y_pred))

#----------------------------------------------------------------------------------------------------------------#
# Model Improvement 2: LinearSVC Model

from sklearn.svm import LinearSVC
print("LinearSVC")

# Initialize and train a Linear Support Vector Classifier model
SVCmodel = LinearSVC()
SVCmodel.fit(x_train, y_train)

# Predict sentiment using the trained SVC model
svc_pred = SVCmodel.predict(x_test)

# Calculate accuracy and display evaluation metrics
svc_acc = accuracy_score(svc_pred, y_test)
print("Test accuracy using LinearSVC: {:.2f}%".format(svc_acc * 100))
print(confusion_matrix(y_test, svc_pred))
print("\n")
print(classification_report(y_test, svc_pred))

#----------------------------------------------------------------------------------------------------------------#
# Model Improvement 3: Hyperparameter Tuning for SVC Model

print("Hyperparameter Tuning for SVC")

# Define parameter grid for SVC model
grid = {
    'C': [0.01, 0.1, 1, 10],
    'kernel': ["linear", "poly", "rbf", "sigmoid"],
    'degree': [1, 3, 5, 7],
    'gamma': [0.01, 1]
}
grid = GridSearchCV(SVCmodel, param_grid)
grid.fit(x_train, y_train)

# Print best parameters from GridSearchCV
print("Best parameters for SVC:", grid.best_params_)

# Predict sentiment using the best model from GridSearchCV
y_pred = grid.predict(x_test)

# Calculate accuracy and display evaluation metrics
logreg_acc = accuracy_score(y_pred, y_test)
print("Test accuracy after tuning SVC: {:.2f}%".format(logreg_acc * 100))
print(confusion_matrix(y_test, y_pred))
print("\n")
print(classification_report(y_test, y_pred))
