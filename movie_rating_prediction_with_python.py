MOVIE RATING PREDICTION WITH PYTHON
"""

# Importing necessary libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.pipeline import make_pipeline

"""Data Collecting and Processing:"""

# Load data from CSV file
Movie = pd.read_csv('Movie.csv', encoding='ISO-8859-1')
Movie.head()

"""Handling Missing Values:"""

# Filling NaN values in 'rating' column with the mean rating
Movie['Rating'].fillna(Movie['Rating'].mean(), inplace=True)

"""Feature Engineering:"""

# Feature Engineering: Combining text features (actors and directors)
Movie ['combined_features'] = Movie ['Genre'] + ' ' + Movie ['Director'] + ' ' + Movie ['Actor 1']

# Ensuring all values in the 'combined_features' column are strings
Movie['combined_features'] = Movie['combined_features'].astype(str)

"""Spliting the Data:"""

# Splitting the data into features (X) and target (y)
X = Movie['combined_features']
y = Movie['Rating']

"""Model Selection and Training:"""

# Splitting the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Creating a pipeline for text processing and linear regression model
model = make_pipeline(CountVectorizer(), LinearRegression())

# Training the model
model.fit(X_train, y_train)

""" Model Evaluation:"""

# Making predictions
predictions = model.predict(X_test)

# Calculating the mean squared error to evaluate the model
mse = mean_squared_error(y_test, predictions)
print(f'Mean Squared Error: {mse}')

"""Prediction:"""

# Taking the example and predicting the ratings for new movie:
new_movie = ["Action Comedy", "John Smith"]
predicted_rating = model.predict(new_movie)
print(f'Predicted Rating for the new movie: {predicted_rating[0]}')
