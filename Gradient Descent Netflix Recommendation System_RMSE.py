#!/usr/bin/env python
# coding: utf-8

# In[36]:


# Loading libraries
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import random

np.random.seed(42) 

# Loading training data
col_names = ['id', 'title', 'show_type', 'description', 'release_year', 'age_certification', 'runtime', 'genres', 'production_countries', 'seasons', 'imdb_id', 'imdb_score', 'imdb_votes', 'tmdb_popularity', 'tmdb_score']
train_data = pd.read_csv(r"C:\Users\gkuma\Downloads\archive\titles1000.csv", names=col_names, skiprows=1)

# Eliminating missing values
train_data = train_data.dropna(subset=['imdb_score'])

# Splitting the data into train and test sets -> 80% training, 20% testing
train_size = 0.8
train_data, test_data = train_test_split(train_data, train_size=train_size, random_state=42)

# Creating user-item matrices for train and test sets
train_user_item_matrix = train_data.pivot_table(index='id', columns='title', values='imdb_score').fillna(0)
test_user_item_matrix = test_data.pivot_table(index='id', columns='title', values='imdb_score').fillna(0)

# Converting the user-item matrices to NumPy arrays
P_train = train_user_item_matrix.values
P_test = test_user_item_matrix.values

# Set the number of latent factors
K = 10

# Initializing user and item matrices with random values for training and testing
P_train = np.random.rand(P_train.shape[0], K)
Q_train = np.random.rand(K, P_train.shape[1])

P_test = np.random.rand(P_test.shape[0], K)
Q_test = np.random.rand(K, P_test.shape[1])

# Learning rate and regularization parameter
alpha = 0.01
beta = 0.02
epochs = 100

# Implementing gradient descent on the training data
for epoch in range(epochs):
    for i in range(len(P_train)):
        for j in range(len(P_train[i])):
            if P_train[i][j] > 0:
                eij_train = P_train[i][j] - np.dot(P_train[i, :], Q_train[:, j])
                for k in range(K):
                    P_train[i][k] = P_train[i][k] + alpha * (2 * eij_train * Q_train[k][j] - beta * P_train[i][k])
                    Q_train[k][j] = Q_train[k][j] + alpha * (2 * eij_train * P_train[i][k] - beta * Q_train[k][j])

# Rating predictions
R_pred_test = np.dot(P_test, Q_test)

# Function to get recommendations for a user from the test set
def get_test_recommendations(user_id, top_n=5):
    if user_id in test_user_item_matrix.index:
        user_index = test_user_item_matrix.index.get_loc(user_id)
        user_ratings = R_pred_test[user_index, :]
        sorted_indices = np.argsort(user_ratings)[::-1]  # Sort in descending order
        top_indices = sorted_indices[:top_n]
        
        recommendations = []
        for index in top_indices:
            title = test_user_item_matrix.columns[index]
            predicted_rating = user_ratings[index]
            recommendations.append({'Title': title, 'Predicted Rating': predicted_rating})
        
        return recommendations
    else:
        return f"User ID {user_id} not found in the test_user_item_matrix.index."


# Get a random user ID from the test set
# random_user_id = random.choice(test_user_item_matrix.index)

# Get recommendations for the randomly selected user ID
# test_recommendations_random = get_test_recommendations(random_user_id)

# Generating recommendations for user
user_id_to_recommend_test = 'tm224577'

test_recommendations_random = get_test_recommendations(user_id_to_recommend_test)

if isinstance(test_recommendations_random, list):
    print(f"Test Set Recommendations for Randomly Selected User ID {user_id_to_recommend_test}:")
    for recommendation in test_recommendations_random:
        print(recommendation)
else:
    print(test_recommendations_random)

# Calculate Mean Squared Error (MSE) on the test set
#mse_test = np.sum((P_test - R_pred_test) ** 2) / np.sum(P_test > 0)

#Function to calculate RMSE
def calculate_rmse(true_ratings, predicted_ratings):
    non_zero_indices = true_ratings.nonzero()
    true_values = true_ratings[non_zero_indices]
    predicted_values = predicted_ratings[non_zero_indices]
    mse = np.mean((true_values - predicted_values) ** 2)
    rmse = np.sqrt(mse)
    return rmse

# Training the recommendation system and calculating RMSE on the test set
for epoch in range(epochs):
    for i in range(len(P_train)):
        for j in range(len(P_train[i])):
            if P_train[i][j] > 0:
                eij_train = P_train[i][j] - np.dot(P_train[i, :], Q_train[:, j])
                for k in range(K):
                    P_train[i][k] = P_train[i][k] + alpha * (2 * eij_train * Q_train[k][j] - beta * P_train[i][k])
                    Q_train[k][j] = Q_train[k][j] + alpha * (2 * eij_train * P_train[i][k] - beta * Q_train[k][j])

# Rating predictions
R_pred_test = np.dot(P_test, Q_test)

# Calculate RMSE on the test set
test_rmse = calculate_rmse(P_test, R_pred_test)
print(f"Root Mean Squared Error (RMSE) on the Test Set: {test_rmse}")


# In[ ]:




