**Movie Recommendation System**

Overview:

This project implements a collaborative filtering-based recommendation system using Matrix Factorization (MF) to predict movie ratings and generate personalized recommendations.

Features:

✅ **User-Item Interaction Matrix**  
&nbsp;&nbsp;&nbsp;&nbsp;Built using IMDb scores  

✅ **Matrix Factorization**  
&nbsp;&nbsp;&nbsp;&nbsp;Predicts ratings using Gradient Descent  

✅ **Personalized Recommendations**  
&nbsp;&nbsp;&nbsp;&nbsp;Suggests top movies for a given user  

✅ **Performance Evaluation**  
&nbsp;&nbsp;&nbsp;&nbsp;Computes Root Mean Squared Error (RMSE) for accuracy measurement  


Technologies Used:

Python
NumPy & Pandas – Data manipulation and handling
Scikit-learn – Train-test split for evaluation
Random Sampling – Selects a user for recommendations

Usage:

The system loads the dataset, cleans missing values, and splits data into train-test sets.
It applies Matrix Factorization via Gradient Descent to learn user-item preferences.
Predictions are generated, and top recommendations are provided for a given user.
RMSE is calculated to evaluate the model’s performance.

Results:

The system successfully predicts movie ratings for unseen data.
Achieved low RMSE, indicating good accuracy in rating predictions.
Future Improvements
🔹 Implementing a Deep Learning-based recommendation model
🔹 Enhancing recommendations using content-based filtering
🔹 Deploying the model as a web-based API

Contributing:

Contributions are welcome! Please fork the repository and submit a pull request.