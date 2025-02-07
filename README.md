🫀 Heart Disease Prediction
<br>
This repository contains a Heart Disease Prediction project using multiple machine learning models. The dataset is preprocessed, and different classification algorithms are evaluated based on performance metrics.

📂 Project Overview
Dataset: A heart disease dataset with multiple features.
Goal: Predict the presence of heart disease using machine learning models.
Tech Stack: Python, Scikit-Learn, Pandas, Matplotlib, Seaborn
🛠 Steps Involved
1️⃣ Data Preprocessing
Read dataset using pandas 📊
Split data into training and testing sets (40% test data)
Apply StandardScaler for scale-sensitive models
2️⃣ Model Training
📌 Scale-Insensitive Models
✅ Random Forest Classifier
✅ Naive Bayes (GaussianNB)
✅ Gradient Boosting Classifier

📌 Scale-Sensitive Models
✅ K-Nearest Neighbors (KNN)
✅ Logistic Regression
✅ Support Vector Machine (SVM)

📊 Model Evaluation
✅ Performance Metrics
Accuracy Score ✅
Recall Score 📏
ROC-AUC Score 🏆
🔍 ROC Curve Analysis
Plotted ROC Curve for various models to analyze performance
📈 Feature Importance (Random Forest)
Identified key features influencing predictions
🔧 Hyperparameter Tuning
GridSearchCV used for optimizing the Random Forest model
Tuned parameters:
n_estimators, max_depth, min_samples_split, min_samples_leaf, max_features
📌 Additional Insights
Correlation Heatmap created using seaborn for feature relationships
Best-performing model identified after evaluation
