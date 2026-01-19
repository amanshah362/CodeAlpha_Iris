# Iris Flower Classification Web App

This is a **Flask web application** for predicting the species of Iris flowers using **Logistic Regression** with polynomial features. The model was trained on the **classic Iris dataset**, and it includes **data preprocessing, feature scaling, and hyperparameter tuning**.  

## Features
- Clean and interactive **Flask web interface** for user input
- **Preprocessing pipeline** with `RobustScaler` and `PolynomialFeatures`
- **Hyperparameter tuning** using `GridSearchCV`
- Multi-class **classification** using `Logistic Regression`
- Model persistence using `pickle` for real-time predictions

## Technologies Used
- **Python** (3.9+)
- **Flask** (Web Framework)
- **Scikit-learn** (ML & preprocessing)
- **Pandas & NumPy** (Data manipulation)
- **Matplotlib & Seaborn** (Data visualization)
- **Pickle** (Model serialization)

## Installation
1. Clone the repository:
```bash
git clone https://github.com/yourusername/iris-flask-app.git
