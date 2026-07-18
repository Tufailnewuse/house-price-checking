# house-price-checking
# 🏡 House Price Prediction using Linear Regression

This project demonstrates how to build a **Machine Learning Regression model** using **Linear Regression** from Scikit-learn to predict house prices. The dataset is automatically downloaded using **KaggleHub**, preprocessed, and used to train and evaluate the model.

## 📌 Project Overview

The program performs the following steps:

1. Downloads the House Price Prediction dataset using KaggleHub.
2. Loads the dataset with Pandas.
3. Cleans the data by removing unnecessary columns.
4. Handles missing values.
5. Removes non-numeric features.
6. Splits the dataset into training and testing sets.
7. Trains a Linear Regression model.
8. Predicts house prices on the test dataset.
9. Evaluates the model using multiple regression metrics.

---

## 📂 Project Structure

```text
.
├── main.py
├── requirements.txt
└── README.md
```

---

## 🚀 Installation

Clone the repository:

```bash
git clone https://github.com/your-username/your-repository.git
cd your-repository
```

Install the required Python packages:

```bash
pip install -r requirements.txt
```

---

## ▶️ Run the Project

```bash
python main.py
```

---

## 📦 Requirements

- Python 3.8 or later
- pandas
- numpy
- scikit-learn
- kagglehub

Install all dependencies using:

```bash
pip install -r requirements.txt
```

---

## 🤖 Machine Learning Model

This project uses the **Linear Regression** algorithm from Scikit-learn.

**Workflow:**

- Load dataset
- Data preprocessing
- Feature selection
- Train/Test split (80% / 20%)
- Model training
- Prediction
- Performance evaluation

---

## 📊 Evaluation Metrics

The trained model is evaluated using:

- ✅ Mean Absolute Error (MAE)
- ✅ Mean Squared Error (MSE)
- ✅ Root Mean Squared Error (RMSE)
- ✅ R-squared (R² Score)

These metrics help measure the prediction accuracy of the regression model.

---

## 🏠 Dataset

**Dataset:** House Price Prediction Dataset

The dataset contains information about houses and their corresponding prices. The model learns the relationship between house features and their market prices to make predictions on unseen data.

The dataset is downloaded automatically using **KaggleHub**, so no manual download is required.

---

## 🛠 Technologies Used

- Python
- Pandas
- NumPy
- Scikit-learn
- KaggleHub

---

## 📚 Learning Objectives

This project helps you understand:

- Data preprocessing
- Feature selection
- Handling missing values
- Linear Regression
- Regression evaluation metrics
- Machine Learning workflow in Python

---

## 📄 License

This project is intended for educational and learning purposes.
