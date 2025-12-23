# California Housing Price Prediction (Scikit-learn)

This project demonstrates a **machine learning regression pipeline** using the **Scikit-learn** library to predict housing prices based on the California Housing dataset.

The project covers:
- Feature engineering with polynomial features
- Model training using Gradient Boosting
- Model evaluation with R² score
- Model persistence using Joblib

---

## 📊 Dataset

- **Source:** `sklearn.datasets.fetch_california_housing`
- **Features:** Median income, house age, average rooms, population, latitude, longitude, etc.
- **Target:** Median house value

---

## 🛠️ Technologies Used

- Python
- Scikit-learn
- Joblib
- Jupyter Notebook

---

## ⚙️ Machine Learning Workflow

1. Load California Housing dataset
2. Expand features using `PolynomialFeatures`
3. Split data into training and testing sets
4. Train a `HistGradientBoostingRegressor`
5. Evaluate performance using **R² score**
6. Save and reload the trained model using Joblib

---

## 🧠 Model Used

```python
HistGradientBoostingRegressor(
    max_iter=350,
    learning_rate=0.05
)
