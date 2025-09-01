# House Price Prediction using Linear Regression

A comprehensive machine learning project that predicts house prices using linear regression models. This project includes data preprocessing, feature engineering, model training, evaluation, and visualization components.

## 🏠 Overview

This project implements a complete machine learning pipeline for predicting house prices based on various features such as:
- Number of rooms and bathrooms
- Living area size
- Location/neighborhood
- House age and condition
- Garage capacity
- And many more features

## 🚀 Features

### Data Processing
- ✅ Automatic handling of missing values
- ✅ Categorical variable encoding
- ✅ Feature scaling and normalization
- ✅ Feature selection and correlation analysis
- ✅ Outlier detection and handling

### Model Training
- ✅ Linear Regression (baseline model)
- ✅ Ridge Regression (L2 regularization)
- ✅ Lasso Regression (L1 regularization + feature selection)
- ✅ Cross-validation for robust evaluation
- ✅ Automatic best model selection

### Analysis & Visualization
- ✅ Exploratory Data Analysis (EDA)
- ✅ Model performance comparison
- ✅ Feature importance analysis
- ✅ Residuals analysis
- ✅ Prediction vs Actual plots

## 📋 Requirements

```txt
pandas>=1.3.0
numpy>=1.20.0
matplotlib>=3.3.0
seaborn>=0.11.0
scikit-learn>=1.0.0
```

## 🛠 Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/your-username/house-price-prediction.git
   cd house-price-prediction
   ```

2. **Create a virtual environment:**
   ```bash
   python -m venv .venv
   
   # On Windows:
   .venv\Scripts\activate
   
   # On macOS/Linux:
   source .venv/bin/activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

## 📊 Dataset

### Recommended Kaggle Datasets

1. **[House Prices: Advanced Regression Techniques](https://www.kaggle.com/c/house-prices-advanced-regression-techniques)**
   - Target: `SalePrice`
   - Features: 79 explanatory variables
   - Difficulty: Intermediate

2. **[California Housing Prices](https://www.kaggle.com/datasets/camnugent/california-housing-prices)**
   - Target: `median_house_value`
   - Features: 10 numerical features
   - Difficulty: Beginner-friendly

3. **[USA Housing](https://www.kaggle.com/datasets/vedavyasv/usa-housing)**
   - Target: `Price`
   - Features: 7 features
   - Difficulty: Beginner

### Dataset Structure
Your dataset should be a CSV file with:
- **Numerical features**: Square footage, number of rooms, year built, etc.
- **Categorical features**: Neighborhood, house style, condition, etc.
- **Target variable**: House price (any column name works)

## 🎯 Quick Start

### Basic Usage

```python
from house_price_predictor import HousePricePredictor

# Initialize the predictor
predictor = HousePricePredictor()

# Load your dataset
data = predictor.load_data('your_dataset.csv', target_column='SalePrice')

# Explore the data
predictor.explore_data()

# Preprocess the data
X_train, X_test, y_train, y_test = predictor.preprocess_data()

# Train models and compare performance
results, best_model = predictor.train_models(X_train, y_train, X_test, y_test)

# Visualize results
predictor.visualize_results(results, y_test, best_model)

# Make predictions for new houses
new_house = {
    'OverallQual': 8,
    'GrLivArea': 2000,
    'GarageCars': 2,
    'YearBuilt': 2010
}
predicted_price = predictor.predict_price(new_house)
print(f"Predicted price: ${predicted_price:,.2f}")
```

### Running the Demo

```bash
python House_Price_Prediction.py
```

This will run the model with synthetic data to demonstrate all features.

## 📈 Model Performance Metrics

The model provides comprehensive evaluation metrics:

- **R² Score**: Coefficient of determination (higher is better, max = 1.0)
- **RMSE**: Root Mean Square Error (lower is better)
- **MAE**: Mean Absolute Error (lower is better)
- **Cross-Validation**: 5-fold CV for robust performance estimation

### Expected Performance
- **Good Model**: R² > 0.8, RMSE < 20% of mean price
- **Excellent Model**: R² > 0.9, RMSE < 15% of mean price

## 🔧 Customization

### Adding New Features

```python
# In the preprocess_data method, add feature engineering:
def create_new_features(self, X):
    # Example: Price per square foot proxy
    if 'GrLivArea' in X.columns and 'SalePrice' in self.data.columns:
        X['Age'] = 2024 - X['YearBuilt']
    
    # Total rooms
    if 'BedroomAbvGr' in X.columns and 'FullBath' in X.columns:
        X['TotalRooms'] = X['BedroomAbvGr'] + X['FullBath']
    
    return X
```

### Hyperparameter Tuning

```python
from sklearn.model_selection import GridSearchCV

# Example for Ridge regression
param_grid = {'alpha': [0.1, 1.0, 10.0, 100.0]}
grid_search = GridSearchCV(Ridge(), param_grid, cv=5)
grid_search.fit(X_train, y_train)
best_alpha = grid_search.best_params_['alpha']
```

## 📊 Visualization Examples

The model generates several visualizations:

1. **Data Distribution**: Target variable distribution and log-transform
2. **Missing Data Heatmap**: Visual representation of missing values
3. **Correlation Analysis**: Top features correlated with price
4. **Model Comparison**: Performance comparison across different models
5. **Prediction Plots**: Actual vs Predicted scatter plots
6. **Residuals Analysis**: Error distribution and patterns
7. **Feature Importance**: Most influential features

## 🔍 Troubleshooting

### Common Issues

1. **ModuleNotFoundError**: Install missing packages with `pip install package_name`

2. **FileNotFoundError**: Check your dataset path
   ```python
   import os
   print(os.getcwd())  # Check current directory
   print(os.listdir('.'))  # List files in current directory
   ```

3. **Memory Error**: For large datasets, try:
   ```python
   # Load only a sample
   data = pd.read_csv('large_dataset.csv', nrows=10000)
   ```

4. **Poor Model Performance**:
   - Check for data leakage
   - Add more relevant features
   - Handle outliers
   - Try feature engineering

### Performance Tips

- **Large Datasets**: Use `chunksize` parameter in `pd.read_csv()`
- **Categorical Variables**: Consider using `pd.get_dummies()` instead of LabelEncoder for better results
- **Outliers**: Use IQR method or Z-score to identify and handle outliers

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

### Contribution Ideas
- Add support for more regression models (XGBoost, Random Forest)
- Implement automated feature engineering
- Add model interpretation tools (SHAP values)
- Create web interface with Flask/Streamlit
- Add support for time series price prediction

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **Kaggle** for providing excellent datasets
- **Scikit-learn** for machine learning tools
- **Pandas** and **NumPy** for data manipulation
- **Matplotlib** and **Seaborn** for visualizations

## 📞 Contact

- **Sudeeksha** - srsudeeksha@gmail.com
- **Project Link**: https://github.com/srsudeeksha/Python_Internship_CodeX/tree/main/CodeX_House_Price_Prediction

## 📚 Further Reading

- [Linear Regression Documentation](https://scikit-learn.org/stable/modules/linear_model.html)
- [Feature Engineering Guide](https://www.kaggle.com/learn/feature-engineering)
- [Cross-Validation Best Practices](https://scikit-learn.org/stable/modules/cross_validation.html)
- [Handling Missing Data](https://pandas.pydata.org/docs/user_guide/missing_data.html)

---

⭐ **Star this repository if you found it helpful!**

*Happy house price predicting! 🏡📊*
