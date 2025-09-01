# House Price Prediction using Linear Regression
# Complete pipeline for data preprocessing, training, and evaluation

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.feature_selection import SelectKBest, f_regression
import warnings
warnings.filterwarnings('ignore')

class HousePricePredictor:
    def __init__(self):
        self.model = None
        self.scaler = StandardScaler()
        self.label_encoders = {}
        self.feature_names = None
        self.target_column = None
        
    def load_data(self, file_path, target_column='SalePrice'):
        """
        Load dataset from CSV file
        For Kaggle datasets, common target columns are: 'SalePrice', 'price', 'median_house_value'
        """
        try:
            self.data = pd.read_csv('D:\CodeX\synthetic_house_data.csv')  # Use the argument, not hardcoded path
            self.target_column = target_column
            print(f"Dataset loaded successfully!")
            print(f"Shape: {self.data.shape}")
            print(f"\nColumns: {list(self.data.columns)}")
            return self.data
        except FileNotFoundError:
            print(f"File {file_path} not found. Please ensure the dataset is in the correct path.")
            return None
    
    def explore_data(self):
        """Perform exploratory data analysis"""
        if self.data is None:
            print("Please load data first using load_data()")
            return
            
        print("=== DATA EXPLORATION ===")
        print(f"\nDataset Info:")
        print(self.data.info())
        
        print(f"\nMissing Values:")
        missing_data = self.data.isnull().sum()
        missing_data = missing_data[missing_data > 0].sort_values(ascending=False)
        print(missing_data)
        
        print(f"\nTarget Variable Statistics:")
        if self.target_column in self.data.columns:
            print(self.data[self.target_column].describe())
        
        # Visualizations
        plt.figure(figsize=(15, 10))
        
        # Target distribution
        plt.subplot(2, 3, 1)
        plt.hist(self.data[self.target_column], bins=50, alpha=0.7)
        plt.title(f'{self.target_column} Distribution')
        plt.xlabel(self.target_column)
        
        # Log-transformed target (often more normal)
        plt.subplot(2, 3, 2)
        plt.hist(np.log1p(self.data[self.target_column]), bins=50, alpha=0.7)
        plt.title(f'Log({self.target_column}) Distribution')
        plt.xlabel(f'Log({self.target_column})')
        
        # Missing data heatmap
        plt.subplot(2, 3, 3)
        missing_matrix = self.data.isnull()
        sns.heatmap(missing_matrix.iloc[:, :20], yticklabels=False, cbar=True)
        plt.title('Missing Data Pattern (First 20 columns)')
        
        # Correlation with target (for numeric columns)
        numeric_cols = self.data.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 1:
            correlations = self.data[numeric_cols].corr()[self.target_column].sort_values(ascending=False)
            
            plt.subplot(2, 3, 4)
            top_corr = correlations[1:11]  # Exclude target itself, top 10
            top_corr.plot(kind='barh')
            plt.title('Top 10 Features Correlated with Price')
            plt.xlabel('Correlation Coefficient')
        
        plt.tight_layout()
        plt.show()
        
        return missing_data
    
    def preprocess_data(self, test_size=0.2, random_state=42):
        """
        Comprehensive data preprocessing pipeline
        """
        if self.data is None:
            print("Please load data first using load_data()")
            return
        
        print("=== DATA PREPROCESSING ===")
        df = self.data.copy()
        
        # Remove rows where target is missing
        if self.target_column in df.columns:
            initial_rows = len(df)
            df = df.dropna(subset=[self.target_column])
            print(f"Removed {initial_rows - len(df)} rows with missing target values")
        
        # Separate features and target
        X = df.drop(columns=[self.target_column])
        y = df[self.target_column]
        
        # Handle missing values
        print("\nHandling missing values...")
        
        # Numeric columns: fill with median
        numeric_cols = X.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            if X[col].isnull().sum() > 0:
                X[col].fillna(X[col].median(), inplace=True)
        
        # Categorical columns: fill with mode or 'Unknown'
        categorical_cols = X.select_dtypes(include=['object']).columns
        for col in categorical_cols:
            if X[col].isnull().sum() > 0:
                mode_value = X[col].mode()
                if len(mode_value) > 0:
                    X[col].fillna(mode_value[0], inplace=True)
                else:
                    X[col].fillna('Unknown', inplace=True)
        
        # Encode categorical variables
        print("Encoding categorical variables...")
        for col in categorical_cols:
            if X[col].dtype == 'object':
                le = LabelEncoder()
                X[col] = le.fit_transform(X[col].astype(str))
                self.label_encoders[col] = le
        
        # Feature selection - remove low variance and highly correlated features
        print("Performing feature selection...")
        
        # Remove constant features
        constant_features = X.columns[X.nunique() <= 1]
        if len(constant_features) > 0:
            X = X.drop(columns=constant_features)
            print(f"Removed {len(constant_features)} constant features")
        
        # Remove highly correlated features
        correlation_matrix = X.corr().abs()
        upper_triangle = correlation_matrix.where(
            np.triu(np.ones(correlation_matrix.shape), k=1).astype(bool)
        )
        high_corr_features = [column for column in upper_triangle.columns 
                            if any(upper_triangle[column] > 0.95)]
        if len(high_corr_features) > 0:
            X = X.drop(columns=high_corr_features)
            print(f"Removed {len(high_corr_features)} highly correlated features")
        
        # Store feature names
        self.feature_names = X.columns.tolist()
        
        # Split the data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )
        
        # Scale features
        print("Scaling features...")
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        print(f"Final dataset shape: {X_train_scaled.shape}")
        print(f"Number of features: {X_train_scaled.shape[1]}")
        
        return X_train_scaled, X_test_scaled, y_train, y_test
    
    def train_models(self, X_train, y_train, X_test, y_test):
        """
        Train multiple linear regression models and compare performance
        """
        print("=== MODEL TRAINING ===")
        
        models = {
            'Linear Regression': LinearRegression(),
            'Ridge Regression': Ridge(alpha=1.0),
            'Lasso Regression': Lasso(alpha=1.0)
        }
        
        results = {}
        
        for name, model in models.items():
            print(f"\nTraining {name}...")
            
            # Train the model
            model.fit(X_train, y_train)
            
            # Make predictions
            y_train_pred = model.predict(X_train)
            y_test_pred = model.predict(X_test)
            
            # Calculate metrics
            train_mse = mean_squared_error(y_train, y_train_pred)
            test_mse = mean_squared_error(y_test, y_test_pred)
            train_r2 = r2_score(y_train, y_train_pred)
            test_r2 = r2_score(y_test, y_test_pred)
            test_mae = mean_absolute_error(y_test, y_test_pred)
            
            # Cross-validation score
            cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='r2')
            
            results[name] = {
                'model': model,
                'train_mse': train_mse,
                'test_mse': test_mse,
                'train_r2': train_r2,
                'test_r2': test_r2,
                'test_mae': test_mae,
                'cv_mean': cv_scores.mean(),
                'cv_std': cv_scores.std(),
                'y_test_pred': y_test_pred
            }
            
            print(f"  Train R²: {train_r2:.4f}")
            print(f"  Test R²: {test_r2:.4f}")
            print(f"  Test RMSE: {np.sqrt(test_mse):.2f}")
            print(f"  Test MAE: {test_mae:.2f}")
            print(f"  CV R² Score: {cv_scores.mean():.4f} (+/- {cv_scores.std()*2:.4f})")
        
        # Select best model based on cross-validation score
        best_model_name = max(results.keys(), key=lambda k: results[k]['cv_mean'])
        self.model = results[best_model_name]['model']
        
        print(f"\n=== BEST MODEL: {best_model_name} ===")
        
        return results, best_model_name
    
    def visualize_results(self, results, y_test, best_model_name):
        """
        Create visualizations for model performance
        """
        plt.figure(figsize=(15, 12))
        
        # Model comparison
        plt.subplot(2, 3, 1)
        model_names = list(results.keys())
        test_r2_scores = [results[name]['test_r2'] for name in model_names]
        bars = plt.bar(model_names, test_r2_scores)
        plt.title('Model Comparison - R² Score')
        plt.ylabel('R² Score')
        plt.xticks(rotation=45)
        for bar, score in zip(bars, test_r2_scores):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f'{score:.3f}', ha='center', va='bottom')
        
        # Best model predictions vs actual
        best_predictions = results[best_model_name]['y_test_pred']
        plt.subplot(2, 3, 2)
        plt.scatter(y_test, best_predictions, alpha=0.6)
        plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
        plt.xlabel('Actual Price')
        plt.ylabel('Predicted Price')
        plt.title(f'{best_model_name} - Predictions vs Actual')
        
        # Residuals plot
        residuals = y_test - best_predictions
        plt.subplot(2, 3, 3)
        plt.scatter(best_predictions, residuals, alpha=0.6)
        plt.axhline(y=0, color='r', linestyle='--')
        plt.xlabel('Predicted Price')
        plt.ylabel('Residuals')
        plt.title('Residuals Plot')
        
        # Feature importance (for linear regression)
        if hasattr(self.model, 'coef_') and self.feature_names:
            plt.subplot(2, 3, 4)
            feature_importance = np.abs(self.model.coef_)
            top_features_idx = np.argsort(feature_importance)[-10:]
            top_features = [self.feature_names[i] for i in top_features_idx]
            top_importance = feature_importance[top_features_idx]
            
            plt.barh(range(len(top_features)), top_importance)
            plt.yticks(range(len(top_features)), top_features)
            plt.xlabel('Absolute Coefficient Value')
            plt.title('Top 10 Most Important Features')
        
        # Error distribution
        plt.subplot(2, 3, 5)
        plt.hist(residuals, bins=30, alpha=0.7)
        plt.xlabel('Residuals')
        plt.ylabel('Frequency')
        plt.title('Residuals Distribution')
        
        # Cross-validation scores
        plt.subplot(2, 3, 6)
        cv_means = [results[name]['cv_mean'] for name in model_names]
        cv_stds = [results[name]['cv_std'] for name in model_names]
        plt.errorbar(range(len(model_names)), cv_means, yerr=cv_stds, fmt='o')
        plt.xticks(range(len(model_names)), model_names, rotation=45)
        plt.ylabel('Cross-Validation R² Score')
        plt.title('Cross-Validation Performance')
        
        plt.tight_layout()
        plt.show()
    
    def predict_price(self, features_dict):
        """
        Make prediction for new house with given features
        """
        if self.model is None:
            print("Please train the model first")
            return None
        
        # Create a sample with all features set to median values
        sample_data = pd.DataFrame([features_dict])
        
        # Apply the same preprocessing
        for col in sample_data.select_dtypes(include=['object']).columns:
            if col in self.label_encoders:
                try:
                    sample_data[col] = self.label_encoders[col].transform(sample_data[col].astype(str))
                except ValueError:
                    # Handle unseen categories
                    sample_data[col] = 0
        
        # Ensure all required features are present
        for feature in self.feature_names:
            if feature not in sample_data.columns:
                sample_data[feature] = 0  # Default value
        
        # Reorder columns to match training data
        sample_data = sample_data[self.feature_names]
        
        # Scale the features
        sample_scaled = self.scaler.transform(sample_data)
        
        # Make prediction
        prediction = self.model.predict(sample_scaled)[0]
        
        return prediction

# Example usage and demonstration
def main():
    # Initialize the predictor
    predictor = HousePricePredictor()
    
    print("=== HOUSE PRICE PREDICTION MODEL ===")
    print("\nTo use this model with your Kaggle dataset:")
    print("1. Download a dataset from Kaggle (e.g., 'House Prices: Advanced Regression Techniques')")
    print("2. Update the file path and target column name below")
    print("3. Run the complete pipeline")
    
    # Example with synthetic data (replace with your Kaggle dataset)
    print("\n=== CREATING SYNTHETIC DATASET FOR DEMONSTRATION ===")
    
    # Generate synthetic house data
    np.random.seed(42)
    n_samples = 1000
    
    synthetic_data = {
        'OverallQual': np.random.randint(1, 11, n_samples),
        'GrLivArea': np.random.randint(500, 5000, n_samples),
        'GarageCars': np.random.randint(0, 4, n_samples),
        'TotalBsmtSF': np.random.randint(0, 2000, n_samples),
        'FullBath': np.random.randint(1, 4, n_samples),
        'YearBuilt': np.random.randint(1900, 2020, n_samples),
        'Neighborhood': np.random.choice(['Downtown', 'Suburb', 'Rural'], n_samples),
        'HouseStyle': np.random.choice(['1Story', '2Story', 'Split'], n_samples),
    }
    
    # Create synthetic target (price) with realistic relationships
    synthetic_data['SalePrice'] = (
        synthetic_data['OverallQual'] * 15000 +
        synthetic_data['GrLivArea'] * 50 +
        synthetic_data['GarageCars'] * 10000 +
        synthetic_data['TotalBsmtSF'] * 20 +
        synthetic_data['FullBath'] * 5000 +
        (synthetic_data['YearBuilt'] - 1900) * 100 +
        np.random.normal(0, 20000, n_samples)
    )
    
    synthetic_df = pd.DataFrame(synthetic_data)
    synthetic_df.to_csv('synthetic_house_data.csv', index=False)
    
    # Load and process the data
    data = predictor.load_data('synthetic_house_data.csv', target_column='SalePrice')
    
    if data is not None:
        # Explore the data
        missing_data = predictor.explore_data()
        
        # Preprocess the data
        X_train, X_test, y_train, y_test = predictor.preprocess_data()
        
        if X_train is not None:
            # Train models
            results, best_model = predictor.train_models(X_train, y_train, X_test, y_test)
            
            # Visualize results
            predictor.visualize_results(results, y_test, best_model)
            
            # Example prediction
            print("\n=== MAKING PREDICTIONS ===")
            example_house = {
                'OverallQual': 8,
                'GrLivArea': 2000,
                'GarageCars': 2,
                'TotalBsmtSF': 1200,
                'FullBath': 2,
                'YearBuilt': 2010,
                'Neighborhood': 'Suburb',
                'HouseStyle': '2Story'
            }
            
            predicted_price = predictor.predict_price(example_house)
            print(f"Predicted price for example house: ${predicted_price:,.2f}")
            
            print(f"\nExample house features:")
            for key, value in example_house.items():
                print(f"  {key}: {value}")

if __name__ == "__main__":
    main()

# Instructions for using with Kaggle datasets:
"""
1. KAGGLE SETUP:
   - Go to kaggle.com and download a house prices dataset
   - Popular datasets:
     * "House Prices: Advanced Regression Techniques" (target: SalePrice)
     * "California Housing Prices" (target: median_house_value)
     * "USA Housing" (target: Price)

2. UPDATE THE CODE:
   - Replace 'synthetic_house_data.csv' with your dataset path
   - Update target_column parameter in load_data()
   - Modify feature engineering if needed

3. FEATURE ENGINEERING IDEAS:
   - Create new features: price per square foot, age of house
   - Combine features: total rooms = bedrooms + bathrooms + living rooms
   - Polynomial features for non-linear relationships
   - Log transform skewed features

4. MODEL IMPROVEMENTS:
   - Try different regularization parameters for Ridge/Lasso
   - Add polynomial features
   - Use feature selection techniques
   - Consider ensemble methods (Random Forest, Gradient Boosting)
   - Handle outliers more carefully

5. EVALUATION METRICS:
   - RMSE: Root Mean Square Error
   - MAE: Mean Absolute Error
   - R²: Coefficient of determination
   - MAPE: Mean Absolute Percentage Error
"""