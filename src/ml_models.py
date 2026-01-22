"""
Machine Learning Models Module for AML Fraud Detection
Implements supervised and unsupervised learning approaches
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.cluster import DBSCAN
from xgboost import XGBClassifier
from sklearn.metrics import (classification_report, confusion_matrix, 
                            roc_auc_score, precision_recall_curve, 
                            average_precision_score)
import joblib
import yaml
from typing import Tuple, Dict, List
import matplotlib.pyplot as plt
import seaborn as sns


class AMLMLModels:
    """
    Machine Learning models for AML fraud detection
    """
    
    def __init__(self, config_path: str = "config/config.yaml"):
        """Initialize with model configurations"""
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.model_config = self.config['ml_models']
        self.models = {}
        self.scaler = StandardScaler()
        self.feature_names = []
    
    def prepare_data(self, df: pd.DataFrame,
                     target_col: str = 'isFraud',
                     test_size: float = 0.2) -> Tuple:
        """
        Prepare data for ML training
        
        Args:
            df: Input DataFrame with features
            target_col: Target variable column
            test_size: Test set proportion
            
        Returns:
            X_train, X_test, y_train, y_test, feature_names
        """
        print("\n" + "="*60)
        print("PREPARING DATA FOR ML")
        print("="*60)
        
        # Separate features and target
        if target_col in df.columns:
            y = df[target_col]
            X = df.drop(columns=[target_col])
        else:
            raise ValueError(f"Target column '{target_col}' not found")
        
        # Remove non-numeric and identifier columns
        exclude_cols = ['transaction_datetime', 'card1', 'card2', 'card3', 
                       'card4', 'card5', 'card6', 'TransactionID', 'date',
                       'hour_window', 'device_first_seen', 'transaction_date']
        
        X = X.select_dtypes(include=[np.number])
        X = X.drop(columns=[col for col in exclude_cols if col in X.columns], errors='ignore')
        
        # Handle any remaining missing values
        X = X.fillna(X.median())
        
        self.feature_names = X.columns.tolist()
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42, stratify=y
        )
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        print(f"✓ Training set: {X_train.shape[0]:,} samples")
        print(f"✓ Test set: {X_test.shape[0]:,} samples")
        print(f"✓ Features: {len(self.feature_names)}")
        print(f"✓ Fraud rate (train): {y_train.mean()*100:.2f}%")
        print(f"✓ Fraud rate (test): {y_test.mean()*100:.2f}%")
        
        return X_train_scaled, X_test_scaled, y_train, y_test, X_train, X_test
    
    def train_logistic_regression(self, X_train, y_train) -> LogisticRegression:
        """
        Train Logistic Regression (baseline model)
        
        Args:
            X_train: Training features
            y_train: Training labels
            
        Returns:
            Trained model
        """
        print("\n" + "-"*60)
        print("Training Logistic Regression (Baseline)")
        print("-"*60)
        
        params = self.model_config['logistic_regression']
        
        model = LogisticRegression(**params)
        model.fit(X_train, y_train)
        
        self.models['logistic_regression'] = model
        
        print("✓ Logistic Regression trained")
        
        return model
    
    def train_random_forest(self, X_train, y_train) -> RandomForestClassifier:
        """
        Train Random Forest classifier
        
        Args:
            X_train: Training features
            y_train: Training labels
            
        Returns:
            Trained model
        """
        print("\n" + "-"*60)
        print("Training Random Forest")
        print("-"*60)
        
        params = self.model_config['random_forest']
        
        model = RandomForestClassifier(**params, n_jobs=-1)
        model.fit(X_train, y_train)
        
        self.models['random_forest'] = model
        
        print("✓ Random Forest trained")
        
        return model
    
    def train_xgboost(self, X_train, y_train) -> XGBClassifier:
        """
        Train XGBoost classifier
        
        Args:
            X_train: Training features
            y_train: Training labels
            
        Returns:
            Trained model
        """
        print("\n" + "-"*60)
        print("Training XGBoost")
        print("-"*60)
        
        params = self.model_config['xgboost']
        
        model = XGBClassifier(**params, n_jobs=-1, eval_metric='logloss')
        model.fit(X_train, y_train)
        
        self.models['xgboost'] = model
        
        print("✓ XGBoost trained")
        
        return model
    
    def train_isolation_forest(self, X_train) -> IsolationForest:
        """
        Train Isolation Forest for anomaly detection (unsupervised)
        
        Args:
            X_train: Training features
            
        Returns:
            Trained model
        """
        print("\n" + "-"*60)
        print("Training Isolation Forest (Anomaly Detection)")
        print("-"*60)
        
        params = self.model_config['isolation_forest']
        
        model = IsolationForest(**params, n_jobs=-1)
        model.fit(X_train)
        
        self.models['isolation_forest'] = model
        
        print("✓ Isolation Forest trained")
        
        return model
    
    def train_dbscan(self, X_train) -> DBSCAN:
        """
        Train DBSCAN for clustering suspicious patterns
        
        Args:
            X_train: Training features
            
        Returns:
            Trained model
        """
        print("\n" + "-"*60)
        print("Training DBSCAN (Clustering)")
        print("-"*60)
        
        params = self.model_config['dbscan']
        
        model = DBSCAN(**params, n_jobs=-1)
        model.fit(X_train)
        
        self.models['dbscan'] = model
        
        n_clusters = len(set(model.labels_)) - (1 if -1 in model.labels_ else 0)
        n_noise = list(model.labels_).count(-1)
        
        print(f"✓ DBSCAN clustering complete")
        print(f"  - Clusters found: {n_clusters}")
        print(f"  - Noise points: {n_noise} ({n_noise/len(X_train)*100:.2f}%)")
        
        return model
    
    def train_all_models(self, X_train, y_train, X_train_original):
        """
        Train all ML models
        
        Args:
            X_train: Scaled training features
            y_train: Training labels
            X_train_original: Original (unscaled) training features
        """
        print("\n" + "="*60)
        print("TRAINING ALL ML MODELS")
        print("="*60)
        
        # Supervised models
        self.train_logistic_regression(X_train, y_train)
        self.train_random_forest(X_train_original, y_train)  # Tree models don't need scaling
        self.train_xgboost(X_train_original, y_train)
        
        # Unsupervised models
        self.train_isolation_forest(X_train)
        self.train_dbscan(X_train)
        
        print("\n" + "="*60)
        print("ALL MODELS TRAINED SUCCESSFULLY")
        print("="*60)
    
    def evaluate_model(self, model, X_test, y_test, model_name: str) -> Dict:
        """
        Evaluate model performance
        
        Args:
            model: Trained model
            X_test: Test features
            y_test: Test labels
            model_name: Name of the model
            
        Returns:
            Dictionary with evaluation metrics
        """
        print("\n" + "-"*60)
        print(f"Evaluating {model_name}")
        print("-"*60)
        
        # Predictions
        y_pred = model.predict(X_test)
        
        # Probability predictions (if available)
        if hasattr(model, 'predict_proba'):
            y_pred_proba = model.predict_proba(X_test)[:, 1]
        else:
            y_pred_proba = y_pred
        
        # Metrics
        from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
        
        metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred),
            'recall': recall_score(y_test, y_pred),
            'f1_score': f1_score(y_test, y_pred),
            'roc_auc': roc_auc_score(y_test, y_pred_proba),
            'avg_precision': average_precision_score(y_test, y_pred_proba)
        }
        
        print(f"Accuracy:  {metrics['accuracy']:.4f}")
        print(f"Precision: {metrics['precision']:.4f}  ← CRITICAL for banks (minimize false positives)")
        print(f"Recall:    {metrics['recall']:.4f}  ← Catch actual fraud")
        print(f"F1-Score:  {metrics['f1_score']:.4f}")
        print(f"ROC-AUC:   {metrics['roc_auc']:.4f}")
        
        # Confusion Matrix
        cm = confusion_matrix(y_test, y_pred)
        print("\nConfusion Matrix:")
        print(f"  TN: {cm[0,0]:,}  |  FP: {cm[0,1]:,}")
        print(f"  FN: {cm[1,0]:,}  |  TP: {cm[1,1]:,}")
        
        return metrics
    
    def evaluate_all_models(self, X_test, y_test, X_test_original) -> pd.DataFrame:
        """
        Evaluate all trained models
        
        Args:
            X_test: Scaled test features
            y_test: Test labels
            X_test_original: Original (unscaled) test features
            
        Returns:
            DataFrame with model comparison
        """
        print("\n" + "="*60)
        print("EVALUATING ALL MODELS")
        print("="*60)
        
        results = []
        
        # Logistic Regression
        if 'logistic_regression' in self.models:
            metrics = self.evaluate_model(
                self.models['logistic_regression'], 
                X_test, y_test, 
                "Logistic Regression"
            )
            results.append({'Model': 'Logistic Regression', **metrics})
        
        # Random Forest
        if 'random_forest' in self.models:
            metrics = self.evaluate_model(
                self.models['random_forest'], 
                X_test_original, y_test, 
                "Random Forest"
            )
            results.append({'Model': 'Random Forest', **metrics})
        
        # XGBoost
        if 'xgboost' in self.models:
            metrics = self.evaluate_model(
                self.models['xgboost'], 
                X_test_original, y_test, 
                "XGBoost"
            )
            results.append({'Model': 'XGBoost', **metrics})
        
        # Results DataFrame
        results_df = pd.DataFrame(results)
        
        print("\n" + "="*60)
        print("MODEL COMPARISON")
        print("="*60)
        print(results_df.to_string(index=False))
        
        return results_df
    
    def predict_fraud_probability(self, X: pd.DataFrame, model_name: str = 'xgboost') -> np.ndarray:
        """
        Predict fraud probability for new transactions
        
        Args:
            X: Features DataFrame
            model_name: Model to use for prediction
            
        Returns:
            Array of fraud probabilities
        """
        if model_name not in self.models:
            raise ValueError(f"Model '{model_name}' not trained")
        
        model = self.models[model_name]
        
        # Scale if needed
        if model_name in ['logistic_regression', 'isolation_forest']:
            X_scaled = self.scaler.transform(X)
            if hasattr(model, 'predict_proba'):
                return model.predict_proba(X_scaled)[:, 1]
            else:
                # For Isolation Forest, convert scores to probabilities
                scores = model.score_samples(X_scaled)
                proba = 1 / (1 + np.exp(scores))  # Sigmoid transformation
                return proba
        else:
            if hasattr(model, 'predict_proba'):
                return model.predict_proba(X)[:, 1]
            else:
                return model.predict(X)
    
    def get_feature_importance(self, model_name: str = 'random_forest', top_n: int = 15) -> pd.DataFrame:
        """
        Get feature importance from tree-based models
        
        Args:
            model_name: Model to extract importance from
            top_n: Number of top features to return
            
        Returns:
            DataFrame with feature importance
        """
        if model_name not in self.models:
            raise ValueError(f"Model '{model_name}' not trained")
        
        model = self.models[model_name]
        
        if not hasattr(model, 'feature_importances_'):
            print(f"Model '{model_name}' does not have feature importances")
            return pd.DataFrame()
        
        importance_df = pd.DataFrame({
            'Feature': self.feature_names,
            'Importance': model.feature_importances_
        }).sort_values('Importance', ascending=False).head(top_n)
        
        print(f"\n{'='*60}")
        print(f"TOP {top_n} FEATURES - {model_name.upper()}")
        print('='*60)
        print(importance_df.to_string(index=False))
        
        return importance_df
    
    def save_models(self, path: str = "models/"):
        """
        Save all trained models
        
        Args:
            path: Directory to save models
        """
        import os
        os.makedirs(path, exist_ok=True)
        
        for model_name, model in self.models.items():
            file_path = os.path.join(path, f"{model_name}.pkl")
            joblib.dump(model, file_path)
            print(f"✓ Saved {model_name} to {file_path}")
        
        # Save scaler
        scaler_path = os.path.join(path, "scaler.pkl")
        joblib.dump(self.scaler, scaler_path)
        print(f"✓ Saved scaler to {scaler_path}")
    
    def load_models(self, path: str = "models/"):
        """
        Load saved models
        
        Args:
            path: Directory containing saved models
        """
        import os
        
        for model_file in os.listdir(path):
            if model_file.endswith('.pkl') and model_file != 'scaler.pkl':
                model_name = model_file.replace('.pkl', '')
                file_path = os.path.join(path, model_file)
                self.models[model_name] = joblib.load(file_path)
                print(f"✓ Loaded {model_name}")
        
        # Load scaler
        scaler_path = os.path.join(path, "scaler.pkl")
        if os.path.exists(scaler_path):
            self.scaler = joblib.load(scaler_path)
            print("✓ Loaded scaler")


if __name__ == "__main__":
    print("AML Machine Learning Module")
    print("="*60)
    print("Supervised Learning:")
    print("  • Logistic Regression (baseline)")
    print("  • Random Forest")
    print("  • XGBoost")
    print("\nUnsupervised Learning:")
    print("  • Isolation Forest (anomaly detection)")
    print("  • DBSCAN (clustering)")
    print("\nBanking Focus: Precision prioritized to minimize false positives")
