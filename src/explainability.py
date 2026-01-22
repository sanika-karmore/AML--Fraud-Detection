"""
Explainability Module for AML Fraud Detection
Provides interpretability for ML model predictions
"""

import pandas as pd
import numpy as np
import shap
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Dict
import yaml


class ModelExplainer:
    """
    Model explainability and interpretability for AML systems
    """
    
    def __init__(self, config_path: str = "config/config.yaml"):
        """Initialize with configuration"""
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.explainability_config = self.config['explainability']
        self.shap_explainer = None
        self.shap_values = None
    
    def get_feature_importance(self, model, feature_names: List[str], 
                              top_n: int = 20) -> pd.DataFrame:
        """
        Extract feature importance from tree-based models
        
        Args:
            model: Trained model with feature_importances_
            feature_names: List of feature names
            top_n: Number of top features to return
            
        Returns:
            DataFrame with feature importance
        """
        if not hasattr(model, 'feature_importances_'):
            print("Model does not have feature importance attributes")
            return pd.DataFrame()
        
        importance_df = pd.DataFrame({
            'Feature': feature_names,
            'Importance': model.feature_importances_
        }).sort_values('Importance', ascending=False).head(top_n)
        
        print("\n" + "="*60)
        print(f"TOP {top_n} MOST IMPORTANT FEATURES")
        print("="*60)
        print(importance_df.to_string(index=False))
        
        return importance_df
    
    def plot_feature_importance(self, importance_df: pd.DataFrame, 
                               title: str = "Feature Importance"):
        """
        Visualize feature importance
        
        Args:
            importance_df: DataFrame with feature importance
            title: Plot title
        """
        plt.figure(figsize=(10, 8))
        
        sns.barplot(data=importance_df, 
                   y='Feature', 
                   x='Importance',
                   palette='viridis')
        
        plt.title(title, fontsize=16, fontweight='bold')
        plt.xlabel('Importance Score', fontsize=12)
        plt.ylabel('Feature', fontsize=12)
        plt.tight_layout()
        
        return plt.gcf()
    
    def initialize_shap_explainer(self, model, X_sample: np.ndarray):
        """
        Initialize SHAP explainer for the model
        
        Args:
            model: Trained model
            X_sample: Sample data for background distribution
        """
        print("\n" + "="*60)
        print("INITIALIZING SHAP EXPLAINER")
        print("="*60)
        
        shap_samples = self.explainability_config['shap_samples']
        
        # Sample data if too large
        if len(X_sample) > shap_samples:
            sample_indices = np.random.choice(len(X_sample), shap_samples, replace=False)
            X_sample = X_sample[sample_indices]
        
        try:
            # Tree-based explainer for tree models
            self.shap_explainer = shap.TreeExplainer(model)
            print("✓ Using TreeExplainer")
        except:
            try:
                # Kernel explainer for other models
                self.shap_explainer = shap.KernelExplainer(
                    model.predict_proba, 
                    X_sample
                )
                print("✓ Using KernelExplainer")
            except Exception as e:
                print(f"✗ Failed to initialize SHAP explainer: {e}")
                self.shap_explainer = None
    
    def calculate_shap_values(self, X: np.ndarray) -> np.ndarray:
        """
        Calculate SHAP values for predictions
        
        Args:
            X: Feature matrix
            
        Returns:
            SHAP values array
        """
        if self.shap_explainer is None:
            print("SHAP explainer not initialized")
            return None
        
        print("\nCalculating SHAP values...")
        
        # Calculate SHAP values
        self.shap_values = self.shap_explainer.shap_values(X)
        
        # For binary classification, get positive class SHAP values
        if isinstance(self.shap_values, list):
            self.shap_values = self.shap_values[1]
        
        print("✓ SHAP values calculated")
        
        return self.shap_values
    
    def plot_shap_summary(self, X: pd.DataFrame, feature_names: List[str]):
        """
        Create SHAP summary plot
        
        Args:
            X: Feature DataFrame
            feature_names: List of feature names
        """
        if self.shap_values is None:
            print("SHAP values not calculated")
            return
        
        plt.figure(figsize=(12, 8))
        
        shap.summary_plot(
            self.shap_values, 
            X, 
            feature_names=feature_names,
            show=False,
            max_display=self.explainability_config['top_features']
        )
        
        plt.title("SHAP Feature Impact on Fraud Prediction", 
                 fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        return plt.gcf()
    
    def plot_shap_waterfall(self, X: pd.DataFrame, 
                           transaction_idx: int,
                           feature_names: List[str]):
        """
        Create SHAP waterfall plot for single transaction
        
        Args:
            X: Feature DataFrame
            transaction_idx: Index of transaction to explain
            feature_names: List of feature names
        """
        if self.shap_values is None:
            print("SHAP values not calculated")
            return
        
        # Create explanation object for waterfall plot
        shap_explanation = shap.Explanation(
            values=self.shap_values[transaction_idx],
            base_values=self.shap_explainer.expected_value if hasattr(self.shap_explainer, 'expected_value') else 0,
            data=X.iloc[transaction_idx].values,
            feature_names=feature_names
        )
        
        plt.figure(figsize=(10, 8))
        shap.waterfall_plot(shap_explanation, show=False)
        plt.title(f"Prediction Explanation - Transaction #{transaction_idx}", 
                 fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        return plt.gcf()
    
    def explain_single_prediction(self, model, X_row: pd.DataFrame,
                                 feature_names: List[str]) -> Dict:
        """
        Explain a single transaction prediction
        
        Args:
            model: Trained model
            X_row: Single transaction features
            feature_names: List of feature names
            
        Returns:
            Dictionary with explanation
        """
        # Prediction
        if hasattr(model, 'predict_proba'):
            fraud_prob = model.predict_proba(X_row)[0][1]
        else:
            fraud_prob = model.predict(X_row)[0]
        
        # Feature values
        feature_values = X_row.iloc[0].to_dict()
        
        # Top contributing features
        if hasattr(model, 'feature_importances_'):
            importance = model.feature_importances_
            top_features_idx = np.argsort(importance)[-5:][::-1]
            top_features = [feature_names[i] for i in top_features_idx]
        else:
            top_features = []
        
        explanation = {
            'fraud_probability': fraud_prob,
            'prediction': 'FRAUD' if fraud_prob > 0.5 else 'LEGITIMATE',
            'confidence': abs(fraud_prob - 0.5) * 2,  # 0-1 scale
            'top_features': top_features,
            'feature_values': {feat: feature_values.get(feat, None) for feat in top_features}
        }
        
        return explanation
    
    def generate_prediction_report(self, row: pd.Series,
                                  explanation: Dict) -> str:
        """
        Generate human-readable prediction report
        
        Args:
            row: Transaction row with all data
            explanation: Explanation dictionary
            
        Returns:
            Formatted report string
        """
        report = []
        report.append("="*60)
        report.append("TRANSACTION FRAUD ANALYSIS REPORT")
        report.append("="*60)
        
        # Transaction details
        report.append("\n[TRANSACTION DETAILS]")
        if 'TransactionID' in row:
            report.append(f"Transaction ID: {row['TransactionID']}")
        if 'TransactionAmt' in row:
            report.append(f"Amount: ${row['TransactionAmt']:.2f}")
        if 'transaction_datetime' in row:
            report.append(f"Date/Time: {row['transaction_datetime']}")
        
        # Prediction
        report.append("\n[FRAUD ASSESSMENT]")
        report.append(f"Prediction: {explanation['prediction']}")
        report.append(f"Fraud Probability: {explanation['fraud_probability']:.1%}")
        report.append(f"Confidence: {explanation['confidence']:.1%}")
        
        # Risk scoring
        if 'risk_score' in row:
            report.append(f"Risk Score: {row['risk_score']:.3f}")
            report.append(f"Risk Category: {row.get('risk_category', 'N/A')}")
        
        # Key factors
        report.append("\n[KEY RISK FACTORS]")
        for i, feature in enumerate(explanation['top_features'], 1):
            value = explanation['feature_values'].get(feature, 'N/A')
            report.append(f"{i}. {feature}: {value}")
        
        # Rule triggers
        report.append("\n[AML RULE TRIGGERS]")
        rule_columns = [col for col in row.index if col.startswith('rule_') and row[col] == 1]
        if rule_columns:
            for rule in rule_columns:
                rule_name = rule.replace('rule_', '').replace('_', ' ').title()
                report.append(f"  ✓ {rule_name}")
        else:
            report.append("  No rules triggered")
        
        # Recommendation
        report.append("\n[RECOMMENDATION]")
        if explanation['fraud_probability'] > 0.7:
            report.append("🔴 HIGH PRIORITY: Immediate investigation required")
            report.append("   Action: Flag for compliance review and potential SAR filing")
        elif explanation['fraud_probability'] > 0.4:
            report.append("🟡 MEDIUM PRIORITY: Enhanced monitoring recommended")
            report.append("   Action: Add to watchlist and review customer profile")
        else:
            report.append("🟢 LOW RISK: Standard processing")
            report.append("   Action: No immediate action required")
        
        report.append("\n" + "="*60)
        
        return "\n".join(report)
    
    def create_correlation_heatmap(self, df: pd.DataFrame, 
                                   feature_cols: List[str],
                                   target_col: str = 'isFraud'):
        """
        Create correlation heatmap with target variable
        
        Args:
            df: DataFrame with features and target
            feature_cols: List of feature columns
            target_col: Target variable column
        """
        # Select top correlated features
        correlations = df[feature_cols + [target_col]].corr()[target_col].abs().sort_values(ascending=False)
        top_features = correlations.head(21).index.tolist()  # +1 for target
        top_features.remove(target_col)
        top_features = top_features[:20] + [target_col]
        
        # Create heatmap
        plt.figure(figsize=(12, 10))
        
        correlation_matrix = df[top_features].corr()
        
        sns.heatmap(correlation_matrix, 
                   annot=False, 
                   cmap='coolwarm',
                   center=0,
                   vmin=-1, vmax=1,
                   square=True,
                   linewidths=0.5)
        
        plt.title("Feature Correlation with Fraud", fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        return plt.gcf()
    
    def save_explanations(self, df: pd.DataFrame, 
                         output_path: str = "reports/model_explanations.html"):
        """
        Save all explanation visualizations
        
        Args:
            df: DataFrame with predictions
            output_path: Output HTML file path
        """
        import os
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # Generate HTML report
        html = []
        html.append("<html><head><title>AML Model Explanations</title></head><body>")
        html.append("<h1>AML Fraud Detection - Model Explainability Report</h1>")
        html.append("<p>Generated using SHAP and feature importance analysis</p>")
        html.append("</body></html>")
        
        with open(output_path, 'w') as f:
            f.write("\n".join(html))
        
        print(f"✓ Explanations saved to {output_path}")


if __name__ == "__main__":
    print("AML Model Explainability Module")
    print("="*60)
    print("Provides interpretability for regulatory compliance:")
    print("  • SHAP (SHapley Additive exPlanations)")
    print("  • Feature importance analysis")
    print("  • Individual prediction explanations")
    print("  • Correlation analysis")
    print("\nBanking Requirement: Explainable AI for regulatory transparency")
