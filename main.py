"""
Main execution script for AML Fraud Detection System
End-to-end pipeline from data loading to risk scoring
"""

import numpy as np
import sys

# Add src to path
sys.path.append('src')

from src.data_processing import DataProcessor
from src.feature_engineering import FeatureEngineer
from src.aml_rules import AMLRuleEngine
from src.ml_models import AMLMLModels
from src.risk_scoring import RiskScoringEngine
from src.explainability import ModelExplainer


def run_aml_pipeline(data_file: str = 'train_transaction.csv',
                     sample_size: int = None):
    """
    Run complete AML fraud detection pipeline
    
    Args:
        data_file: Input data file name
        sample_size: Optional sample size for faster testing
    """
    
    print("\n" + "="*70)
    print("AML FRAUD DETECTION SYSTEM")
    print("Bank-Grade Financial Crime Analytics")
    print("="*70)
    
    # ============================================================
    # STEP 1: DATA LOADING AND CLEANING
    # ============================================================
    
    print("\n[STEP 1/7] DATA LOADING AND CLEANING")
    print("-"*70)
    
    processor = DataProcessor()
    df = processor.load_data(data_file)
    
    if df.empty:
        print("\n❌ ERROR: No data loaded. Please download dataset.")
        print("See data/data_dictionary.md for download instructions.")
        return None
    
    # Sample for faster execution (optional)
    if sample_size and len(df) > sample_size:
        print(f"\n⚠ Sampling {sample_size:,} transactions for demo purposes")
        df = df.sample(n=sample_size, random_state=42)
    
    # Clean data
    df_clean = processor.clean_data(df)
    
    # ============================================================
    # STEP 2: FEATURE ENGINEERING
    # ============================================================
    
    print("\n[STEP 2/7] FEATURE ENGINEERING")
    print("-"*70)
    
    engineer = FeatureEngineer()
    df_features = engineer.engineer_features(
        df_clean,
        customer_id='card1',
        amount_col='TransactionAmt'
    )
    
    # ============================================================
    # STEP 3: RULE-BASED AML DETECTION
    # ============================================================
    
    print("\n[STEP 3/7] RULE-BASED AML DETECTION")
    print("-"*70)
    
    rule_engine = AMLRuleEngine()
    df_rules = rule_engine.apply_all_rules(df_features)
    df_rules = rule_engine.calculate_rule_score(df_rules)
    
    # Generate rule summary
    _ = rule_engine.generate_rule_summary_report(df_rules)
    
    # ============================================================
    # STEP 4: MACHINE LEARNING MODELS
    # ============================================================
    
    print("\n[STEP 4/7] MACHINE LEARNING MODELS")
    print("-"*70)
    
    # Check if fraud labels exist
    if 'isFraud' not in df_rules.columns:
        print("⚠ Warning: No fraud labels found. Skipping supervised ML.")
        print("Using unsupervised anomaly detection only.")
        
        ml_models = AMLMLModels()
        # Prepare features for anomaly detection
        X = df_rules.select_dtypes(include=[np.number])
        X = X.fillna(X.median())
        
        # Scale features
        from sklearn.preprocessing import StandardScaler
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Train Isolation Forest
        ml_models.train_isolation_forest(X_scaled)
        
        # Anomaly scores
        iso_forest = ml_models.models['isolation_forest']
        anomaly_scores = iso_forest.score_samples(X_scaled)
        # Convert to 0-1 probability
        df_rules['anomaly_score'] = 1 / (1 + np.exp(anomaly_scores))
        df_rules['ml_fraud_probability'] = 0  # No supervised model
        
    else:
        ml_models = AMLMLModels()
        
        # Prepare data
        X_train, X_test, y_train, y_test, X_train_orig, X_test_orig = ml_models.prepare_data(
            df_rules,
            target_col='isFraud',
            test_size=0.2
        )
        
        # Train all models
        ml_models.train_all_models(X_train, y_train, X_train_orig)
        
        # Evaluate models
        _ = ml_models.evaluate_all_models(X_test, y_test, X_test_orig)
        
        # Feature importance
        _ = ml_models.get_feature_importance('random_forest', top_n=15)
        
        # Predict on full dataset
        X_full = df_rules.select_dtypes(include=[np.number])
        exclude_cols = ['isFraud', 'TransactionID']
        X_full = X_full.drop(columns=[col for col in exclude_cols if col in X_full.columns], errors='ignore')
        X_full = X_full.fillna(X_full.median())
        
        # Get predictions
        df_rules['ml_fraud_probability'] = ml_models.predict_fraud_probability(
            X_full[ml_models.feature_names],
            model_name='xgboost'
        )
        
        # Anomaly scores
        X_scaled = ml_models.scaler.transform(X_full[ml_models.feature_names])
        if 'isolation_forest' in ml_models.models:
            anomaly_scores = ml_models.models['isolation_forest'].score_samples(X_scaled)
            df_rules['anomaly_score'] = 1 / (1 + np.exp(anomaly_scores))
        else:
            df_rules['anomaly_score'] = 0
    
    # ============================================================
    # STEP 5: RISK SCORING
    # ============================================================
    
    print("\n[STEP 5/7] RISK SCORING")
    print("-"*70)
    
    risk_engine = RiskScoringEngine()
    
    # Calculate composite scores
    df_scored = risk_engine.calculate_composite_score(
        df_rules,
        rule_score_col='rule_score',
        ml_prob_col='ml_fraud_probability',
        anomaly_score_col='anomaly_score'
    )
    
    # Assign risk categories
    df_scored = risk_engine.assign_risk_categories(df_scored)
    
    # Generate summary report
    summary = risk_engine.generate_risk_summary_report(df_scored)
    
    # Prioritize alerts
    high_risk_alerts = risk_engine.prioritize_alerts(df_scored, top_n=100)
    
    # ============================================================
    # STEP 6: EXPLAINABILITY
    # ============================================================
    
    print("\n[STEP 6/7] MODEL EXPLAINABILITY")
    print("-"*70)
    
    explainer = ModelExplainer()
    
    # Feature importance (if available)
    if 'random_forest' in ml_models.models:
        _ = explainer.get_feature_importance(
            ml_models.models['random_forest'],
            ml_models.feature_names,
            top_n=15
        )
    
    # Example: Explain highest risk transaction
    if len(high_risk_alerts) > 0:
        top_transaction = high_risk_alerts.iloc[0]
        print("\n" + "="*70)
        print("SAMPLE TRANSACTION EXPLANATION - HIGHEST RISK")
        print("="*70)
        
        explanation = risk_engine.generate_risk_explanation(top_transaction)
        print(explanation)
    
    # ============================================================
    # STEP 7: SAVE RESULTS
    # ============================================================
    
    print("\n[STEP 7/7] SAVING RESULTS")
    print("-"*70)
    
    # Save processed data
    processor.save_processed_data(df_scored, 'aml_processed_data.csv')
    
    # Export high-risk alerts
    risk_engine.export_alerts(df_scored, output_path='reports/high_risk_alerts.csv')
    
    # Save models
    if 'isFraud' in df_rules.columns:
        ml_models.save_models(path='models/')
    
    # ============================================================
    # FINAL SUMMARY
    # ============================================================
    
    print("\n" + "="*70)
    print("AML PIPELINE EXECUTION COMPLETE ✓")
    print("="*70)
    
    print(f"\n📊 RESULTS SUMMARY:")
    print(f"  • Total transactions processed: {len(df_scored):,}")
    print(f"  • High-risk alerts: {summary['high_risk_count']:,}")
    print(f"  • Medium-risk alerts: {summary['medium_risk_count']:,}")
    print(f"  • Average risk score: {summary['avg_risk_score']:.4f}")
    
    if 'detection_rate' in summary:
        print(f"  • Fraud detection rate: {summary['detection_rate']:.2f}%")
        print(f"  • Alert precision: {summary['alert_precision']:.2f}%")
    
    print(f"\n📁 OUTPUT FILES:")
    print(f"  • Processed data: data/processed/aml_processed_data.csv")
    print(f"  • High-risk alerts: reports/high_risk_alerts.csv")
    if 'isFraud' in df_rules.columns:
        print(f"  • Trained models: models/")
    
    print(f"\n📋 NEXT STEPS:")
    print(f"  1. Review high-risk alerts in reports/high_risk_alerts.csv")
    print(f"  2. Investigate top priority cases")
    print(f"  3. Complete SAR reports for confirmed suspicious activity")
    print(f"  4. Create dashboard visualizations")
    print(f"  5. Present findings to stakeholders")
    
    return df_scored, summary, high_risk_alerts


if __name__ == "__main__":
    
    # Run pipeline with sample data
    # For full dataset, remove sample_size parameter
    
    print("\n🏦 Starting AML Fraud Detection System...")
    print("This may take several minutes depending on dataset size.\n")
    
    try:
        df_results, summary, alerts = run_aml_pipeline(
            data_file='train_transaction.csv',
            sample_size=50000  # Use 50K transactions for demo
            # sample_size=None  # Uncomment for full dataset
        )
        
        print("\n✅ SUCCESS: AML pipeline completed successfully!")
        
    except FileNotFoundError:
        print("\n❌ ERROR: Dataset not found!")
        print("\n📥 Please download the dataset:")
        print("   1. Go to: https://www.kaggle.com/c/ieee-fraud-detection/data")
        print("   2. Download train_transaction.csv")
        print("   3. Place in: AML/data/raw/")
        print("\nSee data/data_dictionary.md for detailed instructions.")
        
    except Exception as e:
        print(f"\n❌ ERROR: {str(e)}")
        import traceback
        traceback.print_exc()
