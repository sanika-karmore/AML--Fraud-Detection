"""
Risk Scoring System for AML Fraud Detection
Combines rule-based scores, ML predictions, and anomaly detection
"""

import pandas as pd
import numpy as np
from typing import Dict, List
import yaml


class RiskScoringEngine:
    """
    Comprehensive risk scoring system combining multiple signals
    """
    
    def __init__(self, config_path: str = "config/config.yaml"):
        """Initialize with scoring configuration"""
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.scoring_config = self.config['risk_scoring']
        self.weights = self.scoring_config['weights']
        self.thresholds = self.scoring_config['thresholds']
    
    def calculate_composite_score(self, df: pd.DataFrame,
                                  rule_score_col: str = 'rule_score',
                                  ml_prob_col: str = 'ml_fraud_probability',
                                  anomaly_score_col: str = 'anomaly_score') -> pd.DataFrame:
        """
        Calculate composite risk score from multiple components
        
        Formula: Final Risk Score = 
            0.4 × Rule Score + 
            0.4 × ML Probability + 
            0.2 × Anomaly Score
        
        Args:
            df: DataFrame with component scores
            rule_score_col: Rule-based score column
            ml_prob_col: ML probability column
            anomaly_score_col: Anomaly score column
            
        Returns:
            DataFrame with final risk scores
        """
        df = df.copy()
        
        print("\n" + "="*60)
        print("CALCULATING COMPOSITE RISK SCORES")
        print("="*60)
        
        # Ensure all component scores exist and are normalized to 0-1
        if rule_score_col not in df.columns:
            print(f"Warning: {rule_score_col} not found, using 0")
            df[rule_score_col] = 0
        
        if ml_prob_col not in df.columns:
            print(f"Warning: {ml_prob_col} not found, using 0")
            df[ml_prob_col] = 0
        
        if anomaly_score_col not in df.columns:
            print(f"Warning: {anomaly_score_col} not found, using 0")
            df[anomaly_score_col] = 0
        
        # Normalize scores to 0-1
        df[rule_score_col] = df[rule_score_col].clip(0, 1)
        df[ml_prob_col] = df[ml_prob_col].clip(0, 1)
        df[anomaly_score_col] = df[anomaly_score_col].clip(0, 1)
        
        # Calculate weighted composite score
        df['risk_score'] = (
            self.weights['rule_score'] * df[rule_score_col] +
            self.weights['ml_probability'] * df[ml_prob_col] +
            self.weights['anomaly_score'] * df[anomaly_score_col]
        )
        
        # Ensure final score is 0-1
        df['risk_score'] = df['risk_score'].clip(0, 1)
        
        print(f"✓ Component weights:")
        print(f"  - Rule Score: {self.weights['rule_score']}")
        print(f"  - ML Probability: {self.weights['ml_probability']}")
        print(f"  - Anomaly Score: {self.weights['anomaly_score']}")
        
        return df
    
    def assign_risk_categories(self, df: pd.DataFrame,
                               score_col: str = 'risk_score') -> pd.DataFrame:
        """
        Assign risk categories based on thresholds
        
        Categories:
        - High Risk: Score > 0.7
        - Medium Risk: Score 0.4 - 0.7
        - Low Risk: Score < 0.4
        
        Args:
            df: DataFrame with risk scores
            score_col: Risk score column
            
        Returns:
            DataFrame with risk categories
        """
        df = df.copy()
        
        # Risk category
        df['risk_category'] = pd.cut(
            df[score_col],
            bins=[0, self.thresholds['medium_risk'], 
                  self.thresholds['high_risk'], 1.0],
            labels=['Low Risk', 'Medium Risk', 'High Risk'],
            include_lowest=True
        )
        
        # Numeric risk level
        df['risk_level'] = df['risk_category'].map({
            'Low Risk': 1,
            'Medium Risk': 2,
            'High Risk': 3
        })
        
        # Risk flags
        df['is_high_risk'] = (df[score_col] > self.thresholds['high_risk']).astype(int)
        df['is_medium_risk'] = (
            (df[score_col] >= self.thresholds['medium_risk']) & 
            (df[score_col] <= self.thresholds['high_risk'])
        ).astype(int)
        df['is_low_risk'] = (df[score_col] < self.thresholds['medium_risk']).astype(int)
        
        # Distribution summary
        risk_distribution = df['risk_category'].value_counts()
        
        print("\n" + "="*60)
        print("RISK CATEGORY DISTRIBUTION")
        print("="*60)
        for category, count in risk_distribution.items():
            pct = (count / len(df)) * 100
            print(f"{category:15s}: {count:,} ({pct:.2f}%)")
        
        return df
    
    def calculate_customer_risk_profile(self, df: pd.DataFrame,
                                       customer_id: str = 'card1') -> pd.DataFrame:
        """
        Calculate customer-level risk profiles
        
        Args:
            df: Transaction-level DataFrame with risk scores
            customer_id: Customer identifier
            
        Returns:
            Customer-level risk profile DataFrame
        """
        customer_risk = df.groupby(customer_id).agg({
            'risk_score': ['mean', 'max', 'std'],
            'is_high_risk': 'sum',
            'is_medium_risk': 'sum',
            'is_low_risk': 'sum',
            'TransactionAmt': ['count', 'sum', 'mean'] if 'TransactionAmt' in df.columns else 'count'
        }).reset_index()
        
        customer_risk.columns = [
            customer_id,
            'avg_risk_score',
            'max_risk_score',
            'risk_score_std',
            'high_risk_txn_count',
            'medium_risk_txn_count',
            'low_risk_txn_count',
            'total_transactions',
            'total_amount',
            'avg_amount'
        ]
        
        # Customer risk category (based on average score)
        customer_risk['customer_risk_category'] = pd.cut(
            customer_risk['avg_risk_score'],
            bins=[0, self.thresholds['medium_risk'], 
                  self.thresholds['high_risk'], 1.0],
            labels=['Low Risk', 'Medium Risk', 'High Risk'],
            include_lowest=True
        )
        
        # High-risk transaction rate
        customer_risk['high_risk_txn_rate'] = (
            customer_risk['high_risk_txn_count'] / customer_risk['total_transactions']
        )
        
        print(f"\n✓ Customer risk profiles calculated for {len(customer_risk):,} customers")
        
        return customer_risk
    
    def prioritize_alerts(self, df: pd.DataFrame,
                         score_col: str = 'risk_score',
                         top_n: int = 100) -> pd.DataFrame:
        """
        Prioritize top high-risk transactions for investigation
        
        Args:
            df: DataFrame with risk scores
            score_col: Risk score column
            top_n: Number of top alerts to return
            
        Returns:
            Top priority alerts DataFrame
        """
        # Filter high-risk only
        high_risk = df[df[score_col] > self.thresholds['high_risk']].copy()
        
        # Sort by risk score descending
        high_risk = high_risk.sort_values(score_col, ascending=False)
        
        # Add priority rank
        high_risk['priority_rank'] = range(1, len(high_risk) + 1)
        
        # Top N
        top_alerts = high_risk.head(top_n)
        
        print(f"\n✓ {len(high_risk):,} high-risk alerts identified")
        print(f"✓ Top {top_n} prioritized for investigation")
        
        return top_alerts
    
    def generate_risk_explanation(self, row: pd.Series) -> str:
        """
        Generate human-readable risk explanation
        
        Args:
            row: Single transaction row with scores
            
        Returns:
            Explanation string
        """
        explanation_parts = []
        
        # Risk category
        risk_cat = row.get('risk_category', 'Unknown')
        risk_score = row.get('risk_score', 0)
        
        explanation_parts.append(f"Risk Score: {risk_score:.3f} ({risk_cat})")
        
        # Component breakdown
        rule_score = row.get('rule_score', 0)
        ml_prob = row.get('ml_fraud_probability', 0)
        anomaly_score = row.get('anomaly_score', 0)
        
        explanation_parts.append(f"Components: Rule={rule_score:.2f}, ML={ml_prob:.2f}, Anomaly={anomaly_score:.2f}")
        
        # Key risk factors
        risk_factors = []
        
        if row.get('rule_amount_anomaly', 0) == 1:
            risk_factors.append("Amount anomaly")
        
        if row.get('rule_velocity', 0) == 1:
            risk_factors.append("High velocity")
        
        if row.get('rule_geographic', 0) == 1:
            risk_factors.append("Geographic risk")
        
        if row.get('is_new_device', 0) == 1:
            risk_factors.append("New device")
        
        if risk_factors:
            explanation_parts.append(f"Risk Factors: {', '.join(risk_factors)}")
        
        return " | ".join(explanation_parts)
    
    def generate_risk_summary_report(self, df: pd.DataFrame) -> Dict:
        """
        Generate comprehensive risk summary statistics
        
        Args:
            df: DataFrame with risk scores
            
        Returns:
            Dictionary with summary statistics
        """
        print("\n" + "="*60)
        print("RISK SCORING SUMMARY REPORT")
        print("="*60)
        
        summary = {
            'total_transactions': len(df),
            'avg_risk_score': df['risk_score'].mean(),
            'median_risk_score': df['risk_score'].median(),
            'high_risk_count': (df['risk_score'] > self.thresholds['high_risk']).sum(),
            'medium_risk_count': (
                (df['risk_score'] >= self.thresholds['medium_risk']) & 
                (df['risk_score'] <= self.thresholds['high_risk'])
            ).sum(),
            'low_risk_count': (df['risk_score'] < self.thresholds['medium_risk']).sum(),
            'high_risk_pct': (df['risk_score'] > self.thresholds['high_risk']).mean() * 100,
            'alert_volume': (df['risk_score'] > self.thresholds['high_risk']).sum(),
        }
        
        # If fraud labels available
        if 'isFraud' in df.columns:
            summary['actual_fraud_count'] = df['isFraud'].sum()
            summary['fraud_rate'] = df['isFraud'].mean() * 100
            
            # Detection rate
            high_risk_fraud = df[df['risk_score'] > self.thresholds['high_risk']]['isFraud'].sum()
            summary['detection_rate'] = (high_risk_fraud / summary['actual_fraud_count']) * 100 if summary['actual_fraud_count'] > 0 else 0
            
            # Precision
            high_risk_df = df[df['risk_score'] > self.thresholds['high_risk']]
            if len(high_risk_df) > 0:
                summary['alert_precision'] = high_risk_df['isFraud'].mean() * 100
            else:
                summary['alert_precision'] = 0
        
        print(f"Total Transactions: {summary['total_transactions']:,}")
        print(f"Average Risk Score: {summary['avg_risk_score']:.4f}")
        print(f"\nRisk Distribution:")
        print(f"  High Risk:   {summary['high_risk_count']:,} ({summary['high_risk_pct']:.2f}%)")
        print(f"  Medium Risk: {summary['medium_risk_count']:,}")
        print(f"  Low Risk:    {summary['low_risk_count']:,}")
        
        if 'detection_rate' in summary:
            print(f"\nPerformance Metrics:")
            print(f"  Actual Fraud: {summary['actual_fraud_count']:,} ({summary['fraud_rate']:.2f}%)")
            print(f"  Detection Rate: {summary['detection_rate']:.2f}%")
            print(f"  Alert Precision: {summary['alert_precision']:.2f}%")
        
        return summary
    
    def export_alerts(self, df: pd.DataFrame, 
                     output_path: str = "reports/high_risk_alerts.csv"):
        """
        Export high-risk alerts to CSV
        
        Args:
            df: DataFrame with risk scores
            output_path: Output file path
        """
        import os
        
        # Filter high-risk
        high_risk = df[df['risk_score'] > self.thresholds['high_risk']].copy()
        
        # Select key columns
        export_cols = [
            'TransactionID', 'card1', 'TransactionAmt', 
            'risk_score', 'risk_category',
            'rule_score', 'ml_fraud_probability', 'anomaly_score',
            'transaction_datetime'
        ]
        
        export_cols = [col for col in export_cols if col in high_risk.columns]
        
        high_risk_export = high_risk[export_cols].sort_values('risk_score', ascending=False)
        
        # Create directory
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # Export
        high_risk_export.to_csv(output_path, index=False)
        
        print(f"\n✓ Exported {len(high_risk_export):,} high-risk alerts to {output_path}")


if __name__ == "__main__":
    print("AML Risk Scoring Engine")
    print("="*60)
    print("Composite Risk Score Formula:")
    print("  Final Risk Score = ")
    print("    0.4 × Rule-Based Score +")
    print("    0.4 × ML Fraud Probability +")
    print("    0.2 × Anomaly Detection Score")
    print("\nRisk Categories:")
    print("  • High Risk:   Score > 0.7")
    print("  • Medium Risk: Score 0.4 - 0.7")
    print("  • Low Risk:    Score < 0.4")
