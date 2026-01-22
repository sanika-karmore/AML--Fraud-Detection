"""
Rule-Based AML Detection Module
Implements compliance rules for suspicious transaction detection
"""

import pandas as pd
import numpy as np
from typing import Dict, List
import yaml


class AMLRuleEngine:
    """
    Rule-based AML detection system
    Implements banking compliance rules for fraud detection
    """
    
    def __init__(self, config_path: str = "config/config.yaml"):
        """Initialize with AML rules from configuration"""
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.rules = self.config['aml_rules']
        self.rule_weights = self.rules['rule_weights']
    
    def rule_amount_anomaly(self, df: pd.DataFrame,
                           amount_col: str = 'TransactionAmt',
                           customer_id: str = 'card1') -> pd.DataFrame:
        """
        Rule 1: Transaction amount exceeds X times customer average
        
        Banking Insight: Large deviations from normal spending indicate 
        potential account takeover or money laundering
        
        Args:
            df: Input DataFrame with transaction data
            amount_col: Transaction amount column
            customer_id: Customer identifier
            
        Returns:
            DataFrame with amount anomaly flag and score
        """
        df = df.copy()
        
        threshold_multiplier = self.rules['amount_threshold_multiplier']
        large_txn_threshold = self.rules['large_transaction_threshold']
        
        # Amount exceeds customer average by threshold
        df['rule_amount_deviation'] = (
            df['amount_deviation'] > threshold_multiplier
        ).astype(int)
        
        # Large absolute amount
        df['rule_large_amount'] = (
            df[amount_col] > large_txn_threshold
        ).astype(int)
        
        # Combined amount anomaly flag
        df['rule_amount_anomaly'] = (
            (df['rule_amount_deviation'] == 1) | 
            (df['rule_large_amount'] == 1)
        ).astype(int)
        
        return df
    
    def rule_velocity_breach(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Rule 2: Transaction velocity exceeds thresholds
        
        Banking Insight: Rapid succession of transactions indicates
        potential automated fraud or money mule activity
        
        Args:
            df: Input DataFrame with velocity features
            
        Returns:
            DataFrame with velocity breach flags
        """
        df = df.copy()
        
        # Already calculated in feature engineering
        # Consolidate into single rule flag
        df['rule_velocity'] = (
            (df['velocity_breach_hourly'] == 1) | 
            (df['velocity_breach_daily'] == 1)
        ).astype(int)
        
        return df
    
    def rule_round_number(self, df: pd.DataFrame,
                         amount_col: str = 'TransactionAmt') -> pd.DataFrame:
        """
        Rule 3: Round number transactions
        
        Banking Insight: Money launderers often use round numbers
        (e.g., $10,000, $50,000) to simplify accounting
        
        Args:
            df: Input DataFrame
            amount_col: Transaction amount column
            
        Returns:
            DataFrame with round number flag
        """
        df = df.copy()
        
        round_numbers = self.rules['round_number_pattern']
        
        # Exact round number matches
        df['rule_round_exact'] = df[amount_col].isin(round_numbers).astype(int)
        
        # Near-round numbers (within 5%)
        df['rule_round_near'] = 0
        for round_num in round_numbers:
            tolerance = round_num * 0.05
            df['rule_round_near'] = (
                df['rule_round_near'] | 
                ((df[amount_col] >= round_num - tolerance) & 
                 (df[amount_col] <= round_num + tolerance))
            ).astype(int)
        
        # Combined round number rule
        df['rule_round_number'] = (
            (df['rule_round_exact'] == 1) | 
            (df['rule_round_near'] == 1)
        ).astype(int)
        
        return df
    
    def rule_geographic_risk(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Rule 4: Geographic anomalies and high-risk locations
        
        Banking Insight: Cross-border transactions and operations in
        high-risk jurisdictions require enhanced due diligence
        
        Args:
            df: Input DataFrame with geographic features
            
        Returns:
            DataFrame with geographic risk flags
        """
        df = df.copy()
        
        # International transaction
        df['rule_international'] = df.get('is_international', 0)
        
        # High-risk country
        df['rule_high_risk_country'] = df.get('is_high_risk_country', 0)
        
        # Multiple countries in short time (if available)
        df['rule_country_hopping'] = (
            df.get('unique_country_count', 0) > 2
        ).astype(int)
        
        # Combined geographic risk
        df['rule_geographic'] = (
            (df['rule_international'] == 1) | 
            (df['rule_high_risk_country'] == 1) | 
            (df['rule_country_hopping'] == 1)
        ).astype(int)
        
        return df
    
    def rule_device_anomaly(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Rule 5: Device-based anomalies
        
        Banking Insight: New devices or multiple devices may indicate
        account compromise or identity theft
        
        Args:
            df: Input DataFrame with device features
            
        Returns:
            DataFrame with device anomaly flags
        """
        df = df.copy()
        
        # New device
        df['rule_new_device'] = df.get('is_new_device', 0)
        
        # Multiple devices
        df['rule_multiple_devices'] = df.get('multiple_devices', 0)
        
        # Combined device rule
        df['rule_device'] = (
            (df['rule_new_device'] == 1) | 
            (df['rule_multiple_devices'] == 1)
        ).astype(int)
        
        return df
    
    def rule_time_anomaly(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Rule 6: Temporal anomalies
        
        Banking Insight: Transactions during unusual hours or patterns
        inconsistent with customer behavior raise red flags
        
        Args:
            df: Input DataFrame with temporal features
            
        Returns:
            DataFrame with time anomaly flags
        """
        df = df.copy()
        
        # Unusual hour (midnight to 6 AM)
        df['rule_unusual_hour'] = df.get('is_unusual_hour', 0)
        
        # Not customer's favorite hour
        df['rule_unusual_time_pattern'] = (
            df.get('is_favorite_hour', 1) == 0
        ).astype(int)
        
        # Combined time rule
        df['rule_time'] = (
            (df['rule_unusual_hour'] == 1) & 
            (df['rule_unusual_time_pattern'] == 1)
        ).astype(int)
        
        return df
    
    def rule_merchant_risk(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Rule 7: High-risk merchant
        
        Banking Insight: Merchants with high fraud rates require
        additional scrutiny
        
        Args:
            df: Input DataFrame with merchant features
            
        Returns:
            DataFrame with merchant risk flag
        """
        df = df.copy()
        
        # High-risk merchant flag (already computed in features)
        df['rule_merchant'] = df.get('is_high_risk_merchant', 0)
        
        return df
    
    def apply_all_rules(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Apply all AML rules to transactions
        
        Args:
            df: Input DataFrame with all features
            
        Returns:
            DataFrame with all rule flags
        """
        print("\n" + "="*60)
        print("APPLYING AML RULES")
        print("="*60)
        
        # Apply each rule
        df = self.rule_amount_anomaly(df)
        print("✓ Rule 1: Amount Anomaly")
        
        df = self.rule_velocity_breach(df)
        print("✓ Rule 2: Velocity Breach")
        
        df = self.rule_round_number(df)
        print("✓ Rule 3: Round Number Pattern")
        
        df = self.rule_geographic_risk(df)
        print("✓ Rule 4: Geographic Risk")
        
        df = self.rule_device_anomaly(df)
        print("✓ Rule 5: Device Anomaly")
        
        df = self.rule_time_anomaly(df)
        print("✓ Rule 6: Time Anomaly")
        
        df = self.rule_merchant_risk(df)
        print("✓ Rule 7: Merchant Risk")
        
        return df
    
    def calculate_rule_score(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate weighted rule-based risk score
        
        Args:
            df: DataFrame with all rule flags
            
        Returns:
            DataFrame with rule_score (0-1)
        """
        df = df.copy()
        
        # Weighted rule score
        df['rule_score'] = (
            df['rule_amount_anomaly'] * self.rule_weights['amount_anomaly'] +
            df['rule_velocity'] * self.rule_weights['velocity_breach'] +
            df['rule_geographic'] * self.rule_weights['geographic_risk'] +
            df['rule_round_number'] * self.rule_weights['round_number'] +
            df['rule_device'] * self.rule_weights['device_anomaly'] +
            df['rule_time'] * self.rule_weights['time_anomaly']
        )
        
        # Add merchant risk if available
        if 'rule_merchant' in df.columns:
            # Adjust weights to include merchant (scale down others slightly)
            df['rule_score'] = df['rule_score'] * 0.9 + df['rule_merchant'] * 0.1
        
        # Normalize to 0-1
        df['rule_score'] = df['rule_score'].clip(0, 1)
        
        # Count total rules triggered
        rule_columns = [col for col in df.columns if col.startswith('rule_') and col != 'rule_score']
        df['total_rules_triggered'] = df[rule_columns].sum(axis=1)
        
        print("\n" + "="*60)
        print("RULE SCORING COMPLETE")
        print("="*60)
        print(f"Average rule score: {df['rule_score'].mean():.4f}")
        print(f"High-risk (score > 0.5): {(df['rule_score'] > 0.5).sum():,} transactions")
        print(f"Average rules triggered: {df['total_rules_triggered'].mean():.2f}")
        
        return df
    
    def get_rule_explanation(self, row: pd.Series) -> Dict[str, any]:
        """
        Generate human-readable explanation for why transaction was flagged
        
        Args:
            row: Single transaction row
            
        Returns:
            Dictionary with rule explanations
        """
        explanations = []
        
        if row.get('rule_amount_anomaly', 0) == 1:
            explanations.append(f"Amount ${row.get('TransactionAmt', 0):.2f} is "
                              f"{row.get('amount_deviation', 0):.1f}× customer average")
        
        if row.get('rule_velocity', 0) == 1:
            explanations.append(f"High transaction velocity: "
                              f"{row.get('txn_per_hour', 0)} txns/hour, "
                              f"{row.get('txn_per_day', 0)} txns/day")
        
        if row.get('rule_round_number', 0) == 1:
            explanations.append("Round number transaction pattern")
        
        if row.get('rule_geographic', 0) == 1:
            explanations.append("Geographic risk: cross-border or high-risk location")
        
        if row.get('rule_device', 0) == 1:
            explanations.append("Device anomaly: new device or multiple devices")
        
        if row.get('rule_time', 0) == 1:
            explanations.append(f"Unusual transaction time: "
                              f"{row.get('transaction_hour', 0)}:00")
        
        if row.get('rule_merchant', 0) == 1:
            explanations.append("High-risk merchant")
        
        return {
            'rule_score': row.get('rule_score', 0),
            'rules_triggered': row.get('total_rules_triggered', 0),
            'explanations': explanations
        }
    
    def generate_rule_summary_report(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Generate summary report of rule performance
        
        Args:
            df: DataFrame with rules applied
            
        Returns:
            Summary DataFrame
        """
        rule_columns = [
            'rule_amount_anomaly', 'rule_velocity', 'rule_round_number',
            'rule_geographic', 'rule_device', 'rule_time', 'rule_merchant'
        ]
        
        summary_data = []
        
        for rule in rule_columns:
            if rule in df.columns:
                triggered_count = df[rule].sum()
                triggered_pct = (triggered_count / len(df)) * 100
                
                # If fraud labels available
                if 'isFraud' in df.columns:
                    fraud_rate = df[df[rule] == 1]['isFraud'].mean() * 100
                else:
                    fraud_rate = None
                
                summary_data.append({
                    'Rule': rule.replace('rule_', '').replace('_', ' ').title(),
                    'Triggered': triggered_count,
                    'Percentage': f"{triggered_pct:.2f}%",
                    'Fraud Rate': f"{fraud_rate:.2f}%" if fraud_rate is not None else "N/A"
                })
        
        summary_df = pd.DataFrame(summary_data)
        
        print("\n" + "="*60)
        print("RULE SUMMARY REPORT")
        print("="*60)
        print(summary_df.to_string(index=False))
        
        return summary_df


if __name__ == "__main__":
    print("AML Rule-Based Detection Engine")
    print("="*60)
    print("Implements 7 core compliance rules:")
    print("  1. Amount Anomaly - Large or unusual transaction amounts")
    print("  2. Velocity Breach - Rapid transaction frequency")
    print("  3. Round Number - Money laundering patterns")
    print("  4. Geographic Risk - Cross-border and high-risk locations")
    print("  5. Device Anomaly - New or multiple devices")
    print("  6. Time Anomaly - Unusual transaction times")
    print("  7. Merchant Risk - High-risk merchant exposure")
    print("\nBanking Compliance: ML supports but does not replace rules")
