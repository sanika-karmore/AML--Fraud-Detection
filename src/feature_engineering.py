"""
Feature Engineering Module for AML Fraud Detection
Creates transaction-level and customer-level features
"""

import pandas as pd
import numpy as np
from typing import List, Dict
import yaml


class FeatureEngineer:
    """
    Feature engineering for AML fraud detection
    """
    
    def __init__(self, config_path: str = "config/config.yaml"):
        """Initialize with configuration"""
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
    
    def create_amount_features(self, df: pd.DataFrame, 
                               amount_col: str = 'TransactionAmt',
                               customer_id: str = 'card1') -> pd.DataFrame:
        """
        Create amount-based features
        
        Args:
            df: Input DataFrame
            amount_col: Transaction amount column
            customer_id: Customer identifier column
            
        Returns:
            DataFrame with amount features
        """
        df = df.copy()
        
        # Customer-level statistics
        customer_stats = df.groupby(customer_id)[amount_col].agg([
            ('customer_avg_amount', 'mean'),
            ('customer_median_amount', 'median'),
            ('customer_std_amount', 'std'),
            ('customer_max_amount', 'max'),
            ('customer_min_amount', 'min')
        ]).reset_index()
        
        df = df.merge(customer_stats, on=customer_id, how='left')
        
        # Amount deviation from customer average
        df['amount_deviation'] = df[amount_col] / (df['customer_avg_amount'] + 1)
        
        # Amount percentile within customer history
        df['amount_vs_max'] = df[amount_col] / (df['customer_max_amount'] + 1)
        
        # Amount z-score
        df['amount_zscore'] = (df[amount_col] - df['customer_avg_amount']) / (df['customer_std_amount'] + 1)
        
        # Round number detection
        round_numbers = self.config['aml_rules']['round_number_pattern']
        df['is_round_number'] = df[amount_col].isin(round_numbers).astype(int)
        
        # Amount range categories
        df['amount_category'] = pd.cut(df[amount_col], 
                                       bins=[0, 50, 200, 500, 1000, float('inf')],
                                       labels=['micro', 'small', 'medium', 'large', 'very_large'])
        
        print(f"✓ Created amount-based features")
        
        return df
    
    def create_velocity_features(self, df: pd.DataFrame,
                                 customer_id: str = 'card1',
                                 time_col: str = 'transaction_datetime') -> pd.DataFrame:
        """
        Create velocity (transaction frequency) features
        
        Args:
            df: Input DataFrame
            customer_id: Customer identifier
            time_col: Timestamp column
            
        Returns:
            DataFrame with velocity features
        """
        df = df.copy()
        df = df.sort_values([customer_id, time_col])
        
        # Transactions per customer
        customer_txn_count = df.groupby(customer_id).size().reset_index(name='customer_total_transactions')
        df = df.merge(customer_txn_count, on=customer_id, how='left')
        
        # Time since last transaction
        df['time_since_last_txn'] = df.groupby(customer_id)[time_col].diff().dt.total_seconds() / 3600
        df['time_since_last_txn'].fillna(0, inplace=True)
        
        # Transactions per hour (rolling window)
        df['hour_window'] = df[time_col].dt.floor('H')
        hourly_velocity = df.groupby([customer_id, 'hour_window']).size().reset_index(name='txn_per_hour')
        df = df.merge(hourly_velocity, on=[customer_id, 'hour_window'], how='left')
        
        # Transactions per day
        df['date'] = df[time_col].dt.date
        daily_velocity = df.groupby([customer_id, 'date']).size().reset_index(name='txn_per_day')
        df = df.merge(daily_velocity, on=[customer_id, 'date'], how='left')
        
        # Velocity breach flags
        max_per_hour = self.config['aml_rules']['max_transactions_per_hour']
        max_per_day = self.config['aml_rules']['max_transactions_per_day']
        
        df['velocity_breach_hourly'] = (df['txn_per_hour'] > max_per_hour).astype(int)
        df['velocity_breach_daily'] = (df['txn_per_day'] > max_per_day).astype(int)
        
        print(f"✓ Created velocity features")
        
        return df
    
    def create_customer_features(self, df: pd.DataFrame,
                                 customer_id: str = 'card1',
                                 amount_col: str = 'TransactionAmt') -> pd.DataFrame:
        """
        Create customer-level aggregated features
        
        Args:
            df: Input DataFrame
            customer_id: Customer identifier
            amount_col: Transaction amount column
            
        Returns:
            DataFrame with customer features
        """
        df = df.copy()
        
        # Customer lifetime statistics
        customer_features = df.groupby(customer_id).agg({
            amount_col: ['count', 'mean', 'sum', 'std', 'min', 'max'],
            'is_weekend': 'mean',
            'is_unusual_hour': 'mean',
            'transaction_hour': lambda x: x.mode()[0] if not x.mode().empty else 0
        }).reset_index()
        
        customer_features.columns = [
            customer_id,
            'customer_txn_count',
            'customer_avg_amount',
            'customer_total_amount',
            'customer_amount_std',
            'customer_min_amount',
            'customer_max_amount',
            'customer_weekend_pct',
            'customer_unusual_hour_pct',
            'customer_favorite_hour'
        ]
        
        df = df.merge(customer_features, on=customer_id, how='left')
        
        # Customer behavior consistency
        df['is_favorite_hour'] = (df['transaction_hour'] == df['customer_favorite_hour']).astype(int)
        
        print(f"✓ Created customer-level features")
        
        return df
    
    def create_merchant_features(self, df: pd.DataFrame,
                                 merchant_col: str = 'P_emaildomain') -> pd.DataFrame:
        """
        Create merchant/email domain risk features
        
        Args:
            df: Input DataFrame
            merchant_col: Merchant or email domain column
            
        Returns:
            DataFrame with merchant features
        """
        df = df.copy()
        
        if merchant_col in df.columns:
            # Merchant transaction statistics
            merchant_stats = df.groupby(merchant_col).agg({
                'TransactionAmt': ['count', 'mean'],
                'isFraud': 'mean' if 'isFraud' in df.columns else lambda x: 0
            }).reset_index()
            
            merchant_stats.columns = [
                merchant_col,
                'merchant_txn_count',
                'merchant_avg_amount',
                'merchant_fraud_rate'
            ]
            
            df = df.merge(merchant_stats, on=merchant_col, how='left')
            
            # High-risk merchant flag (fraud rate > 5%)
            df['is_high_risk_merchant'] = (df['merchant_fraud_rate'] > 0.05).astype(int)
            
            print(f"✓ Created merchant features")
        
        return df
    
    def create_device_features(self, df: pd.DataFrame,
                              customer_id: str = 'card1',
                              device_col: str = 'DeviceInfo') -> pd.DataFrame:
        """
        Create device-based features
        
        Args:
            df: Input DataFrame
            customer_id: Customer identifier
            device_col: Device information column
            
        Returns:
            DataFrame with device features
        """
        df = df.copy()
        
        if device_col in df.columns:
            # Count unique devices per customer
            customer_devices = df.groupby(customer_id)[device_col].nunique().reset_index()
            customer_devices.columns = [customer_id, 'unique_device_count']
            df = df.merge(customer_devices, on=customer_id, how='left')
            
            # Multiple device flag
            df['multiple_devices'] = (df['unique_device_count'] > 1).astype(int)
            
            # Device first seen (proxy for new device detection)
            device_first_seen = df.groupby([customer_id, device_col])['transaction_datetime'].min().reset_index()
            device_first_seen.columns = [customer_id, device_col, 'device_first_seen']
            df = df.merge(device_first_seen, on=[customer_id, device_col], how='left')
            
            # Days since device first used
            df['device_age_days'] = (df['transaction_datetime'] - df['device_first_seen']).dt.days
            
            # New device flag
            new_device_threshold = self.config['aml_rules']['new_device_threshold_days']
            df['is_new_device'] = (df['device_age_days'] < new_device_threshold).astype(int)
            
            print(f"✓ Created device features")
        
        return df
    
    def create_geographic_features(self, df: pd.DataFrame,
                                   customer_id: str = 'card1') -> pd.DataFrame:
        """
        Create geographic/cross-border features
        
        Args:
            df: Input DataFrame
            customer_id: Customer identifier
            
        Returns:
            DataFrame with geographic features
        """
        df = df.copy()
        
        # Country columns (card issuer vs purchaser)
        if 'card4' in df.columns:  # Card issuer country
            # Count unique countries per customer
            customer_countries = df.groupby(customer_id)['card4'].nunique().reset_index()
            customer_countries.columns = [customer_id, 'unique_country_count']
            df = df.merge(customer_countries, on=customer_id, how='left')
            
            # International transaction flag
            df['is_international'] = (df['unique_country_count'] > 1).astype(int)
        
        # High-risk country detection (if available)
        high_risk_countries = self.config['aml_rules'].get('high_risk_countries', [])
        if 'card4' in df.columns and high_risk_countries:
            df['is_high_risk_country'] = df['card4'].isin(high_risk_countries).astype(int)
        
        print(f"✓ Created geographic features")
        
        return df
    
    def engineer_features(self, df: pd.DataFrame,
                         customer_id: str = 'card1',
                         amount_col: str = 'TransactionAmt') -> pd.DataFrame:
        """
        Complete feature engineering pipeline
        
        Args:
            df: Input DataFrame
            customer_id: Customer identifier
            amount_col: Transaction amount column
            
        Returns:
            DataFrame with all engineered features
        """
        print("\n" + "="*60)
        print("STARTING FEATURE ENGINEERING PIPELINE")
        print("="*60)
        
        initial_features = len(df.columns)
        
        # 1. Amount features
        df = self.create_amount_features(df, amount_col, customer_id)
        
        # 2. Velocity features
        if 'transaction_datetime' in df.columns:
            df = self.create_velocity_features(df, customer_id)
        
        # 3. Customer features
        df = self.create_customer_features(df, customer_id, amount_col)
        
        # 4. Merchant features
        df = self.create_merchant_features(df)
        
        # 5. Device features
        df = self.create_device_features(df, customer_id)
        
        # 6. Geographic features
        df = self.create_geographic_features(df, customer_id)
        
        final_features = len(df.columns)
        new_features = final_features - initial_features
        
        print("\n" + "="*60)
        print("FEATURE ENGINEERING COMPLETE")
        print("="*60)
        print(f"Initial features: {initial_features}")
        print(f"Final features: {final_features}")
        print(f"New features created: {new_features}")
        
        return df
    
    def get_feature_list(self) -> Dict[str, List[str]]:
        """
        Get list of all engineered features by category
        
        Returns:
            Dictionary of feature categories and feature names
        """
        features = {
            'amount_features': [
                'customer_avg_amount', 'customer_median_amount', 'customer_std_amount',
                'amount_deviation', 'amount_vs_max', 'amount_zscore',
                'is_round_number', 'amount_category'
            ],
            'velocity_features': [
                'customer_total_transactions', 'time_since_last_txn',
                'txn_per_hour', 'txn_per_day',
                'velocity_breach_hourly', 'velocity_breach_daily'
            ],
            'customer_features': [
                'customer_weekend_pct', 'customer_unusual_hour_pct',
                'customer_favorite_hour', 'is_favorite_hour'
            ],
            'merchant_features': [
                'merchant_txn_count', 'merchant_avg_amount',
                'merchant_fraud_rate', 'is_high_risk_merchant'
            ],
            'device_features': [
                'unique_device_count', 'multiple_devices',
                'device_age_days', 'is_new_device'
            ],
            'geographic_features': [
                'unique_country_count', 'is_international'
            ]
        }
        
        return features


if __name__ == "__main__":
    print("AML Feature Engineering Module")
    print("="*60)
    print("This module creates advanced features for fraud detection:")
    print("  • Transaction amount patterns")
    print("  • Velocity (frequency) analysis")
    print("  • Customer behavior profiling")
    print("  • Merchant risk scoring")
    print("  • Device anomaly detection")
    print("  • Geographic risk factors")
