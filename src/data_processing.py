"""
Data Processing Module for AML Fraud Detection
Handles data loading, cleaning, and preprocessing
"""

import pandas as pd
import numpy as np
from typing import Tuple, Optional
import yaml
import os


class DataProcessor:
    """
    Data processing class for AML transaction data
    """
    
    def __init__(self, config_path: str = "config/config.yaml"):
        """Initialize with configuration"""
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.raw_data_path = self.config['paths']['raw_data']
        self.processed_data_path = self.config['paths']['processed_data']
    
    def load_data(self, file_name: str) -> pd.DataFrame:
        """
        Load raw transaction data
        
        Args:
            file_name: Name of the CSV file
            
        Returns:
            DataFrame with raw transactions
        """
        file_path = os.path.join(self.raw_data_path, file_name)
        
        try:
            df = pd.read_csv(file_path)
            print(f"✓ Loaded {len(df):,} transactions from {file_name}")
            return df
        except FileNotFoundError:
            print(f"✗ File not found: {file_path}")
            print("Please download the dataset and place it in the data/raw/ folder")
            return pd.DataFrame()
    
    def handle_missing_values(self, df: pd.DataFrame, strategy: str = 'smart') -> pd.DataFrame:
        """
        Handle missing values intelligently
        
        Args:
            df: Input DataFrame
            strategy: 'smart', 'drop', or 'fill'
            
        Returns:
            DataFrame with handled missing values
        """
        df = df.copy()
        initial_missing = df.isnull().sum().sum()
        
        if strategy == 'smart':
            # Numeric columns: fill with median
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            for col in numeric_cols:
                if df[col].isnull().sum() > 0:
                    df[col].fillna(df[col].median(), inplace=True)
            
            # Categorical columns: fill with mode or 'Unknown'
            categorical_cols = df.select_dtypes(include=['object']).columns
            for col in categorical_cols:
                if df[col].isnull().sum() > 0:
                    mode_value = df[col].mode()[0] if not df[col].mode().empty else 'Unknown'
                    df[col].fillna(mode_value, inplace=True)
        
        elif strategy == 'drop':
            # Drop rows with any missing values
            df = df.dropna()
        
        elif strategy == 'fill':
            # Simple forward fill
            df = df.fillna(method='ffill')
        
        final_missing = df.isnull().sum().sum()
        print(f"✓ Missing values handled: {initial_missing:,} → {final_missing:,}")
        
        return df
    
    def remove_duplicates(self, df: pd.DataFrame, subset: Optional[list] = None) -> pd.DataFrame:
        """
        Remove duplicate transactions
        
        Args:
            df: Input DataFrame
            subset: Columns to check for duplicates
            
        Returns:
            DataFrame without duplicates
        """
        initial_count = len(df)
        df = df.drop_duplicates(subset=subset, keep='first')
        final_count = len(df)
        
        duplicates_removed = initial_count - final_count
        print(f"✓ Duplicates removed: {duplicates_removed:,} ({duplicates_removed/initial_count*100:.2f}%)")
        
        return df
    
    def normalize_amounts(self, df: pd.DataFrame, amount_col: str = 'TransactionAmt') -> pd.DataFrame:
        """
        Normalize transaction amounts
        
        Args:
            df: Input DataFrame
            amount_col: Name of the amount column
            
        Returns:
            DataFrame with normalized amounts
        """
        df = df.copy()
        
        if amount_col in df.columns:
            # Store original
            df[f'{amount_col}_original'] = df[amount_col]
            
            # Log transformation for skewed distribution
            df[f'{amount_col}_log'] = np.log1p(df[amount_col])
            
            # Min-Max normalization
            from sklearn.preprocessing import MinMaxScaler
            scaler = MinMaxScaler()
            df[f'{amount_col}_normalized'] = scaler.fit_transform(df[[amount_col]])
            
            print(f"✓ Amount normalization complete")
            print(f"  - Original range: ${df[amount_col].min():.2f} to ${df[amount_col].max():.2f}")
            print(f"  - Mean: ${df[amount_col].mean():.2f}, Median: ${df[amount_col].median():.2f}")
        
        return df
    
    def parse_timestamps(self, df: pd.DataFrame, time_col: str = 'TransactionDT') -> pd.DataFrame:
        """
        Parse timestamp and extract temporal features
        
        Args:
            df: Input DataFrame
            time_col: Name of timestamp column
            
        Returns:
            DataFrame with temporal features
        """
        df = df.copy()
        
        if time_col in df.columns:
            # If timestamp is in seconds from epoch
            if df[time_col].dtype in ['int64', 'float64']:
                df['transaction_datetime'] = pd.to_datetime(df[time_col], unit='s')
            else:
                df['transaction_datetime'] = pd.to_datetime(df[time_col])
            
            # Extract temporal features
            df['transaction_hour'] = df['transaction_datetime'].dt.hour
            df['transaction_day'] = df['transaction_datetime'].dt.day
            df['transaction_dayofweek'] = df['transaction_datetime'].dt.dayofweek
            df['transaction_month'] = df['transaction_datetime'].dt.month
            df['transaction_year'] = df['transaction_datetime'].dt.year
            
            # Business day flag
            df['is_weekend'] = df['transaction_dayofweek'].isin([5, 6]).astype(int)
            
            # Unusual hours (midnight to 6 AM)
            unusual_hours = self.config['aml_rules']['unusual_hour_range']
            df['is_unusual_hour'] = ((df['transaction_hour'] >= unusual_hours[0]) & 
                                      (df['transaction_hour'] < unusual_hours[1])).astype(int)
            
            print(f"✓ Timestamp parsing complete")
            print(f"  - Date range: {df['transaction_datetime'].min()} to {df['transaction_datetime'].max()}")
        
        return df
    
    def clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Complete data cleaning pipeline
        
        Args:
            df: Raw DataFrame
            
        Returns:
            Cleaned DataFrame
        """
        print("\n" + "="*60)
        print("STARTING DATA CLEANING PIPELINE")
        print("="*60)
        
        # 1. Remove duplicates
        df = self.remove_duplicates(df)
        
        # 2. Handle missing values
        df = self.handle_missing_values(df, strategy='smart')
        
        # 3. Normalize amounts
        if 'TransactionAmt' in df.columns:
            df = self.normalize_amounts(df, 'TransactionAmt')
        
        # 4. Parse timestamps
        if 'TransactionDT' in df.columns:
            df = self.parse_timestamps(df, 'TransactionDT')
        
        print("\n" + "="*60)
        print("DATA CLEANING COMPLETE")
        print("="*60)
        print(f"Final shape: {df.shape}")
        
        return df
    
    def save_processed_data(self, df: pd.DataFrame, file_name: str):
        """
        Save processed data
        
        Args:
            df: Processed DataFrame
            file_name: Output file name
        """
        output_path = os.path.join(self.processed_data_path, file_name)
        os.makedirs(self.processed_data_path, exist_ok=True)
        
        df.to_csv(output_path, index=False)
        print(f"✓ Saved processed data to {output_path}")
    
    def get_data_summary(self, df: pd.DataFrame) -> dict:
        """
        Get comprehensive data summary
        
        Args:
            df: DataFrame to summarize
            
        Returns:
            Dictionary with summary statistics
        """
        summary = {
            'total_transactions': len(df),
            'total_features': len(df.columns),
            'date_range': (df['transaction_datetime'].min(), df['transaction_datetime'].max()) 
                         if 'transaction_datetime' in df.columns else None,
            'missing_values': df.isnull().sum().to_dict(),
            'numeric_summary': df.describe().to_dict(),
            'fraud_rate': df['isFraud'].mean() if 'isFraud' in df.columns else None
        }
        
        return summary


if __name__ == "__main__":
    # Example usage
    processor = DataProcessor()
    
    print("AML Data Processing Module")
    print("="*60)
    print("This module provides data cleaning and preprocessing for")
    print("AML fraud detection systems.")
    print("\nUsage:")
    print("  processor = DataProcessor()")
    print("  df = processor.load_data('train_transaction.csv')")
    print("  df_clean = processor.clean_data(df)")
