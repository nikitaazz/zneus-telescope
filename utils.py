from scipy.io import arff
import pandas as pd
import numpy as np
from scipy.stats import boxcox


def drop_outliers_iqr(df, columns=None, iqr_multiplier=1.5, verbose=True):
    """
    Drop rows containing outliers based on IQR method.
    
    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame to remove outliers from
    columns : list, optional
        List of columns to check for outliers. If None, checks all numeric columns except 'class'
    iqr_multiplier : float, default=1.5
        Multiplier for IQR to define outlier bounds (1.5 is standard, 3.0 is more conservative)
    verbose : bool, default=True
        Whether to print information about outliers removed
    
    Returns:
    --------
    pd.DataFrame
        DataFrame with outlier rows removed
    dict
        Dictionary containing outlier statistics for each column
    """
    df_original = df.copy()
    original_len = len(df)
    
    # Determine columns to check
    if columns is None:
        # Get all numeric columns except 'class'
        columns = [col for col in df.select_dtypes(include=[np.number]).columns 
                   if col not in ['class', 'ID']]
    
    # Track outliers per column
    outlier_stats = {}
    
    # Create a mask to keep rows without outliers in any of the specified columns
    mask = pd.Series([True] * len(df), index=df.index)
    
    for col in columns:
        if col in df.columns and df[col].dtype in [np.float64, np.int64, np.float32, np.int32]:
            # Calculate IQR bounds
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - iqr_multiplier * IQR
            upper_bound = Q3 + iqr_multiplier * IQR
            
            # Count outliers for this column
            col_outliers = ((df[col] < lower_bound) | (df[col] > upper_bound)).sum()
            
            # Update mask
            mask &= (df[col] >= lower_bound) & (df[col] <= upper_bound)
            
            # Store statistics
            outlier_stats[col] = {
                'outliers_count': col_outliers,
                'outliers_pct': (col_outliers / original_len) * 100,
                'lower_bound': lower_bound,
                'upper_bound': upper_bound,
                'Q1': Q1,
                'Q3': Q3,
                'IQR': IQR
            }
    
    # Apply mask to drop outliers
    df_cleaned = df[mask].copy()
    rows_dropped = original_len - len(df_cleaned)
    
    if verbose:
        print(f"\n{'='*80}")
        print(f"OUTLIER REMOVAL SUMMARY (IQR method with multiplier={iqr_multiplier})")
        print(f"{'='*80}")
        print(f"Original rows: {original_len}")
        print(f"Rows after dropping outliers: {len(df_cleaned)}")
        print(f"Rows dropped: {rows_dropped} ({rows_dropped/original_len*100:.2f}%)")
        print(f"\nOutliers detected per column:")
        print(f"{'-'*80}")
        
        # Sort by outlier count
        sorted_stats = sorted(outlier_stats.items(), 
                            key=lambda x: x[1]['outliers_count'], 
                            reverse=True)
        
        for col, stats in sorted_stats:
            print(f"{col:15s}: {stats['outliers_count']:6d} outliers "
                  f"({stats['outliers_pct']:5.2f}%) - "
                  f"Bounds: [{stats['lower_bound']:8.3f}, {stats['upper_bound']:8.3f}]")
        
        print(f"{'='*80}\n")
    
    return df_cleaned, outlier_stats


def load_data(filepath='MagicTelescope.arff', remove_outliers=False, 
              outlier_columns=None, iqr_multiplier=1.5):
    """
    Load and preprocess the Magic Telescope dataset.
    
    Parameters:
    -----------
    filepath : str
        Path to the ARFF file (default: 'MagicTelescope.arff')
    remove_outliers : bool, default=False
        Whether to remove outliers using IQR method
    outlier_columns : list, optional
        Specific columns to check for outliers. If None, checks all numeric columns.
        Example: ['fAsym', 'fWidth', 'fM3Long', 'fLength', 'fM3Trans', 'fConc1', 'fDist', 'fSize']
    iqr_multiplier : float, default=1.5
        IQR multiplier for outlier detection (1.5 standard, 3.0 more conservative)
    
    Returns:
    --------
    pd.DataFrame
        Preprocessed dataframe ready for analysis/modeling
    dict (optional)
        If remove_outliers=True, also returns outlier statistics
    """
    # Load ARFF file
    data, meta = arff.loadarff(filepath)
    df = pd.DataFrame(data)
    
    # --- Normalize column names ---
    # Strip whitespace, remove special characters, make lowercase
    df.columns = df.columns.astype(str).str.strip() \
                           .str.replace(r'[^0-9A-Za-z_]', '', regex=True)
    
    # --- Decode binary class column ---
    df['class'] = df['class'].str.decode('utf-8')
    
    # --- Strip whitespace from string columns ---
    str_cols = df.select_dtypes(include=['object']).columns
    for col in str_cols:
        df[col] = df[col].str.strip()
    
    # --- Convert class to binary (0/1) ---
    class_mapping = {'g': 1, 'h': 0}
    df['class'] = df['class'].map(class_mapping)
    
    # --- Drop ID column ---
    if 'ID' in df.columns:
        df.drop('ID', axis=1, inplace=True)
    
    # --- Drop duplicates ---
    df.drop_duplicates(inplace=True)
    
    print(f"\nInitial dataset shape: {df.shape}")
    print(f"Columns: {list(df.columns)}")
    
    # --- Remove outliers if requested ---
    outlier_stats = None
    if remove_outliers:
        df, outlier_stats = drop_outliers_iqr(
            df, 
            columns=outlier_columns, 
            iqr_multiplier=iqr_multiplier,
            verbose=True
        )
    
    print(f"Final dataset shape: {df.shape}")
    
    if remove_outliers:
        return df, outlier_stats
    return df


def get_class_mapping():
    """
    Returns the class mapping dictionary.
    
    Returns:
    --------
    dict
        Mapping of original class labels to binary values
    """
    return {'g': 1, 'h': 0}