import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from scipy import stats

class DataProcessor:
    """
    Class for processing and analyzing data
    """
    
    def __init__(self, data):
        """
        Initialize the DataProcessor with a pandas DataFrame
        
        Args:
            data (pd.DataFrame): The data to process
        """
        self.data = data
        
    def drop_missing_rows(self):
        """
        Drop rows with missing values
        
        Returns:
            pd.DataFrame: The data with missing rows dropped
        """
        return self.data.dropna()
    
    def fill_missing_values(self, method="mean"):
        """
        Fill missing values using the specified method
        
        Args:
            method (str): Method to use for filling missing values
                          Options: "mean", "median", "mode", "zero"
        
        Returns:
            pd.DataFrame: The data with missing values filled
        """
        df = self.data.copy()
        
        # Handle numeric columns
        numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns
        
        for col in numeric_cols:
            if df[col].isna().any():
                if method == "mean":
                    df[col] = df[col].fillna(df[col].mean())
                elif method == "median":
                    df[col] = df[col].fillna(df[col].median())
                elif method == "mode":
                    df[col] = df[col].fillna(df[col].mode()[0])
                elif method == "zero":
                    df[col] = df[col].fillna(0)
        
        # Handle categorical columns
        categorical_cols = df.select_dtypes(include=['object']).columns
        
        for col in categorical_cols:
            if df[col].isna().any():
                if method == "mode":
                    df[col] = df[col].fillna(df[col].mode()[0])
                else:
                    df[col] = df[col].fillna("Unknown")
        
        return df
    
    def detect_outliers(self, column, method="zscore", threshold=3):
        """
        Detect outliers in a specific column
        
        Args:
            column (str): The column to check for outliers
            method (str): Method to use for outlier detection
                          Options: "zscore", "iqr", "isolation_forest"
            threshold (float): Threshold for outlier detection
        
        Returns:
            pd.Series: Boolean series indicating outliers
        """
        if column not in self.data.columns:
            raise ValueError(f"Column '{column}' not found in data")
        
        if not pd.api.types.is_numeric_dtype(self.data[column]):
            raise ValueError(f"Column '{column}' is not numeric")
        
        if method == "zscore":
            z_scores = np.abs(stats.zscore(self.data[column].dropna()))
            return pd.Series(
                np.abs(stats.zscore(self.data[column].fillna(self.data[column].median()))) > threshold,
                index=self.data.index
            )
        
        elif method == "iqr":
            Q1 = self.data[column].quantile(0.25)
            Q3 = self.data[column].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - threshold * IQR
            upper_bound = Q3 + threshold * IQR
            return (self.data[column] < lower_bound) | (self.data[column] > upper_bound)
        
        elif method == "isolation_forest":
            model = IsolationForest(contamination=0.1, random_state=42)
            preds = model.fit_predict(self.data[column].values.reshape(-1, 1))
            return pd.Series(preds == -1, index=self.data.index)
        
        else:
            raise ValueError(f"Method '{method}' not recognized")
    
    def generate_insights(self):
        """
        Generate insights about the data
        
        Returns:
            list: A list of insight strings
        """
        insights = []
        
        # Basic data statistics
        insights.append(f"Dataset contains {self.data.shape[0]} rows and {self.data.shape[1]} columns.")
        
        # Missing values
        missing_pct = (self.data.isna().sum() / len(self.data) * 100).round(2)
        columns_with_missing = missing_pct[missing_pct > 0]
        
        if not columns_with_missing.empty:
            missing_str = "Columns with missing values:\n"
            for col, pct in columns_with_missing.items():
                missing_str += f" - {col}: {pct}% missing\n"
            insights.append(missing_str)
        else:
            insights.append("No missing values found in the dataset.")
        
        # Data types
        numeric_cols = self.data.select_dtypes(include=['int64', 'float64']).columns.tolist()
        categorical_cols = self.data.select_dtypes(include=['object']).columns.tolist()
        
        insights.append(f"Dataset contains {len(numeric_cols)} numeric columns and {len(categorical_cols)} categorical columns.")
        
        # Summary statistics for numeric columns
        if numeric_cols:
            for col in numeric_cols:
                stats_str = f"Statistics for {col}:\n"
                stats_str += f" - Min: {self.data[col].min():.2f}\n"
                stats_str += f" - Max: {self.data[col].max():.2f}\n"
                stats_str += f" - Mean: {self.data[col].mean():.2f}\n"
                stats_str += f" - Median: {self.data[col].median():.2f}\n"
                stats_str += f" - Standard Deviation: {self.data[col].std():.2f}"
                insights.append(stats_str)
        
        # Check for skewness in numeric columns
        if numeric_cols:
            skewed_cols = []
            for col in numeric_cols:
                skewness = self.data[col].skew()
                if abs(skewness) > 1:
                    skewed_cols.append((col, skewness))
            
            if skewed_cols:
                skew_str = "Skewed columns that might need transformation:\n"
                for col, skew in skewed_cols:
                    direction = "positively" if skew > 0 else "negatively"
                    skew_str += f" - {col} is {direction} skewed (skewness: {skew:.2f})\n"
                insights.append(skew_str)
        
        # Most frequent values for categorical columns
        if categorical_cols:
            for col in categorical_cols[:3]:  # Limit to first 3 categorical columns to avoid too many insights
                freq = self.data[col].value_counts().head(3)
                freq_str = f"Most frequent values for {col}:\n"
                for val, count in freq.items():
                    freq_str += f" - {val}: {count} occurrences ({count/len(self.data)*100:.1f}%)\n"
                insights.append(freq_str)
        
        # Correlation analysis
        if len(numeric_cols) > 1:
            corr_matrix = self.data[numeric_cols].corr()
            high_corr_pairs = []
            
            for i in range(len(numeric_cols)):
                for j in range(i+1, len(numeric_cols)):
                    if abs(corr_matrix.iloc[i, j]) > 0.7:
                        high_corr_pairs.append((
                            numeric_cols[i],
                            numeric_cols[j],
                            corr_matrix.iloc[i, j]
                        ))
            
            if high_corr_pairs:
                corr_str = "Strong correlations found:\n"
                for col1, col2, corr in high_corr_pairs:
                    direction = "positive" if corr > 0 else "negative"
                    corr_str += f" - {col1} and {col2} have a strong {direction} correlation (r={corr:.2f})\n"
                insights.append(corr_str)
        
        # Recommendations
        recommendations = []
        
        if any(missing_pct > 5):
            cols_high_missing = missing_pct[missing_pct > 5].index.tolist()
            if cols_high_missing:
                recommendations.append(f"Consider how to handle columns with high missing values: {', '.join(cols_high_missing)}.")
        
        if len(insights) > 0:
            insights.append("Recommendations:\n" + "\n".join(f" - {r}" for r in recommendations))
        
        return insights
