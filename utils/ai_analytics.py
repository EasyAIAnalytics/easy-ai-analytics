"""
AI-Powered Analytics Module
"""
import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from prophet import Prophet
import os
import sys
import json
import time
import traceback

# Initialize AI model clients based on available API keys
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY")

if OPENAI_API_KEY:
    from openai import OpenAI
    openai_client = OpenAI(api_key=OPENAI_API_KEY)
else:
    openai_client = None
    
if ANTHROPIC_API_KEY:
    from anthropic import Anthropic
    anthropic_client = Anthropic(api_key=ANTHROPIC_API_KEY)
else:
    anthropic_client = None


class AIAnalytics:
    """
    Class for AI-powered analytics
    """
    
    def __init__(self, data):
        """
        Initialize the AIAnalytics with a pandas DataFrame
        
        Args:
            data (pd.DataFrame): The data to analyze
        """
        self.data = data.copy()
        self.has_openai = openai_client is not None
        self.has_anthropic = anthropic_client is not None
        
    def detect_anomalies(self, column, contamination=0.05):
        """
        Detect anomalies in a numeric column using Isolation Forest
        
        Args:
            column (str): The column to analyze
            contamination (float): The expected proportion of outliers
            
        Returns:
            pd.Series: Boolean series indicating anomalies
        """
        if column not in self.data.columns:
            raise ValueError(f"Column '{column}' not found in data")
            
        # Check if column is numeric
        if not pd.api.types.is_numeric_dtype(self.data[column]):
            raise ValueError(f"Column '{column}' is not numeric")
            
        # Get values without NaN
        values = self.data[column].dropna().values.reshape(-1, 1)
        
        # Initialize and fit Isolation Forest
        iso_forest = IsolationForest(contamination=contamination, random_state=42)
        predictions = iso_forest.fit_predict(values)
        
        # Create a Series with the same index as the original data
        anomalies = pd.Series(False, index=self.data.index)
        anomalies.loc[self.data[column].dropna().index] = predictions == -1
        
        return anomalies
    
    def predict_time_series(self, date_column, value_column, periods=30, method='prophet'):
        """
        Predict future values for a time series
        
        Args:
            date_column (str): The column containing dates
            value_column (str): The column containing values to predict
            periods (int): Number of future periods to predict
            method (str): Forecasting method ('prophet' or 'exponential_smoothing')
            
        Returns:
            dict: Dictionary with dates and predicted values
        """
        # Check if columns exist
        if date_column not in self.data.columns:
            raise ValueError(f"Column '{date_column}' not found in data")
        if value_column not in self.data.columns:
            raise ValueError(f"Column '{value_column}' not found in data")
            
        # Get clean data for forecasting
        forecast_data = self.data[[date_column, value_column]].dropna()
        
        # Ensure value column contains numeric data
        try:
            forecast_data[value_column] = pd.to_numeric(forecast_data[value_column], errors='coerce')
            # Drop rows where conversion to numeric failed
            forecast_data = forecast_data.dropna()
        except Exception as e:
            raise ValueError(f"Failed to convert '{value_column}' to numeric values: {str(e)}")
        
        if len(forecast_data) < 10:
            raise ValueError("Insufficient data points for forecasting (minimum 10 required)")
            
        if method == 'prophet':
            # Prepare data for Prophet
            df_prophet = forecast_data.rename(columns={date_column: 'ds', value_column: 'y'})
            
            # Make sure ds column is datetime
            df_prophet['ds'] = pd.to_datetime(df_prophet['ds'])
            
            # Initialize and fit Prophet model
            model = Prophet(yearly_seasonality=True, weekly_seasonality=True, daily_seasonality=False)
            model.fit(df_prophet)
            
            # Create future dataframe
            future = model.make_future_dataframe(periods=periods)
            
            # Make predictions
            forecast = model.predict(future)
            
            # Prepare results
            results = {
                'dates': forecast['ds'].tail(periods).dt.strftime('%Y-%m-%d').tolist(),
                'predicted_values': forecast['yhat'].tail(periods).tolist(),
                'upper_bound': forecast['yhat_upper'].tail(periods).tolist(),
                'lower_bound': forecast['yhat_lower'].tail(periods).tolist(),
                'historical_dates': df_prophet['ds'].dt.strftime('%Y-%m-%d').tolist(),
                'historical_values': df_prophet['y'].tolist()
            }
            
            return results
            
        elif method == 'exponential_smoothing':
            # Prepare data for Exponential Smoothing
            # Make sure data is sorted by date
            forecast_data = forecast_data.sort_values(date_column)
            
            # Convert to datetime if needed
            forecast_data[date_column] = pd.to_datetime(forecast_data[date_column])
            
            # Prepare time series
            ts = forecast_data[value_column]
            
            # Initialize and fit Exponential Smoothing model
            model = ExponentialSmoothing(
                ts, 
                trend='add', 
                seasonal='add', 
                seasonal_periods=12
            ).fit()
            
            # Make predictions
            predictions = model.forecast(periods)
            
            # Create date range for predictions
            last_date = forecast_data[date_column].max()
            date_range = pd.date_range(start=last_date, periods=periods + 1)[1:]
            
            # Prepare results
            results = {
                'dates': date_range.strftime('%Y-%m-%d').tolist(),
                'predicted_values': predictions.tolist(),
                'historical_dates': forecast_data[date_column].dt.strftime('%Y-%m-%d').tolist(),
                'historical_values': forecast_data[value_column].tolist()
            }
            
            return results
            
        else:
            raise ValueError(f"Unknown method: {method}")
    
    def segment_customers(self, numeric_columns, n_clusters=3):
        """
        Segment customers using K-means clustering
        
        Args:
            numeric_columns (list): List of numeric columns to use for clustering
            n_clusters (int): Number of clusters
            
        Returns:
            dict: Dictionary with cluster information
        """
        # Check if columns exist
        for col in numeric_columns:
            if col not in self.data.columns:
                raise ValueError(f"Column '{col}' not found in data")
            if not pd.api.types.is_numeric_dtype(self.data[col]):
                raise ValueError(f"Column '{col}' is not numeric")
        
        # Get data for clustering
        cluster_data = self.data[numeric_columns].dropna()
        
        if len(cluster_data) < n_clusters * 3:
            raise ValueError(f"Insufficient data points for clustering with {n_clusters} clusters")
        
        # Standardize the data
        scaler = StandardScaler()
        scaled_data = scaler.fit_transform(cluster_data)
        
        # Apply PCA if more than 2 dimensions
        if len(numeric_columns) > 2:
            pca = PCA(n_components=2)
            reduced_data = pca.fit_transform(scaled_data)
            explained_variance = pca.explained_variance_ratio_.sum()
        else:
            reduced_data = scaled_data
            explained_variance = 1.0
        
        # Apply K-means clustering
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        clusters = kmeans.fit_predict(scaled_data)
        
        # Add cluster labels to original data
        cluster_data['Cluster'] = clusters
        
        # Calculate cluster statistics
        cluster_stats = []
        for i in range(n_clusters):
            cluster_i_data = cluster_data[cluster_data['Cluster'] == i]
            stats = {
                'cluster_id': i,
                'size': len(cluster_i_data),
                'percentage': 100 * len(cluster_i_data) / len(cluster_data),
                'features': {}
            }
            
            # Calculate statistics for each feature
            for col in numeric_columns:
                stats['features'][col] = {
                    'mean': cluster_i_data[col].mean(),
                    'median': cluster_i_data[col].median(),
                    'std': cluster_i_data[col].std()
                }
            
            cluster_stats.append(stats)
        
        # Prepare visualization data
        if len(numeric_columns) >= 2:
            # Use the first two columns directly
            viz_data = {
                'x': cluster_data[numeric_columns[0]].tolist(),
                'y': cluster_data[numeric_columns[1]].tolist(),
                'cluster': cluster_data['Cluster'].tolist(),
                'x_label': numeric_columns[0],
                'y_label': numeric_columns[1]
            }
        else:
            # Use PCA results
            viz_data = {
                'x': reduced_data[:, 0].tolist(),
                'y': reduced_data[:, 1].tolist(),
                'cluster': clusters.tolist(),
                'x_label': 'Principal Component 1',
                'y_label': 'Principal Component 2'
            }
        
        # Prepare results
        results = {
            'cluster_stats': cluster_stats,
            'visualization_data': viz_data,
            'explained_variance': explained_variance,
            'n_clusters': n_clusters
        }
        
        return results
    
    def generate_insights_with_ai(self, max_insights=5):
        """
        Generate insights from data using AI models
        
        Args:
            max_insights (int): Maximum number of insights to generate
            
        Returns:
            list: List of AI-generated insights
        """
        if not self.has_openai and not self.has_anthropic:
            return ["AI insight generation requires OpenAI or Anthropic API keys."]
        
        try:
            # Create a data summary
            data_summary = self._create_data_summary()
            
            # Formulate prompt
            prompt = f"""
            As a data analyst, examine this dataset summary and provide {max_insights} key business insights:

            {data_summary}

            Focus on:
            1. Notable patterns and trends
            2. Correlations between variables
            3. Unusual observations or anomalies
            4. Business implications of the findings
            5. Actionable recommendations based on the data

            Format each insight as a separate point.
            """
            
            if self.has_openai:
                # Use OpenAI
                response = openai_client.chat.completions.create(
                    model="gpt-4o",  # the newest OpenAI model is "gpt-4o" which was released May 13, 2024
                    messages=[
                        {"role": "system", "content": "You are an expert data analyst who provides clear, actionable business insights."},
                        {"role": "user", "content": prompt}
                    ],
                    max_tokens=1000,
                    temperature=0.5
                )
                
                insights_text = response.choices[0].message.content
                
            elif self.has_anthropic:
                # Use Anthropic
                response = anthropic_client.messages.create(
                    model="claude-3-5-sonnet-20241022",  # the newest Anthropic model is "claude-3-5-sonnet-20241022" which was released October 22, 2024
                    messages=[
                        {"role": "user", "content": prompt}
                    ],
                    max_tokens=1000,
                    temperature=0.5
                )
                
                insights_text = response.content[0].text
            
            # Process the insights
            insights = []
            for line in insights_text.split('\n'):
                line = line.strip()
                if line and (line.startswith('-') or line.startswith('*') or (len(line) > 2 and line[0].isdigit() and line[1] == '.')):
                    # Remove the list marker
                    clean_line = line[2:].strip() if line[1] in ['.', ')'] else line[1:].strip()
                    insights.append(clean_line)
                    
            return insights[:max_insights]
            
        except Exception as e:
            error_message = str(e)
            
            # Check for common OpenAI API errors
            if "insufficient_quota" in error_message or "quota" in error_message:
                return [f"Error generating AI insights: OpenAI API quota exceeded. Please check your API key billing status and quota limits. You may need to add payment information to your OpenAI account or upgrade your plan."]
            elif "rate limit" in error_message.lower() or "rate_limit" in error_message.lower():
                return [f"Error generating AI insights: API rate limit reached. Please try again in a few minutes, or reduce the frequency of requests."]
            elif "authentication" in error_message.lower() or "auth" in error_message.lower() or "key" in error_message.lower():
                return [f"Error generating AI insights: Authentication error with the API. Please check if your API key is valid and properly configured."]
            elif "context_length_exceeded" in error_message or "maximum context length" in error_message.lower():
                return [f"Error generating AI insights: The dataset is too large for the AI model's context window. Try analyzing fewer columns or a smaller dataset."]
            elif "connection" in error_message.lower() or "timeout" in error_message.lower():
                return [f"Error generating AI insights: Connection error when calling the AI service. Please check your internet connection and try again."]
            else:
                return [f"Error generating AI insights: {error_message}. Please try again or contact support if the issue persists."]
    
    def analyze_sentiment(self, text_column):
        """
        Analyze sentiment in a text column
        
        Args:
            text_column (str): The column containing text to analyze
            
        Returns:
            dict: Dictionary with sentiment analysis results
        """
        if not self.has_openai and not self.has_anthropic:
            return {"error": "Sentiment analysis requires OpenAI or Anthropic API keys."}
        
        if text_column not in self.data.columns:
            return {"error": f"Column '{text_column}' not found in data"}
            
        # Get text data
        text_data = self.data[text_column].dropna().astype(str)
        
        if len(text_data) == 0:
            return {"error": f"No text data found in column '{text_column}'"}
            
        # Sample data if there are too many rows (max 100)
        if len(text_data) > 100:
            text_data = text_data.sample(100, random_state=42)
            
        # Process each text
        results = []
        
        for idx, text in enumerate(text_data):
            try:
                if self.has_openai:
                    # Use OpenAI
                    response = openai_client.chat.completions.create(
                        model="gpt-4o",  # the newest OpenAI model is "gpt-4o" which was released May 13, 2024
                        messages=[
                            {
                                "role": "system",
                                "content": "You are a sentiment analysis expert. Analyze the sentiment of the text and provide a rating from 1 to 5 stars and a confidence score between 0 and 1. Respond with JSON in this format: {'rating': number, 'confidence': number}"
                            },
                            {"role": "user", "content": text}
                        ],
                        response_format={"type": "json_object"},
                        max_tokens=100,
                        temperature=0.3
                    )
                    
                    result = json.loads(response.choices[0].message.content)
                    sentiment = {
                        "text": text[:100] + "..." if len(text) > 100 else text,
                        "rating": max(1, min(5, round(result["rating"]))),
                        "confidence": max(0, min(1, result["confidence"]))
                    }
                    
                elif self.has_anthropic:
                    # Use Anthropic
                    response = anthropic_client.messages.create(
                        model="claude-3-5-sonnet-20241022",  # the newest Anthropic model is "claude-3-5-sonnet-20241022" which was released October 22, 2024
                        messages=[
                            {
                                "role": "user", 
                                "content": f"Analyze the sentiment of this text and provide a rating from 1 to 5 stars (where 1 is very negative and 5 is very positive) and a confidence score between 0 and 1. Respond with only a JSON object in this format: {{\"rating\": number, \"confidence\": number}}\n\nText: {text}"
                            }
                        ],
                        max_tokens=100,
                        temperature=0.3
                    )
                    
                    # Extract JSON from response
                    content = response.content[0].text
                    # Find JSON object in the response
                    json_start = content.find('{')
                    json_end = content.rfind('}')
                    if json_start >= 0 and json_end >= 0:
                        json_str = content[json_start:json_end+1]
                        result = json.loads(json_str)
                        sentiment = {
                            "text": text[:100] + "..." if len(text) > 100 else text,
                            "rating": max(1, min(5, round(result["rating"]))),
                            "confidence": max(0, min(1, result["confidence"]))
                        }
                    else:
                        sentiment = {
                            "text": text[:100] + "..." if len(text) > 100 else text,
                            "rating": 3,
                            "confidence": 0.5,
                            "error": "Failed to parse response"
                        }
                
                results.append(sentiment)
                
                # Avoid rate limiting
                if idx < len(text_data) - 1:
                    time.sleep(0.5)
                    
            except Exception as e:
                error_message = str(e)
                
                # Check for common API errors
                if "insufficient_quota" in error_message or "quota" in error_message:
                    error_detail = "API quota exceeded. Please check your API key billing status and quota limits. You may need to add payment information to your account or upgrade your plan."
                elif "rate limit" in error_message.lower() or "rate_limit" in error_message.lower():
                    error_detail = "API rate limit reached. Please try again in a few minutes, or reduce the frequency of requests."
                elif "authentication" in error_message.lower() or "auth" in error_message.lower() or "key" in error_message.lower():
                    error_detail = "Authentication error with the API. Please check if your API key is valid and properly configured."
                elif "context_length_exceeded" in error_message or "maximum context length" in error_message.lower():
                    error_detail = "The text is too large for the AI model's context window. Try analyzing shorter text."
                elif "connection" in error_message.lower() or "timeout" in error_message.lower():
                    error_detail = "Connection error when calling the AI service. Please check your internet connection and try again."
                else:
                    error_detail = f"{error_message}. Please try again or contact support if the issue persists."
                    
                results.append({
                    "text": text[:100] + "..." if len(text) > 100 else text,
                    "error": error_detail
                })
        
        # Calculate summary statistics
        total_ratings = [r.get("rating") for r in results if "rating" in r]
        
        if total_ratings:
            avg_rating = sum(total_ratings) / len(total_ratings)
            
            # Count ratings by category
            rating_counts = {}
            for i in range(1, 6):
                rating_counts[i] = sum(1 for r in total_ratings if r == i)
                
            summary = {
                "average_rating": avg_rating,
                "rating_counts": rating_counts,
                "total_analyzed": len(results),
                "successful_analyses": len(total_ratings)
            }
        else:
            summary = {
                "error": "No successful sentiment analyses",
                "total_analyzed": len(results),
                "successful_analyses": 0
            }
        
        return {
            "results": results,
            "summary": summary
        }
    
    def generate_hypothesis_test(self, group_column, value_column, alpha=0.05):
        """
        Perform statistical hypothesis testing to compare groups
        
        Args:
            group_column (str): Column defining the groups
            value_column (str): Numeric column to compare across groups
            alpha (float): Significance level
            
        Returns:
            dict: Dictionary with hypothesis test results
        """
        from scipy import stats
        import statsmodels.api as sm
        from statsmodels.formula.api import ols
        
        # Check if columns exist
        if group_column not in self.data.columns:
            return {"error": f"Column '{group_column}' not found in data"}
        if value_column not in self.data.columns:
            return {"error": f"Column '{value_column}' not found in data"}
            
        # Check if value column is numeric
        if not pd.api.types.is_numeric_dtype(self.data[value_column]):
            return {"error": f"Column '{value_column}' must be numeric for hypothesis testing"}
            
        # Get clean data for testing
        test_data = self.data[[group_column, value_column]].dropna()
        
        if len(test_data) < 10:
            return {"error": "Insufficient data points for hypothesis testing (minimum 10 required)"}
            
        # Get unique groups
        groups = test_data[group_column].unique()
        
        if len(groups) < 2:
            return {"error": f"At least 2 unique groups required in '{group_column}' for hypothesis testing"}
            
        if len(groups) == 2:
            # Perform t-test for two groups
            group1 = test_data[test_data[group_column] == groups[0]][value_column]
            group2 = test_data[test_data[group_column] == groups[1]][value_column]
            
            if len(group1) < 5 or len(group2) < 5:
                return {"error": f"Each group needs at least 5 data points for t-test"}
                
            # Check normality (Shapiro-Wilk test)
            if len(group1) <= 5000:  # Shapiro-Wilk limited to 5000 samples
                _, p_norm1 = stats.shapiro(group1)
            else:
                p_norm1 = None
                
            if len(group2) <= 5000:
                _, p_norm2 = stats.shapiro(group2)
            else:
                p_norm2 = None
                
            # Check equality of variances (Levene's test)
            _, p_var = stats.levene(group1, group2)
            
            # Perform t-test
            if p_var > alpha:  # Equal variances
                t_statistic, p_value = stats.ttest_ind(group1, group2, equal_var=True)
                test_type = "Student's t-test (equal variances)"
            else:  # Unequal variances
                t_statistic, p_value = stats.ttest_ind(group1, group2, equal_var=False)
                test_type = "Welch's t-test (unequal variances)"
            
            # Calculate effect size (Cohen's d)
            mean1, mean2 = group1.mean(), group2.mean()
            std1, std2 = group1.std(), group2.std()
            pooled_std = np.sqrt((std1**2 + std2**2) / 2)
            effect_size = abs(mean1 - mean2) / pooled_std if pooled_std > 0 else 0
            
            # Interpret effect size
            if effect_size < 0.2:
                effect_interpretation = "negligible"
            elif effect_size < 0.5:
                effect_interpretation = "small"
            elif effect_size < 0.8:
                effect_interpretation = "medium"
            else:
                effect_interpretation = "large"
            
            # Prepare results
            results = {
                "test_type": test_type,
                "t_statistic": t_statistic,
                "p_value": p_value,
                "significant": p_value < alpha,
                "alpha": alpha,
                "effect_size": effect_size,
                "effect_interpretation": effect_interpretation,
                "group1": {
                    "name": str(groups[0]),
                    "mean": group1.mean(),
                    "std": group1.std(),
                    "n": len(group1),
                },
                "group2": {
                    "name": str(groups[1]),
                    "mean": group2.mean(),
                    "std": group2.std(),
                    "n": len(group2)
                },
                "normality_test": {
                    "p_value_group1": p_norm1,
                    "p_value_group2": p_norm2,
                    "normality_warning": (p_norm1 is not None and p_norm1 < 0.05) or (p_norm2 is not None and p_norm2 < 0.05)
                },
                "variance_test": {
                    "p_value": p_var,
                    "equal_variances": p_var >= alpha
                },
                "conclusion": f"{'Reject' if p_value < alpha else 'Fail to reject'} the null hypothesis that the means are equal."
            }
            
            return results
            
        else:
            # Perform ANOVA for multiple groups
            group_data = []
            group_stats = []
            
            for group in groups:
                group_values = test_data[test_data[group_column] == group][value_column]
                if len(group_values) >= 5:  # Need at least 5 data points per group
                    group_data.append((str(group), group_values))
                    group_stats.append({
                        "name": str(group),
                        "mean": group_values.mean(),
                        "std": group_values.std(),
                        "n": len(group_values)
                    })
            
            if len(group_data) < 2:
                return {"error": "At least 2 groups with sufficient data points (5+) required for ANOVA"}
                
            # One-way ANOVA
            f_statistic, p_value = stats.f_oneway(*(group[1] for group in group_data))
            
            # Calculate effect size (Eta-squared)
            formula = f"{value_column} ~ C({group_column})"
            model = ols(formula, data=test_data).fit()
            anova_table = sm.stats.anova_lm(model, typ=2)
            
            # Calculate effect size (Eta-squared)
            ss_between = anova_table["sum_sq"][0]
            ss_total = ss_between + anova_table["sum_sq"][1]  # Between + Within
            eta_squared = ss_between / ss_total if ss_total > 0 else 0
            
            # Interpret effect size
            if eta_squared < 0.01:
                effect_interpretation = "negligible"
            elif eta_squared < 0.06:
                effect_interpretation = "small"
            elif eta_squared < 0.14:
                effect_interpretation = "medium"
            else:
                effect_interpretation = "large"
            
            # Check homogeneity of variances (Levene's test)
            _, p_levene = stats.levene(*(group[1] for group in group_data))
            
            # Perform post-hoc tests if ANOVA is significant
            posthoc_results = None
            if p_value < alpha:
                # Tukey's HSD test
                from statsmodels.stats.multicomp import pairwise_tukeyhsd
                
                # Prepare data for Tukey's test
                values = []
                group_labels = []
                
                for group_name, group_values in group_data:
                    values.extend(group_values)
                    group_labels.extend([group_name] * len(group_values))
                
                # Perform Tukey's HSD test
                tukey = pairwise_tukeyhsd(values, group_labels, alpha=alpha)
                
                # Extract results
                posthoc_results = []
                for i, (group1, group2, reject, _, _, _) in enumerate(zip(
                    tukey.data[0], tukey.data[1], tukey.reject, 
                    tukey.pvalues, tukey.meandiffs, tukey.confint
                )):
                    posthoc_results.append({
                        "group1": group1,
                        "group2": group2,
                        "p_value_adjusted": float(tukey.pvalues[i]),
                        "mean_difference": float(tukey.meandiffs[i]),
                        "significant": bool(reject),
                        "confidence_interval": [float(tukey.confint[i][0]), float(tukey.confint[i][1])]
                    })
            
            # Prepare results
            results = {
                "test_type": "One-way ANOVA",
                "f_statistic": float(f_statistic),
                "p_value": float(p_value),
                "significant": p_value < alpha,
                "alpha": alpha,
                "effect_size": eta_squared,
                "effect_interpretation": effect_interpretation,
                "groups": group_stats,
                "homogeneity_test": {
                    "p_value": float(p_levene),
                    "equal_variances": p_levene >= alpha,
                    "warning": p_levene < alpha
                },
                "posthoc_tests": posthoc_results,
                "conclusion": f"{'Reject' if p_value < alpha else 'Fail to reject'} the null hypothesis that all group means are equal."
            }
            
            return results
    
    def _create_data_summary(self):
        """
        Create a text summary of the dataset for AI prompt
        
        Returns:
            str: Text summary of the dataset
        """
        data = self.data
        
        # General dataset info
        summary = f"Dataset with {data.shape[0]} rows and {data.shape[1]} columns.\n\n"
        
        # Column information
        summary += "Columns and their data types:\n"
        for col in data.columns:
            summary += f"- {col}: {data[col].dtype}\n"
        
        summary += "\n"
        
        # Numeric column statistics
        numeric_cols = data.select_dtypes(include=['int64', 'float64']).columns
        if len(numeric_cols) > 0:
            summary += "Numeric column statistics:\n"
            for col in numeric_cols[:5]:  # Limit to first 5 numeric columns
                col_data = data[col].dropna()
                summary += (
                    f"- {col}: min={col_data.min():.2f}, max={col_data.max():.2f}, "
                    f"mean={col_data.mean():.2f}, median={col_data.median():.2f}, "
                    f"missing={data[col].isna().sum()} ({100*data[col].isna().sum()/len(data):.1f}%)\n"
                )
            
            if len(numeric_cols) > 5:
                summary += f"- ... and {len(numeric_cols) - 5} more numeric columns\n"
                
            summary += "\n"
        
        # Categorical column information
        cat_cols = data.select_dtypes(include=['object']).columns
        if len(cat_cols) > 0:
            summary += "Categorical column information:\n"
            for col in cat_cols[:5]:  # Limit to first 5 categorical columns
                value_counts = data[col].value_counts()
                if len(value_counts) <= 5:
                    # Show all values if there are 5 or fewer
                    value_info = ", ".join([f"{val}: {count}" for val, count in value_counts.items()])
                else:
                    # Show top 3 values
                    top_values = value_counts.head(3)
                    value_info = ", ".join([f"{val}: {count}" for val, count in top_values.items()])
                    value_info += f", ... and {len(value_counts) - 3} more values"
                
                summary += f"- {col}: {len(value_counts)} unique values ({value_info})\n"
            
            if len(cat_cols) > 5:
                summary += f"- ... and {len(cat_cols) - 5} more categorical columns\n"
                
            summary += "\n"
        
        # Date column information
        date_cols = []
        for col in data.columns:
            try:
                if pd.api.types.is_datetime64_any_dtype(data[col]) or (
                    pd.api.types.is_object_dtype(data[col]) and 
                    pd.to_datetime(data[col], errors='coerce').notna().any()
                ):
                    date_cols.append(col)
            except:
                pass
        
        if date_cols:
            summary += "Date/time column information:\n"
            for col in date_cols:
                try:
                    datetime_col = pd.to_datetime(data[col], errors='coerce')
                    min_date = datetime_col.min()
                    max_date = datetime_col.max()
                    summary += f"- {col}: range from {min_date} to {max_date}\n"
                except:
                    summary += f"- {col}: date/time column (error computing range)\n"
                    
            summary += "\n"
        
        # Correlation information (top 3 correlations)
        if len(numeric_cols) > 1:
            try:
                corr_matrix = data[numeric_cols].corr()
                # Get the most correlated pairs (excluding self-correlations)
                corr_pairs = []
                for i in range(len(numeric_cols)):
                    for j in range(i+1, len(numeric_cols)):
                        col1 = numeric_cols[i]
                        col2 = numeric_cols[j]
                        corr = corr_matrix.loc[col1, col2]
                        if not np.isnan(corr):
                            corr_pairs.append((col1, col2, corr))
                
                if corr_pairs:
                    corr_pairs.sort(key=lambda x: abs(x[2]), reverse=True)
                    summary += "Top correlations between numeric columns:\n"
                    for i, (col1, col2, corr) in enumerate(corr_pairs[:3]):
                        direction = "positive" if corr > 0 else "negative"
                        summary += f"- {col1} and {col2}: {direction} correlation (r = {corr:.2f})\n"
                    
                    summary += "\n"
            except:
                pass
        
        return summary