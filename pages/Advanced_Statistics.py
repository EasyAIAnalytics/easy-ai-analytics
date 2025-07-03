import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from scipy import stats
import statsmodels.api as sm
from statsmodels.formula.api import ols
from statsmodels.stats.multicomp import pairwise_tukeyhsd
from statsmodels.graphics.gofplots import qqplot
import pymc as pm
import arviz as az
import matplotlib.pyplot as plt
from io import BytesIO
import time
import traceback




        
# Page configuration
st.set_page_config(
    page_title="Advanced Statistical Analysis",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Page title with styling
st.markdown('<h1 class="main-header">Advanced Statistical Analysis</h1>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">Powerful statistical tools to validate assumptions and discover deeper insights</p>', unsafe_allow_html=True)

# Initialize session state variables if they don't exist
if 'data' not in st.session_state or st.session_state.data is None:
    st.warning("Please upload data in the main dashboard before using this feature.")
    st.stop()

if 'cleaned_data' not in st.session_state:
    st.session_state.cleaned_data = st.session_state.data.copy()

# Function to plot QQ plot
def create_qq_plot(data, column):
    # Create matplotlib figure
    fig, ax = plt.subplots(figsize=(8, 6))
    qqplot(data[column].dropna(), line='s', ax=ax)
    ax.set_title(f'QQ Plot for {column}')
    
    # Convert matplotlib to plotly
    buf = BytesIO()
    fig.savefig(buf, format="png", dpi=300, bbox_inches='tight')
    buf.seek(0)
    
    # Close matplotlib figure
    plt.close(fig)
    
    return buf

# Create tabs for different statistical analysis functions
tab1, tab2, tab3 = st.tabs([
    "Hypothesis Testing", 
    "Multivariate Analysis",
    "Bayesian Analysis"
])

with tab1:
    st.markdown('<div class="dashboard-card">', unsafe_allow_html=True)
    st.markdown("### Statistical Hypothesis Testing")
    st.markdown("""
    Hypothesis testing helps you determine if there are significant differences between groups in your data 
    or if observed patterns are likely due to chance.
    """)
    
    # Only allow if data is loaded
    if st.session_state.cleaned_data is not None:
        # Select columns for hypothesis testing
        numeric_columns = st.session_state.cleaned_data.select_dtypes(include=['int64', 'float64']).columns.tolist()
        categorical_columns = st.session_state.cleaned_data.select_dtypes(include=['object']).columns.tolist()
        
        if len(numeric_columns) == 0 or len(categorical_columns) == 0:
            st.warning("Hypothesis testing requires both numeric and categorical columns in your data.")
        else:
            col1, col2 = st.columns(2)
            
            with col1:
                # Select group column (categorical)
                group_column = st.selectbox(
                    "Select Group Column (categorical):",
                    categorical_columns,
                    key="hypothesis_group_column"
                )
                
                # Show group counts
                if group_column:
                    group_counts = st.session_state.cleaned_data[group_column].value_counts()
                    st.markdown(f"**Number of groups:** {len(group_counts)}")
                    
                    # Warning if any group has less than 5 elements
                    small_groups = group_counts[group_counts < 5]
                    if len(small_groups) > 0:
                        st.warning(f"{len(small_groups)} groups have fewer than 5 data points, which may affect test reliability.")
            
            with col2:
                # Select value column (numeric)
                value_column = st.selectbox(
                    "Select Value Column (numeric):",
                    numeric_columns,
                    key="hypothesis_value_column"
                )
                
                # Set significance level
                alpha = st.slider(
                    "Significance Level (α):",
                    min_value=0.01,
                    max_value=0.10,
                    value=0.05,
                    step=0.01,
                    format="%.2f"
                )
            
            # Select test type
            test_type = st.radio(
                "Test Type:",
                ["Automatic (based on data)", "t-test (two groups)", "ANOVA (multiple groups)"]
            )
            
            if st.button("Run Hypothesis Test"):
                # Get clean data for test
                test_data = st.session_state.cleaned_data[[group_column, value_column]].dropna()
                
                if len(test_data) < 10:
                    st.error("Insufficient data for hypothesis testing (minimum 10 data points required).")
                else:
                    # Get unique groups
                    groups = test_data[group_column].unique()
                    
                    if len(groups) < 2:
                        st.error(f"At least 2 unique groups required in '{group_column}' for hypothesis testing.")
                    else:
                        # Determine test type if set to automatic
                        if test_type == "Automatic (based on data)":
                            test_type = "t-test (two groups)" if len(groups) == 2 else "ANOVA (multiple groups)"
                        
                        # Run appropriate test
                        if test_type == "t-test (two groups)" and len(groups) >= 2:
                            # Perform t-test for two groups
                            group1_name = groups[0]
                            group2_name = groups[1]
                            group1 = test_data[test_data[group_column] == group1_name][value_column]
                            group2 = test_data[test_data[group_column] == group2_name][value_column]
                            
                            if len(group1) < 5 or len(group2) < 5:
                                st.warning(f"Each group should have at least 5 data points for reliable t-test results.")
                            
                            # Check normality (Shapiro-Wilk test)
                            shapiro_results = []
                            for group_name, group_values in [(group1_name, group1), (group2_name, group2)]:
                                if len(group_values) <= 5000:  # Shapiro-Wilk limited to 5000 samples
                                    _, p_norm = stats.shapiro(group_values)
                                    shapiro_results.append((group_name, p_norm, p_norm >= 0.05))
                                else:
                                    shapiro_results.append((group_name, None, None))
                            
                            # Check equality of variances (Levene's test)
                            _, p_var = stats.levene(group1, group2)
                            equal_variances = p_var >= alpha
                            
                            # Perform t-test
                            if equal_variances:  # Equal variances
                                t_stat, p_value = stats.ttest_ind(group1, group2, equal_var=True)
                                test_name = "Student's t-test (equal variances)"
                            else:  # Unequal variances
                                t_stat, p_value = stats.ttest_ind(group1, group2, equal_var=False)
                                test_name = "Welch's t-test (unequal variances)"
                            
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
                            
                            # Display results
                            st.markdown("#### T-Test Results")
                            
                            # Create two columns for results
                            col1, col2 = st.columns(2)
                            
                            with col1:
                                st.markdown("**Test Information**")
                                st.markdown(f"Test type: {test_name}")
                                st.markdown(f"Significance level (α): {alpha}")
                                st.markdown(f"t-statistic: {t_stat:.4f}")
                                st.markdown(f"p-value: {p_value:.4f}")
                                
                                # Conclusion
                                if p_value < alpha:
                                    st.markdown(f"**Conclusion:** Reject the null hypothesis that the means are equal (p < {alpha}).")
                                    st.markdown(f"There is a statistically significant difference between the groups.")
                                else:
                                    st.markdown(f"**Conclusion:** Fail to reject the null hypothesis that the means are equal (p ≥ {alpha}).")
                                    st.markdown(f"There is not enough evidence of a significant difference between the groups.")
                            
                            with col2:
                                st.markdown("**Effect Size**")
                                st.markdown(f"Cohen's d: {effect_size:.4f}")
                                st.markdown(f"Effect size interpretation: {effect_interpretation}")
                                
                                st.markdown("**Group Statistics**")
                                group_stats_df = pd.DataFrame({
                                    'Group': [group1_name, group2_name],
                                    'Count': [len(group1), len(group2)],
                                    'Mean': [group1.mean(), group2.mean()],
                                    'Std Dev': [group1.std(), group2.std()]
                                })
                                st.dataframe(group_stats_df, use_container_width=True)
                            
                            # Visualize the groups
                            st.markdown("#### Visualization")
                            
                            # Box plot comparing groups
                            fig = px.box(
                                test_data, 
                                x=group_column, 
                                y=value_column, 
                                color=group_column,
                                title=f"Comparison of {value_column} by {group_column}",
                                points="all"
                            )
                            st.plotly_chart(fig, use_container_width=True)
                            
                            # QQ plots for normality check
                            st.markdown("#### Normality Check (QQ Plots)")
                            st.markdown("QQ plots help visualize if your data follows a normal distribution. Points should follow the diagonal line for normally distributed data.")
                            
                            col1, col2 = st.columns(2)
                            
                            with col1:
                                st.markdown(f"**QQ Plot for {group1_name}**")
                                
                                # Create QQ plot
                                qq_buf1 = create_qq_plot(test_data[test_data[group_column] == group1_name], value_column)
                                st.image(qq_buf1)
                                
                                # Display Shapiro-Wilk test result
                                if shapiro_results[0][1] is not None:
                                    st.markdown(f"Shapiro-Wilk test p-value: {shapiro_results[0][1]:.4f}")
                                    if shapiro_results[0][1] < 0.05:
                                        st.markdown("⚠️ Data may not be normally distributed (p < 0.05)")
                                    else:
                                        st.markdown("✅ Data appears normally distributed (p ≥ 0.05)")
                            
                            with col2:
                                st.markdown(f"**QQ Plot for {group2_name}**")
                                
                                # Create QQ plot
                                qq_buf2 = create_qq_plot(test_data[test_data[group_column] == group2_name], value_column)
                                st.image(qq_buf2)
                                
                                # Display Shapiro-Wilk test result
                                if shapiro_results[1][1] is not None:
                                    st.markdown(f"Shapiro-Wilk test p-value: {shapiro_results[1][1]:.4f}")
                                    if shapiro_results[1][1] < 0.05:
                                        st.markdown("⚠️ Data may not be normally distributed (p < 0.05)")
                                    else:
                                        st.markdown("✅ Data appears normally distributed (p ≥ 0.05)")
                            
                            # Additional warnings
                            warnings = []
                            
                            # Check for normality issues
                            if any(result[1] is not None and result[1] < 0.05 for result in shapiro_results):
                                warnings.append("Some groups appear non-normally distributed. Consider non-parametric tests like Mann-Whitney U test.")
                            
                            # Check for unequal variances
                            if not equal_variances:
                                warnings.append(f"Groups have unequal variances (Levene's test p-value: {p_var:.4f} < {alpha}). Welch's t-test was used.")
                            
                            # Check for sample size
                            if len(group1) < 30 or len(group2) < 30:
                                warnings.append("Small sample sizes. Consider non-parametric tests for more reliable results.")
                            
                            if warnings:
                                st.markdown("#### ⚠️ Warnings")
                                for warning in warnings:
                                    st.warning(warning)
                            
                        elif test_type == "ANOVA (multiple groups)":
                            # Perform ANOVA for multiple groups
                            # Prepare data for ANOVA
                            group_data = []
                            group_names = []
                            
                            for group in groups:
                                group_values = test_data[test_data[group_column] == group][value_column]
                                if len(group_values) >= 5:  # Need at least 5 data points per group
                                    group_data.append(group_values)
                                    group_names.append(str(group))
                            
                            if len(group_data) < 2:
                                st.error("At least 2 groups with sufficient data points (5+) required for ANOVA.")
                            else:
                                # One-way ANOVA
                                f_statistic, p_value = stats.f_oneway(*group_data)
                                
                                # Calculate effect size (Eta-squared)
                                # Handle column names with spaces by quoting them with Q()
                                formula = f'Q("{value_column}") ~ C(Q("{group_column}"))'
                                model = ols(formula, data=test_data).fit()
                                anova_table = sm.stats.anova_lm(model, typ=2)
                                
                                # Calculate effect size (Eta-squared)
                                ss_between = anova_table["sum_sq"][0]
                                ss_total = anova_table["sum_sq"].sum()
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
                                _, p_levene = stats.levene(*group_data)
                                equal_variances = p_levene >= alpha
                                
                                # Display results
                                st.markdown("#### ANOVA Results")
                                
                                # Create two columns for results
                                col1, col2 = st.columns(2)
                                
                                with col1:
                                    st.markdown("**Test Information**")
                                    st.markdown("Test type: One-way ANOVA")
                                    st.markdown(f"Significance level (α): {alpha}")
                                    st.markdown(f"F-statistic: {f_statistic:.4f}")
                                    st.markdown(f"p-value: {p_value:.4f}")
                                    
                                    # Conclusion
                                    if p_value < alpha:
                                        st.markdown(f"**Conclusion:** Reject the null hypothesis that all group means are equal (p < {alpha}).")
                                        st.markdown(f"There is a statistically significant difference between at least two groups.")
                                    else:
                                        st.markdown(f"**Conclusion:** Fail to reject the null hypothesis that all group means are equal (p ≥ {alpha}).")
                                        st.markdown(f"There is not enough evidence of a significant difference between the groups.")
                                
                                with col2:
                                    st.markdown("**Effect Size**")
                                    st.markdown(f"Eta-squared: {eta_squared:.4f}")
                                    st.markdown(f"Effect size interpretation: {effect_interpretation}")
                                    
                                    st.markdown("**ANOVA Table**")
                                    st.dataframe(anova_table.round(4), use_container_width=True)
                                
                                # Group statistics
                                st.markdown("**Group Statistics**")
                                group_stats = []
                                
                                for i, (name, values) in enumerate(zip(group_names, group_data)):
                                    group_stats.append({
                                        'Group': name,
                                        'Count': len(values),
                                        'Mean': values.mean(),
                                        'Std Dev': values.std(),
                                        'Min': values.min(),
                                        'Max': values.max()
                                    })
                                
                                group_stats_df = pd.DataFrame(group_stats)
                                st.dataframe(group_stats_df, use_container_width=True)
                                
                                # Visualize the groups
                                st.markdown("#### Visualization")
                                
                                # Box plot comparing groups
                                fig = px.box(
                                    test_data, 
                                    x=group_column, 
                                    y=value_column, 
                                    color=group_column,
                                    title=f"Comparison of {value_column} by {group_column}",
                                    points="all"
                                )
                                st.plotly_chart(fig, use_container_width=True)
                                
                                # Post-hoc tests if ANOVA is significant
                                if p_value < alpha:
                                    st.markdown("#### Post-hoc Tests (Tukey's HSD)")
                                    st.markdown("Since the ANOVA result is significant, we perform post-hoc tests to determine which specific groups differ.")
                                    
                                    # Prepare data for Tukey's test
                                    values = []
                                    labels = []
                                    
                                    for name, group_values in zip(group_names, group_data):
                                        values.extend(group_values)
                                        labels.extend([name] * len(group_values))
                                    
                                    # Perform Tukey's HSD test
                                    tukey = pairwise_tukeyhsd(values, labels, alpha=alpha)
                                    
                                    # Display results
                                    tukey_df = pd.DataFrame(
                                        data=np.column_stack([tukey.groupsunique[tukey.pairindices], 
                                                             tukey.meandiffs, 
                                                             tukey.confint, 
                                                             tukey.pvalues, 
                                                             tukey.reject]),
                                        columns=['Group 1', 'Group 2', 'Mean Diff', 'Lower', 'Upper', 'p-value', 'Significant']
                                    )
                                    tukey_df['p-value'] = tukey_df['p-value'].astype(float)
                                    tukey_df['Significant'] = tukey_df['Significant'].astype(bool)
                                    tukey_df['Lower'] = tukey_df['Lower'].astype(float)
                                    tukey_df['Upper'] = tukey_df['Upper'].astype(float)
                                    tukey_df['Mean Diff'] = tukey_df['Mean Diff'].astype(float)
                                    
                                    st.dataframe(tukey_df, use_container_width=True)
                                    
                                    # Visualize pairs with significant differences
                                    significant_pairs = tukey_df[tukey_df['Significant']]
                                    if not significant_pairs.empty:
                                        st.markdown("#### Significant Differences")
                                        
                                        # Bar chart of significant differences
                                        fig = go.Figure()
                                        
                                        for i, row in significant_pairs.iterrows():
                                            fig.add_trace(go.Bar(
                                                x=[f"{row['Group 1']} vs {row['Group 2']}"],
                                                y=[row['Mean Diff']],
                                                error_y=dict(
                                                    type='data',
                                                    symmetric=False,
                                                    array=[row['Upper'] - row['Mean Diff']],
                                                    arrayminus=[row['Mean Diff'] - row['Lower']]
                                                ),
                                                name=f"{row['Group 1']} vs {row['Group 2']}"
                                            ))
                                        
                                        fig.update_layout(
                                            title="Mean Differences between Groups (with 95% CI)",
                                            xaxis_title="Group Comparison",
                                            yaxis_title="Mean Difference",
                                            showlegend=False
                                        )
                                        
                                        st.plotly_chart(fig, use_container_width=True)
                                
                                # Additional warnings
                                warnings = []
                                
                                # Check for homogeneity of variances
                                if not equal_variances:
                                    warnings.append(f"Groups have unequal variances (Levene's test p-value: {p_levene:.4f} < {alpha}). Consider Welch's ANOVA or non-parametric tests.")
                                
                                # Check for small sample sizes
                                small_groups = [i for i, g in enumerate(group_data) if len(g) < 30]
                                if small_groups:
                                    warnings.append(f"{len(small_groups)} groups have small sample sizes (< 30). Results may be less reliable.")
                                
                                if warnings:
                                    st.markdown("#### ⚠️ Warnings")
                                    for warning in warnings:
                                        st.warning(warning)
    st.markdown('</div>', unsafe_allow_html=True)

with tab2:
    st.markdown('<div class="dashboard-card">', unsafe_allow_html=True)
    st.markdown("### Multivariate Analysis")
    st.markdown("""
    Multivariate analysis examines relationships between multiple variables simultaneously.
    """)
    
    # Only allow if data is loaded
    if st.session_state.cleaned_data is not None:
        # Get numeric columns
        numeric_columns = st.session_state.cleaned_data.select_dtypes(include=['int64', 'float64']).columns.tolist()
        
        if len(numeric_columns) < 3:
            st.warning("Multivariate analysis requires at least 3 numeric columns in your data.")
        else:
            # Create subtabs for different multivariate analyses
            mv_tab1, mv_tab2, mv_tab3 = st.tabs([
                "Principal Component Analysis (PCA)",
                "Correlation Analysis",
                "Multiple Regression"
            ])
            
            with mv_tab1:
                st.markdown("#### Principal Component Analysis (PCA)")
                st.markdown("""
                PCA reduces the dimensionality of your data while preserving as much variance as possible.
                This helps visualize high-dimensional data and identify the most important features.
                """)
                
                # Select columns for PCA
                selected_columns = st.multiselect(
                    "Select Columns for PCA Analysis:",
                    numeric_columns,
                    default=numeric_columns[:min(5, len(numeric_columns))]
                )
                
                if len(selected_columns) < 2:
                    st.warning("PCA requires at least 2 columns.")
                else:
                    # Button to run PCA
                    if st.button("Run PCA Analysis"):
                        # Get clean data for PCA
                        pca_data = st.session_state.cleaned_data[selected_columns].dropna()
                        
                        if len(pca_data) < 10:
                            st.error("Insufficient data for PCA (minimum 10 data points required).")
                        else:
                            from sklearn.preprocessing import StandardScaler
                            from sklearn.decomposition import PCA
                            
                            # Standardize the data
                            scaler = StandardScaler()
                            scaled_data = scaler.fit_transform(pca_data)
                            
                            # Apply PCA
                            pca = PCA()
                            pca_result = pca.fit_transform(scaled_data)
                            
                            # Create DataFrame with PCA results
                            pca_df = pd.DataFrame(
                                data=pca_result[:, :3],  # Take first 3 components
                                columns=['PC1', 'PC2', 'PC3'][:min(3, len(selected_columns))]
                            )
                            
                            # Display explained variance
                            explained_variance = pca.explained_variance_ratio_
                            cumulative_variance = np.cumsum(explained_variance)
                            
                            st.markdown("#### Explained Variance")
                            
                            # Prepare data for the chart
                            variance_data = pd.DataFrame({
                                'Component': [f'PC{i+1}' for i in range(len(explained_variance))],
                                'Explained Variance': explained_variance,
                                'Cumulative Variance': cumulative_variance
                            })
                            
                            # Create the chart
                            fig = go.Figure()
                            
                            # Add bars for individual variance
                            fig.add_trace(go.Bar(
                                x=variance_data['Component'],
                                y=variance_data['Explained Variance'],
                                name='Explained Variance',
                                marker_color='skyblue'
                            ))
                            
                            # Add line for cumulative variance
                            fig.add_trace(go.Scatter(
                                x=variance_data['Component'],
                                y=variance_data['Cumulative Variance'],
                                name='Cumulative Variance',
                                mode='lines+markers',
                                marker=dict(size=8, color='red'),
                                line=dict(color='red', width=2)
                            ))
                            
                            # Update layout
                            fig.update_layout(
                                title='Explained Variance by Principal Component',
                                xaxis_title='Principal Component',
                                yaxis_title='Explained Variance Ratio',
                                yaxis=dict(tickformat='.0%'),
                                legend=dict(
                                    orientation="h",
                                    yanchor="bottom",
                                    y=1.02,
                                    xanchor="right",
                                    x=1
                                )
                            )
                            
                            st.plotly_chart(fig, use_container_width=True)
                            
                            # Calculate number of components needed for 80% variance
                            components_for_80 = np.argmax(cumulative_variance >= 0.8) + 1
                            st.info(f"{components_for_80} principal components explain ≥ 80% of the variance in your data.")
                            
                            # Show contribution of each feature to principal components
                            st.markdown("#### Feature Contributions")
                            
                            # Component loadings
                            loadings = pd.DataFrame(
                                data=pca.components_.T,
                                columns=[f'PC{i+1}' for i in range(len(pca.components_))],
                                index=selected_columns
                            )
                            
                            # Display loadings
                            st.dataframe(loadings.round(4), use_container_width=True)
                            
                            # Visualize feature contributions to PC1 and PC2
                            fig = go.Figure()
                            
                            for i, feature in enumerate(selected_columns):
                                fig.add_trace(go.Scatter(
                                    x=[0, loadings.iloc[i, 0]],
                                    y=[0, loadings.iloc[i, 1]],
                                    mode='lines+markers',
                                    name=feature,
                                    line=dict(color=px.colors.qualitative.Plotly[i % len(px.colors.qualitative.Plotly)], width=2),
                                    marker=dict(size=10)
                                ))
                                
                                # Add text annotations
                                fig.add_annotation(
                                    x=loadings.iloc[i, 0],
                                    y=loadings.iloc[i, 1],
                                    text=feature,
                                    showarrow=False,
                                    xshift=15,
                                    yshift=15
                                )
                            
                            # Add unit circle
                            theta = np.linspace(0, 2*np.pi, 100)
                            fig.add_trace(go.Scatter(
                                x=np.cos(theta),
                                y=np.sin(theta),
                                mode='lines',
                                line=dict(color='lightgray', width=1, dash='dash'),
                                showlegend=False
                            ))
                            
                            # Update layout
                            fig.update_layout(
                                title='Feature Contributions to PC1 and PC2',
                                xaxis_title='PC1',
                                yaxis_title='PC2',
                                xaxis=dict(range=[-1.1, 1.1], zeroline=True, zerolinecolor='lightgray'),
                                yaxis=dict(range=[-1.1, 1.1], zeroline=True, zerolinecolor='lightgray'),
                                showlegend=True
                            )
                            
                            st.plotly_chart(fig, use_container_width=True)
                            
                            # Visualize data in PCA space
                            if len(pca_df.columns) >= 2:
                                st.markdown("#### Data in Principal Component Space")
                                
                                # Add categorical color variable if available
                                categorical_columns = st.session_state.cleaned_data.select_dtypes(include=['object']).columns.tolist()
                                color_col = None
                                
                                if categorical_columns:
                                    color_col = st.selectbox(
                                        "Color points by (optional):",
                                        ["None"] + categorical_columns
                                    )
                                    
                                    if color_col != "None":
                                        # Add color column to PCA DataFrame
                                        pca_df['Color'] = st.session_state.cleaned_data.loc[pca_df.index, color_col]
                                
                                # Create scatter plot
                                if color_col and color_col != "None":
                                    fig = px.scatter(
                                        pca_df, 
                                        x='PC1', 
                                        y='PC2',
                                        color='Color',
                                        title='Data in Principal Component Space',
                                        labels={'PC1': f'PC1 ({explained_variance[0]:.1%} variance)',
                                                'PC2': f'PC2 ({explained_variance[1]:.1%} variance)'},
                                        opacity=0.7
                                    )
                                else:
                                    fig = px.scatter(
                                        pca_df, 
                                        x='PC1', 
                                        y='PC2',
                                        title='Data in Principal Component Space',
                                        labels={'PC1': f'PC1 ({explained_variance[0]:.1%} variance)',
                                                'PC2': f'PC2 ({explained_variance[1]:.1%} variance)'},
                                        opacity=0.7
                                    )
                                
                                st.plotly_chart(fig, use_container_width=True)
                                
                                # 3D PCA visualization if we have 3+ dimensions
                                if len(pca_df.columns) >= 3 and 'PC3' in pca_df.columns:
                                    st.markdown("#### 3D PCA Visualization")
                                    
                                    if color_col and color_col != "None" and 'Color' in pca_df.columns:
                                        fig = px.scatter_3d(
                                            pca_df,
                                            x='PC1',
                                            y='PC2',
                                            z='PC3',
                                            color='Color',
                                            title='3D PCA Visualization',
                                            labels={'PC1': f'PC1 ({explained_variance[0]:.1%} variance)',
                                                    'PC2': f'PC2 ({explained_variance[1]:.1%} variance)',
                                                    'PC3': f'PC3 ({explained_variance[2]:.1%} variance)'},
                                            opacity=0.7
                                        )
                                    else:
                                        fig = px.scatter_3d(
                                            pca_df,
                                            x='PC1',
                                            y='PC2',
                                            z='PC3',
                                            title='3D PCA Visualization',
                                            labels={'PC1': f'PC1 ({explained_variance[0]:.1%} variance)',
                                                    'PC2': f'PC2 ({explained_variance[1]:.1%} variance)',
                                                    'PC3': f'PC3 ({explained_variance[2]:.1%} variance)'},
                                            opacity=0.7
                                        )
                                    
                                    st.plotly_chart(fig, use_container_width=True)
            
            with mv_tab2:
                st.markdown("#### Correlation Analysis")
                st.markdown("""
                Correlation analysis measures the strength and direction of relationships between pairs of variables.
                This helps identify which variables tend to move together and how strongly they are related.
                """)
                
                # Select columns for correlation analysis
                selected_columns = st.multiselect(
                    "Select Columns for Correlation Analysis:",
                    numeric_columns,
                    default=numeric_columns[:min(8, len(numeric_columns))]
                )
                
                if len(selected_columns) < 2:
                    st.warning("Correlation analysis requires at least 2 columns.")
                else:
                    # Select correlation method
                    corr_method = st.radio(
                        "Correlation Method:",
                        ["Pearson (linear)", "Spearman (rank)", "Kendall (ordinal)"],
                        index=0
                    )
                    
                    # Map method to pandas correlation method
                    method_map = {
                        "Pearson (linear)": "pearson",
                        "Spearman (rank)": "spearman",
                        "Kendall (ordinal)": "kendall"
                    }
                    
                    # Button to run correlation analysis
                    if st.button("Run Correlation Analysis"):
                        # Get clean data
                        corr_data = st.session_state.cleaned_data[selected_columns].dropna()
                        
                        if len(corr_data) < 10:
                            st.error("Insufficient data for correlation analysis (minimum 10 data points required).")
                        else:
                            # Calculate correlation matrix
                            corr_matrix = corr_data.corr(method=method_map[corr_method])
                            
                            # Calculate p-values for Pearson correlation
                            p_values = pd.DataFrame(np.zeros_like(corr_matrix), index=corr_matrix.index, columns=corr_matrix.columns)
                            
                            for i, col_i in enumerate(corr_matrix.columns):
                                for j, col_j in enumerate(corr_matrix.columns):
                                    if i != j:  # Skip diagonal (self-correlation)
                                        if method_map[corr_method] == "pearson":
                                            # Pearson correlation p-value
                                            _, p = stats.pearsonr(corr_data[col_i], corr_data[col_j])
                                        elif method_map[corr_method] == "spearman":
                                            # Spearman correlation p-value
                                            _, p = stats.spearmanr(corr_data[col_i], corr_data[col_j])
                                        else:  # Kendall
                                            # Kendall correlation p-value
                                            _, p = stats.kendalltau(corr_data[col_i], corr_data[col_j])
                                            
                                        p_values.loc[col_i, col_j] = p
                            
                            # Display correlation matrix
                            st.markdown("#### Correlation Matrix")
                            
                            # Create heatmap
                            fig = px.imshow(
                                corr_matrix,
                                text_auto='.2f',
                                color_continuous_scale='RdBu_r',
                                zmin=-1,
                                zmax=1,
                                title=f"{corr_method} Correlation Matrix"
                            )
                            
                            st.plotly_chart(fig, use_container_width=True)
                            
                            # Display correlation matrix as dataframe
                            st.dataframe(corr_matrix.round(4), use_container_width=True)
                            
                            # Display p-values
                            st.markdown("#### Correlation P-Values")
                            st.dataframe(p_values.round(4), use_container_width=True)
                            
                            # Find and display significant correlations
                            st.markdown("#### Significant Correlations")
                            
                            # Create a table of significant correlations
                            significant_corrs = []
                            
                            for i, col_i in enumerate(corr_matrix.columns):
                                for j, col_j in enumerate(corr_matrix.columns):
                                    if i < j:  # Only upper triangle to avoid duplicates
                                        corr_value = corr_matrix.loc[col_i, col_j]
                                        p_value = p_values.loc[col_i, col_j]
                                        
                                        significant_corrs.append({
                                            'Variable 1': col_i,
                                            'Variable 2': col_j,
                                            'Correlation': corr_value,
                                            'P-Value': p_value,
                                            'Significant': p_value < 0.05
                                        })
                            
                            # Convert to DataFrame and sort by absolute correlation
                            sig_corr_df = pd.DataFrame(significant_corrs)
                            sig_corr_df['Abs Correlation'] = sig_corr_df['Correlation'].abs()
                            sig_corr_df = sig_corr_df.sort_values('Abs Correlation', ascending=False)
                            sig_corr_df = sig_corr_df.drop('Abs Correlation', axis=1)
                            
                            # Display significant correlations
                            st.dataframe(sig_corr_df.round(4), use_container_width=True)
                            
                            # Scatterplot matrix (limit to 5 columns for readability)
                            if len(selected_columns) <= 5:
                                st.markdown("#### Scatterplot Matrix")
                                
                                fig = px.scatter_matrix(
                                    corr_data,
                                    dimensions=selected_columns,
                                    title="Scatterplot Matrix",
                                    opacity=0.7
                                )
                                
                                # Update layout
                                fig.update_layout(
                                    width=150 * len(selected_columns),
                                    height=150 * len(selected_columns)
                                )
                                
                                st.plotly_chart(fig, use_container_width=True)
                            else:
                                st.info("Scatterplot matrix not shown for more than 5 variables. Select fewer variables for visualization.")
                                
                                # Allow user to select specific pairs for scatter plots
                                st.markdown("#### Pair Scatterplots")
                                st.markdown("Select variable pairs to visualize:")
                                
                                # Get top correlations
                                top_pairs = sig_corr_df.head(5)
                                
                                for _, row in top_pairs.iterrows():
                                    x_col, y_col = row['Variable 1'], row['Variable 2']
                                    corr_val = row['Correlation']
                                    
                                    # Create scatter plot
                                    fig = px.scatter(
                                        corr_data,
                                        x=x_col,
                                        y=y_col,
                                        title=f"{x_col} vs {y_col} (r = {corr_val:.2f})",
                                        trendline="ols"
                                    )
                                    
                                    st.plotly_chart(fig, use_container_width=True)
            
            with mv_tab3:
                st.markdown("#### Multiple Regression Analysis")
                st.markdown("""
                Multiple regression models the relationship between a dependent variable and multiple
                independent variables. This helps identify which factors most strongly influence your outcome variable.
                """)
                
                # Select dependent (target) variable
                target_var = st.selectbox(
                    "Select Dependent Variable (Y):",
                    numeric_columns
                )
                
                # Select independent (predictor) variables
                predictor_vars = st.multiselect(
                    "Select Independent Variables (X):",
                    [col for col in numeric_columns if col != target_var],
                    default=[col for col in numeric_columns[:min(4, len(numeric_columns))] if col != target_var]
                )
                
                if not predictor_vars:
                    st.warning("Multiple regression requires at least one predictor variable.")
                else:
                    # Button to run regression analysis
                    if st.button("Run Regression Analysis"):
                        # Get clean data
                        reg_data = st.session_state.cleaned_data[[target_var] + predictor_vars].dropna()
                        
                        if len(reg_data) < len(predictor_vars) + 5:
                            st.error(f"Insufficient data for regression analysis with {len(predictor_vars)} predictors.")
                        else:
                            # Prepare data
                            X = reg_data[predictor_vars]
                            y = reg_data[target_var]
                            
                            # Add constant term
                            X = sm.add_constant(X)
                            
                            # Fit model
                            model = sm.OLS(y, X).fit()
                            
                            # Display results
                            st.markdown("#### Regression Results")
                            
                            # Create summary table with proper indexing for parameters with spaces
                            coef_df = pd.DataFrame({
                                'Variable': ['Intercept'] + predictor_vars,
                                'Coefficient': [model.params.loc['const']] + [model.params.loc[x] for x in predictor_vars],
                                'Std Error': [model.bse.loc['const']] + [model.bse.loc[x] for x in predictor_vars],
                                't-value': [model.tvalues.loc['const']] + [model.tvalues.loc[x] for x in predictor_vars],
                                'p-value': [model.pvalues.loc['const']] + [model.pvalues.loc[x] for x in predictor_vars],
                                'Significant': [model.pvalues.loc['const'] < 0.05] + [model.pvalues.loc[x] < 0.05 for x in predictor_vars]
                            })
                            
                            # Display model metrics
                            col1, col2, col3 = st.columns(3)
                            
                            with col1:
                                st.metric("R-squared", f"{model.rsquared:.4f}")
                            
                            with col2:
                                st.metric("Adjusted R-squared", f"{model.rsquared_adj:.4f}")
                            
                            with col3:
                                st.metric("F-statistic p-value", f"{model.f_pvalue:.4f}")
                            
                            # Display coefficient table
                            st.dataframe(coef_df.round(4), use_container_width=True)
                            
                            # Create equation string with proper indexing for parameters with spaces
                            equation = f"{target_var} = {model.params.loc['const']:.4f}"
                            for var in predictor_vars:
                                coef = model.params.loc[var]
                                sign = '+' if coef >= 0 else '-'
                                equation += f" {sign} {abs(coef):.4f} × {var}"
                            
                            st.markdown(f"**Regression Equation:**  \n{equation}")
                            
                            # Model summary
                            st.markdown("#### Detailed Model Summary")
                            st.text(model.summary().as_text())
                            
                            # Visualize actual vs predicted values
                            st.markdown("#### Actual vs Predicted Values")
                            
                            # Calculate predicted values
                            reg_data['Predicted'] = model.predict(X)
                            
                            # Create scatter plot
                            fig = px.scatter(
                                reg_data,
                                x=target_var,
                                y='Predicted',
                                title=f"Actual vs Predicted {target_var}",
                                labels={target_var: f"Actual {target_var}", 'Predicted': f"Predicted {target_var}"}
                            )
                            
                            # Add identity line
                            fig.add_trace(
                                go.Scatter(
                                    x=[reg_data[target_var].min(), reg_data[target_var].max()],
                                    y=[reg_data[target_var].min(), reg_data[target_var].max()],
                                    mode='lines',
                                    line=dict(color='red', dash='dash'),
                                    name='y = x'
                                )
                            )
                            
                            st.plotly_chart(fig, use_container_width=True)
                            
                            # Residual analysis
                            st.markdown("#### Residual Analysis")
                            
                            # Calculate residuals
                            reg_data['Residuals'] = reg_data[target_var] - reg_data['Predicted']
                            
                            # Residuals vs Predicted
                            fig = px.scatter(
                                reg_data,
                                x='Predicted',
                                y='Residuals',
                                title="Residuals vs Predicted Values",
                                labels={'Predicted': f"Predicted {target_var}", 'Residuals': 'Residuals'}
                            )
                            
                            # Add horizontal line at y=0
                            fig.add_hline(
                                y=0,
                                line_dash="dash",
                                line_color="red"
                            )
                            
                            st.plotly_chart(fig, use_container_width=True)
                            
                            # QQ plot for residuals
                            st.markdown("#### QQ Plot of Residuals")
                            
                            # Create QQ plot
                            qq_buf = BytesIO()
                            fig, ax = plt.subplots(figsize=(8, 6))
                            stats.probplot(reg_data['Residuals'], plot=ax)
                            ax.set_title('QQ Plot of Residuals')
                            fig.savefig(qq_buf, format='png', dpi=300, bbox_inches='tight')
                            qq_buf.seek(0)
                            
                            # Close matplotlib figure
                            plt.close(fig)
                            
                            # Display QQ plot
                            st.image(qq_buf)
                            
                            # Histogram of residuals
                            fig = px.histogram(
                                reg_data,
                                x='Residuals',
                                title="Distribution of Residuals",
                                labels={'Residuals': 'Residuals'},
                                marginal="box"
                            )
                            
                            st.plotly_chart(fig, use_container_width=True)
                            
                            # Check for multicollinearity
                            st.markdown("#### Multicollinearity Check")
                            
                            # Calculate VIF (Variance Inflation Factor)
                            from statsmodels.stats.outliers_influence import variance_inflation_factor
                            
                            # Calculate VIF for each predictor
                            vif_data = pd.DataFrame()
                            vif_data["Variable"] = X.columns
                            vif_data["VIF"] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
                            
                            # Interpret VIF
                            vif_data["Multicollinearity"] = vif_data["VIF"].apply(
                                lambda x: "Low" if x < 5 else "Moderate" if x < 10 else "High"
                            )
                            
                            st.dataframe(vif_data.round(2), use_container_width=True)
                            
                            if any(vif_data["VIF"] > 10):
                                st.warning("High multicollinearity detected. Some predictor variables are strongly correlated, which can make coefficient estimates unstable.")
    st.markdown('</div>', unsafe_allow_html=True)

with tab3:
    st.markdown('<div class="dashboard-card">', unsafe_allow_html=True)
    st.markdown("### Bayesian Analysis")
    st.markdown("""
    Bayesian analysis provides a probabilistic framework for statistical inference, allowing you to
    incorporate prior knowledge and quantify uncertainty in your estimates.
    """)
    
    # Only allow if data is loaded
    if st.session_state.cleaned_data is not None:
        # Get numeric columns
        numeric_columns = st.session_state.cleaned_data.select_dtypes(include=['int64', 'float64']).columns.tolist()
        categorical_columns = st.session_state.cleaned_data.select_dtypes(include=['object']).columns.tolist()
        
        if len(numeric_columns) == 0:
            st.warning("Bayesian analysis requires numeric columns in your data.")
        else:
            # Create subtabs for different Bayesian analyses
            bay_tab1, bay_tab2 = st.tabs([
                "Bayesian Estimation",
                "A/B Testing"
            ])
            
            with bay_tab1:
                st.markdown("#### Bayesian Parameter Estimation")
                st.markdown("""
                Estimate the parameters of a distribution using Bayesian methods.
                This provides not just point estimates but full probability distributions.
                """)
                
                # Select column for Bayesian estimation
                selected_column = st.selectbox(
                    "Select Column for Bayesian Estimation:",
                    numeric_columns
                )
                
                # Select distribution
                distribution = st.selectbox(
                    "Select Distribution:",
                    ["Normal (Gaussian)", "Exponential", "Poisson", "Bernoulli (for binary data)"]
                )
                
                # Sampling settings
                st.markdown("#### Sampling Settings")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    num_samples = st.number_input(
                        "Number of Samples:",
                        min_value=1000,
                        max_value=10000,
                        value=2000,
                        step=1000
                    )
                
                with col2:
                    num_burnin = st.number_input(
                        "Burn-in Samples:",
                        min_value=500,
                        max_value=5000,
                        value=1000,
                        step=500
                    )
                
                # Prior settings
                st.markdown("#### Prior Settings")
                
                if distribution == "Normal (Gaussian)":
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        mean_prior = st.selectbox(
                            "Mean Prior:",
                            ["Uninformative", "Normal"]
                        )
                        
                        if mean_prior == "Normal":
                            mean_prior_mean = st.number_input("Prior Mean:", value=0.0)
                            mean_prior_sd = st.number_input("Prior Standard Deviation:", min_value=0.1, value=10.0)
                    
                    with col2:
                        sd_prior = st.selectbox(
                            "Standard Deviation Prior:",
                            ["Uniform", "HalfNormal"]
                        )
                        
                        if sd_prior == "HalfNormal":
                            sd_prior_sd = st.number_input("Prior Scale:", min_value=0.1, value=10.0)
                elif distribution == "Exponential":
                    lambda_prior = st.selectbox(
                        "Rate Parameter (λ) Prior:",
                        ["Gamma", "HalfNormal"]
                    )
                    
                    if lambda_prior == "Gamma":
                        lambda_prior_alpha = st.number_input("Prior Shape (α):", min_value=0.1, value=1.0)
                        lambda_prior_beta = st.number_input("Prior Rate (β):", min_value=0.1, value=1.0)
                    elif lambda_prior == "HalfNormal":
                        lambda_prior_sd = st.number_input("Prior Scale:", min_value=0.1, value=10.0)
                elif distribution == "Poisson":
                    lambda_prior = st.selectbox(
                        "Rate Parameter (λ) Prior:",
                        ["Gamma", "Exponential"]
                    )
                    
                    if lambda_prior == "Gamma":
                        lambda_prior_alpha = st.number_input("Prior Shape (α):", min_value=0.1, value=1.0)
                        lambda_prior_beta = st.number_input("Prior Rate (β):", min_value=0.1, value=1.0)
                    elif lambda_prior == "Exponential":
                        lambda_prior_rate = st.number_input("Prior Rate:", min_value=0.001, value=1.0)
                elif distribution == "Bernoulli (for binary data)":
                    p_prior = st.selectbox(
                        "Probability (p) Prior:",
                        ["Beta", "Uniform"]
                    )
                    
                    if p_prior == "Beta":
                        p_prior_alpha = st.number_input("Prior α:", min_value=0.1, value=1.0)
                        p_prior_beta = st.number_input("Prior β:", min_value=0.1, value=1.0)
                
                # Button to run Bayesian estimation
                if st.button("Run Bayesian Estimation"):
                    # Get clean data
                    clean_data = st.session_state.cleaned_data[selected_column].dropna()
                    
                    if len(clean_data) < 10:
                        st.error("Insufficient data for Bayesian estimation (minimum 10 data points required).")
                    else:
                        # Check for data compatibility with selected distribution
                        if distribution == "Bernoulli (for binary data)":
                            # Check if data is binary (0/1)
                            unique_values = clean_data.unique()
                            if not np.all(np.isin(unique_values, [0, 1])):
                                st.error(f"Bernoulli distribution requires binary (0/1) data. Your data contains: {unique_values}")
                                st.stop()
                        elif distribution == "Poisson":
                            # Check if data is non-negative integers
                            if not np.all(clean_data >= 0) or not np.all(clean_data.apply(lambda x: float(x).is_integer())):
                                st.error("Poisson distribution requires non-negative integer data.")
                                st.stop()
                        elif distribution == "Exponential":
                            # Check if data is positive
                            if not np.all(clean_data > 0):
                                st.error("Exponential distribution requires positive data.")
                                st.stop()
                        
                        # Create a progress bar
                        progress_bar = st.progress(0)
                        progress_text = st.empty()
                        progress_text.text("Setting up the Bayesian model...")
                        
                        try:
                            # Define and run the Bayesian model
                            with pm.Model() as model:
                                # Define priors based on distribution
                                if distribution == "Normal (Gaussian)":
                                    # Define priors for mean and standard deviation
                                    if mean_prior == "Uninformative":
                                        mu = pm.Normal("mu", mu=0, sigma=1e6)
                                    else:  # "Normal"
                                        mu = pm.Normal("mu", mu=mean_prior_mean, sigma=mean_prior_sd)
                                    
                                    if sd_prior == "Uniform":
                                        sigma = pm.Uniform("sigma", lower=0, upper=100)
                                    else:  # "HalfNormal"
                                        sigma = pm.HalfNormal("sigma", sigma=sd_prior_sd)
                                    
                                    # Define likelihood
                                    likelihood = pm.Normal("likelihood", mu=mu, sigma=sigma, observed=clean_data.values)
                                    
                                    # Parameter names for summary
                                    param_names = ["Mean (μ)", "Standard Deviation (σ)"]
                                    param_vars = ["mu", "sigma"]
                                
                                elif distribution == "Exponential":
                                    # Define prior for rate parameter
                                    if lambda_prior == "Gamma":
                                        lam = pm.Gamma("lambda", alpha=lambda_prior_alpha, beta=lambda_prior_beta)
                                    else:  # "HalfNormal"
                                        lam = pm.HalfNormal("lambda", sigma=lambda_prior_sd)
                                    
                                    # Define likelihood
                                    likelihood = pm.Exponential("likelihood", lam=lam, observed=clean_data.values)
                                    
                                    # Parameter names for summary
                                    param_names = ["Rate Parameter (λ)"]
                                    param_vars = ["lambda"]
                                
                                elif distribution == "Poisson":
                                    # Define prior for rate parameter
                                    if lambda_prior == "Gamma":
                                        lam = pm.Gamma("lambda", alpha=lambda_prior_alpha, beta=lambda_prior_beta)
                                    else:  # "Exponential"
                                        lam = pm.Exponential("lambda", lam=lambda_prior_rate)
                                    
                                    # Define likelihood
                                    likelihood = pm.Poisson("likelihood", mu=lam, observed=clean_data.values)
                                    
                                    # Parameter names for summary
                                    param_names = ["Rate Parameter (λ)"]
                                    param_vars = ["lambda"]
                                
                                elif distribution == "Bernoulli (for binary data)":
                                    # Define prior for probability
                                    if p_prior == "Beta":
                                        p = pm.Beta("p", alpha=p_prior_alpha, beta=p_prior_beta)
                                    else:  # "Uniform"
                                        p = pm.Uniform("p", lower=0, upper=1)
                                    
                                    # Define likelihood
                                    likelihood = pm.Bernoulli("likelihood", p=p, observed=clean_data.values)
                                    
                                    # Parameter names for summary
                                    param_names = ["Probability (p)"]
                                    param_vars = ["p"]
                                
                                # Update progress
                                progress_bar.progress(0.2)
                                progress_text.text("Starting MCMC sampling...")
                                
                                # Sample using MCMC
                                trace = pm.sample(
                                    draws=num_samples,
                                    tune=num_burnin,
                                    return_inferencedata=True,
                                    progressbar=False
                                )
                                
                                # Update progress
                                progress_bar.progress(0.8)
                                progress_text.text("Analyzing results...")
                                
                                # Compute summary statistics
                                summary = az.summary(trace, var_names=param_vars)
                                
                                # Update progress
                                progress_bar.progress(1.0)
                                progress_text.text("Done!")
                                
                                # Clear progress bar and text
                                time.sleep(0.5)
                                progress_bar.empty()
                                progress_text.empty()
                                
                                # Display results
                                st.markdown("#### Bayesian Estimation Results")
                                
                                # Create a nicer summary table
                                # Create a results dataframe with appropriate columns
                                result_df = pd.DataFrame({
                                    'Parameter': param_names,
                                    'Mean': summary['mean'].values,
                                    'Std Dev': summary['sd'].values
                                })
                                
                                # Add HDI intervals if available
                                if 'hdi_2.5%' in summary.columns:
                                    result_df['2.5%'] = summary['hdi_2.5%'].values
                                    result_df['97.5%'] = summary['hdi_97.5%'].values
                                elif 'hdi 2.5%' in summary.columns:
                                    result_df['2.5%'] = summary['hdi 2.5%'].values
                                    result_df['97.5%'] = summary['hdi 97.5%'].values
                                elif '2.5%' in summary.columns:
                                    result_df['2.5%'] = summary['2.5%'].values
                                    result_df['97.5%'] = summary['97.5%'].values
                                
                                st.dataframe(result_df.round(4), use_container_width=True)
                                
                                # Create posterior distribution plots
                                st.markdown("#### Posterior Distributions")
                                
                                for i, var in enumerate(param_vars):
                                    # Create matplotlib figure for posterior
                                    fig, ax = plt.subplots(figsize=(10, 6))
                                    az.plot_posterior(trace, var_names=[var], ax=ax)
                                    ax.set_title(f"Posterior Distribution for {param_names[i]}")
                                    
                                    # Convert to image
                                    buf = BytesIO()
                                    fig.savefig(buf, format='png', dpi=300, bbox_inches='tight')
                                    buf.seek(0)
                                    
                                    # Close matplotlib figure
                                    plt.close(fig)
                                    
                                    # Display image
                                    st.image(buf)
                                
                                # Create trace plot
                                st.markdown("#### MCMC Sampling Traces")
                                st.markdown("These plots show how the MCMC sampler explored the parameter space.")
                                
                                # Create matplotlib figure for trace plot
                                fig, ax = plt.subplots(figsize=(10, 6 * len(param_vars)))
                                az.plot_trace(trace, var_names=param_vars, compact=False, figsize=(10, 6 * len(param_vars)))
                                
                                # Convert to image
                                buf = BytesIO()
                                fig.savefig(buf, format='png', dpi=300, bbox_inches='tight')
                                buf.seek(0)
                                
                                # Close matplotlib figure
                                plt.close(fig)
                                
                                # Display image
                                st.image(buf)
                                
                                # Create posterior predictive check plot
                                st.markdown("#### Posterior Predictive Check")
                                st.markdown("This plot compares your observed data to simulated data from the posterior distribution.")
                                
                                # Create posterior predictive samples
                                with model:
                                    ppc = pm.sample_posterior_predictive(trace, var_names=["likelihood"], random_seed=42)
                                
                                # Create matplotlib figure for posterior predictive check
                                fig, ax = plt.subplots(figsize=(10, 6))
                                
                                # Plot histogram of observed data
                                ax.hist(clean_data, bins=30, alpha=0.5, label="Observed Data", density=True)
                                
                                # Plot histogram of simulated data
                                ax.hist(ppc.posterior_predictive["likelihood"].values.flatten(), bins=30, alpha=0.5, label="Simulated Data", density=True)
                                
                                ax.set_title("Posterior Predictive Check")
                                ax.set_xlabel(selected_column)
                                ax.set_ylabel("Density")
                                ax.legend()
                                
                                # Convert to image
                                buf = BytesIO()
                                fig.savefig(buf, format='png', dpi=300, bbox_inches='tight')
                                buf.seek(0)
                                
                                # Close matplotlib figure
                                plt.close(fig)
                                
                                # Display image
                                st.image(buf)
                                
                                # Interpret results
                                st.markdown("#### Interpretation")
                                
                                if distribution == "Normal (Gaussian)":
                                    # Get parameter means
                                    mu_mean = summary.loc["mu", "mean"]
                                    sigma_mean = summary.loc["sigma", "mean"]
                                    
                                    # Handle different HDI column naming formats
                                    if 'hdi_2.5%' in summary.columns:
                                        mu_hdi_low = summary.loc["mu", "hdi_2.5%"]
                                        mu_hdi_high = summary.loc["mu", "hdi_97.5%"]
                                        sigma_hdi_low = summary.loc["sigma", "hdi_2.5%"]
                                        sigma_hdi_high = summary.loc["sigma", "hdi_97.5%"]
                                    elif 'hdi 2.5%' in summary.columns:
                                        mu_hdi_low = summary.loc["mu", "hdi 2.5%"]
                                        mu_hdi_high = summary.loc["mu", "hdi 97.5%"]
                                        sigma_hdi_low = summary.loc["sigma", "hdi 2.5%"]
                                        sigma_hdi_high = summary.loc["sigma", "hdi 97.5%"]
                                    elif '2.5%' in summary.columns:
                                        mu_hdi_low = summary.loc["mu", "2.5%"]
                                        mu_hdi_high = summary.loc["mu", "97.5%"]
                                        sigma_hdi_low = summary.loc["sigma", "2.5%"]
                                        sigma_hdi_high = summary.loc["sigma", "97.5%"]
                                    else:
                                        # Fallback if HDI columns aren't found
                                        mu_hdi_low = mu_mean - 2 * summary.loc["mu", "sd"]
                                        mu_hdi_high = mu_mean + 2 * summary.loc["mu", "sd"]
                                        sigma_hdi_low = sigma_mean - 2 * summary.loc["sigma", "sd"]
                                        sigma_hdi_high = sigma_mean + 2 * summary.loc["sigma", "sd"]
                                    
                                    st.markdown(f"""
                                    Based on the Bayesian analysis:
                                    
                                    - The mean of {selected_column} is estimated to be **{mu_mean:.4f}** with a 95% highest density interval of [{mu_hdi_low:.4f}, {mu_hdi_high:.4f}].
                                    - The standard deviation is estimated to be **{sigma_mean:.4f}** with a 95% highest density interval of [{sigma_hdi_low:.4f}, {sigma_hdi_high:.4f}].
                                    
                                    This means we are 95% confident that the true mean lies within the interval [{mu_hdi_low:.4f}, {mu_hdi_high:.4f}].
                                    """)
                                
                                elif distribution == "Exponential":
                                    # Get parameter mean
                                    lambda_mean = summary.loc["lambda", "mean"]
                                    
                                    # Handle different HDI column naming formats
                                    if 'hdi_2.5%' in summary.columns:
                                        lambda_hdi_low = summary.loc["lambda", "hdi_2.5%"]
                                        lambda_hdi_high = summary.loc["lambda", "hdi_97.5%"]
                                    elif 'hdi 2.5%' in summary.columns:
                                        lambda_hdi_low = summary.loc["lambda", "hdi 2.5%"]
                                        lambda_hdi_high = summary.loc["lambda", "hdi 97.5%"]
                                    elif '2.5%' in summary.columns:
                                        lambda_hdi_low = summary.loc["lambda", "2.5%"]
                                        lambda_hdi_high = summary.loc["lambda", "97.5%"]
                                    else:
                                        # Fallback if HDI columns aren't found
                                        lambda_hdi_low = lambda_mean - 2 * summary.loc["lambda", "sd"]
                                        lambda_hdi_high = lambda_mean + 2 * summary.loc["lambda", "sd"]
                                    
                                    mean_value = 1 / lambda_mean
                                    
                                    st.markdown(f"""
                                    Based on the Bayesian analysis:
                                    
                                    - The rate parameter (λ) of the exponential distribution is estimated to be **{lambda_mean:.4f}** with a 95% highest density interval of [{lambda_hdi_low:.4f}, {lambda_hdi_high:.4f}].
                                    - This corresponds to a mean value of **{mean_value:.4f}** for {selected_column}.
                                    
                                    This means we are 95% confident that the true rate parameter lies within the interval [{lambda_hdi_low:.4f}, {lambda_hdi_high:.4f}].
                                    """)
                                
                                elif distribution == "Poisson":
                                    # Get parameter mean
                                    lambda_mean = summary.loc["lambda", "mean"]
                                    
                                    # Handle different HDI column naming formats
                                    if 'hdi_2.5%' in summary.columns:
                                        lambda_hdi_low = summary.loc["lambda", "hdi_2.5%"]
                                        lambda_hdi_high = summary.loc["lambda", "hdi_97.5%"]
                                    elif 'hdi 2.5%' in summary.columns:
                                        lambda_hdi_low = summary.loc["lambda", "hdi 2.5%"]
                                        lambda_hdi_high = summary.loc["lambda", "hdi 97.5%"]
                                    elif '2.5%' in summary.columns:
                                        lambda_hdi_low = summary.loc["lambda", "2.5%"]
                                        lambda_hdi_high = summary.loc["lambda", "97.5%"]
                                    else:
                                        # Fallback if HDI columns aren't found
                                        lambda_hdi_low = lambda_mean - 2 * summary.loc["lambda", "sd"]
                                        lambda_hdi_high = lambda_mean + 2 * summary.loc["lambda", "sd"]
                                    
                                    st.markdown(f"""
                                    Based on the Bayesian analysis:
                                    
                                    - The rate parameter (λ) of the Poisson distribution is estimated to be **{lambda_mean:.4f}** with a 95% highest density interval of [{lambda_hdi_low:.4f}, {lambda_hdi_high:.4f}].
                                    - This rate represents both the mean and variance of the Poisson distribution.
                                    
                                    This means we are 95% confident that the true rate parameter lies within the interval [{lambda_hdi_low:.4f}, {lambda_hdi_high:.4f}].
                                    """)
                                
                                elif distribution == "Bernoulli (for binary data)":
                                    # Get parameter mean
                                    p_mean = summary.loc["p", "mean"]
                                    
                                    # Handle different HDI column naming formats
                                    if 'hdi_2.5%' in summary.columns:
                                        p_hdi_low = summary.loc["p", "hdi_2.5%"]
                                        p_hdi_high = summary.loc["p", "hdi_97.5%"]
                                    elif 'hdi 2.5%' in summary.columns:
                                        p_hdi_low = summary.loc["p", "hdi 2.5%"]
                                        p_hdi_high = summary.loc["p", "hdi 97.5%"]
                                    elif '2.5%' in summary.columns:
                                        p_hdi_low = summary.loc["p", "2.5%"]
                                        p_hdi_high = summary.loc["p", "97.5%"]
                                    else:
                                        # Fallback if HDI columns aren't found
                                        p_hdi_low = max(0.0, p_mean - 2 * summary.loc["p", "sd"])
                                        p_hdi_high = min(1.0, p_mean + 2 * summary.loc["p", "sd"])
                                    
                                    st.markdown(f"""
                                    Based on the Bayesian analysis:
                                    
                                    - The probability parameter (p) of the Bernoulli distribution is estimated to be **{p_mean:.4f}** with a 95% highest density interval of [{p_hdi_low:.4f}, {p_hdi_high:.4f}].
                                    - This means that the estimated probability of observing a value of 1 is {p_mean:.1%}.
                                    
                                    This means we are 95% confident that the true probability lies within the interval [{p_hdi_low:.4f}, {p_hdi_high:.4f}].
                                    """)
                        
                        except Exception as e:
                            st.error(f"Error during Bayesian estimation: {str(e)}")
                            st.error(traceback.format_exc())
            
            with bay_tab2:
                st.markdown("#### Bayesian A/B Testing")
                st.markdown("""
                Compare two groups using a Bayesian approach to determine which performs better.
                This provides probabilities of one group outperforming the other, rather than just p-values.
                """)
                
                if len(categorical_columns) == 0:
                    st.warning("Bayesian A/B testing requires at least one categorical column to define groups.")
                else:
                    # Select columns for A/B testing
                    group_column = st.selectbox(
                        "Select Group Column (categorical):",
                        categorical_columns,
                        key="bayesian_group_column"
                    )
                    
                    metric_column = st.selectbox(
                        "Select Metric Column (numeric):",
                        numeric_columns,
                        key="bayesian_metric_column"
                    )
                    
                    # Show group counts
                    if group_column and st.session_state.cleaned_data is not None:
                        group_counts = st.session_state.cleaned_data[group_column].value_counts()
                        
                        if len(group_counts) < 2:
                            st.warning(f"A/B testing requires at least 2 groups. Your data has only {len(group_counts)} group(s) in the '{group_column}' column.")
                        elif len(group_counts) > 2:
                            st.info(f"Found {len(group_counts)} groups. Please select exactly 2 groups for A/B testing.")
                            
                            # Let user select exactly 2 groups
                            group_options = sorted(group_counts.index.tolist())
                            group_a = st.selectbox("Select Group A:", group_options, index=0, key="bayesian_group_a")
                            
                            # Filter options for group B to prevent selecting same as group A
                            group_b_options = [g for g in group_options if g != group_a]
                            group_b = st.selectbox("Select Group B:", group_b_options, index=0, key="bayesian_group_b")
                        else:
                            # Exactly 2 groups, use them directly
                            group_options = sorted(group_counts.index.tolist())
                            group_a = group_options[0]
                            group_b = group_options[1]
                            
                            st.info(f"Found exactly 2 groups: '{group_a}' ({group_counts[group_a]} data points) and '{group_b}' ({group_counts[group_b]} data points)")
                    
                    # Sampling settings
                    st.markdown("#### Sampling Settings")
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        num_samples = st.number_input(
                            "Number of Samples for A/B Testing:",
                            min_value=1000,
                            max_value=10000,
                            value=2000,
                            step=1000
                        )
                    
                    with col2:
                        num_burnin = st.number_input(
                            "Burn-in Samples for A/B Testing:",
                            min_value=500,
                            max_value=5000,
                            value=1000,
                            step=500
                        )
                    
                    # Prior settings
                    st.markdown("#### Prior Settings")
                    
                    prior_type = st.selectbox(
                        "Prior Type:",
                        ["Uninformative", "Weakly Informative"],
                        key="bayesian_prior_type"
                    )
                    
                    # Button to run A/B testing
                    if st.button("Run Bayesian A/B Test"):
                        # Get data for both groups
                        if 'group_a' not in locals() or 'group_b' not in locals():
                            st.error("Please select two groups for A/B testing.")
                        else:
                            # Get clean data for both groups
                            group_a_data = st.session_state.cleaned_data[
                                st.session_state.cleaned_data[group_column] == group_a
                            ][metric_column].dropna()
                            
                            group_b_data = st.session_state.cleaned_data[
                                st.session_state.cleaned_data[group_column] == group_b
                            ][metric_column].dropna()
                            
                            if len(group_a_data) < 5 or len(group_b_data) < 5:
                                st.error(f"Insufficient data for A/B testing. Group A: {len(group_a_data)} points, Group B: {len(group_b_data)} points. Each group needs at least 5 data points.")
                            else:
                                # Create a progress bar
                                progress_bar = st.progress(0)
                                progress_text = st.empty()
                                progress_text.text("Setting up the Bayesian model...")
                                
                                try:
                                    # Define and run the Bayesian A/B testing model
                                    with pm.Model() as model:
                                        # Define priors based on prior type
                                        if prior_type == "Uninformative":
                                            # Uninformative priors
                                            group_a_mean = pm.Normal("group_a_mean", mu=0, sigma=1e6)
                                            group_b_mean = pm.Normal("group_b_mean", mu=0, sigma=1e6)
                                            group_a_sigma = pm.HalfNormal("group_a_sigma", sigma=1e6)
                                            group_b_sigma = pm.HalfNormal("group_b_sigma", sigma=1e6)
                                        else:  # "Weakly Informative"
                                            # Weakly informative priors based on data
                                            all_data = pd.concat([group_a_data, group_b_data])
                                            data_mean = all_data.mean()
                                            data_std = all_data.std()
                                            
                                            group_a_mean = pm.Normal("group_a_mean", mu=data_mean, sigma=data_std * 5)
                                            group_b_mean = pm.Normal("group_b_mean", mu=data_mean, sigma=data_std * 5)
                                            group_a_sigma = pm.HalfNormal("group_a_sigma", sigma=data_std * 5)
                                            group_b_sigma = pm.HalfNormal("group_b_sigma", sigma=data_std * 5)
                                        
                                        # Define likelihoods
                                        group_a_likelihood = pm.Normal("group_a_likelihood", mu=group_a_mean, sigma=group_a_sigma, observed=group_a_data.values)
                                        group_b_likelihood = pm.Normal("group_b_likelihood", mu=group_b_mean, sigma=group_b_sigma, observed=group_b_data.values)
                                        
                                        # Calculate difference between means
                                        diff = pm.Deterministic("diff", group_b_mean - group_a_mean)
                                        
                                        # Calculate percent improvement
                                        percent_diff = pm.Deterministic("percent_diff", (group_b_mean - group_a_mean) / pm.abs(group_a_mean) * 100)
                                        
                                        # Calculate probability of improvement
                                        prob_improvement = pm.Deterministic("prob_improvement", pm.math.switch(diff > 0, 1, 0))
                                        
                                        # Update progress
                                        progress_bar.progress(0.2)
                                        progress_text.text("Starting MCMC sampling...")
                                        
                                        # Sample using MCMC
                                        trace = pm.sample(
                                            draws=num_samples,
                                            tune=num_burnin,
                                            return_inferencedata=True,
                                            progressbar=False
                                        )
                                        
                                        # Update progress
                                        progress_bar.progress(0.8)
                                        progress_text.text("Analyzing results...")
                                        
                                        # Compute summary statistics
                                        summary = az.summary(trace, var_names=["group_a_mean", "group_b_mean", "diff", "percent_diff"])
                                        
                                        # Calculate probability that B > A
                                        prob_b_better = (trace.posterior["diff"] > 0).mean().item()
                                        
                                        # Update progress
                                        progress_bar.progress(1.0)
                                        progress_text.text("Done!")
                                        
                                        # Clear progress bar and text
                                        time.sleep(0.5)
                                        progress_bar.empty()
                                        progress_text.empty()
                                        
                                        # Display results
                                        st.markdown("#### Bayesian A/B Test Results")
                                        
                                        # Basic statistics
                                        col1, col2 = st.columns(2)
                                        
                                        with col1:
                                            st.markdown(f"**Group A ({group_a})**")
                                            st.markdown(f"Count: {len(group_a_data)}")
                                            st.markdown(f"Mean: {group_a_data.mean():.4f}")
                                            st.markdown(f"Std Dev: {group_a_data.std():.4f}")
                                        
                                        with col2:
                                            st.markdown(f"**Group B ({group_b})**")
                                            st.markdown(f"Count: {len(group_b_data)}")
                                            st.markdown(f"Mean: {group_b_data.mean():.4f}")
                                            st.markdown(f"Std Dev: {group_b_data.std():.4f}")
                                        
                                        # Create a summary table
                                        # Create a results dataframe with appropriate columns
                                        result_df = pd.DataFrame({
                                            'Parameter': ["Group A Mean", "Group B Mean", "Difference (B - A)", "Percent Difference"],
                                            'Mean': summary['mean'].values,
                                            'Std Dev': summary['sd'].values
                                        })
                                        
                                        # Add HDI intervals if available
                                        if 'hdi_2.5%' in summary.columns:
                                            result_df['2.5%'] = summary['hdi_2.5%'].values
                                            result_df['97.5%'] = summary['hdi_97.5%'].values
                                        elif 'hdi 2.5%' in summary.columns:
                                            result_df['2.5%'] = summary['hdi 2.5%'].values
                                            result_df['97.5%'] = summary['hdi 97.5%'].values
                                        elif '2.5%' in summary.columns:
                                            result_df['2.5%'] = summary['2.5%'].values
                                            result_df['97.5%'] = summary['97.5%'].values
                                        
                                        st.dataframe(result_df.round(4), use_container_width=True)
                                        
                                        # Probability metrics
                                        st.markdown("#### Probability Analysis")
                                        
                                        # Create metrics
                                        col1, col2 = st.columns(2)
                                        
                                        with col1:
                                            st.metric(
                                                f"Probability {group_b} > {group_a}",
                                                f"{prob_b_better:.1%}"
                                            )
                                        
                                        with col2:
                                            st.metric(
                                                f"Probability {group_a} > {group_b}",
                                                f"{1 - prob_b_better:.1%}"
                                            )
                                        
                                        # Create posterior distribution plots
                                        st.markdown("#### Posterior Distributions")
                                        
                                        # Create matplotlib figure for group means
                                        fig, ax = plt.subplots(figsize=(10, 6))
                                        az.plot_posterior(
                                            trace, 
                                            var_names=["group_a_mean", "group_b_mean"], 
                                            ax=ax,
                                            hdi_prob=0.95
                                        )
                                        ax.set_title(f"Posterior Distributions of Group Means")
                                        
                                        # Convert to image
                                        buf = BytesIO()
                                        fig.savefig(buf, format='png', dpi=300, bbox_inches='tight')
                                        buf.seek(0)
                                        
                                        # Close matplotlib figure
                                        plt.close(fig)
                                        
                                        # Display image
                                        st.image(buf)
                                        
                                        # Create matplotlib figure for difference
                                        fig, ax = plt.subplots(figsize=(10, 6))
                                        az.plot_posterior(
                                            trace, 
                                            var_names=["diff"], 
                                            ax=ax,
                                            hdi_prob=0.95,
                                            ref_val=0
                                        )
                                        ax.set_title(f"Posterior Distribution of Difference (B - A)")
                                        
                                        # Convert to image
                                        buf = BytesIO()
                                        fig.savefig(buf, format='png', dpi=300, bbox_inches='tight')
                                        buf.seek(0)
                                        
                                        # Close matplotlib figure
                                        plt.close(fig)
                                        
                                        # Display image
                                        st.image(buf)
                                        
                                        # Create matplotlib figure for percent difference
                                        fig, ax = plt.subplots(figsize=(10, 6))
                                        az.plot_posterior(
                                            trace, 
                                            var_names=["percent_diff"], 
                                            ax=ax,
                                            hdi_prob=0.95,
                                            ref_val=0
                                        )
                                        ax.set_title(f"Posterior Distribution of Percent Difference")
                                        
                                        # Convert to image
                                        buf = BytesIO()
                                        fig.savefig(buf, format='png', dpi=300, bbox_inches='tight')
                                        buf.seek(0)
                                        
                                        # Close matplotlib figure
                                        plt.close(fig)
                                        
                                        # Display image
                                        st.image(buf)
                                        
                                        # Interpret results
                                        st.markdown("#### Interpretation")
                                        
                                        # Get the mean for diff and percent_diff
                                        diff_mean = summary.loc["diff", "mean"]
                                        percent_diff_mean = summary.loc["percent_diff", "mean"]
                                        
                                        # Handle different HDI column naming formats
                                        if 'hdi_2.5%' in summary.columns:
                                            diff_hdi_low = summary.loc["diff", "hdi_2.5%"]
                                            diff_hdi_high = summary.loc["diff", "hdi_97.5%"]
                                        elif 'hdi 2.5%' in summary.columns:
                                            diff_hdi_low = summary.loc["diff", "hdi 2.5%"]
                                            diff_hdi_high = summary.loc["diff", "hdi 97.5%"]
                                        elif '2.5%' in summary.columns:
                                            diff_hdi_low = summary.loc["diff", "2.5%"]
                                            diff_hdi_high = summary.loc["diff", "97.5%"]
                                        else:
                                            # Fallback if HDI columns aren't found
                                            diff_hdi_low = diff_mean - 2 * summary.loc["diff", "sd"]
                                            diff_hdi_high = diff_mean + 2 * summary.loc["diff", "sd"]
                                        
                                        if diff_hdi_low > 0:
                                            # B is definitely better
                                            strength = "strong" if prob_b_better > 0.95 else "moderate"
                                            st.markdown(f"""
                                            Based on the Bayesian A/B test:
                                            
                                            - There is {prob_b_better:.1%} probability that **{group_b}** is better than **{group_a}** for the metric **{metric_column}**.
                                            - The expected difference is **{diff_mean:.4f}** with a 95% highest density interval of [{diff_hdi_low:.4f}, {diff_hdi_high:.4f}].
                                            - This represents a **{percent_diff_mean:.1f}%** improvement.
                                            
                                            This is {strength} evidence that **{group_b}** outperforms **{group_a}**.
                                            """)
                                        elif diff_hdi_high < 0:
                                            # A is definitely better
                                            strength = "strong" if prob_b_better < 0.05 else "moderate"
                                            st.markdown(f"""
                                            Based on the Bayesian A/B test:
                                            
                                            - There is {1-prob_b_better:.1%} probability that **{group_a}** is better than **{group_b}** for the metric **{metric_column}**.
                                            - The expected difference is **{diff_mean:.4f}** with a 95% highest density interval of [{diff_hdi_low:.4f}, {diff_hdi_high:.4f}].
                                            - This represents a **{-percent_diff_mean:.1f}%** decrease.
                                            
                                            This is {strength} evidence that **{group_a}** outperforms **{group_b}**.
                                            """)
                                        else:
                                            # Inconclusive
                                            st.markdown(f"""
                                            Based on the Bayesian A/B test:
                                            
                                            - The results are inconclusive. The 95% highest density interval for the difference [{diff_hdi_low:.4f}, {diff_hdi_high:.4f}] includes zero.
                                            - There is {prob_b_better:.1%} probability that **{group_b}** is better than **{group_a}** for the metric **{metric_column}**.
                                            - The expected difference is **{diff_mean:.4f}**, but we cannot be confident about the direction of the effect.
                                            
                                            More data may be needed to reach a definitive conclusion.
                                            """)
                                
                                except Exception as e:
                                    st.error(f"Error during Bayesian A/B testing: {str(e)}")
                                    st.error(traceback.format_exc())
    st.markdown('</div>', unsafe_allow_html=True)

with st.sidebar:
    if st.button("Logout", key="logout_btn_advanced_statistics"):
        for k in list(st.session_state.keys()):
            del st.session_state[k]
        st.rerun()