import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

# Set page config
st.set_page_config(page_title="Statistical Methods Demo", layout="wide")

# Title and introduction
st.title("Statistical Methods Demonstration")
st.write("This app demonstrates various statistical measures using a sample dataset.")

# Generate sample data
np.random.seed(42)
data = np.concatenate([
    np.random.normal(100, 15, 1000),  # Normal distribution
    np.random.normal(150, 20, 500)    # Second peak for bimodality
])

# Create two columns
col1, col2 = st.columns(2)

# Calculate statistics
mean = np.mean(data)
median = np.median(data)
mode_result = stats.mode(data, keepdims=True)  # Use keepdims=True for consistent output
mode = mode_result.mode[0] if mode_result.count[0] > 1 else mean  # Fallback to mean if no clear mode
std_dev = np.std(data)
variance = np.var(data)
kurt = stats.kurtosis(data)
skew = stats.skew(data)
range_val = np.max(data) - np.min(data)
minimum = np.min(data)
maximum = np.max(data)
sum_val = np.sum(data)
count = len(data)

# Display statistics in first column
with col1:
    st.subheader("Statistical Measures")
    st.write(f"Mean: {mean:.2f}")
    st.write(f"Median: {median:.2f}")
    st.write(f"Mode: {mode:.2f}")
    st.write(f"Standard Deviation: {std_dev:.2f}")
    st.write(f"Sample Variance: {variance:.2f}")
    st.write(f"Kurtosis: {kurt:.2f}")
    st.write(f"Skewness: {skew:.2f}")
    st.write(f"Range: {range_val:.2f}")
    st.write(f"Minimum: {minimum:.2f}")
    st.write(f"Maximum: {maximum:.2f}")
    st.write(f"Sum: {sum_val:.2f}")
    st.write(f"Count: {count}")

# Create visualizations in second column
with col2:
    st.subheader("Visualizations")
    
    # Histogram
    fig1, ax1 = plt.subplots()
    sns.histplot(data, bins=30, kde=True, ax=ax1)
    ax1.axvline(mean, color='r', linestyle='--', label=f'Mean: {mean:.2f}')
    ax1.axvline(median, color='g', linestyle='--', label=f'Median: {median:.2f}')
    ax1.axvline(mode, color='y', linestyle='--', label=f'Mode: {mode:.2f}')
    ax1.legend()
    ax1.set_title("Distribution with Key Statistics")
    st.pyplot(fig1)
    
    # Box plot
    fig2, ax2 = plt.subplots()
    sns.boxplot(x=data, ax=ax2)
    ax2.set_title("Box Plot showing Spread and Outliers")
    st.pyplot(fig2)

# Explanations
st.subheader("Explanations of Statistical Measures")
with st.expander("Click to see explanations"):
    st.write("""
    - **Mean**: Arithmetic average of all values
    - **Median**: Middle value when data is sorted
    - **Mode**: Most frequent value in the dataset (falls back to mean if no clear mode)
    - **Standard Deviation**: Measure of data dispersion from the mean
    - **Sample Variance**: Square of standard deviation
    - **Kurtosis**: Measure of tailedness of the distribution
    - **Skewness**: Measure of asymmetry of the distribution
    - **Range**: Difference between maximum and minimum values
    - **Minimum**: Smallest value in the dataset
    - **Maximum**: Largest value in the dataset
    - **Sum**: Total of all values
    - **Count**: Number of values in the dataset
    """)

# Data table
st.subheader("Sample Data Preview")
df = pd.DataFrame(data, columns=['Values'])
st.dataframe(df.head(10))

# Add interactive feature
st.subheader("Try Your Own Numbers")
user_input = st.text_area("Enter numbers separated by commas (e.g., 1, 2, 3, 4, 5)")
if user_input:
    try:
        user_data = [float(x.strip()) for x in user_input.split(',')]
        if len(user_data) > 1:
            user_mode_result = stats.mode(user_data, keepdims=True)
            user_mode = user_mode_result.mode[0] if user_mode_result.count[0] > 1 else np.mean(user_data)
            st.write(f"Mean: {np.mean(user_data):.2f}")
            st.write(f"Median: {np.median(user_data):.2f}")
            st.write(f"Mode: {user_mode:.2f}")
            st.write(f"Standard Deviation: {np.std(user_data):.2f}")
    except:
        st.error("Please enter valid numbers separated by commas")
