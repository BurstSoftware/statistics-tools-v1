import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

# Attempt to import Julia, but continue if it fails
julia_available = False
try:
    from julia import Main
    import julia
    jl = julia.Julia(compiled_modules=False)
    jl.eval("using Statistics; using StatsBase")
    julia_available = True
except Exception as e:
    st.warning(f"Julia integration unavailable: {e}. Julia page will be disabled. "
               "To use Julia features, install Julia locally and run this app with 'streamlit run your_app.py'.")

# Set page config
st.set_page_config(page_title="Statistical Methods Demo", layout="wide", initial_sidebar_state="expanded")

# Sidebar for page selection
pages = ["Python Stats"]
if julia_available:
    pages.append("Julia Stats")
page = st.sidebar.selectbox("Choose a page", pages)

# Generate sample data
np.random.seed(42)
data = np.concatenate([
    np.random.normal(100, 15, 1000),
    np.random.normal(150, 20, 500)
])

if page == "Python Stats":
    # --- Python Page ---
    st.title("Statistical Methods Demonstration (Python)")
    st.write("This page demonstrates statistical measures using Python.")

    col1, col2 = st.columns(2)

    # Calculate statistics
    mean = np.mean(data)
    median = np.median(data)
    mode_result = stats.mode(data, keepdims=True)
    mode = mode_result.mode[0] if mode_result.count[0] > 1 else mean
    std_dev = np.std(data)
    variance = np.var(data)
    kurt = stats.kurtosis(data)
    skew = stats.skew(data)
    range_val = np.max(data) - np.min(data)
    minimum = np.min(data)
    maximum = np.max(data)
    sum_val = np.sum(data)
    count = len(data)

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

    with col2:
        st.subheader("Visualizations")
        fig1, ax1 = plt.subplots()
        sns.histplot(data, bins=30, kde=True, ax=ax1)
        ax1.axvline(mean, color='r', linestyle='--', label=f'Mean: {mean:.2f}')
        ax1.axvline(median, color='g', linestyle='--', label=f'Median: {median:.2f}')
        ax1.axvline(mode, color='y', linestyle='--', label=f'Mode: {mode:.2f}')
        ax1.legend()
        ax1.set_title("Distribution with Key Statistics")
        st.pyplot(fig1)

elif page == "Julia Stats" and julia_available:
    # --- Julia Page ---
    st.title("Statistical Methods Demonstration (Julia)")
    st.write("This page demonstrates statistical measures using Julia.")

    # Pass data to Julia
    Main.data = data

    # Julia statistical calculations
    jl.eval("""
    mean_val = mean(data)
    median_val = median(data)
    mode_val = length(unique(data)) < length(data) ? mode(data) : mean_val
    std_val = std(data)
    var_val = var(data)
    kurt_val = kurtosis(data)
    skew_val = skewness(data)
    range_val = maximum(data) - minimum(data)
    min_val = minimum(data)
    max_val = maximum(data)
    sum_val = sum(data)
    count_val = length(data)
    """)

    # Retrieve results
    mean = Main.mean_val
    median = Main.median_val
    mode = Main.mode_val
    std_dev = Main.std_val
    variance = Main.var_val
    kurt = Main.kurt_val
    skew = Main.skew_val
    range_val = Main.range_val
    minimum = Main.min_val
    maximum = Main.max_val
    sum_val = Main.sum_val
    count = Main.count_val

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Statistical Measures (Julia)")
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

    with col2:
        st.subheader("Visualizations")
        fig1, ax1 = plt.subplots()
        sns.histplot(data, bins=30, kde=True, ax=ax1)
        ax1.axvline(mean, color='r', linestyle='--', label=f'Mean: {mean:.2f}')
        ax1.axvline(median, color='g', linestyle='--', label=f'Median: {median:.2f}')
        ax1.axvline(mode, color='y', linestyle='--', label=f'Mode: {mode:.2f}')
        ax1.legend()
        ax1.set_title("Distribution with Key Statistics (Julia)")
        st.pyplot(fig1)

# Shared explanations
st.subheader("Explanations of Statistical Measures")
with st.expander("Click to see explanations"):
    st.write("""
    - **Mean**: Arithmetic average of all values
    - **Median**: Middle value when data is sorted
    - **Mode**: Most frequent value (falls back to mean if no clear mode)
    - **Standard Deviation**: Measure of data dispersion
    - **Sample Variance**: Square of standard deviation
    - **Kurtosis**: Measure of tailedness
    - **Skewness**: Measure of asymmetry
    - **Range**: Difference between maximum and minimum
    - **Minimum**: Smallest value
    - **Maximum**: Largest value
    - **Sum**: Total of all values
    - **Count**: Number of values
    """)
