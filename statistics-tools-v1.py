import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

# Set page config as the first Streamlit command
st.set_page_config(page_title="Statistical Methods Demo", layout="wide", initial_sidebar_state="expanded")

# Sidebar for page selection
page = st.sidebar.selectbox("Choose a page", ["Python Stats", "Python Expressions", "Julia Expressions", "R Expressions"])

# Generate sample data
np.random.seed(42)
data = np.concatenate([
    np.random.normal(100, 15, 1000),
    np.random.normal(150, 20, 500)
])

if page == "Python Stats":
    # --- Python Stats Page (Live Calculations) ---
    st.title("Statistical Methods Demonstration (Python)")
    st.write("This page demonstrates statistical measures using Python with live calculations.")

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

elif page == "Python Expressions":
    # --- Python Expressions Page ---
    st.title("Python Statistical Expressions")
    st.write("This page shows how statistical measures are calculated in Python (as code examples, not executed here).")
    st.write("These are the expressions used in the 'Python Stats' page to compute the live results.")

    # Python code as text
    python_code = """
    import numpy as np
    from scipy import stats

    # Sample data would be defined as:
    # data = np.concatenate([np.random.normal(100, 15, 1000), np.random.normal(150, 20, 500)])

    mean = np.mean(data)                            # Mean
    median = np.median(data)                        # Median
    mode_result = stats.mode(data, keepdims=True)   # Mode calculation
    mode = mode_result.mode[0] if mode_result.count[0] > 1 else mean  # Mode with fallback
    std_dev = np.std(data)                          # Standard Deviation
    variance = np.var(data)                         # Sample Variance
    kurt = stats.kurtosis(data)                     # Kurtosis
    skew = stats.skew(data)                         # Skewness
    range_val = np.max(data) - np.min(data)         # Range
    minimum = np.min(data)                          # Minimum
    maximum = np.max(data)                          # Maximum
    sum_val = np.sum(data)                          # Sum
    count = len(data)                               # Count
    """

    st.subheader("Python Code Examples")
    st.code(python_code, language="python")

    st.subheader("Explanation")
    st.write("""
    These expressions show how to calculate statistics in Python using NumPy and SciPy:
    - Requires `numpy` for basic numerical operations and `scipy.stats` for advanced statistics.
    - The 'Python Stats' page executes these exact expressions on the sample data.
    - To run this code yourself:
      1. Install dependencies: `pip install numpy scipy`
      2. Define your `data` array
      3. Execute in a Python environment
    """)

elif page == "Julia Expressions":
    # --- Julia Expressions Page ---
    st.title("Julia Statistical Expressions")
    st.write("This page shows how statistical measures would be calculated in Julia (as code examples, not executed).")

    # Julia code as text
    julia_code = """
    using Statistics
    using StatsBase

    # Sample data would be defined as:
    # data = [your_data_here]

    mean_val = mean(data)                           # Mean
    median_val = median(data)                       # Median
    mode_val = length(unique(data)) < length(data) ? mode(data) : mean_val  # Mode with fallback
    std_val = std(data)                            # Standard Deviation
    var_val = var(data)                            # Sample Variance
    kurt_val = kurtosis(data)                      # Kurtosis
    skew_val = skewness(data)                      # Skewness
    range_val = maximum(data) - minimum(data)      # Range
    min_val = minimum(data)                        # Minimum
    max_val = maximum(data)                        # Maximum
    sum_val = sum(data)                            # Sum
    count_val = length(data)                       # Count
    """

    st.subheader("Julia Code Examples")
    st.code(julia_code, language="julia")

    st.subheader("Explanation")
    st.write("""
    These expressions show how to calculate statistics in Julia:
    - Requires `Statistics` (standard library) and `StatsBase` (additional package).
    - To run this code:
      1. Install Julia (https://julialang.org/downloads/)
      2. Install packages: `using Pkg; Pkg.add("StatsBase")`
      3. Replace `data` with your actual dataset
    """)

elif page == "R Expressions":
    # --- R Expressions Page ---
    st.title("R Statistical Expressions")
    st.write("This page shows how statistical measures would be calculated in R (as code examples, not executed).")

    # R code as text
    r_code = """
    # Sample data would be defined as:
    # data <- c(your_data_here)

    mean_val <- mean(data)                          # Mean
    median_val <- median(data)                      # Median
    mode_val <- as.numeric(names(sort(table(data), decreasing=TRUE)[1]))  # Mode (basic approach)
    # Note: If no unique mode, might need custom function or fallback
    if(length(unique(data)) == length(data)) mode_val <- mean_val  # Fallback to mean
    std_val <- sd(data)                             # Standard Deviation
    variance_val <- var(data)                       # Sample Variance
    kurt_val <- moments::kurtosis(data)             # Kurtosis (requires 'moments' package)
    skew_val <- moments::skewness(data)             # Skewness (requires 'moments' package)
    range_val <- max(data) - min(data)              # Range
    min_val <- min(data)                            # Minimum
    max_val <- max(data)                            # Maximum
    sum_val <- sum(data)                            # Sum
    count_val <- length(data)                       # Count
    """

    st.subheader("R Code Examples")
    st.code(r_code, language="r")

    st.subheader("Explanation")
    st.write("""
    These expressions show how to calculate statistics in R:
    - Base R provides most functions (`mean`, `median`, `sd`, `var`, etc.).
    - Mode calculation is basic; R doesn't have a built-in `mode` function, so this uses `table()` to find the most frequent value.
    - Kurtosis and Skewness require the `moments` package for simplicity.
    - To run this code:
      1. Install R (https://www.r-project.org/)
      2. Install the `moments` package: `install.packages("moments")`
      3. Load the package: `library(moments)`
      4. Replace `data` with your actual dataset
    """)

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
