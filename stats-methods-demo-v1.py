import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

# Set page config as the first Streamlit command
st.set_page_config(page_title="Statistical Methods Demo", layout="wide", initial_sidebar_state="expanded")

# Sidebar for page selection
page = st.sidebar.selectbox("Choose a page", ["Python Stats", "Python Expressions", "Julia Expressions", "R Expressions", "Graph Types"])

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

elif page == "Graph Types":
    # --- Graph Types Page ---
    st.title("Statistical Graph Types in Python")
    st.write("This page demonstrates various statistical graph types, their use cases, and visualizations using the sample data.")

    # List of graph descriptions
    st.subheader("Graph Type Descriptions")
    st.write("""
    - **Histogram**: Shows the distribution of a single variable, revealing shape, spread, and central tendency.
    - **Box Plot**: Displays summary statistics (median, quartiles, outliers) and compares distributions.
    - **Violin Plot**: Combines box plot with kernel density, showing distribution shape and summary stats.
    - **Bar Plot**: Shows counts or summaries for categorical or binned data.
    - **Scatter Plot**: Examines relationships between two continuous variables.
    - **Line Plot (ECDF)**: Shows cumulative distribution or trends over a continuous variable.
    - **Q-Q Plot**: Compares data distribution to a theoretical distribution (e.g., normal) for normality assessment.
    - **Heatmap**: Visualizes correlation or intensity across two dimensions (here, simulated multivariate data).
    - **Pie Chart**: Displays proportions or percentages of categories (here, binned data).
    - **Area Plot**: Shows cumulative or stacked data trends (here, cumulative distribution alternative).
    """)

    # Histogram
    st.subheader("1. Histogram")
    st.write("**Use Case**: Distribution analysis")
    fig1, ax1 = plt.subplots()
    sns.histplot(data, bins=30, kde=True, ax=ax1)
    ax1.set_title("Histogram with KDE")
    st.pyplot(fig1)

    # Box Plot
    st.subheader("2. Box Plot")
    st.write("**Use Case**: Summary statistics and outlier detection")
    fig2, ax2 = plt.subplots()
    sns.boxplot(x=data, ax=ax2)
    ax2.set_title("Box Plot")
    st.pyplot(fig2)

    # Violin Plot
    st.subheader("3. Violin Plot")
    st.write("**Use Case**: Distribution shape and summary stats")
    fig3, ax3 = plt.subplots()
    sns.violinplot(x=data, ax=ax3)
    ax3.set_title("Violin Plot")
    st.pyplot(fig3)

    # Bar Plot (using binned data)
    st.subheader("4. Bar Plot")
    st.write("**Use Case**: Binned data counts")
    bins = np.histogram_bin_edges(data, bins=10)
    hist, _ = np.histogram(data, bins=bins)
    fig4, ax4 = plt.subplots()
    ax4.bar(bins[:-1], hist, width=np.diff(bins), edgecolor='black')
    ax4.set_title("Bar Plot of Binned Data")
    st.pyplot(fig4)

    # Scatter Plot (simulated paired data)
    st.subheader("5. Scatter Plot")
    st.write("**Use Case**: Relationship between variables")
    x_data = np.random.normal(0, 1, len(data))
    fig5, ax5 = plt.subplots()
    ax5.scatter(x_data, data, alpha=0.5)
    ax5.set_title("Scatter Plot (Simulated Paired Data)")
    ax5.set_xlabel("X Variable")
    ax5.set_ylabel("Sample Data")
    st.pyplot(fig5)

    # Line Plot (ECDF)
    st.subheader("6. Line Plot (ECDF)")
    st.write("**Use Case**: Cumulative distribution")
    sorted_data = np.sort(data)
    y = np.arange(1, len(data) + 1) / len(data)
    fig6, ax6 = plt.subplots()
    ax6.plot(sorted_data, y)
    ax6.set_title("Empirical Cumulative Distribution Function (ECDF)")
    ax6.set_xlabel("Value")
    ax6.set_ylabel("Cumulative Probability")
    st.pyplot(fig6)

    # Q-Q Plot
    st.subheader("7. Q-Q Plot")
    st.write("**Use Case**: Normality assessment")
    fig7, ax7 = plt.subplots()
    stats.probplot(data, dist="norm", plot=ax7)
    ax7.set_title("Q-Q Plot vs Normal Distribution")
    st.pyplot(fig7)

    # Heatmap (simulated correlation)
    st.subheader("8. Heatmap")
    st.write("**Use Case**: Correlation visualization (simulated multivariate data)")
    sim_data = pd.DataFrame({
        'Var1': data,
        'Var2': np.random.normal(0, 1, len(data)),
        'Var3': np.random.normal(10, 2, len(data))
    })
    corr = sim_data.corr()
    fig8, ax8 = plt.subplots()
    sns.heatmap(corr, annot=True, cmap='coolwarm', ax=ax8)
    ax8.set_title("Heatmap of Simulated Correlation Matrix")
    st.pyplot(fig8)

    # Pie Chart (binned data)
    st.subheader("9. Pie Chart")
    st.write("**Use Case**: Proportion of categories")
    hist, bin_edges = np.histogram(data, bins=5)
    labels = [f"{bin_edges[i]:.0f}-{bin_edges[i+1]:.0f}" for i in range(len(hist))]
    fig9, ax9 = plt.subplots()
    ax9.pie(hist, labels=labels, autopct='%1.1f%%', startangle=90)
    ax9.set_title("Pie Chart of Binned Data")
    st.pyplot(fig9)

    # Area Plot (cumulative distribution)
    st.subheader("10. Area Plot")
    st.write("**Use Case**: Cumulative or stacked trends")
    fig10, ax10 = plt.subplots()
    ax10.fill_between(sorted_data, y, alpha=0.5)
    ax10.set_title("Area Plot of Cumulative Distribution")
    ax10.set_xlabel("Value")
    ax10.set_ylabel("Cumulative Probability")
    st.pyplot(fig10)

    st.subheader("Additional Notes")
    st.write("""
    - These visualizations use Matplotlib and Seaborn, included in the app's dependencies.
    - The sample data is continuous and bimodal, but these graphs can adapt to other data types.
    - Some graphs (e.g., Heatmap, Scatter) use simulated additional data for demonstration.
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
