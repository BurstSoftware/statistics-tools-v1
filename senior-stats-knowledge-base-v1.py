import streamlit as st

# Streamlit app configuration
st.set_page_config(page_title="Senior Statistician Knowledge Base", layout="wide")

# Title and introduction
st.title("Senior Statistician Knowledge Base")
st.markdown("Explore the comprehensive knowledge areas required for senior statisticians, broken down into detailed subcomponents.")

# Dictionary to organize content (condensed for brevity)
knowledge_areas = {
    "1. Foundational Statistical Theory": {
        "Probability Theory": {
            "Basic Concepts": "Detailed content for Basic Concepts: Sample spaces (Ω), events, axioms (P(E) ≥ 0, P(Ω) = 1)...",
            "Random Variables": "Discrete (PMF: P(X = x)), Continuous (PDF: ∫f(x)dx = 1), Mixed...",
            "Distributions": "Normal (f(x) = (1/√(2πσ²))e^(-(x-μ)²/(2σ²))), Binomial (P(X = k) = (n choose k) p^k (1-p)^(n-k))...",
            # Add other subtopics from convo
        },
        "Statistical Inference": {
            "Point Estimation": "Bias (E[θ̂] - θ), Variance (Var(θ̂)), MSE = Var + Bias²...",
            "Interval Estimation": "CI for mean (X̄ ± z_(α/2) * σ/√n), coverage (1-α)...",
            # Add other subtopics
        },
        # Add other subsections (Regression Analysis, etc.)
    },
    "2. Advanced Statistical Methods": {
        "Nonparametric Statistics": {
            "Rank-Based Tests": "Wilcoxon (W = min(W⁺, W⁻)), Mann-Whitney (U = n₁n₂ + n₁(n₁+1)/2 - R₁)...",
            "Kernel Density Estimation": "f(x) = (1/(nh)) * ΣK((x - x_i)/h), bandwidth (h ≈ 1.06σn^(-1/5))...",
        },
        "Bayesian Statistics": {
            "Priors": "Conjugate (Beta for binomial), Non-informative (uniform)...",
            "Posterior Computation": "Analytical (normal posterior), MCMC (Gibbs, Metropolis-Hastings)...",
        },
        # Add other subsections
    },
    "3. Mathematical Foundations": {
        "Linear Algebra": {
            "Vectors and Matrices": "Dot product (v·w = Σv_i w_i), Matrix mult (AB)_{ij} = ΣA_{ik}B_{kj}...",
            "Determinants and Eigenvalues": "det(A) = ad - bc, solve det(A - λI) = 0...",
        },
        "Calculus": {
            "Derivatives": "∂f/∂x_i, Gradient (∇f = (∂f/∂x₁, …))...",
            "Integrals": "∫_a^b f(x)dx, Trapezoidal rule ((b-a)/2)(f(a) + f(b))...",
        },
    },
    "4. Programming and Computational Skills": {
        "Statistical Software": {
            "R": "lm(y ~ x₁ + x₂), dplyr::filter()...",
            "SAS": "PROC REG MODEL y = x₁ x₂ / VIF...",
        },
        "Programming Languages": {
            "Python": "pandas (df.groupby()), scikit-learn (LogisticRegression())...",
            "Julia": "A = rand(3,3), pdf(Normal(0,1), x)...",
        },
    },
    "5. Data Management and Preprocessing": {
        "Data Cleaning": {
            "Missing Data": "Mean imputation (X̄), MICE (pooled estimate = (1/m)Σθ̂_j)...",
            "Outliers": "Z-scores (z = (x - X̄)/s), IQR (Q1 - 1.5*IQR)...",
        },
        "Data Transformation": {
            "Normalization": "Min-Max (x’ = (x - min)/(max - min)), Z-score (x’ = (x - X̄)/s)...",
            "Encoding": "One-hot ([1,0,0]), Label (Red → 0)...",
        },
    },
    "6. Domain-Specific Knowledge": {
        "Industry Expertise": {
            "Biostatistics": "Clinical trials (Phase I-IV), power calc (n = (z_(1-α/2) + z_(1-β))² * 2σ² / δ²)...",
            "Econometrics": "Cointegration (Engle-Granger), panel data (xtreg y x₁ x₂, fe)...",
        },
        "Regulatory Standards": {
            "Pharmaceuticals": "ICH E9, GCP (data integrity)...",
            "Finance": "Basel III (VaR), SOX compliance...",
        },
    },
    "7. Communication and Collaboration": {
        "Technical Writing": {
            "Reports": "Structure (Intro, Methods, Results), clarity (avoid jargon)...",
            "Papers": "Abstract (150-250 words), LaTeX (\hat{\beta})...",
        },
        "Data Storytelling": {
            "Visualization": "Bar (categorical), line (time)...",
            "Simplification": "P-value as ‘5% chance by luck’...",
        },
    },
    "8. Problem-Solving and Critical Thinking": {
        "Model Selection": {
            "Criteria": "AIC (-2log(L) + 2k), BIC (-2log(L) + k*log(n))...",
            "Trade-offs": "Bias-variance, interpretability vs. predictive power...",
        },
        "Diagnostics": {
            "Residual Analysis": "Q-Q plot, Breusch-Pagan (χ² on squared residuals)...",
            "Model Fit": "Likelihood ratio (-2(log L_red - log L_full))...",
        },
    },
    "9. Emerging Trends and Tools": {
        "AI and Machine Learning Integration": {
            "Statistical ML": "Overfitting (train-test split), Ridge as prior...",
            "Deep Learning": "MSE ((1/n)Σ(y_i - ŷ_i)²), SGD (θ - η∇L)...",
        },
        "Reproducible Research": {
            "Jupyter Notebooks": "Python cells (import pandas), Markdown...",
            "R Markdown": "```{r} lm(y ~ x) ```, Shiny apps...",
        },
    },
    "10. Practical Experience": {
        "Project Management": {
            "Planning": "Hypotheses (H₀: β₁ = 0), milestones (Week 1: data)...",
            "Execution": "API calls, refine based on residuals...",
        },
        "Real-World Applications": {
            "Messy Data": "Text (tf-idf), Images (PCA)...",
            "Case Studies": "Sales (ARIMA, RMSE = 50), Clinical (Cox, HR = 1.3)...",
        },
    }
}

# Display content in Streamlit
for main_section, subsections in knowledge_areas.items():
    with st.expander(main_section, expanded=False):
        for subsection, subtopics in subsections.items():
            st.header(subsection)
            for subtopic, content in subtopics.items():
                with st.expander(subtopic, expanded=False):
                    st.write(content)

# Footer
st.markdown("---")
st.markdown("Built By Burst Software Development | Date: April 15, 2025")
