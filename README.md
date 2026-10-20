# What Drives Housing Burden in Canada?

A machine learning analysis of how much of their income Canadian neighbourhoods spend on housing, and why.

Group project for DATASCI 3000B (Introduction to Machine Learning), Western University, Winter 2026.

## The question

What makes some neighbourhoods spend a larger share of their income on shelter than others?

The target is the **Housing Burden Ratio (HBR)**: shelter spending divided by disposable income, measured for roughly 58,000 Canadian dissemination areas (small census neighbourhoods). In Canada, spending more than 30% of income on shelter counts as being housing burdened.

## Approach

**1. Data preparation**
- Merged household spending and demographic data on dissemination area (about 1,180 features)
- Removed every variable mathematically tied to shelter spending or income, to prevent target leakage
- 80/20 train/test split, with scaling and feature selection fitted on training data only

**2. Unsupervised learning**
- K-Means clustering (k = 2 to 12, chosen with elbow and silhouette scores) on log-transformed features
- PCA and UMAP to visualise the structure in 2D

**3. Supervised learning**
- Elastic Net regression on a normalised target (Pearson Type III to normal transform), tuned with 5-fold GridSearchCV
- XGBoost regression, tuned with 5-fold RandomizedSearchCV
- Bootstrapped 95% confidence intervals for R² (1,000 iterations)
- SHAP to explain which features drive XGBoost's predictions

## Key findings

| Model | Test R² | Test RMSE |
|---|---|---|
| Elastic Net | 0.60 | 0.039 |
| XGBoost | 0.79 | 0.029 |

- **Income is the main driver.** High burden comes mostly from low income, not from unusually high shelter costs.
- **Geography matters less than expected.** Clustering found an urban / suburban / rural spectrum, but average HBR was similar across all three. Rural areas had a much wider spread.
- **The relationship is non-linear.** XGBoost clearly beat the linear model, because features interact (for example, household size matters differently depending on income).
- **Spending acts as an income proxy.** Discretionary purchases such as recreational vehicles were strong predictors of low burden, because they signal spare income. They are proxies, not causes.

## Limitations

- Both models underpredict the highest-burden areas
- Confidence intervals for R² are wide (XGBoost 0.69 to 0.95), largely because of extreme outliers
- Clusters overlap heavily (silhouette score about 0.24), so they are better read as a continuum than as distinct segments
- The data is cross-sectional, so results are diagnostic rather than causal

## Data

The data was provided by Environics Analytics under a non-disclosure agreement for course use only. **It is not included in this repository and cannot be shared.** Notebook outputs have been cleared for the same reason, so the code will not run without access to the original files.

## Tech stack

Python, pandas, Polars, scikit-learn, XGBoost, SHAP, UMAP, distfit, Matplotlib, Seaborn. Originally run in Google Colab.

## Team

Keith Monga, Aiden Wesseling, Logan Peterson, Gwan Woo Kim
