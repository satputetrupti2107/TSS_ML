# Delivery Time Prediction using Simple Linear Regression

Predicting delivery time from sorting time using simple linear regression and several transformed variants (log, exponential, quadratic), then comparing them on RMSE and R².

## Problem Statement
A delivery company wants to understand how the time spent sorting orders affects total delivery time. This project builds regression models to predict **Delivery Time** from **Sorting Time** and identifies the best-fitting model.

## Dataset
- **File:** `delivery_time.csv`
- **Size:** 21 rows x 2 columns
- **Features:**
  - `SortingTime` (independent variable)
  - `DeliveryTime` (target variable)
- No missing values.

## Approach
1. **Exploratory data analysis**: summary statistics, boxplot, histograms, distribution plot, scatter plot.
2. **Correlation analysis**: Pearson correlation between Sorting Time and Delivery Time is **0.826**, indicating a strong positive relationship.
3. **Model building** with `statsmodels` (OLS):
   - Linear: `DeliveryTime ~ SortingTime`
   - Log: `DeliveryTime ~ log(SortingTime)`
   - Exponential: `log(DeliveryTime) ~ SortingTime`
   - Quadratic (on log target): `log(DeliveryTime) ~ SortingTime + SortingTime²`
4. **Evaluation** using RMSE (on the original delivery time scale) and R², with actual vs predicted plots for each model.

## Results

| Model | RMSE | R² |
|-------|------|-----|
| Linear | 2.7917 | 0.6823 |
| Log | 2.7332 | 0.6954 |
| Exponential | 2.9403 | 0.7109 |
| Quadratic (exp + square) | 2.7990 | 0.7649 |

**Baseline linear equation:** `DeliveryTime = 6.5827 + 1.6490 x SortingTime` (p < 0.05, so Sorting Time is a statistically significant predictor).

**Takeaways**
- The quadratic model achieves the highest R² (about 76.5%) and was chosen as the final model.
- The log model has the lowest RMSE, so the models are close in predictive error on this small dataset.
- Note: R² for the exponential and quadratic models is computed on `log(DeliveryTime)`, so it is not strictly comparable with the linear and log models. RMSE (computed on the original scale) is the fairer comparison.

## Project Structure
```
├── 4_TSS_SLR_Deliverytime_Assignment.ipynb   # Full analysis and modelling
├── delivery_time.csv                          # Dataset
├── requirements.txt
└── README.md
```

## How to Run
1. Clone the repository
   ```bash
   git clone https://github.com/<your-username>/<repo-name>.git
   cd <repo-name>
   ```
2. Install dependencies
   ```bash
   pip install -r requirements.txt
   ```
3. Launch the notebook
   ```bash
   jupyter notebook 4_TSS_SLR_Deliverytime_Assignment.ipynb
   ```

## Tech Stack
Python, pandas, NumPy, Matplotlib, Seaborn, statsmodels

## Limitations and Future Improvements
- Only 21 observations, so results may not generalise; a larger dataset would help.
- Use a train/test split or cross-validation instead of evaluating on the training data.
- Check residual diagnostics (normality, homoscedasticity) and influential points.
- Try additional features (distance, order size, etc.) with multiple regression.
