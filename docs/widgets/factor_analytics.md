# Factor Analytics

This section requires two different datasets: return data and historical Fama-French factors.

## Current Factor Exposures

The factor exposures are calculated using the Fama-French Global Four-Factor Model and reflect exposures over the most recent calendar year.

<img src="../../images/widgets/SCR-20250514-cltg.png" alt="Current Factor Exposures"/>

- **Market**:  
  Risk associated with the general movement of the market.

- **Momentum**:  
  Indicates that the portfolio includes assets with a history of strong performance that continue to exhibit positive momentum.

- **Value**:  
  Measures the impact of investing in undervalued assets relative to their book value.

- **Size**:  
  Captures the performance difference between small-cap and large-cap stocks, with a focus on smaller companies.

The percentages in the graph represent the strength of the relationship between the returns of the strategy/product and each factor.

### 🧮 Formula

This approach estimates how much of the excess returns can be attributed to well-known risk factors using linear regression.

---

**🔹 Step 1: Calculate Excess Returns**

Calculate the excess returns:

$$
R_{\text{excess}} = R_{\text{strat}} - R_f
$$

Where:  
- $R_{\text{strat}}$: Strategy returns  
- $R_f$: Risk-free rate

---

**🔹 Step 2: Run Linear Regression on Factor Returns**

Estimate factor exposures ($\beta$) by regressing the excess returns on the Fama-French factor returns:

$$
R_{\text{excess}} =
\beta_{\text{mkt}} X_1 +
\beta_{\text{size}} X_2 +
\beta_{\text{momentum}} X_3 +
\beta_{\text{value}} X_4 +
R_f + \varepsilon
$$

Where:  
- $X_1$: Market factor (MKT-RF)  
- $X_2$: Size factor (SMB)  
- $X_3$: Momentum factor (MOM)  
- $X_4$: Value factor (HML)  
- $\beta_i$: Factor exposure (regression coefficient) for each risk factor  
- $\varepsilon$: Residual (unexplained return)

---

This regression helps decompose returns into exposures attributable to well-known systematic risks versus alpha or residual return.

### 🧪 Python Code Example

```python
import pandas as pd
import numpy as np
from statsmodels.api import OLS, add_constant
from typing import Tuple

def calculate_performance_attribution(
    data: pd.DataFrame,
    strategy_col: str,
    risk_free_col: str,
    factor_cols: Tuple[str, str, str, str],
    data_length: int
) -> Tuple[pd.Series, pd.Series, pd.Series]:
    """
    Calculate factor-based performance attribution using linear regression.

    Args:
        data: DataFrame containing strategy returns, risk-free rate, and factor returns.
        strategy_col: Column name for strategy returns.
        risk_free_col: Column name for risk-free rate.
        factor_cols: Tuple of factor column names (e.g., 'Market', 'Size', 'Value', 'Momentum').
        data_length: Number of recent observations to use for regression.

    Returns:
        Tuple containing:
            - risk_expos: Regression coefficients including Alpha
            - ret_attrs: Return contribution from each factor (coefficients × means)
            - risk_attrs: Risk exposures excluding Alpha
    """
    # Prepare regression dataset
    data = data.copy()
    data['excess_rets'] = data[strategy_col] - data[risk_free_col]

    # Keep only relevant columns and slice recent data
    regr_data = data[list(factor_cols) + [risk_free_col, 'excess_rets']].tail(data_length)

    # Define independent and dependent variables
    y = regr_data['excess_rets']
    X = add_constant(regr_data[list(factor_cols)])
    X = X.rename(columns={'const': 'Alpha'})

    # Fit linear regression
    model = OLS(y, X).fit()
    risk_expos = model.params

    # Compute return attributions (coefficient × average factor value)
    factor_means = X.mean()
    ret_attrs = risk_expos * factor_means

    # Drop alpha from risk exposure for cleaner view
    risk_attrs = risk_expos.drop('Alpha')

    return risk_expos, ret_attrs, risk_attrs

risk_expos, ret_attrs, risk_attrs = calculate_performance_attribution(
    data=data,
    strategy_col='Strategy',
    risk_free_col='RiskFree',
    factor_cols=('Market', 'Size', 'Value', 'Momentum'),
    data_length=60
)

```

## Current Factor Return Attributions

## Current Factor Risk Attributions

## Rolling Factor Exposures

## Rolling Factor Return Attributions

## Rolling Factor Risk Attributions