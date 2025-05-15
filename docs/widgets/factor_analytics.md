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

### ⚙️ Widget Options

<img src="../../images/widgets/SCR-20250514-lppk.png" alt="Current Factor Exposures options" width="25%"/>

- **Market** - Represents the market factor used in performance attribution or comparison, typically reflecting the excess return of the overall market.
- **Start, End Date** - Specifies the time window over which returns and factor exposures are calculated and analyzed.

## Current Factor Return Attributions

Factor Return Attribution attributes excess returns to common risk factors using the Fama-French model. Factor exposures (betas) are estimated via linear regression, and return attribution is calculated accordingly.

<img src="../../images/widgets/SCR-20250515-btqy.png" alt="Current Factor Exposures options" width="25%"/>

### 🧮 Formula

---

**🔹 Regression Model with Alpha**

Once factor exposures (β) are estimated, the constant term from the regression is treated as Alpha:

$$
R_{\text{excess}} =
\beta_{\text{mkt}} X_1 +
\beta_{\text{size}} X_2 +
\beta_{\text{momentum}} X_3 +
\beta_{\text{value}} X_4 +
R_f + \text{Alpha}
$$

Where:
- $R_{\text{excess}}$: Strategy or product excess returns
- $X_i$: Factor returns (e.g., Market, Size, Momentum, Value)
- $\beta_i$: Exposure to each factor
- $R_f$: Risk-free rate
- $\text{Alpha}$: Constant term representing unexplained return

---

**🔹 Mean Return of Each Factor**

Calculate the average return of each factor:

$$
\mu_x = \frac{R_1 + R_2 + \dots + R_T}{T}
$$

Where:  
- $\mu_x$: Mean return of factor $x$  
- $R_t$: Return of factor $x$ at time $t$  
- $T$: Total number of observations

---

**🔹 Return Attribution per Factor**

Determine the return attribution for each factor:

$$
A_x = \mu_x \cdot \beta_x \cdot (\text{Total No. of Factors} + 1)
$$

Where:
- $A_x$: Return attribution of factor $x$  
- $\mu_x$: Mean return of factor $x$  
- $\beta_x$: Exposure to factor $x$  
- $\text{Total No. of Factors}$: In this case, **5** (including Alpha)

---

This framework helps explain how much of the product’s return is driven by each systematic factor versus residual Alpha.

### 🧪 Python Code Example

```python
import pandas as pd
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
    Perform factor return attribution using a linear regression model based on the Fama-French framework.

    Args:
        data: DataFrame with daily return data, including strategy returns, risk-free rate, and factor returns.
        strategy_col: Column name for the strategy or product returns.
        risk_free_col: Column name for the risk-free rate.
        factor_cols: Tuple of factor column names (e.g., Market, Size, Value, Momentum).
        data_length: Number of most recent rows to use for the regression.

    Returns:
        Tuple containing:
            - risk_expos: Regression coefficients including Alpha
            - ret_attrs: Return attribution for each factor and Alpha
            - risk_attrs: Risk exposures excluding Alpha
    """
    data = data.copy()
    data['excess_rets'] = data[strategy_col] - data[risk_free_col]

    # Select relevant columns and limit to recent data
    regr_data = data[list(factor_cols) + [risk_free_col, 'excess_rets']].tail(data_length)

    # Linear regression
    y = regr_data['excess_rets']
    X = add_constant(regr_data[list(factor_cols)])
    X = X.rename(columns={'const': 'Alpha'})
    model = OLS(y, X).fit()
    risk_expos = model.params

    # Return attribution (Step 2 & 3 from formula)
    ret_attrs = X.mean() * risk_expos
    ret_attrs['RiskFree'] = regr_data[risk_free_col].mean()
    ret_attrs = ret_attrs * len(regr_data)  # Scale by period count

    risk_attrs = risk_expos.drop('Alpha')

    return risk_expos, ret_attrs, risk_attrs

risk_expos, ret_attrs, risk_attrs = calculate_performance_attribution(
    data=data,
    strategy_col=stgy_rets.name,
    risk_free_col='RiskFree',
    factor_cols=('Market', 'Size', 'Value', 'Momentum'),
    data_length=data_length
)

```

## Current Factor Risk Attributions

## Rolling Factor Exposures

## Rolling Factor Return Attributions

## Rolling Factor Risk Attributions