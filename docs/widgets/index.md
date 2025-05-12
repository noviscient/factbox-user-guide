# Widgets

Factsheets can be built using a selected set of widgets: Basic, Performance Analytics, and Factor Analytics. The Performance and Factor Analytics widgets are pre-calculated based on the uploaded return data and the selected benchmark.

## Basic

### CSV Key-Value Data

CSV format with two columns: `key`, `value`.

### CSV Table Data

CSV format with multiple columns representing tabular data.

### Factsheet Title

Displays a title and description for the factsheet.

### Free Text

Plain text input, up to 200 characters.

### Image

Upload `.jpg`, `.jpeg`, or `.png` images (maximum size: 2 MB).

### Time Series Chart

Upload time series data in CSV format.

## Performance Analytics

### Cumulative Performance Chart

The cumulative performance of an investment represents the total percentage change in the asset’s price over a specific period.

The data for this chart will be pre-calculated based on the provided return data.

![Cumulative Performance Chart widget](../images/widgets/SCR-20250506-cwyt.png)

#### Widget Options

<img src="../images/widgets/SCR-20250506-cyaz.png" alt="Cumulative Performance Chart widget options" width="25%"/>

- **Benchmarks** - Select a benchmark from the list.
- **Start, End Date** - Set the date range for the displayed data.

### Distribution of Monthly Returns

The chart displays the number of months in which a portfolio's monthly performance historically has fallen within varying performance increments.

The data for this chart will be pre-calculated based on the provided return data.

![Distribution of Monthly Returns widget](../images/widgets/SCR-20250506-lgdf.png)

#### Widget Options

<img src="../images/widgets/SCR-20250506-lhax.png" alt="Distribution of Monthly Returns widget options" width="25%"/>

- **Benchmarks** - Select a benchmark from the list.
- **Start, End Date** - Set the date range for the displayed data.

### Drawdown Report

A drawdown is defined as a loss of equity from a peak to a trough within a single month or over a consecutive period of months.

The data for this chart will be pre-calculated based on the provided return data.

![Drawdown Report Table widget](../images/widgets/SCR-20250506-lmlr.png)

#### Widget Options

<img src="../images/widgets/SCR-20250506-lmuw.png" alt="Drawdown Report widget options" width="25%"/>

- **Top N Drawdowns** - Number of drawdown periods to display.
- **Start, End Date** - Set the date range for the displayed data.

### Expected Shortfall

Represents the expected shortfall risk associated with the returns data.

The data for this chart will be pre-calculated based on the provided return data.

![Expected Shortfall widget](../images/widgets/SCR-20250506-lqnz.png)

#### Widget Options

<img src="../images/widgets/SCR-20250506-lskv.png" alt="Expected Shortfall options" width="25%"/>

- **Confidence Level** - The probability that losses will not exceed the expected shortfall threshold. For example, a 95% confidence level means there is a 5% chance that losses will exceed the calculated expected shortfall.
- **Start, End Date** - Defines the time range over which the expected shortfall is calculated, based on the provided return data.

### Historical Performance Table

Analyzing historical performance data cah help you identify trends, by comparing historical data, you can spot upward and downward performance trends.

The data for this chart will be pre-calculated based on the provided return data.

![Historical Performance Table widget](../images/widgets/SCR-20250506-ufnn.png)

#### Widget Options

<img src="../images/widgets/SCR-20250506-uglz.png" alt="Historical Performance Table options" width="25%"/>

- **Start, End Date** - Sets the period for displaying historical performance metrics based on available return data.

### Performance and Risk Metrics

Performance and risk metrics are widely used to evaluate the performance of a portfolio, and forms a major component of portfolio management.

The data for this chart will be pre-calculated based on the provided return data.

![Performance and Risk Metrics widget](../images/widgets/SCR-20250506-uhys.png)

#### Widget Options

<img src="../images/widgets/SCR-20250506-ujba.png" alt="Performance and Risk Metrics options" width="25%"/>

-**Benchmarks** - Compare performance metrics against selected benchmarks. Multiple selections allowed.
-**Prinmary Benchmark** - The main benchmark used for comparison.
- **Start, End Date** - Sets the time period for displaying historical performance metrics based on the available return data.

### Return Report

The return report represents best, worst, average, median and last returns of different rolling period.

The data for this chart will be pre-calculated based on the provided return data.

![Performance and Risk Metrics widget](../images/widgets/SCR-20250506-urbb.png)

#### Widget Options

<img src="../images/widgets/SCR-20250507-bals.png" alt="Return Report options" width="25%"/>

- **Start, End Date** - Defines the time window used to calculate and display rolling period return statistics.

### Return Statistics

Return statistics show statistical measures for the return data provided.

- **CAGR** - Compound Annual Growth Rate; the annualized rate of return assuming profits are reinvested over the period.
- **3 Month ROR** - Return on investment over the last 3 months, showing short-term performance.
- **6 Month ROR** - Return over the past 6 months, capturing medium-term performance trends.
- **1 Year ROR** - Return over the last 12 months, indicating recent yearly performance.
- **3 Year ROR** - Cumulative return over the past 3 years, useful for evaluating longer-term results.
- **Year to Date ROR** - Return from the beginning of the calendar year up to the current date.
- **Total Return** -  The overall return over the entire period, including both capital gains and income.
- **Winning Month** - The percentage of months with positive returns during the evaluated period.
- **Avg Winning Month** - The average return in months where the performance was positive.
- **Avg Losing Month** - The average return in months where the performance was negative.

The data for this widget will be pre-calculated based on the provided return data.

![Performance and Risk Metrics widget](../images/widgets/SCR-20250507-bdwu.png)

#### Widget Options

<img src="../images/widgets/SCR-20250507-bfrp.png" alt="Return Statistics options" width="25%"/>

- **Start, End Date** - Defines the time window used to calculate and display rolling period return statistics.

### Risk Statistics

Display risk statistics properties.

- **Volatility** - Measures the standard deviation of returns, indicating the overall risk or variability in investment performance.
- **Downside Volatility** - Measure of downside risk that focuses on returns that fall below the risk-free benchmark. The risk-free benchmark will depend on the geography where the strategy/product is denominated and the market traded. For US and Global strategies/products, we will be using the 13 week Treasury Bill rate.

!!! note
    $$
    \text{Annual. Downside Volatility} =
    \sqrt{
    \frac{
    \sum_{t=1}^{n} \left[ \min(R_{st} - R_{ft}, 0) \right]^2
    }{n}
    \times \text{Trading Days per Year}
    }
    $$

    **Where:**

    - n: Total number of return observations  
    - min(X, Y): Returns the smaller of X and Y; used to isolate negative excess returns  
    - R_{st}: Strategy/Product return at time t  
    - R_{ft}: Risk-free return at time t
    - Trading Days per Year: 252

- **Maximum Drawdown** - The largest peak-to-trough decline in value during a specific period, showing the worst potential loss.

!!! note
    📈 Max Drawdown Calculation

    Compute the cumulative returns series, $C$:

    $$
    C = [C_1, C_2, \dots, C_T]
    $$

    Where the cumulative return at each time point $t$ is calculated as:

    $$
    C_t = \prod_{i=0}^{t} (1 + R_i)
    $$

    - $R_i$: Return at time $i$  
    - $t$: Time index in the return series

    ---

    📉 Drawdown Series

    Calculate the drawdown series, $D$:

    $$
    D = [D_1, D_2, \dots, D_T]
    $$

    Where drawdown at each time point $t$ is:

    $$
    D_t = \frac{C_t}{\max_{i=0}^{t}(C_i)} - 1
    $$

    - $\max_{i=0}^{t}(C_i)$: Maximum cumulative return observed up to time $t$

    ---

    📉 Maximum Drawdown

    Finally, compute the **maximum drawdown** as:

    $$
    \text{Max Drawdown} = \left| \min(D) \right|
    $$

    Where:
    - $D$: Full drawdown time series  
    - $\min(D)$: The lowest drawdown observed over the time period

- **Value at Risk** - Measures the extent of possible financial losses within the strategy/product over a specific time frame given a certain significance level (alpha). For the VaR, we will using the monthly returns as the input and the alpha specified will be 0.05.

!!! note

    The **Value at Risk** at a given significance level is calculated as:

    $$
    \text{Value at Risk} = Q(\alpha, \text{rets})
    $$

    Where:
    - $\alpha$: The significance level (e.g., 0.05 for 5%)
    - $\text{rets}$: All historical returns of the strategy
    - $Q$: Quantile function that returns the $\alpha$-th percentile of the return distribution

    ---

    🧪 Python Code Example

    ```python
        import numpy as np
        import numpy.typing as npt
        from typing import Dict

        def calculate_var(rets: npt.ArrayLike, alpha: float = 0.05) -> float:
            """
            Calculate Value at Risk (VaR) at a given significance level.

            Args:
                rets: A NumPy array-like of strategy returns.
                alpha: Significance level (default is 0.05 for 5% VaR).

            Returns:
                The VaR value (a negative number indicating potential loss).
            """
            rets_array = np.asarray(rets)
            clean_rets = rets_array[~np.isnan(rets_array)]
            var = np.quantile(clean_rets, alpha)

            return var
    ```

- **Expected Shortfall** - Measures the weighted average of the "extreme" losses in the tail of the distribution of possible returns, beyond the VaR cutoff point and given a certain significance level (alpha).

!!! note

    The **Expected Shortfall** (also called Conditional Value at Risk) is the **average loss** in the worst-case $\alpha$ fraction of return outcomes.

    ---

    Given $\alpha < 0.05$:

    $$
    \text{ES} = \frac{1}{N_<} \sum_{i=1}^{N_<} x_i
    $$

    Where:
    - $N_<$: Number of returns less than the $\alpha$-quantile
    - $x_i$: Each return in that worst $\alpha$ tail of the distribution

    ---

    🧪 Python Code Example

    ```python
    import numpy as np
    import numpy.typing as npt

    def calculate_empirical_expected_shortfall(rets: npt.ArrayLike, alpha: float = 0.05) -> float:
        """
        Calculate the empirical Expected Shortfall (ES) at a given significance level.

        Args:
            rets: A NumPy array-like of strategy returns.
            alpha: Significance level (default is 0.05).

        Returns:
            The ES value (mean of worst-case losses).
        """
        rets_array = np.asarray(rets)
        clean_rets = rets_array[~np.isnan(rets_array)]
        quantile = np.quantile(clean_rets, alpha)

        if alpha >= 0.5:
            es = clean_rets[clean_rets >= quantile].mean()
        else:
            es = clean_rets[clean_rets <= quantile].mean()

        return es
    ```

- **Beta (Market Index)** - Indicates sensitivity to market movements; a beta above 1 implies higher volatility than the market.

!!! note

    Beta measures the return data's sensitivity to market movements. It is derived from the **linear regression** of the return data against market returns.

    $$
    R_i = \beta R_m + \varepsilon
    $$

    Where:
    - $R_i$: Strategy returns  
    - $R_m$: Market returns  
    - $\beta$: Beta coefficient (our objective)  
    - $\varepsilon$: Error term or residual, capturing the portion of returns not explained by the market

    ---

    🧪 Python Code Example

    ```python
        from statsmodels.api import OLS, add_constant

        def calculate_risk_statistics(self):
            ...
            risk_stats['Beta (Market Index)'] = OLS(
                stgy_rets.values,
                add_constant(rets_all[self.market_rets.name].values)
            ).fit().params[1]
            ...
    ```

    🔍 The beta value is obtained from the fitted regression model. It corresponds to the coefficient of the market return (i.e., params[1]). A beta above 1 indicates greater volatility than the market; below 1 indicates lower sensitivity.


- **Correlation (Market Index)** - Measures the degree to which the investment moves in relation to the market index.
- **Tail Correlation (Market Index)** - Measures correlation during extreme market events, focusing on co-movement in the tails of the return distribution.
- **Sharpe Ratio** - Assesses risk-adjusted return by comparing excess return over the risk-free rate to volatility.
- **Calmar Ratio** - Evaluates performance relative to risk by dividing annualized return by maximum drawdown.

The data for this widget will be pre-calculated based on the provided return data.

![Risk Statistics widget](../images/widgets/SCR-20250508-llcd.png)

#### Widget Options

<img src="../images/widgets/SCR-20250508-lliz.png" alt="Risk Statistics options" width="25%"/>

- **Primary Benchmark** - The main benchmark used for comparison.
- **Start, End Date** - Defines the time window used to calculate and display rolling period return statistics.