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

<img src="../images/widgets/SCR-20250506-bals.png" alt="Return Report options" width="25%"/>

- **Start, End Date** - Defines the time window used to calculate and display rolling period return statistics.