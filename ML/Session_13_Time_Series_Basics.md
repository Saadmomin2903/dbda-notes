# Session 13 – Time Series Basics

## 📚 Table of Contents
1. [Time Series Fundamentals](#time-series-fundamentals)
2. [Stationarity](#stationarity)
3. [Moving Averages](#moving-averages)
4. [Exponential Smoothing](#exponential-smoothing)
5. [Time Series Decomposition](#time-series-decomposition)
6. [MCQs](#mcqs)
7. [Common Mistakes](#common-mistakes)
8. [One-Line Exam Facts](#one-line-exam-facts)

---

# Time Series Fundamentals

## 📘 Concept Overview

**Time Series**: Sequence of observations ordered by time.

```
y₁, y₂, y₃, ..., yₜ
```

**Key characteristic**: Temporal dependence (current value depends on past)

## 🧮 Components

Time series = **Trend + Seasonal + Cyclic + Irregular**

### 1. Trend (T)
Long-term increase/decrease in data.

**Example**: Population growth, GDP increase

### 2. Seasonal (S)
Regular, periodic fluctuations.

**Example**: Ice cream sales higher in summer, retail sales spike in December

**Period**: Fixed (daily, weekly, monthly, quarterly, yearly)

### 3. Cyclic (C)
Long-term oscillations (no fixed period).

**Example**: Business cycles, economic recessions

**Difference from seasonal**: Variable length cycles

### 4. Irregular/Random (I)
Unpredictable, random fluctuations (noise).

## 📊 Additive vs Multiplicative

### Additive Model
```
Y_t = T_t + S_t + I_t
```

**When**: Seasonal variation constant over time

### Multiplicative Model
```
Y_t = T_t × S_t × I_t
```

**When**: Seasonal variation proportional to level

**Can convert to additive**: log(Y_t) = log(T_t) + log(S_t) + log(I_t)

## 🧪 Python Setup

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

# Generate sample time series
np.random.seed(42)
dates = pd.date_range('2020-01-01', periods=365, freq='D')
trend = np.linspace(100, 150, 365)
seasonal = 10 * np.sin(2 * np.pi * np.arange(365) / 365)
noise = np.random.normal(0, 2, 365)
y = trend + seasonal + noise

ts = pd.Series(y, index=dates)

# Plot
plt.figure(figsize=(12, 4))
plt.plot(ts)
plt.title('Sample Time Series')
plt.xlabel('Date')
plt.ylabel('Value')
plt.show()
```

---

# Stationarity

## 📘 Concept Overview

**Stationary time series**: Statistical properties (mean, variance, autocorrelation) constant over time.

**Why important**: Most time series models assume stationarity!

## 🧮 Types of Stationarity

### Strict Stationarity
Joint distribution unchanged by time shifts.

**Too restrictive** for practical use.

### Weak (Covariance) Stationarity

Requires:
1. **Constant mean**: E[Y_t] = μ for all t
2. **Constant variance**: Var(Y_t) = σ² for all t
3. **Autocovariance depends only on lag**: Cov(Y_t, Y_{t-k}) = γ_k

## 📊 Testing Stationarity

### Visual Inspection

```python
# Plot over time
plt.figure(figsize=(12, 6))

plt.subplot(2, 1, 1)
plt.plot(ts)
plt.title('Original Series')

# Rolling statistics
rolling_mean = ts.rolling(window=30).mean()
rolling_std = ts.rolling(window=30).std()

plt.subplot(2, 1, 2)
plt.plot(ts, label='Original')
plt.plot(rolling_mean, label='Rolling Mean', color='red')
plt.plot(rolling_std, label='Rolling Std', color='black')
plt.legend()
plt.title('Rolling Statistics')
plt.tight_layout()
plt.show()
```

### Augmented Dickey-Fuller (ADF) Test

**Null hypothesis (H₀)**: Series has unit root (non-stationary)

```python
from statsmodels.tsa.stattools import adfuller

def adf_test(series, name=''):
    """Perform ADF test."""
    result = adfuller(series, autolag='AIC')
    
    print(f'ADF Test for {name}:')
    print(f'  ADF Statistic: {result[0]:.4f}')
    print(f'  p-value: {result[1]:.4f}')
    print(f'  Critical Values:')
    for key, value in result[4].items():
        print(f'    {key}: {value:.4f}')
    
    if result[1] < 0.05:
        print('  → Reject H₀: Series is STATIONARY')
    else:
        print('  → Fail to reject H₀: Series is NON-STATIONARY')

adf_test(ts, 'Original Series')
```

**Interpretation**:
- p-value < 0.05 → Reject H₀ → Stationary
- p-value ≥ 0.05 → Non-stationary

## 🔄 Making Series Stationary

### 1. Differencing

```
∇Y_t = Y_t - Y_{t-1}
```

```python
# First-order differencing
ts_diff = ts.diff().dropna()

plt.figure(figsize=(12, 4))
plt.plot(ts_diff)
plt.title('First Difference')
plt.show()

adf_test(ts_diff, 'First Difference')
```

**Second-order differencing**:
```
∇²Y_t = ∇Y_t - ∇Y_{t-1}
```

### 2. Log Transformation

Stabilizes variance:
```
Y'_t = log(Y_t)
```

```python
ts_log = np.log(ts)
```

### 3. Detrending

Remove trend component:
```python
from scipy import signal

detrended = signal.detrend(ts)
```

### 4. Seasonal Differencing

For seasonal data with period s:
```
∇_sY_t = Y_t - Y_{t-s}
```

```python
# Monthly data with yearly seasonality
ts_seasonal_diff = ts.diff(12).dropna()  # s=12
```

---

# Moving Averages

## 📘 Simple Moving Average (SMA)

**Average** of last k observations:

```
SMA_t = (1/k) Σ_{i=0}^{k-1} Y_{t-i}
```

## 🧮 Properties

- **Smooths** short-term fluctuations
- **Lags** behind actual trend
- Equal weight to all k observations

## 🧪 Python Implementation

```python
def simple_moving_average(series, window):
    """Compute SMA."""
    return series.rolling(window=window).mean()

# Different window sizes
windows = [7, 30, 90]

plt.figure(figsize=(12, 6))
plt.plot(ts, label='Original', alpha=0.5)

for w in windows:
    sma = simple_moving_average(ts, w)
    plt.plot(sma, label=f'SMA-{w}', linewidth=2)

plt.legend()
plt.title('Simple Moving Averages')
plt.show()
```

**Observation**: 
- Small window (7): Follows data closely (less smoothing)
- Large window (90): Very smooth (more lag)

## 📊 Weighted Moving Average (WMA)

Give **different weights** to observations:

```
WMA_t = Σ_{i=0}^{k-1} w_i Y_{t-i}
```

Where Σw_i = 1

**Example**: More weight to recent observations
```
w = [0.5, 0.3, 0.2]  # Most recent gets 0.5
```

```python
def weighted_moving_average(series, weights):
    """Compute WMA."""
    weights = np.array(weights)
    weights = weights / weights.sum()  # Normalize
    
    def wma(x):
        return np.dot(x, weights[::-1])
    
    return series.rolling(window=len(weights)).apply(wma, raw=True)

wma = weighted_moving_average(ts, [0.5, 0.3, 0.2])
```

---

# Exponential Smoothing

## 📘 Single Exponential Smoothing (SES)

**Exponentially decreasing weights** for past observations.

## 🧮 Mathematical Foundation

### Recursive Formula

```
Ŷ_{t+1} = α Y_t + (1-α) Ŷ_t
```

Where:
- α ∈ (0, 1) = smoothing parameter
- Ŷ_t = forecast/smoothed value at time t
- Y_t = actual observation at time t

**Initial condition**: Ŷ_1 = Y_1 (or average of first few)

### Explicit Form

Expanding recursion:

```
Ŷ_{t+1} = α Y_t + α(1-α) Y_{t-1} + α(1-α)² Y_{t-2} + ...
        = α Σ_{i=0}^{∞} (1-α)^i Y_{t-i}
```

**Exponentially decreasing weights**: (1-α)^i

## 🧠 Effect of α

```
α close to 1:
  - More weight to recent observations
  - Less smoothing (reactive)
  - Good for rapidly changing data

α close to 0:
  - More weight to past observations
  - More smoothing (stable)
  - Good for stable data
```

## 🧪 Python Implementation

```python
def single_exponential_smoothing(series, alpha):
    """Compute SES."""
    result = [series.iloc[0]]  # Initialize with first value
    
    for i in range(1, len(series)):
        forecast = alpha * series.iloc[i-1] + (1 - alpha) * result[-1]
        result.append(forecast)
    
    return pd.Series(result, index=series.index)

# Compare different alpha values
alphas = [0.1, 0.5, 0.9]

plt.figure(figsize=(12, 6))
plt.plot(ts, label='Original', alpha=0.5)

for alpha in alphas:
    ses = single_exponential_smoothing(ts, alpha)
    plt.plot(ses, label=f'SES α={alpha}', linewidth=2)

plt.legend()
plt.title('Single Exponential Smoothing')
plt.show()
```

### Using Statsmodels

```python
from statsmodels.tsa.holtwinters import SimpleExpSmoothing

model = SimpleExpSmoothing(ts)
fit = model.fit(smoothing_level=0.5, optimized=False)

# Or optimize alpha
fit_opt = model.fit(optimized=True)
print(f"Optimized α: {fit_opt.params['smoothing_level']:.4f}")

# Forecast
forecast = fit_opt.forecast(steps=30)

plt.figure(figsize=(12, 6))
plt.plot(ts, label='Original')
plt.plot(fit_opt.fittedvalues, label='Fitted', linewidth=2)
plt.plot(forecast, label='Forecast', linewidth=2, linestyle='--')
plt.legend()
plt.title('SES with Optimized α')
plt.show()
```

## 📊 Double Exponential Smoothing (Holt's Method)

Handles **trend** in addition to level.

### Formulas

```
Level:    ℓ_t = α Y_t + (1-α)(ℓ_{t-1} + b_{t-1})
Trend:    b_t = β (ℓ_t - ℓ_{t-1}) + (1-β) b_{t-1}
Forecast: Ŷ_{t+h} = ℓ_t + h × b_t
```

Where:
- α = level smoothing parameter
- β = trend smoothing parameter
- h = forecast horizon

```python
from statsmodels.tsa.holtwinters import ExponentialSmoothing

# Generate data with trend
trend_data = pd.Series(np.linspace(100, 200, 365) + np.random.normal(0, 5, 365),
                       index=dates)

# Holt's method
model_holt = ExponentialSmoothing(trend_data, trend='add', seasonal=None)
fit_holt = model_holt.fit()

print(f"α (level): {fit_holt.params['smoothing_level']:.4f}")
print(f"β (trend): {fit_holt.params['smoothing_trend']:.4f}")

# Forecast
forecast_holt = fit_holt.forecast(steps=30)

plt.figure(figsize=(12, 6))
plt.plot(trend_data, label='Original')
plt.plot(fit_holt.fittedvalues, label='Fitted')
plt.plot(forecast_holt, label='Forecast', linestyle='--')
plt.legend()
plt.title("Holt's Linear Trend Method")
plt.show()
```

## 📊 Triple Exponential Smoothing (Holt-Winters)

Handles **trend + seasonality**.

### Formulas (Additive Seasonality)

```
Level:    ℓ_t = α(Y_t - s_{t-m}) + (1-α)(ℓ_{t-1} + b_{t-1})
Trend:    b_t = β(ℓ_t - ℓ_{t-1}) + (1-β)b_{t-1}
Seasonal: s_t = γ(Y_t - ℓ_t) + (1-γ)s_{t-m}
Forecast: Ŷ_{t+h} = ℓ_t + h×b_t + s_{t+h-m}
```

Where:
- m = seasonal period
- γ = seasonal smoothing parameter

### Multiplicative Seasonality

```
Level:    ℓ_t = α(Y_t / s_{t-m}) + (1-α)(ℓ_{t-1} + b_{t-1})
Seasonal: s_t = γ(Y_t / ℓ_t) + (1-γ)s_{t-m}
Forecast: Ŷ_{t+h} = (ℓ_t + h×b_t) × s_{t+h-m}
```

```python
# Generate seasonal data
seasonal_data = pd.Series(
    100 + np.linspace(0, 50, 365) + 
    20 * np.sin(2 * np.pi * np.arange(365) / 365) + 
    np.random.normal(0, 3, 365),
    index=dates
)

# Holt-Winters
model_hw = ExponentialSmoothing(
    seasonal_data,
    trend='add',
    seasonal='add',
    seasonal_periods=365
)
fit_hw = model_hw.fit()

print(f"α (level): {fit_hw.params['smoothing_level']:.4f}")
print(f"β (trend): {fit_hw.params['smoothing_trend']:.4f}")
print(f"γ (seasonal): {fit_hw.params['smoothing_seasonal']:.4f}")

# Forecast
forecast_hw = fit_hw.forecast(steps=90)

plt.figure(figsize=(14, 6))
plt.plot(seasonal_data, label='Original')
plt.plot(fit_hw.fittedvalues, label='Fitted')
plt.plot(forecast_hw, label='Forecast', linestyle='--', linewidth=2)
plt.legend()
plt.title('Holt-Winters Method')
plt.show()
```

---

# Time Series Decomposition

## 📘 Concept Overview

**Decomposition**: Separate time series into components (Trend, Seasonal, Residual).

## 🧮 Methods

### 1. Additive Decomposition

```
Y_t = T_t + S_t + R_t
```

```python
from statsmodels.tsa.seasonal import seasonal_decompose

# Additive decomposition
decomposition = seasonal_decompose(ts, model='additive', period=30)

fig = decomposition.plot()
fig.set_size_inches(12, 8)
plt.tight_layout()
plt.show()

# Access components
trend = decomposition.trend
seasonal = decomposition.seasonal
residual = decomposition.resid
```

### 2. Multiplicative Decomposition

```
Y_t = T_t × S_t × R_t
```

```python
decomposition_mult = seasonal_decompose(ts, model='multiplicative', period=30)
```

## 📊 STL Decomposition

**STL**: Seasonal and Trend decomposition using Loess

**Advantages**:
- Handles any seasonal pattern
- Robust to outliers
- Allows changing seasonal component

```python
from statsmodels.tsa.seasonal import STL

stl = STL(ts, seasonal=13)  # seasonal window
result = stl.fit()

fig = result.plot()
fig.set_size_inches(12, 8)
plt.tight_layout()
plt.show()
```

## 📊 Autocorrelation

### ACF (Autocorrelation Function)

Correlation between Y_t and Y_{t-k}:

```
ρ_k = Cov(Y_t, Y_{t-k}) / Var(Y_t)
```

```python
from statsmodels.graphics.tsaplots import plot_acf

fig, ax = plt.subplots(figsize=(12, 4))
plot_acf(ts, lags=40, ax=ax)
plt.title('Autocorrelation Function (ACF)')
plt.show()
```

**Interpretation**:
- Slow decay → Trend present
- Periodic pattern → Seasonality present
- Rapid decay → Stationary

### PACF (Partial Autocorrelation Function)

Correlation between Y_t and Y_{t-k} **removing** effect of intermediate lags.

```python
from statsmodels.graphics.tsaplots import plot_pacf

fig, ax = plt.subplots(figsize=(12, 4))
plot_pacf(ts, lags=40, ax=ax)
plt.title('Partial Autocorrelation Function (PACF)')
plt.show()
```

**Use**: 
- ACF + PACF help identify ARIMA orders
- Will cover in Session 14

---

# 🔥 MCQs

### Q1. Stationary series has:
**Options:**
- A) Increasing trend
- B) Constant mean and variance ✓
- C) Seasonality
- D) Random walk

**Explanation**: Stationarity requires constant statistical properties over time.

---

### Q2. ADF test null hypothesis is:
**Options:**
- A) Series is stationary
- B) Series is non-stationary (has unit root) ✓
- C) Series has trend
- D) Series has seasonality

**Explanation**: H₀: Unit root present (non-stationary). Reject H₀ if p < 0.05.

---

### Q3. SMA with window k gives weight:
**Options:**
- A) Exponentially decreasing
- B) 1/k to all observations ✓
- C) More to recent
- D) Random

**Explanation**: Simple MA gives equal weight = 1/k to all k observations.

---

### Q4. In SES, α close to 1 means:
**Options:**
- A) More smoothing
- B) Less smoothing (reactive) ✓
- C) No effect
- D) Constant forecast

**Explanation**: High α gives more weight to recent observations (less smoothing).

---

### Q5. Holt's method adds:
**Options:**
- A) Seasonality
- B) Trend ✓
- C) Cyclicity
- D) Noise

**Explanation**: Double exponential smoothing handles level + trend.

---

### Q6. Holt-Winters handles:
**Options:**
- A) Level only
- B) Level + trend
- C) Level + trend + seasonality ✓
- D) Trend only

**Explanation**: Triple exponential smoothing (level + trend + seasonal).

---

### Q7. Additive model is:
**Options:**
- A) Y = T × S × I
- B) Y = T + S + I ✓
- C) Y = T - S
- D) Y = log(T + S)

**Explanation**: Additive: components added together.

---

### Q8. Differencing is used to:
**Options:**
- A) Add trend
- B) Remove trend/make stationary ✓
- C) Smooth data
- D) Add seasonality

**Explanation**: ∇Y_t = Y_t - Y_{t-1} removes trend.

---

### Q9. ACF measures:
**Options:**
- A) Variance
- B) Correlation between Y_t and Y_{t-k} ✓
- C) Trend
- D) Forecast error

**Explanation**: Autocorrelation at lag k.

---

### Q10. STL decomposition is:
**Options:**
- A) Simple Time Linear
- B) Seasonal and Trend using Loess ✓
- C) Statistical Testing Level
- D) Smooth Trend Line

**Explanation**: STL = Seasonal and Trend decomposition using Loess.

---

### Q11. For seasonal data with period 12, seasonal differencing is:
**Options:**
- A) Y_t - Y_{t-1}
- B) Y_t - Y_{t-12} ✓
- C) Y_t - Y_{t-6}
- D) Y_t / Y_{t-12}

**Explanation**: Seasonal differencing uses lag = seasonal period.

---

### Q12. SMA lag refers to:
**Options:**
- A) Delay in capturing trend ✓
- B) Autocorrelation
- C) Forecast horizon
- D) Window size

**Explanation**: MA lags behind actual trend (smooths past data).

---

### Q13. Multiplicative model appropriate when:
**Options:**
- A) Constant seasonal variation
- B) Seasonal variation proportional to level ✓
- C) No seasonality
- D) Linear trend only

**Explanation**: Multiplicative when seasonal amplitude grows with level.

---

### Q14. PACF removes:
**Options:**
- A) Trend
- B) Effect of intermediate lags ✓
- C) Seasonality
- D) Noise

**Explanation**: Partial ACF controls for lags 1 to k-1.

---

### Q15. Log transformation stabilizes:
**Options:**
- A) Mean
- B) Variance ✓
- C) Autocorrelation
- D) Trend

**Explanation**: Log(Y) reduces heteroscedasticity (varying variance).

---

# ⚠️ Common Mistakes

1. **Not checking stationarity before modeling**: Most models assume stationarity

2. **Using SMA for forecasting**: SMA smooths but lags (better methods exist)

3. **Forgetting to difference back**: After differencing, must integrate predictions

4. **Wrong decomposition model**: Additive vs multiplicative depends on data

5. **Over-differencing**: Differencing too many times adds unnecessary noise

6. **Ignoring seasonal differencing**: Need both regular and seasonal for seasonal data

7. **Not optimizing smoothing parameters**: Always use `optimized=True` or grid search

8. **Confusing ACF and PACF**: ACF shows all correlations, PACF shows direct only

9. **Using too small window for decomposition**: Need at least 2 seasonal periods

10. **Extrapolating SMA/SES too far**: Good for short-term, poor for long-term

---

# ⭐ One-Line Exam Facts

1. **Stationary**: Constant mean, variance, and autocovariance over time

2. **ADF test**: H₀ = unit root (non-stationary), reject if p < 0.05

3. **Differencing**: ∇Y_t = Y_t - Y_{t-1} (removes trend)

4. **SMA**: Equal weight 1/k to all k observations

5. **SES formula**: Ŷ_{t+1} = αY_t + (1-α)Ŷ_t

6. **High α**: More weight to recent (less smoothing, reactive)

7. **Low α**: More weight to past (more smoothing, stable)

8. **Holt's method**: Level + trend (double exponential)

9. **Holt-Winters**: Level + trend + seasonality (triple exponential)

10. **Additive**: Y = T + S + I (constant seasonal variation)

11. **Multiplicative**: Y = T × S × I (proportional seasonal variation)

12. **ACF**: Correlation between Y_t and Y_{t-k}

13. **PACF**: Direct correlation (removes intermediate lags)

14. **STL**: Robust decomposition using Loess smoothing

15. **Seasonal differencing**: Y_t - Y_{t-s} where s = period

---

**End of Session 13**
