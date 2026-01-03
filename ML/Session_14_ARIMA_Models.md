# Session 14 – ARIMA Models

## 📚 Table of Contents
1. [Autoregressive (AR) Models](#autoregressive-ar-models)
2. [Moving Average (MA) Models](#moving-average-ma-models)
3. [ARMA Models](#arma-models)
4. [ARIMA Models](#arima-models)
5. [Model Identification](#model-identification)
6. [SARIMA Models](#sarima-models)
7. [MCQs](#mcqs)
8. [Common Mistakes](#common-mistakes)
9. [One-Line Exam Facts](#one-line-exam-facts)

---

# Autoregressive (AR) Models

## 📘 Concept Overview

**AR(p)**: Current value depends on **p previous values** plus noise.

## 🧮 Mathematical Foundation

### AR(p) Model

```
Y_t = c + φ₁Y_{t-1} + φ₂Y_{t-2} + ... + φₚY_{t-p} + ε_t
```

Where:
- c = constant term
- φᵢ = AR coefficients
- p = order (number of lags)
- ε_t ~ N(0, σ²) = white noise

### AR(1) Model

Simplest case (p=1):

```
Y_t = c + φ₁Y_{t-1} + ε_t
```

**Stationarity condition**: |φ₁| < 1

**Mean**: μ = c/(1 - φ₁)

**Variance**: σ²_y = σ²/(1 - φ₁²)

**Autocorrelation**:
```
ρ_k = φ₁^k  (exponential decay)
```

### AR(2) Model

```
Y_t = c + φ₁Y_{t-1} + φ₂Y_{t-2} + ε_t
```

**Stationarity conditions**:
- φ₁ + φ₂ < 1
- φ₂ - φ₁ < 1
- |φ₂| < 1

## 📊 ACF and PACF Patterns

### ACF for AR(p)
- **Decays** exponentially or sinusoidally
- Never cuts off sharply

### PACF for AR(p)
- **Cuts off** after lag p
- PACF(k) = 0 for k > p

**Identification**: PACF shows order p!

## 🧪 Python Implementation

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.ar_model import AutoReg
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

# Simulate AR(1) process
np.random.seed(42)
n = 500
phi1 = 0.7
c = 5

y = [c]
for t in range(1, n):
    y.append(c + phi1 * y[-1] + np.random.normal(0, 1))

y = np.array(y)
ts = pd.Series(y)

# Plot series
plt.figure(figsize=(12, 4))
plt.plot(ts)
plt.title('AR(1) Process (φ₁=0.7)')
plt.show()

# ACF and PACF
fig, axes = plt.subplots(1, 2, figsize=(14, 4))
plot_acf(ts, lags=20, ax=axes[0])
plot_pacf(ts, lags=20, ax=axes[1])
axes[0].set_title('ACF (exponential decay)')
axes[1].set_title('PACF (cuts off at lag 1)')
plt.tight_layout()
plt.show()

# Fit AR(1) model
model = AutoReg(ts, lags=1)
fit = model.fit()

print(fit.summary())
print(f"\nEstimated φ₁: {fit.params[1]:.4f} (true: {phi1})")
print(f"Estimated constant: {fit.params[0]:.4f} (true: {c})")

# Forecast
forecast = fit.forecast(steps=10)
print(f"\nForecast (next 10 steps):\n{forecast}")
```

### From Scratch (AR(1))

```python
class AR1:
    def fit(self, y):
        """Fit AR(1) using OLS."""
        # Y_t = c + φ₁ Y_{t-1} + ε_t
        Y = y[1:]
        X = y[:-1]
        
        # Add constant
        X_with_const = np.column_stack([np.ones(len(X)), X])
        
        # OLS: β = (X'X)^{-1} X'Y
        self.params = np.linalg.inv(X_with_const.T @ X_with_const) @ X_with_const.T @ Y
        self.c = self.params[0]
        self.phi1 = self.params[1]
        
        return self
    
    def predict(self, y, steps=1):
        """Forecast future values."""
        forecasts = []
        last_value = y[-1]
        
        for _ in range(steps):
            forecast = self.c + self.phi1 * last_value
            forecasts.append(forecast)
            last_value = forecast
        
        return np.array(forecasts)

# Test
ar1 = AR1()
ar1.fit(ts.values)
print(f"φ₁: {ar1.phi1:.4f}, c: {ar1.c:.4f}")
```

---

# Moving Average (MA) Models

## 📘 Concept Overview

**MA(q)**: Current value depends on **q previous error terms** plus noise.

## 🧮 Mathematical Foundation

### MA(q) Model

```
Y_t = μ + ε_t + θ₁ε_{t-1} + θ₂ε_{t-2} + ... + θ_qε_{t-q}
```

Where:
- μ = mean
- θᵢ = MA coefficients
- q = order
- ε_t ~ N(0, σ²) = white noise

### MA(1) Model

```
Y_t = μ + ε_t + θ₁ε_{t-1}
```

**Always stationary** (no conditions needed!)

**Mean**: E[Y_t] = μ

**Variance**: Var(Y_t) = σ²(1 + θ₁²)

**Autocorrelation**:
```
ρ₁ = θ₁/(1 + θ₁²)
ρ_k = 0 for k > 1  (cuts off!)
```

### MA(2) Model

```
Y_t = μ + ε_t + θ₁ε_{t-1} + θ₂ε_{t-2}
```

**Autocorrelation**:
```
ρ₁ = (θ₁ + θ₁θ₂)/(1 + θ₁² + θ₂²)
ρ₂ = θ₂/(1 + θ₁² + θ₂²)
ρ_k = 0 for k > 2
```

## 📊 ACF and PACF Patterns

### ACF for MA(q)
- **Cuts off** after lag q
- ACF(k) = 0 for k > q

### PACF for MA(q)
- **Decays** exponentially or sinusoidally
- Never cuts off sharply

**Identification**: ACF shows order q!

## 🧪 Python Implementation

```python
from statsmodels.tsa.arima.model import ARIMA

# Simulate MA(1) process
np.random.seed(42)
n = 500
theta1 = 0.8
mu = 10

epsilon = np.random.normal(0, 1, n+1)
y = [mu + epsilon[0]]

for t in range(1, n):
    y.append(mu + epsilon[t] + theta1 * epsilon[t-1])

y = np.array(y)
ts_ma = pd.Series(y)

# Plot
plt.figure(figsize=(12, 4))
plt.plot(ts_ma)
plt.title('MA(1) Process (θ₁=0.8)')
plt.show()

# ACF and PACF
fig, axes = plt.subplots(1, 2, figsize=(14, 4))
plot_acf(ts_ma, lags=20, ax=axes[0])
plot_pacf(ts_ma, lags=20, ax=axes[1])
axes[0].set_title('ACF (cuts off at lag 1)')
axes[1].set_title('PACF (exponential decay)')
plt.tight_layout()
plt.show()

# Fit MA(1) model using ARIMA(0,0,1)
model_ma = ARIMA(ts_ma, order=(0, 0, 1))
fit_ma = model_ma.fit()

print(fit_ma.summary())
print(f"\nEstimated θ₁: {fit_ma.params['ma.L1']:.4f} (true: {theta1})")
```

---

# ARMA Models

## 📘 Concept Overview

**ARMA(p, q)**: Combines **AR(p)** and **MA(q)** components.

## 🧮 Mathematical Foundation

### ARMA(p, q) Model

```
Y_t = c + φ₁Y_{t-1} + ... + φₚY_{t-p} + ε_t + θ₁ε_{t-1} + ... + θ_qε_{t-q}
```

**Combines**:
- AR part: φ₁Y_{t-1} + ... + φₚY_{t-p}
- MA part: θ₁ε_{t-1} + ... + θ_qε_{t-q}

### ARMA(1, 1) Model

```
Y_t = c + φ₁Y_{t-1} + ε_t + θ₁ε_{t-1}
```

**Most commonly used** simple ARMA model.

## 📊 ACF and PACF Patterns

### ACF for ARMA(p, q)
- Decays exponentially (not sharp cutoff)

### PACF for ARMA(p, q)
- Decays exponentially (not sharp cutoff)

**Both decay** → suggests ARMA (not pure AR or MA)

## 🧪 Python Implementation

```python
# Simulate ARMA(1,1)
from statsmodels.tsa.arima_process import ArmaProcess

np.random.seed(42)
ar_params = np.array([1, -0.7])  # 1, -φ₁
ma_params = np.array([1, 0.5])   # 1, θ₁

arma_process = ArmaProcess(ar_params, ma_params)
y_arma = arma_process.generate_sample(nsample=500)

ts_arma = pd.Series(y_arma)

# Plot
plt.figure(figsize=(12, 4))
plt.plot(ts_arma)
plt.title('ARMA(1,1) Process')
plt.show()

# ACF and PACF
fig, axes = plt.subplots(1, 2, figsize=(14, 4))
plot_acf(ts_arma, lags=20, ax=axes[0])
plot_pacf(ts_arma, lags=20, ax=axes[1])
axes[0].set_title('ACF (both decay)')
axes[1].set_title('PACF (both decay)')
plt.tight_layout()
plt.show()

# Fit ARMA(1,1)
model_arma = ARIMA(ts_arma, order=(1, 0, 1))
fit_arma = model_arma.fit()

print(fit_arma.summary())
```

---

# ARIMA Models

## 📘 Concept Overview

**ARIMA(p, d, q)**: ARMA on **differenced** series.

**d** = number of differencing operations to make series stationary

## 🧮 Mathematical Foundation

### ARIMA(p, d, q) Model

```
Step 1: Difference d times to get stationary series Z_t
        Z_t = ∇^d Y_t

Step 2: Fit ARMA(p, q) to Z_t
        Z_t = φ₁Z_{t-1} + ... + φₚZ_{t-p} + ε_t + θ₁ε_{t-1} + ... + θ_qε_{t-q}
```

**Common values**:
- d = 0: No differencing (stationary series) → ARMA
- d = 1: First difference (remove trend)
- d = 2: Second difference (rare)

### Example: ARIMA(1, 1, 0)

```
Step 1: Z_t = Y_t - Y_{t-1}  (first difference)
Step 2: Z_t = φ₁Z_{t-1} + ε_t  (AR(1) on difference)
```

Equivalently:
```
Y_t = Y_{t-1} + φ₁(Y_{t-1} - Y_{t-2}) + ε_t
```

## ⚙️ Model Selection Process

```
1. Check stationarity (ADF test)
   - If non-stationary → difference (d=1)
   - Repeat until stationary

2. Plot ACF and PACF of stationary series

3. Identify p and q:
   - PACF cuts at p → AR(p) → ARIMA(p, d, 0)
   - ACF cuts at q → MA(q) → ARIMA(0, d, q)
   - Both decay → ARMA(p,q) → ARIMA(p, d, q)

4. Fit multiple models, compare AIC/BIC

5. Check residuals (should be white noise)

6. Forecast
```

## 🧪 Python Implementation

```python
from statsmodels.tsa.stattools import adfuller

# Generate non-stationary series (random walk with drift)
np.random.seed(42)
n = 200
drift = 0.5
y_rw = np.cumsum(np.random.normal(drift, 1, n))

ts_rw = pd.Series(y_rw)

# Plot
plt.figure(figsize=(12, 4))
plt.plot(ts_rw)
plt.title('Random Walk with Drift (Non-stationary)')
plt.show()

# Check stationarity
result = adfuller(ts_rw)
print(f"ADF Statistic: {result[0]:.4f}")
print(f"p-value: {result[1]:.4f}")
if result[1] > 0.05:
    print("Non-stationary → Need differencing")

# First difference
ts_diff = ts_rw.diff().dropna()

# Check stationarity of differenced series
result_diff = adfuller(ts_diff)
print(f"\nAfter differencing:")
print(f"ADF Statistic: {result_diff[0]:.4f}")
print(f"p-value: {result_diff[1]:.4f}")
if result_diff[1] < 0.05:
    print("Stationary → Can use d=1")

# ACF and PACF of differenced series
fig, axes = plt.subplots(1, 2, figsize=(14, 4))
plot_acf(ts_diff, lags=20, ax=axes[0])
plot_pacf(ts_diff, lags=20, ax=axes[1])
plt.tight_layout()
plt.show()

# Fit ARIMA(0,1,0) - random walk
model_010 = ARIMA(ts_rw, order=(0, 1, 0))
fit_010 = model_010.fit()

# Fit ARIMA(1,1,0)
model_110 = ARIMA(ts_rw, order=(1, 1, 0))
fit_110 = model_110.fit()

# Fit ARIMA(0,1,1)
model_011 = ARIMA(ts_rw, order=(0, 1, 1))
fit_011 = model_011.fit()

# Compare AIC
print("\nModel Comparison:")
print(f"ARIMA(0,1,0) AIC: {fit_010.aic:.2f}")
print(f"ARIMA(1,1,0) AIC: {fit_110.aic:.2f}")
print(f"ARIMA(0,1,1) AIC: {fit_011.aic:.2f}")

# Use best model
best_model = fit_110  # Example: assume (1,1,0) is best

print(f"\nBest model: ARIMA(1,1,0)")
print(best_model.summary())

# Forecast
forecast = best_model.forecast(steps=20)
forecast_ci = best_model.get_forecast(steps=20).conf_int()

# Plot with forecast
plt.figure(figsize=(12, 6))
plt.plot(ts_rw, label='Observed')
plt.plot(range(len(ts_rw), len(ts_rw)+20), forecast, 'r--', label='Forecast')
plt.fill_between(range(len(ts_rw), len(ts_rw)+20),
                forecast_ci.iloc[:, 0], forecast_ci.iloc[:, 1],
                alpha=0.3, color='red', label='95% CI')
plt.legend()
plt.title('ARIMA Forecast')
plt.show()
```

### Auto ARIMA

```python
from pmdarima import auto_arima

# Automatically select best ARIMA model
auto_model = auto_arima(
    ts_rw,
    start_p=0, start_q=0,
    max_p=5, max_q=5,
    d=None,  # Auto-determine d
    seasonal=False,
    stepwise=True,
    suppress_warnings=True,
    information_criterion='aic',
    trace=True
)

print(auto_model.summary())
```

---

# Model Identification

## 📊 ACF and PACF Decision Table

| ACF Pattern | PACF Pattern | Model |
|-------------|--------------|-------|
| Cuts off at lag q | Decays | MA(q) |
| Decays | Cuts off at lag p | AR(p) |
| Decays | Decays | ARMA(p,q) |

## 🧮 Information Criteria

### AIC (Akaike Information Criterion)

```
AIC = -2·log(L) + 2k
```

Where:
- L = likelihood
- k = number of parameters

**Lower AIC = better model**

### BIC (Bayesian Information Criterion)

```
BIC = -2·log(L) + k·log(n)
```

**BIC penalizes complexity more** than AIC

**Rule**: Prefer simpler model if AIC/BIC close

## 📊 Residual Diagnostics

### Requirements for Good Model

Residuals should be **white noise**:
1. Mean ≈ 0
2. Constant variance
3. No autocorrelation
4. Normally distributed

### Ljung-Box Test

**H₀**: Residuals are white noise (no autocorrelation)

```python
from statsmodels.stats.diagnostic import acorr_ljungbox

# Test residuals
residuals = best_model.resid
lb_test = acorr_ljungbox(residuals, lags=20)

print(lb_test)
# p-value > 0.05 → residuals are white noise (good!)
```

### Residual Plots

```python
# Diagnostic plots
best_model.plot_diagnostics(figsize=(14, 8))
plt.tight_layout()
plt.show()
```

---

# SARIMA Models

## 📘 Concept Overview

**SARIMA(p,d,q)(P,D,Q)ₘ**: ARIMA with **seasonal** components.

## 🧮 Mathematical Foundation

```
Non-seasonal: (p, d, q)
Seasonal: (P, D, Q) with period m
```

**P** = seasonal AR order
**D** = seasonal differencing order
**Q** = seasonal MA order
**m** = seasonal period (12 for monthly, 4 for quarterly, etc.)

### Example: SARIMA(1,1,1)(1,1,1)₁₂

Monthly data with:
- AR(1), first difference, MA(1) non-seasonal
- SAR(1), seasonal difference (lag 12), SMA(1)

## 🧪 Python Implementation

```python
from statsmodels.tsa.statespace.sarimax import SARIMAX

# Generate seasonal data
np.random.seed(42)
n = 144  # 12 years of monthly data
trend = np.linspace(100, 150, n)
seasonal = 20 * np.sin(2 * np.pi * np.arange(n) / 12)
noise = np.random.normal(0, 3, n)
y_seasonal = trend + seasonal + noise

ts_seasonal = pd.Series(y_seasonal)

# Plot
plt.figure(figsize=(12, 4))
plt.plot(ts_seasonal)
plt.title('Seasonal Time Series')
plt.show()

# Fit SARIMA
model_sarima = SARIMAX(
    ts_seasonal,
    order=(1, 1, 1),         # Non-seasonal
    seasonal_order=(1, 1, 1, 12)  # Seasonal with period 12
)

fit_sarima = model_sarima.fit()
print(fit_sarima.summary())

# Forecast
forecast_sarima = fit_sarima.forecast(steps=24)

plt.figure(figsize=(12, 6))
plt.plot(ts_seasonal, label='Observed')
plt.plot(range(len(ts_seasonal), len(ts_seasonal)+24), forecast_sarima,
        'r--', label='Forecast')
plt.legend()
plt.title('SARIMA Forecast')
plt.show()
```

---

# 🔥 MCQs

### Q1. AR(p) model depends on:
**Options:**
- A) Past errors
- B) Past p values ✓
- C) Future values
- D) Trend only

**Explanation**: AR(p) = autoregressive with p lagged values.

---

### Q2. For AR(p), PACF:
**Options:**
- A) Decays slowly
- B) Cuts off after lag p ✓
- C) Never cuts off
- D) Oscillates

**Explanation**: PACF identifies AR order (cuts off at p).

---

### Q3. MA(q) model depends on:
**Options:**
- A) Past values
- B) Past q error terms ✓
- C) Future values
- D) Seasonal components

**Explanation**: MA(q) = moving average of q past errors.

---

### Q4. For MA(q), ACF:
**Options:**
- A) Decays slowly
- B) Cuts off after lag q ✓
- C) Never cuts off
- D) Increases

**Explanation**: ACF identifies MA order (cuts off at q).

---

### Q5. In ARIMA(p,d,q), d represents:
**Options:**
- A) AR order
- B) MA order
- C) Number of differences ✓
- D) Seasonal period

**Explanation**: d = differencing order to make series stationary.

---

### Q6. ARIMA(0,1,0) is:
**Options:**
- A) White noise
- B) Random walk ✓
- C) AR(1)
- D) MA(1)

**Explanation**: First difference only (no AR/MA terms) = random walk.

---

### Q7. Lower AIC means:
**Options:**
- A) Worse model
- B) Better model ✓
- C) More parameters
- D) Overfitting

**Explanation**: AIC balances fit and complexity; lower = better.

---

### Q8. Ljung-Box test H₀ is:
**Options:**
- A) Residuals are autocorrelated
- B) Residuals are white noise ✓
- C) Series is stationary
- D) Model is correct

**Explanation**: Tests for autocorrelation in residuals; want p > 0.05.

---

### Q9. SARIMA(p,d,q)(P,D,Q)ₘ has seasonal period:
**Options:**
- A) p
- B) q
- C) m ✓
- D) P

**Explanation**: m = seasonal period (e.g., 12 for monthly).

---

### Q10. AR(1) is stationary if:
**Options:**
- A) φ₁ > 1
- B) |φ₁| < 1 ✓
- C) φ₁ = 1
- D) φ₁ < -1

**Explanation**: |φ₁| < 1 ensures stationarity for AR(1).

---

### Q11. MA models are:
**Options:**
- A) Conditionally stationary
- B) Always stationary ✓
- C) Never stationary
- D) Stationary if θ < 1

**Explanation**: MA models always stationary (no restrictions on θ).

---

### Q12. If both ACF and PACF decay, use:
**Options:**
- A) AR only
- B) MA only
- C) ARMA ✓
- D) White noise

**Explanation**: Both decay suggests mixed ARMA model.

---

### Q13. BIC vs AIC:
**Options:**
- A) BIC penalizes complexity more ✓
- B) AIC penalizes complexity more
- C) Both identical
- D) BIC ignores parameters

**Explanation**: BIC has k·log(n) penalty vs AIC's 2k.

---

### Q14. ARIMA(1,0,0) is same as:
**Options:**
- A) MA(1)
- B) AR(1) ✓
- C) Random walk
- D) ARMA(1,1)

**Explanation**: d=0 means no differencing; just AR(1).

---

### Q15. Seasonal differencing for monthly data uses lag:
**Options:**
- A) 1
- B) 4
- C) 12 ✓
- D) 365

**Explanation**: Monthly seasonality has period 12.

---

# ⚠️ Common Mistakes

1. **Not differencing non-stationary series**: ARMA requires stationarity (use d > 0)

2. **Over-differencing**: Too much differencing adds noise (check ADF after each difference)

3. **Confusing AR and MA**: AR uses past values, MA uses past errors

4. **Ignoring residual diagnostics**: Always check residuals are white noise

5. **Using AIC/BIC on different data**: Only compare on same dataset

6. **Forgetting seasonal component**: Use SARIMA for seasonal data

7. **Not checking stationarity assumptions**: Verify |φ| < 1 for AR

8. **Overfitting with too many parameters**: Start simple, add complexity if needed

9. **Interpreting ACF/PACF incorrectly**: PACF for AR order, ACF for MA order

10. **Forecasting too far ahead**: ARIMA uncertainty grows with horizon

---

# ⭐ One-Line Exam Facts

1. **AR(p)**: Y_t = φ₁Y_{t-1} + ... + φₚY_{t-p} + ε_t

2. **MA(q)**: Y_t = ε_t + θ₁ε_{t-1} + ... + θ_qε_{t-q}

3. **ARMA(p,q)**: Combines AR(p) and MA(q)

4. **ARIMA(p,d,q)**: ARMA on d-th differenced series

5. **AR(p) PACF**: Cuts off at lag p

6. **MA(q) ACF**: Cuts off at lag q

7. **AR(1) stationary**: |φ₁| < 1

8. **MA models**: Always stationary (no conditions)

9. **AIC**: -2log(L) + 2k (lower = better)

10. **BIC**: -2log(L) + k·log(n) (penalizes complexity more)

11. **Ljung-Box**: Tests residual autocorrelation (want p > 0.05)

12. **SARIMA(p,d,q)(P,D,Q)ₘ**: Adds seasonal AR, differencing, MA

13. **ARIMA(0,1,0)**: Random walk (first difference only)

14. **d = 0**: Stationary series (ARMA)

15. **Seasonal period m**: 12 (monthly), 4 (quarterly), 7 (weekly)

---

**End of Session 14**

**Progress: 14/30 sessions completed (47%)!** Time Series phase complete. Ready to continue with Recommendation Systems next.
