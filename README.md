# Stock Price Forecasting: Hybrid LSTM + ARIMA

Stock market prediction is challenging due to the complexity and volatility of financial
data. Traditional models like ARIMA (Autoregressive Integrated Moving Average) capture
linear patterns, while deep learning models like LSTM (Long Short-Term Memory) can model
non-linear relationships. This project builds a hybrid ARIMA-LSTM model that combines
the two, and tests it against each model on its own and against a naive baseline.

## Method

1. An **LSTM** looks at the last 20 daily log returns and forecasts the next day's return.
2. An **ARIMA** model is fitted on the LSTM's errors (residuals) to pick up any linear
   pattern the LSTM missed.
3. **Hybrid forecast** = LSTM forecast + ARIMA's predicted error. The forecast return is
   converted back to a price: `P̂(t) = P(t-1) · exp(r̂(t))`.

Data: daily closing prices from Yahoo Finance, last 1,500 trading days up to 9 Oct 2026,
split in time order:

| Block | Share | Used for |
|---|---|---|
| Train | 70% | fitting the LSTM and the standalone ARIMA |
| Validation | 15% | fitting the ARIMA on the LSTM's out-of-sample residuals |
| Test | 15% | evaluation only |

No model sees test data while it is being fitted. Every forecast for day *t* uses only
data up to day *t-1*.

**Models compared**

| Model | Description |
|---|---|
| Naive | tomorrow's price = today's price |
| ARIMA | `auto_arima` on log prices |
| LSTM | 1 LSTM layer (32 units), averaged over 3 random seeds |
| Hybrid | LSTM + ARIMA on LSTM residuals |

## Assets

| Asset | Yahoo ticker | Note |
|---|---|---|
| S&P 500 | `^GSPC` | |
| NASDAQ Composite | `^IXIC` | |
| Zomato | `ETERNAL.NS` | renamed Eternal Ltd in 2025 |
| Adani Wilmar | `AWL.NS` | |
| IRCTC | `IRCTC.NS` | |
| SBI | `SBIN.NS` | |
| Tata Motors | `TMPV.NS` | prices before the 14 Oct 2025 demerger are rescaled |
| Jio Financial | `JIOFIN.NS` | listed Aug 2023, so a shorter history |
| Gold | `GOLDBEES.NS` | Nippon India gold ETF |

## Results

Test-period RMSE as a ratio of the naive forecast's RMSE (below 1.000 beats naive):

| Asset | ARIMA | LSTM | Hybrid |
|---|---|---|---|
| S&P 500 | 0.997 | 0.997 | 1.056 |
| NASDAQ Composite | 1.000 | 0.997 | 1.068 |
| Zomato | 1.000 | 0.999 | 0.999 |
| Adani Wilmar | 1.066 | 1.008 | 1.008 |
| IRCTC | 1.020 | 1.007 | 1.058 |
| SBI | 1.005 | 1.006 | 1.006 |
| Tata Motors | 1.010 | 1.027 | 1.027 |
| Jio Financial | 1.000 | 1.003 | 0.975 |
| Gold | 1.000 | 0.998 | 1.000 |
| **Average** | **1.011** | **1.004** | **1.022** |

![RMSE relative to naive](rmse_vs_naive.png)

Findings:

- **No model reliably beats the naive forecast.** Every model's error is between 2.5% better
  and 6.8% worse than naive, and on average all three are slightly worse. The hybrid beats
  naive on only 2 assets (Zomato, Jio Financial), and neither difference is statistically
  significant (Diebold-Mariano test, p = 0.64 and 0.46).
- **The hybrid does not outperform the LSTM.** On 5 of 9 assets the residual ARIMA found
  no pattern to correct, so the hybrid equals the LSTM. It has lower error on Jio Financial
  (not significant, p = 0.40) and significantly higher error on S&P 500, NASDAQ and IRCTC
  (p < 0.05), where the residual ARIMA fitted noise in the validation period.
- **Direction accuracy is close to a coin flip**: 48.7% ARIMA, 49.5% LSTM, 50.6% hybrid on average.
- For most assets `auto_arima` picks ARIMA(0,1,0), which is a random walk, the same as the naive forecast.

These results are consistent with daily stock prices behaving close to a random walk:
past prices alone carry little information about the next day's move. The forecast lines
in the notebook look accurate on a chart because they mostly follow the previous day's price,
one day late.

An earlier version of this project reported that the hybrid reduced error. That result
came from fitting the ARIMA on residuals from the test period, which let the model see the
answers. The current version removes that leak.

## Files

| File | Contents |
|---|---|
| `Hybrid_Model.ipynb` | full pipeline: data, four models, evaluation, charts |
| `LSTM_Model.ipynb` | first experiment: stacked LSTM on Apple price levels, compared with naive |
| `results.csv` | all metrics per asset and model (RMSE, MAE, MAPE, RMSE vs naive, direction hit rate) |
| `rmse_vs_naive.png`, `hybrid_last60.png` | charts from the notebook |

## How to run

```bash
pip install -r requirements.txt
jupyter notebook Hybrid_Model.ipynb
```

The data end date and random seeds are fixed. A full run takes a few minutes on a laptop CPU.

## Future work

- Add exogenous inputs such as economic indicators, FX rates and news sentiment.
- Forecast targets that are more predictable than price direction, such as volatility or trading volume.
- Use walk-forward evaluation (re-fitting the models as time moves on) over several test periods.
