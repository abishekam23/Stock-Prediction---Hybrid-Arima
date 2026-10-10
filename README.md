# Stock-Prediction-Hybrid-Arima

Stock market prediction is challenging due to the complexity and volatility of financial
data. Traditional models like ARIMA (Autoregressive Integrated Moving Average)
capture linear trends, while deep learning models like LSTM (Long Short-Term
Memory) can model non-linear relationships. This study develops a hybrid ARIMA-LSTM
model to enhance prediction accuracy by combining the strengths of both
methods.

The model combines LSTM's ability to learn complex patterns with ARIMA's linear
forecasting: the LSTM produces the main forecast, and an ARIMA model fitted on the
LSTM's residual errors corrects it. The hybrid model was tested on a range of
stocks, including GSPC, NASDAQ, Zomato, Adani Wilmar, IRCTC, SBI, Tata Motors,
Jio Finance, and Gold Stock. Results show that the hybrid model outperforms
standalone ARIMA and LSTM models, reducing prediction errors (RMSE and MAE).

Future work should focus on developing adaptive hybridization methods and incorporating
exogenous factors like economic indicators and news sentiment for improved accuracy.
