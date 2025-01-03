# Stock-Prediction-Hybrid-Arima

Stock market prediction is challenging due to the complexity and volatility of financial
data. Traditional models like ARIMA (Autoregressive Integrated Moving Average)
capture linear trends, while deep learning models like LSTM (Long Short-Term
Memory) can model non-linear relationships. This study develops a hybrid ARIMALSTM
model to enhance prediction accuracy by combining the strengths of both
methods. 

The model integrates ARIMA's linear forecasting with LSTM’s ability to
learn complex patterns, using a novel Degree of Hybridization (DoH) parameter to
dynamically balance their contributions. The hybrid model was tested on a range of
stocks, including GSPC, NASDAQ, Zomato, Adani Wilmar, IRCTC, SBI, Tata Motors,
Jio Finance, and Gold Stock. Results show that the hybrid model outperforms
standalone ARIMA and LSTM models, reducing prediction errors (RMSE and MAE).

The DoH parameter provides flexibility in adjusting the balance between linear and
non-linear components, adapting to varying market conditions. Future work should
focus on developing adaptive hybridization methods and incorporating exogenous
factors like economic indicators and news sentiment for improved accuracy.
