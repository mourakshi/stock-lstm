# 📈 Stock Price Predictor & Investment Simulator

An interactive Streamlit web app that predicts next-day stock prices using a deep learning model and simulates investment outcomes. Built with a Bidirectional LSTM trained on technical indicators and historical price data.

![image](https://github.com/user-attachments/assets/c9f79b7e-3687-4ce6-a5d9-0119cea27007)
![image](https://github.com/user-attachments/assets/46e97997-07b7-43dd-ac0a-c08a792f1875)


---

## 🚀 Key Features

- **Real-time Data** - Fetches live market data from Yahoo Finance (`yfinance`)
- **Technical Analysis** - Calculates 13+ indicators including:
  - Moving Averages (SMA, EMA)
  - Momentum Indicators (RSI, MACD)
  - Volatility Measures (Bollinger Bands)
- **AI Prediction** - Bidirectional LSTM model trained on 60-day windows
- **Investment Simulator** - Calculates potential profit/loss scenarios
- **Trading Strategy** - Provides Buy/Hold/Sell recommendations
- **Interactive Charts** - Visualizes historical prices and predictions

---

## 🧠 Model Architecture

- **Model Type**: 2-layer Bidirectional LSTM
- **Input**: 60 days × 13 technical features
- **Output**: Next day's predicted price
- **Training**:
  ```python
  model.fit(X_train, y_train, epochs=20, batch_size=32)
  stock-lstm-app/
  
##📦 Project Structure
├── app.py                   # Main Streamlit application
├── requirements.txt         # Python dependencies
├── model/
│   ├── model.h5             # Pretrained LSTM model
│   ├── feature_scaler.save  # Feature normalization
│   └── target_scaler.save   # Price normalization
##🔧 Installation & Usage
### Clone repository
git clone https://github.com/yourusername/stock-lstm-app.git
cd stock-lstm-app

###Install dependencies
pip install -r requirements.txt

### Launch application
streamlit run app.py
##✨ Credits
-Developed by Mourakshi Thakuria

-Built with TensorFlow, Streamlit, and yfinance

-Visualization using Plotly
##📄 License
MIT License - Free for use, modification, and distribution
