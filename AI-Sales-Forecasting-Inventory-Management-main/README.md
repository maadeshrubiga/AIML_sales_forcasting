🤖 AI-Based Sales Forecasting and Inventory Management

AI-Based Sales Forecasting and Inventory Management is an intelligent system developed to predict future sales and optimize inventory decisions using Artificial Intelligence and Machine Learning techniques. The project helps businesses forecast demand, reduce overstocking or stock shortages, and improve inventory planning through data-driven insights.


Commands to run:
pip install -r requirements.txt
pip install reportlab
streamlit run app.py


📌 Project Overview

The system analyzes historical sales data, seasonal patterns, holidays, and economic factors to predict future sales. It integrates advanced forecasting techniques such as LSTM (Long Short-Term Memory) and ARIMA (AutoRegressive Integrated Moving Average) to compare model performance and generate accurate predictions.

The project also includes an interactive Streamlit dashboard for visualization, comparison, forecasting, and report generation.

🎯 Objectives

• Predict future sales using AI and Machine Learning models  
• Compare forecasting performance between LSTM and ARIMA  
• Improve inventory planning and stock management  
• Analyze seasonal trends and sales patterns  
• Reduce inventory wastage and stock shortages  
• Provide interactive visualizations and reports

✨ Features

• Sales forecasting using LSTM and ARIMA models  
• Comparative analysis between forecasting models  
• Inventory management support system  
• Forecast days selection for future sales prediction  
• Interactive Streamlit dashboard  
• Data visualization with charts and graphs  
• Performance evaluation metrics  
• Report generation and analytics

📂 Dataset Used

The project uses historical retail sales datasets along with supporting datasets:

• train.csv – Historical sales records  
• test.csv – Data used for forecasting and testing  
• features.csv – Economic and environmental features such as CPI, temperature, fuel price, unemployment, and holidays  
• stores.csv – Store-related information

📅 Engineered Temporal Features

The system generates temporal features to improve forecasting accuracy:

• Week / Month / Quarter – Captures seasonality  
• Lag_1 – Sales from previous week  
• Lag_4 – Monthly sales trend  
• Lag_52 – Same week previous year  
• RollMean_4 – 4-week moving average  
• RollStd_4 – Sales volatility signal  
• RollMean_12 – Quarterly trend  
• IsHoliday – Holiday demand spikes

🧠 Algorithms and Techniques Used

1️⃣ LSTM (Long Short-Term Memory)  
Used for deep learning-based sequential forecasting and long-term trend prediction.

2️⃣ ARIMA (AutoRegressive Integrated Moving Average)  
Used for time-series forecasting based on historical statistical patterns.

3️⃣ Feature Engineering  
Used to create temporal and rolling statistical features for improved model performance.

4️⃣ Comparative Analysis  
Used to compare prediction accuracy between LSTM and ARIMA models.

🛠️ Tools and Technologies Used

• Python  
• Streamlit  
• TensorFlow / Keras  
• Pandas  
• NumPy  
• Matplotlib  
• Scikit-learn  
• Statsmodels  
• VS Code

📊 Model Performance

The system achieved strong forecasting accuracy:

• MAE: 1,595,194.47  
• RMSE: 2,060,132.28  
• MAPE: 3.49%  
• Accuracy: 96.51%

✅ Results

The system successfully predicts future sales with high accuracy and provides meaningful inventory insights. It helps businesses make better inventory decisions, understand demand patterns, and reduce operational risks.

🚀 Future Enhancements

• Real-time sales forecasting integration  
• Cloud deployment support  
• Multi-product forecasting  
• AI chatbot for business insights  
• Demand alert notifications  
• Advanced deep learning techniques

📁 Project Structure

• app.py – Main Streamlit application  
• lstm_model.py – LSTM forecasting model  
• arima_model.py – ARIMA forecasting model  
• rl_inventory.py – Inventory management logic  
• data/ – Dataset files  
• requirements.txt – Required dependencies



