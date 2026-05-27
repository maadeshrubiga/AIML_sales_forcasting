import pandas as pd
import numpy as np

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

# ---------------- LOAD DATA ----------------
df = pd.read_csv("data/train.csv")

print("\n" + "="*50)
print("📊 SALES FORECASTING SYSTEM")
print("="*50)

print("\n✅ Data loaded successfully")
print(f"📁 Total records: {len(df)}")

# ---------------- PREPROCESS ----------------
df['Date'] = pd.to_datetime(df['Date'])
df = df.sort_values('Date')

df = df.groupby('Date')['Weekly_Sales'].sum().reset_index()
df.rename(columns={'Weekly_Sales': 'Sales'}, inplace=True)

# ---------------- LSTM ----------------
data = df['Sales'].values.reshape(-1, 1)

scaler = MinMaxScaler()
data_scaled = scaler.fit_transform(data)

X, y = [], []
for i in range(10, len(data_scaled)):
    X.append(data_scaled[i-10:i])
    y.append(data_scaled[i])

X, y = np.array(X), np.array(y)

split = int(0.8 * len(X))
X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]

# ---------------- MODEL ----------------
model = Sequential()
model.add(LSTM(50, input_shape=(X.shape[1], 1)))
model.add(Dense(1))

model.compile(optimizer='adam', loss='mse')

print("\n🚀 Training LSTM model...")
model.fit(X_train, y_train, epochs=10, batch_size=32, verbose=0)
print("✅ Training completed")

# ---------------- PREDICTION ----------------
pred = model.predict(X_test)
pred = scaler.inverse_transform(pred)
y_test = scaler.inverse_transform(y_test)

mae = mean_absolute_error(y_test, pred)

# ---------------- FUTURE FORECAST ----------------
last_seq = data_scaled[-10:]
future = []

for _ in range(5):  # next 5 days
    pred_input = last_seq.reshape(1, 10, 1)
    next_val = model.predict(pred_input, verbose=0)
    future.append(next_val[0][0])
    last_seq = np.append(last_seq[1:], next_val)

future = scaler.inverse_transform(np.array(future).reshape(-1,1))

# ---------------- TERMINAL OUTPUT ----------------
print("\n📊 Sample Predictions:")
for i, val in enumerate(pred[:5]):
    print(f"Day {i+1}: {int(val[0])}")

print(f"\n📉 MAE Error: {round(mae, 2)}")

print("\n🔮 Future Predictions:")
for i, val in enumerate(future[:5]):
    print(f"Day {i+1}: {int(val[0])}")

print("\n" + "="*50)
print("✅ Process Completed Successfully")
print("="*50)