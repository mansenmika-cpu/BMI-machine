import pandas as pd
import csv
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

# 1. TEST DATA (35 readings with 2 intentional anomalies)
with open('BMI_data.csv', mode='r') as file:
    reader = csv.reader(file)
    for row in reader:
        if row != "":
            data = row

BMI_values_list = [float(x) for x in data if x != '']

times = [8, 13, 18, 22]
# Map all values in the list to the time slots
lst = [[times[i % 4], BMI_values_list[i]] for i in range(len(BMI_values_list))]
df = pd.DataFrame(lst, columns=['Hour', 'BMI'])

# 2. ANOMALY DETECTION (Z-Score)
window = 28 # Looking back at the last 7 days (4 readings/day)
df['Rolling_Mean'] = df['BMI'].rolling(window=window).mean()
df['Rolling_Std'] = df['BMI'].rolling(window=window).std()
df['Z_Score'] = (df['BMI'] - df['Rolling_Mean']) / df['Rolling_Std']
df['Is_Anomaly'] = df['Z_Score'].abs() > 2.0

# 3. PREDICTIVE MODELING
df['Target_BMI'] = df['BMI'].shift(-1)
model_df = df.dropna(subset=['Target_BMI', 'BMI']).copy()

X = model_df[['Hour', 'BMI']]
y = model_df['Target_BMI']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# 4. RESULTS & VISUALIZATION
anomalies = df[df['Is_Anomaly'] == True]
print(f"Detected {len(anomalies)} biological anomalies.")

plt.figure(figsize=(12, 6))
plt.plot(df.index, df['BMI'], label='BMI Readings', color='blue', marker='o', markersize=4)
plt.scatter(anomalies.index, anomalies['BMI'], color='red', label='Anomalies', s=100, edgecolors='black')

plt.title('BMI Readings with Anomaly Detection')
plt.xlabel('Reading Number')
plt.ylabel('BMI Value')
plt.legend()
plt.grid(True, linestyle='--', alpha=0.6)
plt.show()