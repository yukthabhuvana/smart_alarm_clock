import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random

# Generate 100 rows of sample sleep data
def generate_sleep_data(n=100):
    data = []
    for _ in range(n):
        day = random.randint(1, 28)
        bed_hour = random.randint(21, 23)
        bed_min = random.randint(0, 59)
        bed_time = datetime(2025, 4, day, bed_hour, bed_min)

        sleep_duration = round(random.uniform(6.0, 9.0), 2)
        wake_time = bed_time + timedelta(hours=sleep_duration)

        rem = round(random.uniform(15.0, 25.0), 2)
        nrem = round(random.uniform(60.0, 75.0), 2)
        quality = random.randint(2, 5)

        data.append([bed_time, wake_time, quality, sleep_duration, rem, nrem])

    columns = ['bed_time', 'wake_time', 'sleep_quality', 'duration', 'REM_percent', 'NREM_percent']
    return pd.DataFrame(data, columns=columns)

# Create and save the dataset
df = generate_sleep_data()
df.to_csv("sleep_data.csv", index=False)

print("✅ sleep_data.csv file created successfully!")
df.head()
data = pd.read_csv("sleep_data.csv")

# Convert to datetime
data['bed_time'] = pd.to_datetime(data['bed_time'])
data['wake_time'] = pd.to_datetime(data['wake_time'])

# Continue with preprocessing..
# Calculate actual sleep duration from timestamps (in hours)
data['sleep_duration'] = (data['wake_time'] - data['bed_time']).dt.total_seconds() / 3600

# Extract hour (in decimal) for bed and wake time
data['bed_hour'] = data['bed_time'].dt.hour + data['bed_time'].dt.minute / 60
data['wake_hour'] = data['wake_time'].dt.hour + data['wake_time'].dt.minute / 60

# Final features and label
features = data[['bed_hour', 'sleep_duration', 'REM_percent', 'NREM_percent', 'sleep_quality']]
labels = data['wake_hour']

features.head()
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

# Split the data
X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42)

# Train the model
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Evaluate
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)


print(f"📉 Root Mean Squared Error: {rmse:.2f}")
# Simulate a user input
new_user_input = pd.DataFrame({
    'bed_hour': [22.0],         # 10:00 PM
    'sleep_duration': [7.5],    # 7.5 hours
    'REM_percent': [20.0],
    'NREM_percent': [65.0],
    'sleep_quality': [4]
})

# Predict wake time
predicted_time = model.predict(new_user_input)[0]

# Convert decimal hour to HH:MM format
wake_hr = int(predicted_time)
wake_min = int((predicted_time - wake_hr) * 60)
print(f"⏰ Your predicted optimal wake-up time: {wake_hr:02d}:{wake_min:02d}")
import ipywidgets as widgets
from IPython.display import display

alarm_widget = widgets.HTML(
    value=f"<h3 style='color:green;'>✅ Smart Alarm Set for: {wake_hr:02d}:{wake_min:02d}</h3>"
)
display(alarm_widget)
