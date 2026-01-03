import pandas as pd
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
import joblib
import matplotlib.pyplot as plt

print("--- FINAL MODEL TRAINING SCRIPT (Ridge Regression) ---")

# --- Step 1: Load Real-Scenario Dataset ---
print("Step 1: Loading 'hvac_refrigerant_leak_dataset.csv'...")
try:
    df = pd.read_csv(r'E:\Minor Project\hvac_refrigerant_leak_dataset.csv')
except FileNotFoundError:
    print("\nERROR: 'hvac_refrigerant_leak_dataset.csv' not found.")
    exit()

# Simulate humidity data if not present
if 'Hum_ambient' not in df.columns:
    print("Simulating Humidity data as 'Hum_ambient'...")
    humidity = 50.0 + (df['T_ambient'] - df['T_ambient'].mean()) * 0.5
    humidity_noise = np.random.normal(0, 2.0, len(df))
    df['Hum_ambient'] = humidity + humidity_noise
    df['Hum_ambient'] = df['Hum_ambient'].clip(30, 90)

print("Dataset loaded and prepared.")
print("-" * 30)

# --- Step 2: Define Features and Target ---
features = ['T2_discharge', 'T_ambient', 'Hum_ambient']
target = 'T1_suction'

print(f"Features (X): {features}")
print(f"Target (y): {target}")
print("-" * 30)

# --- Step 3: Prepare Data with Scaling ---
print("Step 3: Preparing and scaling data...")

# Filter healthy data only
df_healthy = df[df['leak_status'] == 0].copy()
X_train = df_healthy[features]
y_train = df_healthy[target]

# Create and fit scaler
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)

print(f"Training samples: {len(X_train)}")
print(f"Feature means: {scaler.mean_}")
print(f"Feature stds: {scaler.scale_}")
print("-" * 30)

# --- Step 4: Model Training ---
print("Step 4: Training Ridge Regression model...")

# Train Ridge model (alpha=1.0 provides good regularization)
model = Ridge(alpha=1.0)
model.fit(X_train_scaled, y_train)

print("Model training complete.")
print(f"Model R^2 score on healthy training data: {model.score(X_train_scaled, y_train):.4f}")
print(f"\nModel Coefficients: {model.coef_}")
print(f"Model Intercept: {model.intercept_:.4f}")
print("-" * 30)

# --- Step 5: Test Model on Leak Data ---
print("Step 5: Testing the model on LEAK data...")

df_faulty = df[df['leak_status'] == 1].copy()
X_test_faulty = df_faulty[features]
y_test_faulty_actual = df_faulty[target]

# Scale test data
X_test_faulty_scaled = scaler.transform(X_test_faulty)

# Get predictions
y_test_faulty_predicted = model.predict(X_test_faulty_scaled)
y_train_predicted = model.predict(X_train_scaled)

# Calculate errors
faulty_error = np.abs(y_test_faulty_actual - y_test_faulty_predicted)
healthy_error = np.abs(y_train - y_train_predicted)

print(f"Average Prediction Error on Healthy Data: {healthy_error.mean():.4f} °C")
print(f"Average Prediction Error on Leak Data:    {faulty_error.mean():.4f} °C")
print(f"Max Error on Healthy Data: {healthy_error.max():.4f} °C")
print(f"Max Error on Leak Data: {faulty_error.max():.4f} °C")

# Suggest threshold
suggested_threshold = healthy_error.mean() + 2 * healthy_error.std()
print(f"\nSuggested Anomaly Threshold: {suggested_threshold:.2f} °C")
print("-" * 30)

# --- Step 6: Save Model and Scaler ---
print("Step 6: Saving the trained model and scaler...")

# Save both model and scaler
joblib.dump(model, r'E:\Minor Project\refrigerant_model_FINAL.pkl')
joblib.dump(scaler, r'E:\Minor Project\scaler_FINAL.pkl')

print("✓ Model saved to: refrigerant_model_FINAL.pkl")
print("✓ Scaler saved to: scaler_FINAL.pkl")
print("-" * 30)

# --- Step 7: Visualization ---
print("Step 7: Visualizing anomaly scores...")

plt.figure(figsize=(15, 6))
plt.subplot(1, 2, 1)
plt.hist(healthy_error, bins=30, alpha=0.7, label='Healthy', color='green')
plt.hist(faulty_error, bins=30, alpha=0.7, label='Leak', color='red')
plt.axvline(suggested_threshold, color='black', linestyle='--', label=f'Threshold={suggested_threshold:.2f}')
plt.xlabel('Prediction Error (°C)')
plt.ylabel('Frequency')
plt.title('Error Distribution')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(df_healthy.index, healthy_error, label='Healthy Error', alpha=0.6, color='green')
plt.plot(df_faulty.index, faulty_error, label='Leak Error', alpha=0.6, color='red')
plt.axhline(suggested_threshold, color='black', linestyle='--', label=f'Threshold={suggested_threshold:.2f}')
plt.xlabel('Sample Index')
plt.ylabel('Prediction Error (°C)')
plt.title('Prediction Error Over Time')
plt.legend()

plt.tight_layout()
plt.show()

print("\n" + "="*60)
print("TRAINING COMPLETE!")
print("="*60)
print("\nNext Steps:")
print("1. Run the converter script to generate the Arduino header")
print("2. Update your Arduino code with the new threshold")
print("3. Upload to ESP32 and test!")
print("="*60)