import numpy as np
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt

# Define the ground truth function for gauge block thickness vs temperature
def ground_truth_function(temperature):
    thickness = 3.0 - 0.0002 * (temperature - 23) / 0.4
    return thickness


# Define the temperature function with noise
def temperature_function(time):
    frequency = 2 * np.pi / 30  # Period of 30 minutes
    amplitude = 1  # Temperature variance
    noise = np.random.uniform(-1, 1, size=time.shape)  # Uniform noise between -1 and 1
    temperature = 23 + amplitude * np.sin(frequency * time) + noise
    return temperature


# Generate training data with temperature and corresponding gauge block thickness
num_samples = 1000
time_range = np.linspace(0, 300, num_samples)  # Time range in minutes
temperature_range = temperature_function(time_range)
thickness_data = ground_truth_function(temperature_range)

# Define the PINN model
model = keras.Sequential([
    keras.layers.Dense(20, activation='tanh', input_shape=(1,)),
    keras.layers.Dense(20, activation='tanh'),
    keras.layers.Dense(1)
])


# Define the loss function for the PINN model
def loss_fn(model, temperature, thickness):
    predicted_thickness = model(temperature)
    loss = tf.reduce_mean(tf.square(predicted_thickness - thickness))
    return loss


# Training
optimizer = tf.keras.optimizers.Adam()
losses = []

for i in range(10000):
    with tf.GradientTape() as tape:
        time_batch = np.random.choice(time_range, size=32)
        temperature_batch = temperature_function(time_batch)
        thickness_batch = ground_truth_function(temperature_batch)
        loss = loss_fn(model, thickness_batch, temperature_batch)

    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))

# Prediction
time_test = np.linspace(0, 300, num_samples)
temperature_test = temperature_function(time_test)
thickness_predicted = model.predict(temperature_test)

# Plotting
fig, ax1 = plt.subplots(figsize=(8, 6))

color1 = 'tab:red'
ax1.set_xlabel('Time (minutes)')
ax1.set_ylabel('Gauge Block Thickness (mm)', color=color1)
ax1.plot(time_range, thickness_data, color=color1, label='Ground Truth')
ax1.plot(time_test, thickness_predicted, color='orange', linestyle='--', label='Predicted')
ax1.set_ylim(2.95, 3.05)  # Y-axis range for thickness
ax1.tick_params(axis='y', labelcolor=color1)
ax1.grid(True)

ax2 = ax1.twinx()  # Create a second y-axis sharing the same x-axis

color2 = 'tab:blue'
ax2.set_ylabel('Temperature (°C)', color=color2)
ax2.plot(time_range, temperature_range, color=color2, label='Temperature')
ax2.tick_params(axis='y', labelcolor=color2)

fig.tight_layout()
plt.title('Gauge Block Thickness and Temperature over Time')

# Plot loss function
fig, ax = plt.subplots(figsize=(8, 6))
ax.plot(losses)
ax.set_xlabel('Iteration')
ax.set_ylabel('Loss')
ax.set_title('Loss Function')
ax.grid(True)
plt.legend()
plt.show()