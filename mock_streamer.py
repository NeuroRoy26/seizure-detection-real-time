import numpy as np
import time
import random

# Define the number of channels and the sampling rate
num_channels = 23
sampling_rate = 1000  # Hz

# Define the duration of the simulation
duration = 60 * 60  # seconds

# Define the frequency of the sine waves
frequency = 10  # Hz

# Define the amplitude of the sine waves
amplitude = 1

# Define the duration of the mock seizure
seizure_duration = 10  # seconds

# Initialize the time array
time_array = np.arange(0, duration, 1 / sampling_rate)

# Initialize the data array
data = np.zeros((len(time_array), num_channels))

# Initialize the seizure counter
seizure_counter = 0

# Simulate the EEG data
for i in range(len(time_array)):
    # Generate random sine waves for each channel
    for j in range(num_channels):
        data[i, j] = amplitude * np.sin(2 * np.pi * frequency * time_array[i] + random.uniform(0, 2 * np.pi))
    
    # Inject a mock seizure every few minutes
    if time_array[i] > seizure_counter * 180 and time_array[i] < seizure_counter * 180 + seizure_duration:
        for j in range(num_channels):
            data[i, j] = 10 * amplitude * np.sin(2 * np.pi * frequency * time_array[i] + random.uniform(0, 2 * np.pi))
    elif time_array[i] > seizure_counter * 180 + seizure_duration:
        seizure_counter += 1
    
    # Print the data to the terminal
    print(data[i, :])

    # Sleep for a short duration to simulate real-time data
    time.sleep(1 / sampling_rate)
