import streamlit as st
import requests
import time
import numpy as np
import matplotlib.pyplot as plt

# Define the API endpoint URL
url = "http://127.0.0.1:8000/latest"

# Define the polling rate
polling_rate = 0.1  # seconds

# Define the rolling buffer size
buffer_size = 10  # seconds

# Initialize the rolling buffer
buffer = []

# Initialize the connection status
connection_status = "Disconnected"

# Initialize the last update time
last_update_time = None

# Create a Streamlit app
st.title("EEG Dashboard")

# Create a start/stop button
start_button = st.button("Start")
stop_button = st.button("Stop")

# Create a polling loop
if start_button:
    while True:
        # Send a GET request to the API endpoint
        response = requests.get(url)

        # Check if the response is successful
        if response.status_code == 200:
            # Parse the response JSON
            data = response.json()

            # Check if the data is available
            if "data" in data:
                # Append the data to the rolling buffer
                buffer.append(data["data"])

                # Limit the buffer size
                if len(buffer) > buffer_size * int(1 / polling_rate):
                    buffer.pop(0)

                # Update the connection status
                connection_status = "Connected"

                # Update the last update time
                last_update_time = time.time()

                # Plot the EEG data
                st.subheader("EEG Data")
                fig, ax = plt.subplots()
                for i in range(23):
                    ax.plot([x[i] for x in buffer], label=f"Channel {i+1}")
                ax.legend()
                st.pyplot(fig)

                # Plot the seizure probability
                st.subheader("Seizure Probability")
                fig, ax = plt.subplots()
                ax.plot([x["seizure_probability"] for x in buffer])
                st.pyplot(fig)

                # Display the connection status and last update time
                st.subheader("Connection Status")
                st.write(connection_status)
                st.write(f"Last Update Time: {last_update_time}")

            else:
                # Display an error message
                st.error("No data available")

        else:
            # Display an error message
            st.error("Failed to connect to the API endpoint")

        # Sleep for the polling rate
        time.sleep(polling_rate)

        # Check if the stop button is clicked
        if stop_button:
            break
