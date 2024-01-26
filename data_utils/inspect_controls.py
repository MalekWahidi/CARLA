import numpy as np

def load_and_inspect(file_path):
    # Load the numpy file
    controls_data = np.load(file_path)

    # Display some basic information
    print("Shape of the Data:", controls_data.shape)
    print("First few entries:\n", controls_data[:5])

    # Basic statistics
    print("\nStatistics:")
    print("Mean Steer:", np.mean(controls_data[:, 0]))
    print("Mean Throttle:", np.mean(controls_data[:, 1]))
    print("Mean Brake:", np.mean(controls_data[:, 2]))

    # Checking for any anomalies
    print("\nAnomalies Check:")
    if np.any(controls_data[:, 0] > 1) or np.any(controls_data[:, 0] < -1):
        print("Anomaly detected in Steer values")
    if np.any(controls_data[:, 1] > 1) or np.any(controls_data[:, 1] < 0):
        print("Anomaly detected in Throttle values")
    if np.any(controls_data[:, 2] > 1) or np.any(controls_data[:, 2] < 0):
        print("Anomaly detected in Brake values")
    else:
        print("No anomalies detected")

# Path to your 'all_controls.npy' file
controls_folder = "/home/malek/Documents/CARLA/datasets/town01_straight/controls"
file_path = f"{controls_folder}/all_controls.npy"

# Run the inspection
load_and_inspect(file_path)
