import os
import sys
import numpy as np

# Add the parent directory to sys.path
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)

from utils.utils import load_config

def load_and_inspect(file_path):
    # Load the numpy file
    controls_data = np.load(file_path)

    # Display some basic info
    print("Shape of the Data:", controls_data.shape)
    print("First few entries:\n", controls_data[350:500])
    print("\nStats:")
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

if __name__ == "__main__":
    config = load_config('config.json')['data_collection']

    # Get path to 'all_controls.npy' file
    datasets_path = os.path.join(current_dir, '..', 'datasets')
    controls_path = os.path.join(datasets_path, config["dataset_name"], "controls/all_controls.npy")

    # Run the inspection
    load_and_inspect(controls_path)
