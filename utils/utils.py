import json
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

# Load config json file
def load_config(config_path):
    with open(config_path, 'r') as file:
        return json.load(file)
    
# Visualize image sequences in input batch
def visualize_sequence(batch_size, sequence_length, inputs):
    # Set up a figure with subplots
    fig, axs = plt.subplots(batch_size, sequence_length, figsize=(15, 10))
    axs = axs.reshape(batch_size, sequence_length)  # Ensure axs is 2D array

    # Plot each image and set row labels (sequence numbers)
    for i in range(batch_size):
        for j in range(sequence_length):
            # Check if axs is a 2D array (when batch_size and sequence_length > 1)
            if batch_size > 1 and sequence_length > 1:
                ax = axs[i, j]
            elif batch_size > 1:  # Only batch_size is > 1
                ax = axs[i]
            else:  # Only sequence_length is > 1
                ax = axs[j]

            # Plot each image in the subplot
            ax.imshow(inputs[i][j].permute(1, 2, 0).cpu().numpy())
            ax.axis('off')  # Turn off the axis

            # Set the column title for the first row
            if i == 0:
                ax.set_title(f'Image {j + 1}')

        # Set the row title once per row
        axs[i, 0].set_ylabel(f'Seq {i + 1}', rotation=0, labelpad=20)

    plt.tight_layout()
    plt.show()

# Exponential weighted MSE loss to weight high steering scenarios more heavily
def exponential_weighted_mse_loss(outputs, labels):
    # Calculate MSE loss for all controls
    mse_loss = (outputs - labels) ** 2  # Calculate squared errors for each control dimension

    # Apply exponential weighting only to the steering angle loss
    steering_weights = torch.exp(torch.abs(labels[:, 0]))  # Exponential weights for steering angles
    weighted_steering_loss = mse_loss[:, 0] * steering_weights  # Apply weights only to steering loss

    # Combine the weighted steering loss with unweighted throttle and brake losses
    total_loss = (weighted_steering_loss + mse_loss[:, 1] + mse_loss[:, 2]).mean()  # Average over all samples and controls

    return total_loss

# Exponential weighted MSE loss to weight high steering scenarios more heavily
def standard_weighted_mae_loss(outputs, labels):
    # Calculate MSE loss for all controls
    mae_loss = abs(outputs - labels)  # Calculate squared errors for each control dimension

    # Weighted sum of MAE losses
    total_loss = ((mae_loss[:, 0] * 0.50) + (mae_loss[:, 1] * 0.45) + (mae_loss[:, 2] * 0.05)).mean()  # Average over all samples and controls

    return total_loss
