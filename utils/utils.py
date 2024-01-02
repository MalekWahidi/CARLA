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

# Exponential weighted MSE loss to weight high steering/braking scenarios more heavily
def exponential_weighted_mse_loss(outputs, labels, alpha_steering=2, alpha_braking=2, alpha_throttle=1):
    loss_fn = nn.L1Loss()
    
    mse_loss = loss_fn(outputs, labels)

    # Calculate the weighting factors for steering and braking
    weights_steering = torch.exp(torch.abs(labels[:, 0]) ** alpha_steering)
    weights_braking = torch.exp(labels[:, 2] ** alpha_braking)

    # Apply weights to steering and braking losses
    steering_loss = mse_loss[:, 0] * weights_steering
    braking_loss = mse_loss[:, 2] * weights_braking

    # Throttle loss
    throttle_loss = mse_loss[:, 1] * alpha_throttle

    # Combine the losses
    combined_loss = steering_loss + braking_loss + throttle_loss

    # Return the mean of the combined loss
    return combined_loss.mean()
