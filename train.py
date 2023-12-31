import os
import wandb
import psutil
import platform
import warnings
from tqdm import tqdm
import multiprocessing

import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader

from models.ncp.NCP import NCP_CfC
from models.perception.Conv import ConvHead
from models.perception.DinoV2 import DinoV2

from utils.utils import load_config
from data_utils.carla_dataloader import CarlaDataset

import matplotlib.pyplot as plt

# Ignore specific warnings
warnings.filterwarnings("ignore", message="xFormers is available")

# Pytorch optimizations
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.benchmark = True

def train(perception_model, ncp_model, criterion, optimizer, trainloader, num_epochs, checkpoint_path, pretrained, device):
    if not pretrained:
        perception_model.train()

    ncp_model.train()
    running_loss = 0.0

    wandb.watch(ncp_model, criterion, log='all', log_freq=10)
    if not pretrained:
        wandb.watch(perception_model, criterion, log='all', log_freq=10)

    for epoch in range(num_epochs):
        pbar = tqdm(total=len(trainloader), desc=f"Epoch {epoch+1}/{num_epochs}")

        for i, (inputs, labels) in enumerate(trainloader):
            ## Visualize image sequences in input batch
            # selected_images = [0, 2, 6, 9]  # Indices for images 1, 3, 7, 10

            # # Set up a figure with subplots
            # _, axs = plt.subplots(config['batch_size'], len(selected_images), figsize=(15, 10))

            # # Titles for the columns
            # column_titles = ['Image 1', 'Image 3', 'Image 7', 'Image 10']

            # # Set column titles
            # for ax, col in zip(axs[0], column_titles):
            #     ax.set_title(col)

            # # Plot each image and set row labels (sequence numbers)
            # for i in range(batch_size):
            #     # Set the row title once per row
            #     axs[i, 0].set_ylabel(f'Seq {i + 1}', rotation=0, size='large', labelpad=20)
                
            #     for j, img_idx in enumerate(selected_images):
            #         # Plot each selected image in the subplot
            #         axs[i, j].imshow(inputs[i][img_idx].permute(1, 2, 0).cpu().numpy())
            #         axs[i, j].axis('off')  # Turn off the axis

            # plt.tight_layout()
            # plt.show()

            # Convert inputs and labels to float32 and move to GPU
            inputs, labels = inputs.float().cuda(non_blocking=True) , labels.float().cuda(non_blocking=True) 

            # Reset gradients and hidden state for next iteration
            optimizer.zero_grad(set_to_none=True)

            # Inference
            features = perception_model(inputs)
            outputs, _ = ncp_model(features)

            # Backprop
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

            # Log trian loss and learning rate after each batch
            wandb.log({
                "train/loss": loss.item(),
                "train/lr": optimizer.param_groups[0]['lr'],
                "epoch": epoch
            }, commit=False)  # Delay logging until all items are ready

            # Update progress bar
            pbar.set_description(f"Epoch {epoch+1} - Loss: {running_loss / (i + 1):.6f}")
            pbar.update(1)

        # Log average loss after each epoch
        average_loss = running_loss / len(trainloader)
        wandb.log({
            "epoch/average_loss": average_loss,
            "epoch": epoch
        })

        pbar.close()
        running_loss = 0.0

    # Finish the wandb run after training is complete
    wandb.finish()

    # Save combined model state (end-to-end perception+NCP)
    combined_state = {
        'perception_model': perception_model.state_dict(),
        'ncp_model': ncp_model.state_dict()
    }

    torch.save(combined_state, checkpoint_path)
    print(f"Model checkpoint saved")
    
def get_dataloader_params():
    num_cpu_cores = multiprocessing.cpu_count()
    
    # Select num_workers based on heuristic (approx num_cpu_cores)
    num_workers = max(1, num_cpu_cores - 2)
    prefetch_factor = 2
    
    return num_workers, prefetch_factor


if __name__ == "__main__":
    config_path = 'config.json'
    config = load_config(config_path)['train']

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Initialize wandb
    wandb.init(project="CARLA", name="NCP-DinoV2")

    # Model checkpoint save path
    checkpoint_folder = config['checkpoint_folder']
    checkpoint_name = config['checkpoint_name']
    checkpoint_path = os.path.join(checkpoint_folder, checkpoint_name)

    # Select the perception model
    if config['vision_backbone'] == 'cnn':
        perception_model = ConvHead(n_features=config['ncp_inputs']).to(device)
        pretrained = False
    elif config['vision_backbone'] == 'dinov2':
        perception_model = DinoV2()
        pretrained = True
    else:
        print(f"Perception model {config['vision_backbone']} not found!")

    # Dataset paths
    datasets_path = config['datasets_path']
    dataset_name = config['dataset_name']
    img_folder = os.path.join(datasets_path, dataset_name, 'rgb')
    controls_folder = os.path.join(datasets_path, dataset_name, 'controls')

    # Get dataloader parameters
    num_workers, prefetch_factor = get_dataloader_params()

    # Create dataset and dataloader
    dataset = CarlaDataset(img_folder, controls_folder, config['seq_len'], backbone=config['vision_backbone'])
    dataloader = DataLoader(dataset, batch_size=config['batch_size'], shuffle=True, num_workers=num_workers, pin_memory=True, prefetch_factor=prefetch_factor, persistent_workers=True)
    
    # Display all training details
    print(f"Training".center(50, "="))
    print()
    print(f"Model Architecture".center(50, "="))
    print(f"Vision Backbone: {config['vision_backbone']}")
    print(f"Pretrained: {pretrained}")
    print(f"NCP: [{config['ncp_inputs']}, {config['ncp_neurons']}, {config['ncp_outputs']}]")
    print()
    print(f"Training Hyperparameters".center(50, "="))
    print(f"Epochs: {config['epochs']}")
    print(f"Learning rate (NCP): {config['lr_ncp']}")
    if not pretrained:
        print(f"Learning rate (vision backbone): {config['lr_vision']}")
    print(f"Batch size: {config['batch_size']}")
    print(f"Sequence length: {config['seq_len']}")
    print()
    print(f"Hardware Specs".center(50, "="))
    if device.type == 'cuda':
        print(f"Device: GPU")
        device_id = torch.cuda.current_device()
        print(f"Name: {torch.cuda.get_device_name(device_id)}")
        print(f"Memory: {torch.cuda.get_device_properties(device_id).total_memory / (1024 ** 3):.2f} GB")
    else:
        print(f"Device: CPU")
        print(f"Name: {platform.uname().processor}")
        print(f"Number of cores: {psutil.cpu_count(logical=False)}")
    print()
    print(f"Dataset Details".center(50, "="))
    print(f"Dataset: {config['dataset_name']}")
    print(f"Size: {dataset.size()} frames")
    print(f"Num Workers: {num_workers}")
    print(f"Prefetch Factor: {prefetch_factor}")
    print()

    # Instantiate the CfC RNN model with appropriate dimensions
    ncp_model = NCP_CfC(config['ncp_inputs'], config['ncp_neurons'], config['ncp_outputs']).to(device)
    
    # Pretrained perception vs end-to-end training
    if pretrained:
        # Freeze perception model parameters
        for param in perception_model.parameters():
            param.requires_grad = False

        # Define optimizer parameters (only NCP parameters are optimized)
        optimizer_params = [
            {'params': ncp_model.parameters(), 'lr': config['lr_ncp']}
        ]
    else:
        # Train both models end-to-end with different learning rates
        optimizer_params = [
            {'params': perception_model.parameters(), 'lr': config['lr_vision']},
            {'params': ncp_model.parameters(), 'lr': config['lr_ncp']}
        ]

    loss_fn = nn.MSELoss()
    optimizer = optim.Adam(optimizer_params)

    # Optimize models and train loop if possible
    # train_opt = torch.compile(train)
    # perception_model = torch.compile(perception_model)
    # ncp_model = torch.compile(ncp_model)

    train(perception_model, ncp_model, loss_fn, optimizer, dataloader, config['epochs'], checkpoint_path, pretrained, device)