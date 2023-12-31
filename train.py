import os
import wandb
import psutil
import platform
import warnings
from tqdm import tqdm

import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader

from models.ncp.NCP import NCP_CfC
from models.perception.Conv import ConvHead
from models.perception.DinoV2 import DinoV2

from utils.utils import load_config, visualize_sequence
from data_utils.carla_dataloader import CarlaDataset

# Ignore specific warnings
warnings.filterwarnings("ignore", message="xFormers is available")

# Pytorch optimizations
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.benchmark = True

def train(perception_model, ncp_model, criterion, optimizer, trainloader, num_epochs, checkpoint_path, pretrained, wandb_enable, device):
    if not pretrained:
        perception_model.train()

    ncp_model.train()
    running_loss = 0.0

    for epoch in range(num_epochs):
        pbar = tqdm(total=len(trainloader), desc=f"Epoch {epoch+1}/{num_epochs}")

        for i, (inputs, labels) in enumerate(trainloader):
            # Convert inputs and labels to float32 and move to GPU
            inputs, labels = inputs.float().cuda(non_blocking=True) , labels.float().cuda(non_blocking=True) 

            # visualize_sequence(inputs.shape[0], inputs.shape[1], inputs)

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

            if wandb_enable:
                # Log trian loss and learning rate after each batch
                wandb.log({
                    "train loss": loss.item(),
                    "epoch": epoch
                }, commit=False)  # Delay logging until all items are ready

            # Update progress bar
            pbar.set_description(f"Epoch {epoch+1} - Loss: {running_loss / (i + 1):.6f}")
            pbar.update(1)

        if wandb_enable:
            # Log average loss after each epoch
            average_loss = running_loss / len(trainloader)
            wandb.log({
                "average loss": average_loss,
                "epoch": epoch
            })

        if (epoch + 1) % 10 == 0:
            # Save combined model state (end-to-end perception+NCP)
            combined_state = {
                'perception_model': perception_model.state_dict(),
                'ncp_model': ncp_model.state_dict()
            }

            torch.save(combined_state, checkpoint_path)
            print(f"Model checkpoint saved")

        pbar.close()
        running_loss = 0.0

    # Save combined model state (end-to-end perception+NCP)
    combined_state = {
        'perception_model': perception_model.state_dict(),
        'ncp_model': ncp_model.state_dict()
    }

    torch.save(combined_state, checkpoint_path)
    print(f"Model checkpoint saved")

if __name__ == "__main__":
    config_path = 'config.json'
    config = load_config(config_path)['train']

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Initialize wandb
    if config['wandb']:
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

    # Create dataloader
    dataset = CarlaDataset(img_folder, controls_folder, config['seq_len'], backbone=config['vision_backbone'])
    dataloader = DataLoader(dataset, 
                            batch_size=config['batch_size'], 
                            shuffle=True, 
                            num_workers=4, 
                            pin_memory=True, 
                            prefetch_factor=2, 
                            persistent_workers=True)
    
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
        print(f"Learning rate (vision backbone): {config['lr_ncp']/10}")
    print(f"Batch size: {config['batch_size']}")
    print(f"Sequence length: {config['seq_len']}")
    print()
    print(f"Dataset Details".center(50, "="))
    print(f"Dataset: {config['dataset_name']}")
    print(f"Size: {dataset.size()} frames")
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

    # Instantiate NCP model
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
            {'params': perception_model.parameters(), 'lr': config['lr_ncp']/10},
            {'params': ncp_model.parameters(), 'lr': config['lr_ncp']}
        ]

    loss_fn = nn.MSELoss()
    optimizer = optim.Adam(optimizer_params)

    train(perception_model, ncp_model, loss_fn, optimizer, dataloader, config['epochs'], checkpoint_path, pretrained, config['wandb'], device)