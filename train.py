import os
import wandb
import psutil
import platform
import warnings
from tqdm import tqdm

import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader, ConcatDataset

from models.control.NCP import NCP
from models.control.cond_NCP import Cond_NCP
from models.control.LSTM import LSTM_Net
from models.control.cond_LSTM import Cond_LSTM
from models.control.MLP import MLP_Net

from models.perception.VC1 import VC1
from models.perception.conv import ConvHead
from models.perception.DinoV2 import DinoV2
from models.perception.resnet50 import ResNet50

from utils import utils
from data_utils.carla_dataloader import CarlaData
from data_utils.conditional_dataloader import ConditionalCarlaData

# Ignore specific warnings
warnings.filterwarnings("ignore", message="xFormers is available")
warnings.filterwarnings('ignore', category=UserWarning, module='torchvision.transforms.functional')

# Pytorch settings for optimization and reproducibility
torch.manual_seed(0)
torch.cuda.manual_seed(0)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True
torch.backends.cuda.matmul.allow_tf32 = True


# Used as worker_init_fn for deterministic data loading
def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32

def train(perception_model, control_model, optimizer, trainloader, num_epochs, checkpoint_path, pretrained, wandb_enable, resume):
    if not pretrained:
        perception_model.train()

    control_model.train()

    if wandb_enable:
        wandb.watch(perception_model, log='gradients', log_freq=100)
        wandb.watch(control_model, log='gradients')

    cumulative_loss = 0.0
    start_epoch = 0
    train_step = 0

    # Resume previous training run from saved checkpoint
    if resume and os.path.exists(checkpoint_path):
        print("Resuming from saved checkpoint...")
        checkpoint = torch.load(checkpoint_path)
        perception_model.load_state_dict(checkpoint['perception_model'])
        control_model.load_state_dict(checkpoint['ncp_model'])
        start_epoch = checkpoint['epoch'] + 1  # Resume from the next epoch
        train_step = checkpoint['train_step'] + 1 # Resume from the next training step
        optimizer.load_state_dict(checkpoint['optimizer'])

    for epoch in range(start_epoch, num_epochs):
        pbar = tqdm(total=len(trainloader), desc=f"Epoch {epoch+1}/{num_epochs}")

        for i, (images, controls) in enumerate(trainloader):

            # Convert tensors to float32 and move to GPU
            images = images.float().cuda(non_blocking=True)
            controls = controls.float().cuda(non_blocking=True)

            # visualize_sequence(images.shape[0], images.shape[1], images)

            # Reset gradients for next iteration
            optimizer.zero_grad(set_to_none=True)

            # Inference
            features = perception_model(images)
            outputs, _ = control_model(features)

            # Weight loss by steering angle
            loss = utils.standard_weighted_mae_loss(outputs, controls)
            # loss = nn.functional.mse_loss(outputs, controls)

            # Backprop
            loss.backward()
            optimizer.step()

            cumulative_loss += loss.item()
            running_average_loss = cumulative_loss / (i + 1)  # Calculate running average loss (cumulative/n_batches)

            if wandb_enable:
                # Log batch loss with training step
                wandb.log({"batch loss": loss.item(), "train_step": train_step, "epoch": epoch}, commit=True)
                train_step += 1  # Increment training step counter

            # Update progress bar
            pbar.set_description(f"Epoch {epoch+1} - Running Avg Loss: {running_average_loss:.4f}")
            pbar.update(1)

        if wandb_enable:
            # Log average loss after each epoch
            average_loss = cumulative_loss / len(trainloader)
            wandb.log({
                "epoch loss": average_loss,
                "epoch": epoch
            })

        if (epoch + 1) % 5 == 0 or epoch + 1 == num_epochs:
            # Save combined model state (end-to-end perception+NCP) with additional information
            combined_state = {
                'perception_model': perception_model.state_dict(),
                'ncp_model': control_model.state_dict(),
                'epoch': epoch,  # Save the current epoch
                'optimizer': optimizer.state_dict()  # Save optimizer state
            }

            torch.save(combined_state, checkpoint_path)
            print(f"Checkpoint saved at epoch {epoch + 1}")

        pbar.close()
        cumulative_loss = 0.0 # Reset cumulative loss for next epoch

    # Log final train_step value to wandb for reference in case of resumed training later on
    if wandb_enable:
        wandb.config.train_step = train_step
    
    print(f"Training completed!")


if __name__ == "__main__":
    config_path = 'config.json'
    config = utils.load_config(config_path)['train']

    # Get root dir for setting relative paths of datasets and checkpoints
    root_dir = os.path.dirname(os.path.abspath(__file__))

    # Set device to GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Initialize wandb
    if config['wandb']:
        wandb.init(project="CARLA", name=config["checkpoint_name"].split(".")[0])

        # Log hyperparams
        wandb.config.dataset = config['dataset_name']
        wandb.config.epochs = config['epochs']
        wandb.config.lr_ncp = config['lr_ncp']
        wandb.config.batch_size = config['batch_size']
        wandb.config.seq_len = config['seq_len']
        wandb.config.ncp_type = config['control_cells']
        wandb.config.ncp_inputs = config['control_inputs']
        wandb.config.ncp_hidden = config['control_neurons']
        wandb.config.ncp_outputs = config['control_outputs']
        wandb.config.conditional = config['conditional']

    # Model checkpoint save path
    checkpoint_name = config['checkpoint_name']
    weights_path = os.path.join(root_dir, "weights/e2e")
    # Create the weights dir if it don't exist yet
    os.makedirs(weights_path, exist_ok=True)
    checkpoint_path = os.path.join(weights_path, checkpoint_name + ".pth")
    
    # Select the perception model based on config settings
    if config['vision_backbone'] == 'cnn':
        pretrained = False
        perception_model = ConvHead(n_features=config['control_inputs']).to(device)
    elif config['vision_backbone'] == 'resnet':
        pretrained = False
        perception_model = ResNet50(n_features=config['control_inputs']).to(device)
    elif config['vision_backbone'] == 'dinov2':
        pretrained = True
        config["control_inputs"] = 384
        perception_model = DinoV2().to(device)
    elif config['vision_backbone'] == 'vc1':
        pretrained = True
        config["control_inputs"] = 768
        perception_model = VC1().to(device)
    else:
        raise ValueError(f"Perception model '{config['vision_backbone']}' not found!")

    # Select the control model based on config settings
    if config['control_head'] == 'ncp' and not config['conditional']:
        control_model = NCP(config['control_inputs'], config['control_neurons'], config['control_outputs'], cell_type=config['control_cells']).to(device)
    elif config['control_head'] == 'ncp' and config['conditional']:
        control_model = Cond_NCP(config['control_inputs'], config['num_commands'], config['control_neurons'], config['control_outputs']).to(device)
    
    elif config['control_head'] == 'lstm' and not config['conditional']:
        control_model = LSTM_Net(config['control_inputs'], config['control_neurons'], config['control_outputs']).to(device)
    elif config['control_head'] == 'lstm' and config['conditional']:
        control_model = Cond_LSTM(config['control_inputs'], config['num_commands'], config['control_neurons'], config['control_outputs']).to(device)

    elif config['control_head'] == 'mlp':
        control_model = MLP_Net(config['control_inputs'], config['control_neurons'], config['control_outputs']).to(device)
    else:
        print(f"Control model '{config['control_head']}' not found!")

    # Dataset paths
    dataset_name = config['dataset_name']
    dataset_path = os.path.join(root_dir, "datasets", dataset_name)
    img_folder = os.path.join(dataset_path, 'rgb')
    controls_folder = os.path.join(dataset_path, 'controls')

    # Create dataloader
    if not config['conditional']:
        dataset = CarlaData(img_folder, 
                            controls_folder, 
                            config['seq_len'], 
                            backbone=config['vision_backbone'], 
                            n_outputs=config['control_outputs'],
                            start_idx=config['start_idx'],
                            end_idx=config['end_idx'])
    else:
        print("Conditional data loader not setup yet!")

    # Generator for deterministic shuffling
    g = torch.Generator()
    g.manual_seed(0)

    dataloader = DataLoader(dataset,
                            batch_size=config['batch_size'], 
                            shuffle=True, 
                            num_workers=4, 
                            pin_memory=True,
                            persistent_workers=True,
                            worker_init_fn=seed_worker,
                            generator=g)
    
    # Display all training details
    print(f"Training".center(50, "="))
    print()
    print(f"Model Architecture".center(50, "="))
    print(f"Vision Backbone: {config['vision_backbone']}")
    print(f"Pretrained: {pretrained}")
    print(f"Control Model: {config['control_head']} [{config['control_inputs']}, {config['control_neurons']}, {config['control_outputs']}]")
    print(f"Conditional: {config['conditional']}")
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

    pretrained=False # Added to try finetuning PVRs
    
    # Pretrained perception vs end-to-end training
    if pretrained:
        # Freeze perception model parameters
        for param in perception_model.parameters():
            param.requires_grad = False

        # Define optimizer parameters (only NCP parameters are optimized)
        optimizer_params = [
            {'params': control_model.parameters(), 'lr': config['lr_ncp']}
        ]
    else:
        # Previous work on NCPs has shown that training end-to-end with perception
        # works better with lower learning rates for the vision backbone than the NCP
        lr_ncp = config['lr_ncp']
        lr_vision = lr_ncp/100
        
        if config['wandb']:
            wandb.config.lr_vision = lr_vision

        # Train both models end-to-end with different learning rates
        optimizer_params = [
            {'params': perception_model.parameters(), 'lr': lr_vision},
            {'params': control_model.parameters(), 'lr': lr_ncp}
        ]

    optimizer = optim.Adam(optimizer_params)

    train(perception_model, control_model, optimizer, dataloader, config['epochs'], checkpoint_path, pretrained, config['wandb'], config['resume'])
    
    wandb.finish()