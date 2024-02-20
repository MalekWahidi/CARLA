import os
import sys
import json

import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

import torch
from torchvision import transforms
from torchvision.utils import save_image

from captum.attr import Saliency
from captum.attr import visualization as viz

# Add the parent directory to the path to import the models and utils
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from models.perception.resnet50 import ResNet50  
from models.control.NCP import NCP
from utils import utils  


def load_model(config_path, checkpoint_path, device):
    # Load the configuration
    config = utils.load_config(config_path)['train']

    # Initialize the perception model
    if config['vision_backbone'] == 'resnet':
        perception_model = ResNet50(n_features=config['control_inputs']).to(device)
    else:
        raise NotImplementedError(f"Model '{config['vision_backbone']}' not supported")

    # Initialize the control model
    if config['control_head'] == 'ncp':
        control_model = NCP(config['control_inputs'], config['control_neurons'], config['control_outputs'], cell_type=config['control_cells']).to(device)
    else:
        raise NotImplementedError(f"Control model '{config['control_head']}' not supported")

    # Load the trained model weights
    checkpoint = torch.load(checkpoint_path, map_location=device)
    perception_model.load_state_dict(checkpoint['perception_model'])
    control_model.load_state_dict(checkpoint['ncp_model'])

    return perception_model, control_model

def generate_saliency(perception_model, input_image):
    perception_model.eval()
    input_image.requires_grad = True

    # Print to check the shape; assuming shape [1, 1, 32] based on your output
    print("Features shape before squeezing:", perception_model(input_image).shape)

    # Define the target feature index
    features = perception_model(input_image)
    
    # Adjust the lambda function to correctly handle the output shape
    # Assuming the desired feature is within a tensor of shape [1, 1, 32], 
    # you need to squeeze out the singleton dimensions before indexing
    saliency = Saliency(lambda img: perception_model(img).squeeze())
    
    # Generate saliency map with respect to the target class index
    # The target is now an integer index, assuming the squeezed output is a 1D tensor
    saliency_map = saliency.attribute(input_image, target=features.squeeze().squeeze())
    saliency_map = saliency_map.detach().cpu().numpy()

    return saliency_map

def visualize_saliency(saliency_map, original_image, save_path=None):
    # Convert the saliency map and original image to suitable format for visualization
    saliency_map = np.transpose(saliency_map[0], (1, 2, 0))
    original_image_np = np.transpose(original_image.squeeze().cpu().numpy(), (1, 2, 0))

    # Plotting
    fig, ax = plt.subplots(1, 2)
    ax[0].imshow(original_image_np)
    ax[0].axis('off')
    ax[1].imshow(np.mean(saliency_map, axis=2), cmap='hot')
    ax[1].axis('off')
    
    if save_path:
        plt.savefig(save_path)
    plt.show()

if __name__ == "__main__":
    # Configuration and checkpoint
    config_path = 'config.json'
    checkpoint_path = 'weights/e2e/ResNet-wMAE-straight-minimal.pth'
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load model
    perception_model, ncp_model = load_model(config_path, checkpoint_path, device)

    # Load an example input image
    img_path = 'datasets/town01_straight/rgb/00000.png'
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])
    img = transform(Image.open(img_path)).unsqueeze(0).unsqueeze(0).to(device)
    
    # Generate and visualize saliency map
    saliency_map = generate_saliency(perception_model, img)
    visualize_saliency(saliency_map, img, save_path='saliency_map.png')
