import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch.nn.functional import conv_transpose2d
from torchvision import transforms
from torchvision.models import resnet50
from torchvision.models.resnet import Bottleneck

from utils import utils
from models.control.NCP import NCP
from models.perception.resnet50 import ResNet50  


def load_model(config_path, checkpoint_path, device):
    # Load the configuration
    config = utils.load_config(config_path)['train']

    # Initialize the perception model
    if config['vision_backbone'] == 'resnet':
        perception_model = ResNet50(n_features=1000).to(device)
    else:
        raise NotImplementedError(f"Model '{config['vision_backbone']}' not supported")

    # Initialize the control model
    if config['control_head'] == 'ncp':
        control_model = NCP(config['control_inputs'], config['control_neurons'], config['control_outputs'], cell_type=config['control_cells']).to(device)
    else:
        raise NotImplementedError(f"Control model '{config['control_head']}' not supported")

    # Load the trained model weights
    # checkpoint = torch.load(checkpoint_path, map_location=device)
    # perception_model.load_state_dict(checkpoint['perception_model'])
    # control_model.load_state_dict(checkpoint['ncp_model'])

    return perception_model.eval().to(device), control_model.eval().to(device)


class ResnetVisualizer(nn.Module):
    def __init__(self, resnet):
        super(ResnetVisualizer, self).__init__()
        self.model = resnet.resnet50

        for name, child in self.model.named_children():
            if 'layer' in name:
                setattr(self, name, LayerVisualizer(child))
        
        # For Deconv
        self.k7x7 = torch.ones((1,1,7,7)).to('cuda')
        self.k3x3 = torch.ones((1,1,3,3)).to('cuda')

    def forward(self, x):
        x = self.model.conv1(x)
        x = self.model.bn1(x)
        x = self.model.relu(x)
        act1 = x.mean(1, keepdim=True)
        x = self.model.maxpool(x)

        x, vis1 = self.layer1(x)
        x, vis2 = self.layer2(x)
        x, vis3 = self.layer3(x)
        x, vis4 = self.layer4(x)

        x = self.model.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.model.fc(x)
        
        vis = list(reversed([act1] + vis1 + vis2 + vis3 + vis4))
        prod = vis[0]
        for i in range(1, len(vis)):
            act = vis[i]
            if prod.shape != act.shape:
                prod = conv_transpose2d(
                        prod, self.k3x3, 
                        stride=2, padding=1, 
                        output_padding=1)
            #print(f'{i} prod:{tuple(prod.shape)}, act:{tuple(act.shape)}')
            prod *= act

        # Resize to input image
        prod = conv_transpose2d(prod, self.k7x7, stride=2, padding=3, output_padding=1)
        
        return x, prod #* input.mean(1, keepdim=True)

class LayerVisualizer(nn.Module):
    def __init__(self, layer):
        super(LayerVisualizer, self).__init__()
        self.layer = layer

        for name, child in self.layer.named_children():
            setattr(self, name, BottleneckVisualizer(child))

    def forward(self, x):

        vis=[]
        for name, child in self.layer.named_children():
            block = getattr(self, name)
            x, prod = block(x)
            vis += prod

        """
        vis = []   # Activations
        for block in self.layer.children():
            x = block(x)
            vis.append(x.mean(1,keepdim=True))   # Average channels
            print(f'LayerVis: {tuple(vis[-1].shape)}')
        """
        return x, vis

class BottleneckVisualizer(nn.Module):
    def __init__(self, block):
        super(BottleneckVisualizer, self).__init__()
        self.block = block
        self.k3x3 = torch.ones((1,1,3,3))
        self.k1x1 = torch.ones((1,1,1,1))

    def forward(self, x):
        vis = []

        residual = x

        out = self.block.conv1(x)
        out = self.block.bn1(out)
        out = self.block.relu(out)

        vis += [out.mean(1,keepdim=True)]


        out = self.block.conv2(out)
        out = self.block.bn2(out)
        out = self.block.relu(out)

        vis += [out.mean(1,keepdim=True)]


        out = self.block.conv3(out)
        out = self.block.bn3(out)

        if self.block.downsample is not None:
            residual = self.block.downsample(x)

        out += residual
        out = self.block.relu(out)
        vis += [out.mean(1,keepdim=True)]

        return out, [vis[-1]] #vis#[out.mean(1,keepdim=True)]


if __name__ == "__main__":
    # Configuration and checkpoint
    config_path = 'config.json'
    checkpoint_path = 'weights/e2e/ResNet-wMAE-lanekeep-minimal.pth'
    device = torch.device("cuda")

    # Load model
    perception_model, ncp_model = load_model(config_path, checkpoint_path, device)
    model_vis = ResnetVisualizer(perception_model.eval()).to(device)

    for i in range(300, 400):
        # Load an example input image
        img_path = f'datasets/town01_10min_lanekeep_minimal/rgb/{i:05d}.png'
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
        ])
        img = transform(Image.open(img_path)).unsqueeze(0).to(device)

        with torch.no_grad():
            _, vis = model_vis(img)

        # Visualization
        vis = vis.squeeze().cpu().numpy()  # Adjust dimensions as needed
        vis = np.interp(vis, (vis.min(), vis.max()), (0, 1))

        img = img.squeeze().cpu().numpy().transpose(1, 2, 0)
        img = np.interp(img, (img.min(), img.max()), (0, 1))

        # Display images
        plt.subplot(1, 2, 1)
        plt.imshow(img)
        plt.title("Input Image")
        plt.axis('off')

        plt.subplot(1, 2, 2)
        plt.imshow(vis, cmap='bwr')
        plt.title("Visualization")
        plt.axis('off')

        plt.draw()
        plt.pause(0.05)  # Pause for a brief moment to create a video effect

    plt.show()