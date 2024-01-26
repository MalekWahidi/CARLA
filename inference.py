import os
import carla
import queue
import random
import warnings

import cv2
import numpy as np

import torch
from torchvision import transforms

from models.control.NCP import NCP_CfC
from models.perception.Conv import ConvHead
from models.perception.DinoV2 import DinoV2
from models.perception.VC1 import VC1

from utils.utils import load_config

# Ignore specific warnings
warnings.filterwarnings("ignore", message="xFormers is available")


def load_models(checkpoint_path, device, config):
    # Initialize perception model
    if "CNN" in config["checkpoint_name"]:
        perception_model = ConvHead().to(device)
    elif "VC" in config["checkpoint_name"]:
        perception_model = VC1().to(device)
    elif "Dino" in config["checkpoint_name"]:
        perception_model = DinoV2().to(device)

    # Initialize NCP model
    ncp_model = NCP_CfC(config['control_inputs'], config['control_neurons'], config['control_outputs']).to(device)

    # Load the combined checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)

    # Load state dictionaries into the respective models
    perception_model.load_state_dict(checkpoint['perception_model'])
    ncp_model.load_state_dict(checkpoint['ncp_model'])

    # Set models to evaluation mode
    perception_model.eval()
    ncp_model.eval()

    return perception_model, ncp_model


def process_img(img, cam_w, cam_h):
    # Convert raw image to int8 numpy array
    img = np.frombuffer(img.raw_data, dtype=np.uint8)

    # Reshape into a 2D RGBA image
    img = img.reshape((cam_h, cam_w, 4))

    # Visualize
    cv2.imshow("Camera View", img)
    cv2.waitKey(1)

    # Remove alpha channel and normalize
    img = img[:, :, :3]

    # Reshape image and normalize
    img = img.astype(np.float32) / 255.0
    
    # Convert to torch tensor and normalize
    img = transforms.ToTensor()(img)

    # Add batch and time dimensions
    img = img.unsqueeze(0).unsqueeze(0)

    return img


def run_closed_loop():
    try:
        config_path = 'config.json'
        config = load_config(config_path)['inference']

        model_path = os.path.join(config["checkpoint_folder"], config["checkpoint_name"])
        
        actors = [] # Keep track of all simulated actors for cleanup

        # Initialize CARLA client
        client = carla.Client('localhost', 2000)
        client.set_timeout(2.0)

        # Load trained model to GPU
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        perception_model, ncp_model = load_models(model_path, device, config)
        
        # Initialize world, blueprint, map, and traffic manager objects
        world = client.get_world()
        bp = world.get_blueprint_library()
        m = world.get_map()
        traffic_manager = client.get_trafficmanager()

        # Set everything to sync mode
        settings = world.get_settings()
        settings.synchronous_mode = True
        settings.fixed_delta_seconds = config["time_step"]
        world.apply_settings(settings)
        traffic_manager.set_synchronous_mode(True)

        # Store all spawn points available for this map
        spawn_points = m.get_spawn_points()

        print(f"Spawning {config['num_cars']} NPC vehicles...")
        # Spawn NPC vehicles at random locations and set to autopilot
        for _ in range(config["num_cars"]):
            npc_bp = random.choice(bp.filter('vehicle.*'))
            spawn_point = random.choice(spawn_points)
            npc = world.try_spawn_actor(npc_bp, spawn_point)
            
            if npc:
                npc.set_autopilot(True)
                actors.append(npc)

        # Spawn ego car (AI-controlled)
        print("Spawning ego car...")
        ego_bp = bp.filter("model3")[0]
        ego_bp.set_attribute('role_name','ego')
        
        # Define the location to spawn the ego car
        ego_spawn_point = carla.Transform(carla.Location(x=config["spawn_point"][0], 
                                                         y=config["spawn_point"][1], 
                                                         z=config["spawn_point"][2]))
        
        # Keep trying to spawn agent until successful
        while True:
            ego_car = world.try_spawn_actor(ego_bp, ego_spawn_point)
            if ego_car:
                break

        # ego_car.set_autopilot(True)
        actors.append(ego_car)
            
        # Spawn front-facing rgb camera on ego car hood
        print("Setting up ego camera parameters...")
        cam_bp = bp.find('sensor.camera.rgb')
        cam_bp.set_attribute('image_size_x', config["cam_w"])
        cam_bp.set_attribute('image_size_y', config["cam_h"])
        cam_bp.set_attribute('fov', config["cam_fov"])

        print("Spawning ego camera...")
        spawn_point = carla.Transform(carla.Location(x=config["cam_x"], y=config["cam_y"], z=config["cam_z"]))
        ego_cam = world.try_spawn_actor(cam_bp, spawn_point, attach_to=ego_car)
        actors.append(ego_cam)

        # Synchronous mode requires queueing images from the camera
        print("Listening to ego camera...")
        image_queue = queue.Queue()
        ego_cam.listen(image_queue.put)

        # Initialize NCP hidden state
        hx = None

        while True:
            world.tick()
            
            # Preprocess image for model input and move to GPU
            img = process_img(image_queue.get(), int(config["cam_w"]), int(config["cam_h"])).to(device)

            # Model inference
            with torch.no_grad():
                features = perception_model(img)
                controls, hx = ncp_model(features, hx)

            # Remove batch and time dimensions
            controls = controls.squeeze(0).squeeze(0).cpu().numpy()

            # Extract steering, throttle, and brake values
            if config["control_outputs"] == 3:
                steer, throttle, brake = controls
            elif config["control_outputs"] == 1:
                steer, throttle, brake = controls[0], 0.3, 0.0

            print(f"Steer: {steer:.4f} | Throttle: {throttle:.4f} | Brake: {brake:.4f}", end='\r', flush=True)

            # Apply model output to control the ego car
            control_command = carla.VehicleControl(steer=float(steer),
                                                   throttle=float(throttle),
                                                   brake=float(brake))
            ego_car.apply_control(control_command)

    except Exception as e:
        print(f"ERROR: {e}")

    finally:
        print("\nCleaning up...")

        # Reset world settings
        settings = world.get_settings()
        settings.synchronous_mode = False
        settings.no_rendering_mode = False
        settings.fixed_delta_seconds = None
        world.apply_settings(settings)

        # Destroy all actors in the simulation
        client.apply_batch([carla.command.DestroyActor(a) for a in actors])

        print('Done.')


if __name__ == '__main__':
    run_closed_loop()