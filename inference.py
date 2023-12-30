import carla
import queue
import random

import cv2
import numpy as np

import torch
from torchvision import transforms

from models.ncp.NCP import ConvCfC

# Simulation parameters
NUM_CARS = 10
NUM_PED = 0

# Camera parameters
CAM_WIDTH = 640
CAM_HEIGHT = 360

def process_img(img):
    # Convert raw image to int8 numpy array
    img = np.frombuffer(img.raw_data, dtype=np.uint8)

    # Reshape into a 2D RGBA image
    img = img.reshape((CAM_HEIGHT, CAM_WIDTH, 4))

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


def run_closed_loop(model_path):
    try:
        actors = [] # Keep track of all simulated actors for cleanup

        # Initialize CARLA client
        client = carla.Client('localhost', 2000)
        client.set_timeout(2.0)

        # Load trained model
        model = ConvCfC(n_actions=3)
        model.load_state_dict(torch.load(model_path))
        model.eval()
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)
        hx = None
        
        # Initialize the world, blueprint, map, and traffic manager objects
        world = client.get_world()
        bp = world.get_blueprint_library()
        m = world.get_map()
        traffic_manager = client.get_trafficmanager()

        # Set everything to sync mode
        settings = world.get_settings()
        settings.synchronous_mode = True
        settings.fixed_delta_seconds = 0.05 # 20 FPS
        world.apply_settings(settings)
        traffic_manager.set_synchronous_mode(True)

        # Store all spawn points available for this map
        spawn_points = m.get_spawn_points()

        print(f"Spawning {NUM_CARS} NPC vehicles...")
        # Spawn NPC vehicles at random locations and set to autopilot
        for _ in range(NUM_CARS):
            npc_bp = random.choice(bp.filter('vehicle.*'))
            spawn_point = random.choice(spawn_points)
            npc = world.try_spawn_actor(npc_bp, spawn_point)
            
            if npc:
                npc.set_autopilot(True)
                actors.append(npc)

        print("Spawning ego car...")
        # Spawn main agents vehicle and set to autopilot
        ego_bp = bp.filter("model3")[0]
        ego_bp.set_attribute('role_name','ego')
        spawn_point = random.choice(spawn_points)
        
        # Keep trying to spawn agent until successful
        while True:
            ego_car = world.try_spawn_actor(ego_bp, spawn_point)
            if ego_car:
                break
        ego_car.set_autopilot(True)
        actors.append(ego_car)
            
        print("Spawning ego camera...")
        cam_bp = bp.find('sensor.camera.rgb')
        cam_bp.set_attribute('image_size_x', str(CAM_WIDTH))
        cam_bp.set_attribute('image_size_y', str(CAM_HEIGHT))
        cam_bp.set_attribute('fov', '110')

        spawn_point = carla.Transform(carla.Location(x=1.25, y=0, z=1.1))
        ego_cam = world.try_spawn_actor(cam_bp, spawn_point, attach_to=ego_car)
        actors.append(ego_cam)

        image_queue = queue.Queue()
        ego_cam.listen(image_queue.put)

        while True:
            world.tick()

            # Preprocess image for model input and move to GPU
            img = process_img(image_queue.get()).to(device)

            # Model inference
            with torch.no_grad():
                controls, hx = model(img, hx)

            # Remove batch and time dimensions
            controls = controls.squeeze(0).squeeze(0).cpu().numpy()

            steer, throttle, brake = controls
            brake = 0

            print(f"Steer: {steer:.4f} | Throttle: {throttle:.4f} | Brake: {brake:.4f}")

            # Apply model output to control the ego car
            control_command = carla.VehicleControl(steer=float(steer),
                                                   throttle=float(throttle),
                                                   brake=float(brake))
            ego_car.apply_control(control_command)

    except Exception as e:
        print(f"An error occurred: {e}")

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
    model_path = '/home/malek/Documents/CARLA/weights/e2e/ConvCfC.pth'
    run_closed_loop(model_path)
