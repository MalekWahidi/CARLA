import os
import sys
import json
import queue
import random
from tqdm import tqdm

import cv2
import numpy as np

import carla

# Add the parent directory to sys.path
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)

from utils.utils import load_config


def save_metadata(config, root_path, filename="metadata.json"):
    """
    Saves metadata from the config dict in the dataset root dir

    :param config: Dictionary containing hyperparameters and configuration settings.
    :param dataset_root_dir: The root directory where the dataset is stored.
    :param filename: The name of the file to save the configuration to. Default is 'config_metadata.json'.
    """

    # Define the full path to the configuration file
    file_path = os.path.join(root_path, filename)
    
    # Serialize config dictionary to JSON and write it to the file
    with open(file_path, 'w') as f:
        json.dump(config, f, indent=4)  # Use `indent` for pretty-printing


def process_img(img, cam_w, cam_h):
    # Convert raw image to int8 numpy array
    img = np.frombuffer(img.raw_data, dtype=np.uint8)

    # Reshape into a 2D RGBA image
    img = img.reshape((int(cam_h), int(cam_w), 4))

    # Visualize
    cv2.imshow("Camera View", img)
    cv2.waitKey(1)

    # Remove alpha channel and normalize
    img = img[:, :, :3]

    return img

def save_image(counter, img, img_folder, config):
    # Preprocess raw image data for model training
    img = process_img(img, config["cam_w"], config["cam_h"])
    img_file = os.path.join(img_folder, f"{counter:05d}.png")
    cv2.imwrite(img_file, img)

def apply_perturbations(steer, throttle, brake, perturbation_chance=0.01, max_steer_perturb=0.5, max_throttle_perturb=0.1, max_brake_perturb=0.0):
    """
    Apply random perturbations to steer, throttle, and brake with a given chance.
    
    :param steer: Original steering value
    :param throttle: Original throttle value
    :param brake: Original brake value
    :param perturbation_chance: Chance to apply perturbation
    :param max_steer_perturb: Maximum perturbation that can be applied to steering
    :param max_throttle_perturb: Maximum perturbation that can be applied to throttle
    :param max_brake_perturb: Maximum perturbation that can be applied to brake

    :return: Tuple of (perturbed_steer, perturbed_throttle, perturbed_brake)
    """
    if random.random() < perturbation_chance:
        perturbed_steer = steer + random.uniform(-max_steer_perturb, max_steer_perturb)
        # perturbed_throttle = throttle + random.uniform(-max_throttle_perturb, max_throttle_perturb)
        # perturbed_brake = brake + random.uniform(-max_brake_perturb, max_brake_perturb)
        
        # Ensure values remain within acceptable ranges
        perturbed_steer = np.clip(perturbed_steer, -1.0, 1.0)
        # perturbed_throttle = np.clip(perturbed_throttle, 0.0, 1.0)
        # perturbed_brake = np.clip(perturbed_brake, 0.0, 1.0)
        
        return perturbed_steer, throttle, brake
    else:
        return steer, throttle, brake

def run_simulation():
    try:
        config = load_config('config.json')['data_collection']

        actors = [] # Keep track of all simulated actors for cleanup
        control_data = [] # Store all outputs controls for training dataset

        # Initialize CARLA client
        client = carla.Client('localhost', 2000)
        client.set_timeout(5.0)

        # Initialize the world, blueprint, map, and traffic manager objects
        world = client.get_world()
        bp = world.get_blueprint_library()
        m = world.get_map()
        traffic_manager = client.get_trafficmanager()

        # Toggle all buildings off
        world.unload_map_layer(carla.MapLayer.Buildings)
        world.unload_map_layer(carla.MapLayer.ParkedVehicles)
        world.unload_map_layer(carla.MapLayer.Props)
        world.unload_map_layer(carla.MapLayer.StreetLights)
        world.unload_map_layer(carla.MapLayer.Foliage)
        world.unload_map_layer(carla.MapLayer.Foliage)

        # Set everything to sync mode
        settings = world.get_settings()
        settings.synchronous_mode = True
        settings.fixed_delta_seconds = config["time_step"]
        world.apply_settings(settings)
        traffic_manager.set_synchronous_mode(True)

        # Make traffic manager and weather conditions reprodiucible
        traffic_manager.set_random_device_seed(0)
        weather = carla.WeatherParameters(
            cloudiness=0.0,
            precipitation=0.0,
            sun_altitude_angle=90.0)
        world.set_weather(weather)

        # Store all spawn points available for this map
        spawn_points = m.get_spawn_points()

        # Spawn NPC vehicles at random locations and set to autopilot
        print(f"Spawning {config['num_cars']} NPC vehicles...")
        for _ in range(config["num_cars"]):
            npc_bp = random.choice(bp.filter('vehicle.*'))
            spawn_point = random.choice(spawn_points)
            npc = world.try_spawn_actor(npc_bp, spawn_point)
            
            if npc:
                npc.set_autopilot(True)
                actors.append(npc)
                
        # print(f"Spawning {NUM_PED} NPC pedestrians...")
        # # Spawn AI-controlled NPC pedestrians at random locations
        # for _ in range(NUM_PED):
        #     ped_bp = random.choice(bp.filter('*walker.pedestrian*'))

        #     ped_spawn_point = carla.Transform()
        #     ped_spawn_point.location = world.get_random_location_from_navigation()

        #     if ped_spawn_point.location:
        #         ped = world.try_spawn_actor(ped_bp, ped_spawn_point)

        #     if ped:
        #         actors.append(ped)

        #         # Initialize pedestrian AI controller
        #         controller_bp = bp.find('controller.ai.walker')
        #         controller = world.try_spawn_actor(controller_bp, ped.get_transform(), ped)
        #         actors.append(controller)

        #         # Set random target location for NPC pedestrian
        #         controller.start()
        #         controller.go_to_location(world.get_random_location_from_navigation())
        #         controller.set_max_speed(1 + random.random())

        # Spawn main agents vehicle and set to autopilot
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

        actors.append(ego_car)
        ego_car.set_autopilot(True)

        # Set the ego car to ignore all traffic lights
        traffic_manager.ignore_lights_percentage(ego_car, 100)
        
        # Set the ego car to always go straight at intersections
        traffic_manager.set_route(ego_car, ['Straight']*100)
        
        print("Spawning ego camera...")
        cam_bp = bp.find('sensor.camera.rgb')
        cam_bp.set_attribute('image_size_x', config["cam_w"])
        cam_bp.set_attribute('image_size_y', config["cam_h"])
        cam_bp.set_attribute('fov', config["cam_fov"])
        print("Camera attributes set...")

        spawn_point = carla.Transform(carla.Location(x=config["cam_x"], y=config["cam_y"], z=config["cam_z"]))
        print("Camera spawn point set...")

        ego_cam = world.try_spawn_actor(cam_bp, spawn_point, attach_to=ego_car)
        actors.append(ego_cam)
        print("Camera spawned...")

        image_queue = queue.Queue()
        ego_cam.listen(image_queue.put)

        # Define path to store rgb frames and control data
        datasets_parent = os.path.join(current_dir, '..', 'datasets')
        dataset_path = os.path.join(datasets_parent, config["dataset_name"])
        rgb_path = os.path.join(dataset_path, "rgb")
        controls_path = os.path.join(dataset_path, "controls")
        
        # Create the data dirs if they don't exist yet
        os.makedirs(rgb_path, exist_ok=True)
        os.makedirs(controls_path, exist_ok=True)

        for i in tqdm(range(config["max_steps"]), desc="Collecting Data", unit="frame"):
            world.tick()
            
            # Collect output control data for training supervision
            controls = ego_car.get_control()

            # Apply perturbations with a 5% chance
            perturbed_controls = apply_perturbations(controls.steer, controls.throttle, controls.brake)

            # Apply the perturbed controls to the car
            ego_car.apply_control(carla.VehicleControl(steer=perturbed_controls[0], throttle=perturbed_controls[1], brake=perturbed_controls[2]))

            outputs = [perturbed_controls[0], perturbed_controls[1], perturbed_controls[2]]

            control_data.append(outputs)
            
            # Process image frame and save as PNG
            save_image(i, image_queue.get(), rgb_path, config)

        # Save all control data to a single npy file at the end
        np.save(f"{controls_path}/all_controls.npy", control_data)

        save_metadata(config, dataset_path)

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
    # Set random seeds for reproducibility
    random.seed(0)
    np.random.seed(0)

    run_simulation()
