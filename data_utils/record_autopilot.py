import os
import queue
import random
from tqdm import tqdm

import cv2
import numpy as np

import carla

from utils.utils import load_config


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

    return img

def save_image(counter, img, img_folder, config):
    # Preprocess raw image data for model training
    img = process_img(img, config["cam_w"], config["cam_h"])
    img_file = os.path.join(img_folder, f"{counter:05d}.png")
    cv2.imwrite(img_file, img)

def run_simulation():
    try:
        config = load_config('config.json')['data_collection']

        actors = [] # Keep track of all simulated actors for cleanup
        control_data = [] # Store all outputs controls for training dataset

        # Initialize CARLA client
        client = carla.Client('localhost', 2000)
        client.set_timeout(2.0)

        # Initialize the world, blueprint, map, and traffic manager objects
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

        # Define the location to spawn the ego car (replace with your coordinates)
        ego_spawn_point = carla.Transform(carla.Location(x=-70, y=-60, z=2))
        
        # Keep trying to spawn agent until successful
        while True:
            ego_car = world.try_spawn_actor(ego_bp, ego_spawn_point)
            if ego_car:
                break

        actors.append(ego_car)
        ego_car.set_autopilot(True)
            
        print("Spawning ego camera...")
        cam_bp = bp.find('sensor.camera.rgb')
        cam_bp.set_attribute('image_size_x', config["cam_w"])
        cam_bp.set_attribute('image_size_y', config["cam_h"])
        cam_bp.set_attribute('fov', config["cam_fov"])

        spawn_point = carla.Transform(carla.Location(x=config["cam_x"], y=config["cam_y"], z=config["cam_z"]))
        ego_cam = world.try_spawn_actor(cam_bp, spawn_point, attach_to=ego_car)
        actors.append(ego_cam)

        image_queue = queue.Queue()
        ego_cam.listen(image_queue.put)

        # Define path to store rgb frames
        rgb_path = os.path.join(config["datasets_path"], config["dataset_name"], "rgb")

        for i in tqdm(range(config["max_steps"]), desc="Collecting Data", unit="frame"):
            world.tick()
            
            # Collect output control data for training supervision
            controls = ego_car.get_control()
            outputs = [controls.steer, controls.throttle, controls.brake]

            # Check if the agent is steering to cut the recording loop
            if controls.steer != 0:
                break

            control_data.append(outputs)
            
            # Process image frame and save as PNG
            save_image(i, image_queue.get(), rgb_path, config)


        controls_path = os.path.join(config["datasets_path"], config["dataset_name"], "controls")
        
        # Save all control data to a single npy file at the end
        np.save(f"{controls_path}/all_controls.npy", control_data)

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
    run_simulation()
