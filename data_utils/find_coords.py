import time
import carla

# Connect to the CARLA server
client = carla.Client('localhost', 2000)
client.set_timeout(10.0)
world = client.get_world()

# Get the spectator object
spectator = world.get_spectator()

while True:
    # Get the location of the spectator
    location = spectator.get_location()
    print(f"Spectator Coordinates: x={location.x:.2f}, y={location.y:.2f}, z={location.z:.2f}", end='\r', flush=True)

    time.sleep(0.1)