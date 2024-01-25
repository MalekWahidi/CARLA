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
    print(f"Spectator Location: x={location.x}, y={location.y}, z={location.z}", end='\r', flush=True)

    time.sleep(0.1)