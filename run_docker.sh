#!/bin/bash

# GUI enabled
sudo docker run --privileged --gpus all --net=host -e DISPLAY=$DISPLAY --name carla_sim --rm carlasim/carla:0.9.14 /bin/bash ./CarlaUE4.sh

# Headless mode
# sudo docker run --privileged --gpus all --net=host --name carla_sim --rm carlasim/carla:0.9.14 /bin/bash ./CarlaUE4.sh -opengl -RenderOffscreen