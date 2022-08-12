# A Dockerfile that sets up a full Gym install with test dependencies
ARG PYTHON_VERSION
FROM python:$PYTHON_VERSION

SHELL ["/bin/bash", "-o", "pipefail", "-c"]

RUN apt-get -y update
RUN apt-get install -y freeglut3-dev

COPY . /usr/local/gym_miniworld/
WORKDIR /usr/local/gym_miniworld/

RUN pip install --no-cache-dir
RUN pip install torch --no-cache-dir

RUN cd pytorch-a2c-ppo-acktr
RUN xvfb-run -a -s "-screen 0 1024x768x24 -ac +extension GLX +render -noreset" time python3 main.py --no-cuda --algo a2c --log-interval 1 --num-frames 200 --num-processes 1 --num-steps 80 --lr 0.00005 --env-name MiniWorld-Hallway-v0
