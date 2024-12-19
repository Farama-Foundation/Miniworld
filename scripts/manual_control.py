#!/usr/bin/env python3

"""
This script allows you to manually control the simulator
using the keyboard arrows.
"""

import argparse

import gymnasium as gym

import miniworld
from miniworld.manual_control import ManualControl


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--env-name", default="MiniWorld-Hallway-v0")
    parser.add_argument(
        "--domain-rand", action="store_true", help="enable domain randomization"
    )
    parser.add_argument(
        "--no-time-limit", action="store_true", help="ignore time step limits"
    )
    parser.add_argument(
        "--top_view",
        action="store_true",
        help="show the top view instead of the agent view",
    )
    args = parser.parse_args()
    view_mode = "top" if args.top_view else "agent"

    env = gym.make(args.env_name, view=view_mode, render_mode="human")
    miniworld_version = miniworld.__version__

    print(f"Miniworld v{miniworld_version}, Env: {args.env_name}")

    manual_control = ManualControl(env, args.no_time_limit, args.domain_rand)
    manual_control.run()

    if isinstance(obs, dict):
        if "mission" in obs:
            print(f"Mission: {obs['mission']}")


if __name__ == "__main__":
    main()
