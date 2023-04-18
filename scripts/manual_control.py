#!/usr/bin/env python3

"""
This script allows you to manually control the simulator
using the keyboard arrows.
"""

import argparse
import math

import gymnasium as gym
import pyglet
from pyglet.window import key

import miniworld


class ManualControl:
    def __init__(self, env, no_time_limit, domain_rand):
        self.env = env

        if no_time_limit:
            self.env.max_episode_steps = math.inf
        if domain_rand:
            self.env.domain_rand = True

    def run(self):
        print("============")
        print("Instructions")
        print("============")
        print("move: arrow keys\npickup: P\ndrop: D\ndone: ENTER\nquit: ESC")
        print("============")

        self.env.reset()

        # Create the display window
        self.env.render()

        env = self.env

        @env.unwrapped.window.event
        def on_key_press(symbol, modifiers):
            """
            This handler processes keyboard commands that
            control the simulation
            """

            if symbol == key.BACKSPACE or symbol == key.SLASH:
                print("RESET")
                self.env.reset()
                self.env.render()
                return

            if symbol == key.ESCAPE:
                self.env.close()

            if symbol == key.UP:
                self.step(self.env.actions.move_forward)
            elif symbol == key.DOWN:
                self.step(self.env.actions.move_back)
            elif symbol == key.LEFT:
                self.step(self.env.actions.turn_left)
            elif symbol == key.RIGHT:
                self.step(self.env.actions.turn_right)
            elif symbol == key.PAGEUP or symbol == key.P:
                self.step(self.env.actions.pickup)
            elif symbol == key.PAGEDOWN or symbol == key.D:
                self.step(self.env.actions.drop)
            elif symbol == key.ENTER:
                self.step(self.env.actions.done)

        @env.unwrapped.window.event
        def on_key_release(symbol, modifiers):
            pass

        @env.unwrapped.window.event
        def on_draw():
            self.env.render()

        @env.unwrapped.window.event
        def on_close():
            pyglet.app.exit()

        # Enter main event loop
        pyglet.app.run()

        self.env.close()

    def step(self, action):
        print(
            "step {}/{}: {}".format(
                self.env.step_count + 1,
                self.env.max_episode_steps,
                self.env.actions(action).name,
            )
        )

        obs, reward, termination, truncation, info = self.env.step(action)

        if reward > 0:
            print(f"reward={reward:.2f}")

        if termination or truncation:
            print("done!")
            self.env.reset()

        self.env.render()


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


if __name__ == "__main__":
    main()
