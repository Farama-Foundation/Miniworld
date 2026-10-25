"""Check reset rendering with real MiniWorld environments and OpenGL frames."""

from unittest.mock import Mock

import gymnasium as gym
import numpy as np
import pytest

import miniworld  # noqa: F401


@pytest.mark.parametrize("env_id", ["MiniWorld-OneRoom-v0", "MiniWorld-Hallway-v0"])
@pytest.mark.parametrize("view", ["agent", "top"])
@pytest.mark.parametrize("render_mode", [None, "rgb_array", "human"])
def test_reset_renders_only_in_human_mode(env_id, view, render_mode, monkeypatch):
    env = gym.make(env_id, render_mode=render_mode, view=view)
    base = env.unwrapped
    try:
        # Construction also resets the environment and should display its first frame.
        assert (base.window is not None) == (render_mode == "human")
        render = Mock(wraps=base.render)
        monkeypatch.setattr(base, "render", render)

        first_obs, first_info = env.reset(seed=0)
        assert base.observation_space.contains(first_obs)
        assert first_info == {}
        assert render.call_count == int(render_mode == "human")

        env.step(base.actions.turn_left)
        render.reset_mock()
        second_obs, second_info = env.reset(seed=0)
        assert base.step_count == 0
        assert render.call_count == int(render_mode == "human")
        np.testing.assert_array_equal(second_obs, first_obs)
        assert second_info == first_info

        if render_mode == "rgb_array":
            frame = env.render()
            assert frame.ndim == 3 and frame.shape[-1] == 3
            assert frame.dtype == np.uint8
            assert base.window is None
    finally:
        env.close()
