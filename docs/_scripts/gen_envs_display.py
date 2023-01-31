import os

import gymnasium as gym

import miniworld


def create_grid_cell(env_id):
    env_name = env_id.split("-")[-2]
    return f"""
            <a href="/environments/{env_name}">
                <div class="env-grid__cell">
                    <div class="cell__image-container">
                        <img src="/_static/environments/{env_name.lower()}.jpg">
                    </div>
                    <div class="cell__title">
                        <span>{env_name}</span>
                    </div>
                </div>
            </a>
    """


def generate_page(env_list):
    cells = [create_grid_cell(env_id) for env_id in env_list]
    cells = "\n".join(cells)

    return f"""
<div class="env-grid">
    {cells}
</div>
    """


if __name__ == "__main__":
    miniworld_env_ids = sorted(list(miniworld.envs.env_ids))
    filtered_env_ids = []
    previous_env_name = None

    for env_id in miniworld_env_ids:
        print(env_id)
        env_spec = gym.spec(env_id)

        # MiniWorld-Hallway -> Hallway
        env_name = env_spec.name.split("-")[1]
        # We are not adding sub envs like YMazeLeft
        if previous_env_name is None or previous_env_name not in env_name:
            previous_env_name = env_name
            filtered_env_ids.append(env_id)

    file_path = os.path.join(
        os.path.dirname(os.path.dirname(__file__)),
        "content",
        "env_list.html",
    )

    res = generate_page(filtered_env_ids)

    file = open(file_path, "w+", encoding="utf-8")
    file.write(res)
    file.close()
