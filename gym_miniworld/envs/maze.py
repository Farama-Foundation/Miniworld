import math
from collections import deque, defaultdict
import numpy as np
from gym import spaces

from gym_miniworld.entity import Box
from gym_miniworld.miniworld import MiniWorldEnv
from gym_miniworld.params import DEFAULT_PARAMS


class Maze(MiniWorldEnv):
    """
    Maze environment in which the agent has to reach a red box
    """

    def __init__(
        self, num_rows=8, num_cols=8, room_size=3, max_episode_steps=None, **kwargs
    ):
        self.num_rows = num_rows
        self.num_cols = num_cols
        self.room_size = room_size
        self.gap_size = 0.25

        super().__init__(
            max_episode_steps=max_episode_steps or num_rows * num_cols * 24, **kwargs
        )

        # Allow only the movement actions
        self.action_space = spaces.Discrete(self.actions.move_forward + 1)

    def _gen_world(self):
        rows = []

        # For each row
        for j in range(self.num_rows):
            row = []

            # For each column
            for i in range(self.num_cols):

                min_x = i * (self.room_size + self.gap_size)
                max_x = min_x + self.room_size

                min_z = j * (self.room_size + self.gap_size)
                max_z = min_z + self.room_size

                room = self.add_rect_room(
                    min_x=min_x,
                    max_x=max_x,
                    min_z=min_z,
                    max_z=max_z,
                    wall_tex="brick_wall",
                    # floor_tex='asphalt'
                )
                row.append(room)

            rows.append(row)

        visited = set()

        def visit(i, j):
            """
            Recursive backtracking maze construction algorithm
            https://stackoverflow.com/questions/38502
            """

            room = rows[j][i]

            visited.add(room)

            # Reorder the neighbors to visit in a random order
            neighbors = self.rand.subset([(0, 1), (0, -1), (-1, 0), (1, 0)], 4)

            # For each possible neighbor
            for dj, di in neighbors:
                ni = i + di
                nj = j + dj

                if nj < 0 or nj >= self.num_rows:
                    continue
                if ni < 0 or ni >= self.num_cols:
                    continue

                neighbor = rows[nj][ni]

                if neighbor in visited:
                    continue

                if di == 0:
                    self.connect_rooms(
                        room, neighbor, min_x=room.min_x, max_x=room.max_x
                    )
                elif dj == 0:
                    self.connect_rooms(
                        room, neighbor, min_z=room.min_z, max_z=room.max_z
                    )

                visit(ni, nj)

        # Generate the maze starting from the top-left corner
        visit(0, 0)

        self.box = self.place_entity(Box(color="red"))

        self.place_agent()

    def step(self, action):
        obs, reward, done, info = super().step(action)

        if self.near(self.box):
            reward += self._reward()
            done = True

        return obs, reward, done, info


class MazeS2(Maze):
    def __init__(self):
        super().__init__(num_rows=2, num_cols=2)


class MazeS3(Maze):
    def __init__(self):
        super().__init__(num_rows=3, num_cols=3)


class MazeS3Fast(Maze):
    def __init__(self, forward_step=0.7, turn_step=45):

        # Parameters for larger movement steps, fast stepping
        params = DEFAULT_PARAMS.no_random()
        params.set("forward_step", forward_step)
        params.set("turn_step", turn_step)

        max_steps = 300

        super().__init__(
            num_rows=3,
            num_cols=3,
            params=params,
            max_episode_steps=max_steps,
            domain_rand=False,
        )


class MazeS4(Maze):
    def __init__(self):
        super().__init__(num_rows=4, num_cols=4)


class CustomMaze(MiniWorldEnv):
    def __init__(
        self, num_rows=8, num_cols=8, room_size=3, gap_size=0.25, forward_step=0.4, turn_step=22.5,
        center_room=False, fixed_agent_dir=False, random_texture=False, deep_maze=False,
        image_noise_scale=0.0, max_episode_steps=None, **kwargs
    ):
        self.num_rows = num_rows
        self.num_cols = num_cols
        self.room_size = room_size
        self.gap_size = gap_size

        params = DEFAULT_PARAMS.no_random()
        params.set("forward_step", forward_step)
        params.set("turn_step", turn_step)

        self.center_room = center_room
        self.fixed_agent_dir = fixed_agent_dir
        self.random_texture = random_texture
        self.deep_maze = deep_maze
        self.maze_diameter = 0
        self.image_noise_scale = image_noise_scale

        self.wall_textures = [
            'brick_wall', 'drywall', 'wood', 'marble', 'metal_grill', 'picket_fence',
            'water', 'cardboard', 'asphalt', 'airduct_grate', 'cinder_blocks'
        ]
        self.floor_textures = ['floor_tiles_bw', 'asphalt', 'grass']

        super().__init__(
            max_episode_steps=max_episode_steps or num_rows * num_cols * 24,
            params=params, domain_rand=False, **kwargs
        )

        # Allow only the movement actions
        self.action_space = spaces.Discrete(self.actions.move_forward + 1)

    def _gen_world(self):
        rows = []
        graph = defaultdict(list)

        # For each row
        for j in range(self.num_rows):
            row = []

            # For each column
            for i in range(self.num_cols):

                min_x = i * (self.room_size + self.gap_size)
                max_x = min_x + self.room_size

                min_z = j * (self.room_size + self.gap_size)
                max_z = min_z + self.room_size

                room = self.add_rect_room(
                    min_x=min_x,
                    max_x=max_x,
                    min_z=min_z,
                    max_z=max_z,
                    wall_tex=self.rand.choice(self.wall_textures) if self.random_texture else 'brick_wall',
                    floor_tex=self.rand.choice(self.floor_textures) if self.random_texture else 'floor_tiles_bw',
                )
                row.append(room)

            rows.append(row)

        visited = set()

        def visit(i, j):
            """
            Recursive backtracking maze construction algorithm
            https://stackoverflow.com/questions/38502
            """

            room = rows[j][i]

            visited.add(room)

            # Reorder the neighbors to visit in a random order
            neighbors = self.rand.subset([(0, 1), (0, -1), (-1, 0), (1, 0)], 4)

            # For each possible neighbor
            for dj, di in neighbors:
                ni = i + di
                nj = j + dj

                if nj < 0 or nj >= self.num_rows:
                    continue
                if ni < 0 or ni >= self.num_cols:
                    continue

                neighbor = rows[nj][ni]

                if neighbor in visited:
                    continue

                if di == 0:
                    self.connect_rooms(
                        room, neighbor, min_x=room.min_x, max_x=room.max_x
                    )
                elif dj == 0:
                    self.connect_rooms(
                        room, neighbor, min_z=room.min_z, max_z=room.max_z
                    )
                graph[(j, i)].append((nj, ni))
                graph[(nj, ni)].append((j, i))

                visit(ni, nj)

        # Generate the maze starting from the top-left corner
        visit(0, 0)

        box_room, agent_room = None, None

        if self.deep_maze:

            def bfs(start):
                distances = {node: -1 for node in graph}
                distances[start] = 0
                queue = deque([start])

                while queue:
                    node = queue.popleft()
                    for neighbor in graph[node]:
                        if distances[neighbor] == -1:
                            distances[neighbor] = distances[node] + 1
                            queue.append(neighbor)

                max_distance = max(distances.values())
                farthest_node = max(distances, key=distances.get)

                return farthest_node, max_distance

            # Find graph diameter
            # Step 1: Choose an arbitrary start node
            start_node = next(iter(graph))
            # Step 2: Perform BFS from the start node
            node_a, _ = bfs(start_node)
            # Step 3: Perform BFS from node_a
            node_b, self.maze_diameter = bfs(node_a)

            box_room = rows[node_b[0]][node_b[1]]
            agent_room = rows[node_a[0]][node_a[1]]

        # Place the box & agent in center of the corresponding room
        if self.center_room:
            self.box = self.place_entity_at_room_center(Box(color="red"), box_room)
            self.agent = self.place_entity_at_room_center(self.agent, agent_room, dir=0 if self.fixed_agent_dir else None)
        else:
            self.box = self.place_entity(Box(color="red"), box_room)
            self.agent = self.place_agent(agent_room, dir=0 if self.fixed_agent_dir else None)

    def place_entity_at_room_center(self, ent, room, dir=None):
        """
        Place an entity/object at the center of the room.
        """

        assert len(self.rooms) > 0, "create rooms before calling place_entity"
        assert ent.radius is not None, "entity must have physical size defined"
        assert room in self.rooms, "room must be in the maze"

        pos = np.asarray([room.mid_x, 0, room.mid_z])
        ent.pos = pos
        ent.dir = dir if dir is not None else self.rand.float(-math.pi, math.pi)
        self.entities.append(ent)
        return ent

    def step(self, action):
        obs, reward, done, info = super().step(action)

        if self.near(self.box):
            reward += self._reward()
            done = True

        return obs, reward, done, info

    def render_obs(self, frame_buffer=None):
        obs = super().render_obs(frame_buffer)
        if self.image_noise_scale > 0:
            obs_noise = self.np_random.normal(loc=0.0, scale=self.image_noise_scale, size=obs.shape)
            return (obs + obs_noise).astype(obs.dtype)
        else:
            return obs

    def render(self, mode="human", close=False, view="agent"):
        if mode == "rgb_array":
            img_agent = super().render(mode, close, view="agent")
            img_top = super().render(mode, close, view="top")
            return np.concatenate([img_agent, img_top], axis=1)
        else:
            return super().render(mode, close, view)


class CornerMaze(CustomMaze):
    def __init__(self, num_rows=8, num_cols=8, image_noise_scale=0.0, **kwargs):
        super().__init__(num_rows=num_rows, num_cols=num_cols, center_room=True, fixed_agent_dir=True,
                         image_noise_scale=image_noise_scale, **kwargs)


class CornerMazeS3(CornerMaze):
    def __init__(self, image_noise_scale=0.0):
        super().__init__(num_rows=3, num_cols=3, image_noise_scale=image_noise_scale)


class CornerMazeS4(CornerMaze):
    def __init__(self, image_noise_scale=0.0):
        super().__init__(num_rows=4, num_cols=4, image_noise_scale=image_noise_scale)


class CornerMazeS5(CornerMaze):
    def __init__(self, image_noise_scale=0.0):
        super().__init__(num_rows=5, num_cols=5, image_noise_scale=image_noise_scale)


class TextureMaze(CustomMaze):
    def __init__(self, num_rows=8, num_cols=8, image_noise_scale=0.0, **kwargs):
        super().__init__(num_rows=num_rows, num_cols=num_cols, center_room=True, fixed_agent_dir=True,
                         random_texture=True, image_noise_scale=image_noise_scale, **kwargs)


class TextureMazeS3(TextureMaze):
    def __init__(self, image_noise_scale=0.0):
        super().__init__(num_rows=3, num_cols=3, image_noise_scale=image_noise_scale)


class TextureMazeS4(TextureMaze):
    def __init__(self, image_noise_scale=0.0):
        super().__init__(num_rows=4, num_cols=4, image_noise_scale=image_noise_scale)


class TextureMazeS5(TextureMaze):
    def __init__(self, image_noise_scale=0.0):
        super().__init__(num_rows=5, num_cols=5, image_noise_scale=image_noise_scale)


class FullRandomMaze(CustomMaze):
    def __init__(self, num_rows=8, num_cols=8, image_noise_scale=0.0, **kwargs):
        super().__init__(num_rows=num_rows, num_cols=num_cols, center_room=True, fixed_agent_dir=True,
                         random_texture=True, deep_maze=True, image_noise_scale=image_noise_scale, **kwargs)


class FullRandomMazeS3(FullRandomMaze):
    def __init__(self, image_noise_scale=0.0):
        super().__init__(num_rows=3, num_cols=3, image_noise_scale=image_noise_scale)


class FullRandomMazeS4(FullRandomMaze):
    def __init__(self, image_noise_scale=0.0):
        super().__init__(num_rows=4, num_cols=4, image_noise_scale=image_noise_scale)


class FullRandomMazeS5(FullRandomMaze):
    def __init__(self, image_noise_scale=0.0):
        super().__init__(num_rows=5, num_cols=5, image_noise_scale=image_noise_scale)
