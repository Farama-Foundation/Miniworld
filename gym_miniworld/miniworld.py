import math
from enum import IntEnum
import numpy as np
import gym
from gym import spaces
from .random import *
from .opengl import *
from .objmesh import *
from .entity import *
from .math import *
from .params import *

# Default wall height for room
DEFAULT_WALL_HEIGHT=2.74

# Texture size/density in texels/meter
TEX_DENSITY = 512

def gen_texcs_wall(
    tex,
    min_x,
    min_y,
    width,
    height
):
    """
    Generate texture coordinates for a wall quad
    """

    xc = (TEX_DENSITY / tex.width)
    yc = (TEX_DENSITY / tex.height)

    min_u = (min_x) * xc
    max_u = (min_x + width) * xc
    min_v = (min_y) * yc
    max_v = (min_y + height) * yc

    return np.array(
        [
            [min_u, min_v],
            [min_u, max_v],
            [max_u, max_v],
            [max_u, min_v],
        ],
        dtype=np.float32
    )

def gen_texcs_floor(
    tex,
    poss
):
    """
    Generate texture coordinates for the floor or ceiling
    This is done by mapping x,z positions directly to texture
    coordinates
    """

    texc_mul = np.array(
        [
            TEX_DENSITY / tex.width,
            TEX_DENSITY / tex.height
        ],
        dtype=float
    )

    coords = np.stack([poss[:,0], poss[:,2]], axis=1) * texc_mul

    return coords

class Room:
    """
    Represent an individual room and its contents
    """

    def __init__(
        self,
        outline,
        wall_height=DEFAULT_WALL_HEIGHT,
        floor_tex='floor_tiles_bw',
        wall_tex='concrete',
        ceil_tex='concrete_tiles',
        no_ceiling=False
    ):
        # The outlien should have shape Nx2
        assert len(outline.shape) == 2
        assert outline.shape[1] == 2
        assert outline.shape[0] >= 3

        # Add a Y coordinate to the outline points
        outline = np.insert(outline, 1, 0, axis=1)

        # Number of outline vertices / walls
        self.num_walls = outline.shape[0]

        # List of 2D points forming the outline of the room
        # Shape is Nx3
        self.outline = outline

        # Compute the min and max x, z extents
        self.min_x = self.outline[:, 0].min()
        self.max_x = self.outline[:, 0].max()
        self.min_z = self.outline[:, 2].min()
        self.max_z = self.outline[:, 2].max()

        # Compute midpoint coordinates
        self.mid_x = (self.max_x + self.min_x) / 2
        self.mid_z = (self.max_z + self.min_z) / 2

        # Compute approximate surface area
        self.area = (self.max_x - self.min_x) * (self.max_z - self.min_z)

        # Compute room edge directions and normals
        # Compute edge vectors (p1 - p0)
        # For the first point, p0 is the last
        # For the last point, p0 is p_n-1
        next_pts = np.concatenate([self.outline[1:], np.expand_dims(self.outline[0], axis=0)], axis=0)
        self.edge_dirs = next_pts - self.outline
        self.edge_dirs = (self.edge_dirs.T / np.linalg.norm(self.edge_dirs, axis=1)).T
        self.edge_norms = -np.cross(self.edge_dirs, Y_VEC)
        self.edge_norms = (self.edge_norms.T / np.linalg.norm(self.edge_norms, axis=1)).T

        # Height of the room walls
        self.wall_height = wall_height

        # No ceiling flag
        self.no_ceiling = no_ceiling

        # Texture names
        self.wall_tex_name = wall_tex
        self.floor_tex_name = floor_tex
        self.ceil_tex_name = ceil_tex

        # Lists of portals, indexed by wall/edge index
        self.portals = [[] for i in range(self.num_walls)]

        # List of neighbor rooms
        # Same length as list of portals
        self.neighbors = []

    def add_portal(
        self,
        edge,
        start_pos=None,
        end_pos=None,
        min_x=None,
        max_x=None,
        min_z=None,
        max_z=None,
        min_y=0,
        max_y=None

    ):
        """
        Create a new portal/opening in a wall of this room
        """

        if max_y == None:
            max_y = self.wall_height

        assert edge <= self.num_walls
        assert max_y > min_y

        # Get the edge points, compute the direction vector
        e_p0 = self.outline[edge]
        e_p1 = self.outline[(edge+1) % self.num_walls]
        e_len = np.linalg.norm(e_p1 - e_p0)
        e_dir = (e_p1 - e_p0) / e_len
        x0, _, z0 = e_p0
        x1, _, z1 = e_p1
        dx, _, dz = e_dir

        # If the portal extents are specified by x coordinates
        if min_x != None:
            assert min_z == None and max_z == None
            assert start_pos == None and end_pos == None
            assert x0 != x1

            m0 = (min_x - x0) / dx
            m1 = (max_x - x0) / dx

            if m1 < m0:
                m0, m1 = m1, m0

            start_pos, end_pos = m0, m1

        # If the portal extents are specified by z coordinates
        elif min_z != None:
            assert min_x == None and max_x == None
            assert start_pos == None and end_pos == None
            assert z0 != z1

            m0 = (min_z - z0) / dz
            m1 = (max_z - z0) / dz

            if m1 < m0:
                m0, m1 = m1, m0

            start_pos, end_pos = m0, m1

        else:
            assert min_x == None and max_x == None
            assert min_z == None and max_z == None

        assert end_pos > start_pos
        assert start_pos >= 0, "portal outside of wall extents"
        assert end_pos <= e_len, "portal outside of wall extents"

        self.portals[edge].append({
            'start_pos': start_pos,
            'end_pos': end_pos,
            'min_y': min_y,
            'max_y': max_y
        })

        # Sort the portals by start position
        self.portals[edge].sort(key=lambda e: e['start_pos'])

        return start_pos, end_pos

    def point_inside(self, p):
        """
        Test if a point is inside the room
        """

        # Vector from edge start to test point
        ap = p - self.outline

        # Compute the dot products of normals to AP vectors
        dotNAP = np.sum(self.edge_norms * ap, axis=1)

        # The point is inside if all the dot products are greater than zero
        return np.all(np.greater(dotNAP, 0))

    def _gen_static_data(self, params, rng):
        """
        Generate polygons and static data for this room
        Needed for rendering and collision detection
        Note: the wall polygons are quads, but the floor and
              ceiling can be arbitrary n-gons
        """

        # Load the textures and do texture randomization
        self.wall_tex = Texture.get(self.wall_tex_name, rng)
        self.floor_tex = Texture.get(self.floor_tex_name, rng)
        self.ceil_tex = Texture.get(self.ceil_tex_name, rng)

        # Generate the floor vertices
        self.floor_verts = self.outline
        self.floor_texcs = gen_texcs_floor(
            self.floor_tex,
            self.floor_verts
        )

        # Generate the ceiling vertices
        # Flip the ceiling vertex order because of backface culling
        self.ceil_verts = np.flip(self.outline, axis=0) + self.wall_height * Y_VEC
        self.ceil_texcs = gen_texcs_floor(
            self.ceil_tex,
            self.ceil_verts
        )

        self.wall_verts = []
        self.wall_norms = []
        self.wall_texcs = []
        self.wall_segs = []

        def gen_seg_poly(
            edge_p0,
            side_vec,
            seg_start,
            seg_end,
            min_y,
            max_y
        ):
            if seg_end == seg_start:
                return

            if min_y == max_y:
                return

            s_p0 = edge_p0 + seg_start * side_vec
            s_p1 = edge_p0 + seg_end * side_vec

            # If this polygon starts at ground level, add a collidable segment
            if min_y == 0:
                self.wall_segs.append(np.array([s_p1, s_p0]))

            # Generate the vertices
            # Vertices are listed in counter-clockwise order
            self.wall_verts.append(s_p0 + min_y * Y_VEC)
            self.wall_verts.append(s_p0 + max_y * Y_VEC)
            self.wall_verts.append(s_p1 + max_y * Y_VEC)
            self.wall_verts.append(s_p1 + min_y * Y_VEC)

            # Compute the normal for the polygon
            normal = np.cross(s_p1 - s_p0, Y_VEC)
            normal = -normal / np.linalg.norm(normal)
            for i in range(4):
                self.wall_norms.append(normal)

            # Generate the texture coordinates
            texcs = gen_texcs_wall(
                self.wall_tex,
                seg_start,
                min_y,
                seg_end - seg_start,
                max_y - min_y
            )
            self.wall_texcs.append(texcs)

        # For each wall
        for wall_idx in range(self.num_walls):
            edge_p0 = self.outline[wall_idx, :]
            edge_p1 = self.outline[(wall_idx+1) % self.num_walls, :]
            wall_width = np.linalg.norm(edge_p1 - edge_p0)
            side_vec = (edge_p1 - edge_p0) / wall_width

            if len(self.portals[wall_idx]) > 0:
                seg_end = self.portals[wall_idx][0]['start_pos']
            else:
                seg_end = wall_width

            # Generate the first polygon (going up to the first portal)
            gen_seg_poly(
                edge_p0,
                side_vec,
                0,
                seg_end,
                0,
                self.wall_height
            )

            # For each portal in this wall
            for portal_idx, portal in enumerate(self.portals[wall_idx]):
                portal = self.portals[wall_idx][portal_idx]
                start_pos = portal['start_pos']
                end_pos = portal['end_pos']
                min_y = portal['min_y']
                max_y = portal['max_y']

                # Generate the bottom polygon
                gen_seg_poly(
                    edge_p0,
                    side_vec,
                    start_pos,
                    end_pos,
                    0,
                    min_y
                )

                # Generate the top polygon
                gen_seg_poly(
                    edge_p0,
                    side_vec,
                    start_pos,
                    end_pos,
                    max_y,
                    self.wall_height
                )

                if portal_idx < len(self.portals[wall_idx]) - 1:
                    next_portal = self.portals[wall_idx][portal_idx+1]
                    next_portal_start = next_portal['start_pos']
                else:
                    next_portal_start = wall_width

                # Generate the polygon going up to the next portal
                gen_seg_poly(
                    edge_p0,
                    side_vec,
                    end_pos,
                    next_portal_start,
                    0,
                    self.wall_height
                )

        self.wall_verts = np.array(self.wall_verts)
        self.wall_norms = np.array(self.wall_norms)

        if len(self.wall_segs) > 0:
            self.wall_segs = np.array(self.wall_segs)
        else:
            self.wall_segs = np.array([]).reshape(0, 2, 3)

        if len(self.wall_texcs) > 0:
            self.wall_texcs = np.concatenate(self.wall_texcs)
        else:
            self.wall_texcs = np.array([]).reshape(0, 2)

    def _render(self):
        """
        Render the static elements of the room
        """

        glColor3f(1, 1, 1)

        # Draw the floor
        self.floor_tex.bind()
        glBegin(GL_POLYGON)
        glNormal3f(0, 1, 0)
        for i in range(self.floor_verts.shape[0]):
            glTexCoord2f(*self.floor_texcs[i, :])
            glVertex3f(*self.floor_verts[i, :])
        glEnd()

        # Draw the ceiling
        if not self.no_ceiling:
            self.ceil_tex.bind()
            glBegin(GL_POLYGON)
            glNormal3f(0, -1, 0)
            for i in range(self.ceil_verts.shape[0]):
                glTexCoord2f(*self.ceil_texcs[i, :])
                glVertex3f(*self.ceil_verts[i, :])
            glEnd()

        # Draw the walls
        self.wall_tex.bind()
        glBegin(GL_QUADS)
        for i in range(self.wall_verts.shape[0]):
            glNormal3f(*self.wall_norms[i, :])
            glTexCoord2f(*self.wall_texcs[i, :])
            glVertex3f(*self.wall_verts[i, :])
        glEnd()

class MiniWorldEnv(gym.Env):
    """
    Base class for MiniWorld environments. Implements the procedural
    world generation and simulation logic.
    """

    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second' : 30
    }

    # Enumeration of possible actions
    class Actions(IntEnum):
        # Turn left or right by a small amount
        turn_left = 0
        turn_right = 1

        # Move forward or back by a small amount
        move_forward = 2
        move_back = 3

        # Pick up or drop an object being carried
        pickup = 4
        drop = 5

        # Toggle/activate an object
        toggle = 6

        # Done completing task
        done = 7

    def __init__(
        self,
        max_episode_steps=1500,
        obs_width=80,
        obs_height=60,
        window_width=800,
        window_height=600,
        params=DEFAULT_PARAMS,
        domain_rand=False
    ):
        # Action enumeration for this environment
        self.actions = MiniWorldEnv.Actions

        # Actions are discrete integer values
        self.action_space = spaces.Discrete(len(self.actions))

        # Observations are RGB images with pixels in [0, 255]
        self.observation_space = spaces.Box(
            low=0,
            high=255,
            shape=(obs_height, obs_width, 3),
            dtype=np.uint8
        )

        self.reward_range = (-math.inf, math.inf)

        # Maximum number of steps per episode
        self.max_episode_steps = max_episode_steps

        # Simulation parameters, used for domain randomization
        self.params = params

        # Domain randomization enable/disable flag
        self.domain_rand = domain_rand

        # Window for displaying the environment to humans
        self.window = None

        # Invisible window to render into (shadow OpenGL context)
        self.shadow_window = pyglet.window.Window(width=1, height=1, visible=False)

        # Enable depth testing and backface culling
        glEnable(GL_DEPTH_TEST)
        glEnable(GL_CULL_FACE)

        # Frame buffer used to render observations
        self.obs_fb = FrameBuffer(obs_width, obs_height, 8)

        # Frame buffer used for human visualization
        self.vis_fb = FrameBuffer(window_width, window_height, 16)

        # Compute the observation display size
        self.obs_disp_width = 256
        self.obs_disp_height = obs_height * (self.obs_disp_width / obs_width)

        # For displaying text
        self.text_label = pyglet.text.Label(
            font_name="Arial",
            font_size=14,
            multiline=True,
            width=400,
            x = window_width + 5,
            y = window_height - (self.obs_disp_height + 19)
        )

        # Initialize the state
        self.seed()
        self.reset()

    def close(self):
        pass

    def seed(self, seed=None):
        self.rand = RandGen(seed)
        return [seed]

    def reset(self):
        """
        Reset the simulation at the start of a new episode
        This also randomizes many environment parameters (domain randomization)
        """

        # Step count since episode start
        self.step_count = 0

        # Create the agent
        self.agent = Agent()

        # List of entities contained
        self.entities = []

        # List of rooms in the world
        self.rooms = []

        # Wall segments for collision detection
        # Shape is (N, 2, 3)
        self.wall_segs = []

        # Generate the world
        self._gen_world()

        # Check if domain randomization is enabled or not
        rand = self.rand if self.domain_rand else None

        # Randomize elements of the world (domain randomization)
        self.params.sample_many(rand, self, [
            'sky_color',
            'light_pos',
            'light_color',
            'light_ambient'
        ])

        # Get the max forward step distance
        self.max_forward_step = self.params.get_max('forward_step')

        # Randomize parameters of the entities
        for ent in self.entities:
            ent.randomize(self.params, rand)

        # Compute the min and max x, z extents of the whole floorplan
        self.min_x = min([r.min_x for r in self.rooms])
        self.max_x = max([r.max_x for r in self.rooms])
        self.min_z = min([r.min_z for r in self.rooms])
        self.max_z = max([r.max_z for r in self.rooms])

        # Generate static data
        if len(self.wall_segs) == 0:
            self._gen_static_data()

        # Pre-compile static parts of the environment into a display list
        self._render_static()

        # Generate the first camera image
        obs = self.render_obs()

        # Return first observation
        return obs

    def _get_carry_pos(self, agent_pos, ent):
        """
        Compute the position at which to place an object being carried
        """

        dist = self.agent.radius + ent.radius + self.max_forward_step
        pos = agent_pos + self.agent.dir_vec * 1.05 * dist

        # Adjust the Y-position so the object is visible while being carried
        y_pos = max(self.agent.cam_height - ent.height - 0.3, 0)
        pos = pos + Y_VEC * y_pos

        return pos

    def move_agent(self, fwd_dist, fwd_drift):
        """
        Move the agent forward
        """

        next_pos = (
            self.agent.pos +
            self.agent.dir_vec * fwd_dist +
            self.agent.right_vec * fwd_drift
        )

        if self.intersect(self.agent, next_pos, self.agent.radius):
            return False

        carrying = self.agent.carrying
        if carrying:
            next_carrying_pos = self._get_carry_pos(next_pos, carrying)

            if self.intersect(carrying, next_carrying_pos, carrying.radius):
                return False

            carrying.pos = next_carrying_pos

        self.agent.pos = next_pos

        return True

    def turn_agent(self, turn_angle):
        """
        Turn the agent left or right
        """

        turn_angle *= (math.pi / 180)
        orig_dir = self.agent.dir

        self.agent.dir += turn_angle

        carrying = self.agent.carrying
        if carrying:
            pos = self._get_carry_pos(self.agent.pos, carrying)

            if self.intersect(carrying, pos, carrying.radius):
                self.agent.dir = orig_dir
                return False

            carrying.pos = pos
            carrying.dir = self.agent.dir

        return True

    def step(self, action):
        """
        Perform one action and update the simulation
        """

        self.step_count += 1

        rand = self.rand if self.domain_rand else None
        fwd_step = self.params.sample(rand, 'forward_step')
        fwd_drift = self.params.sample(rand, 'forward_drift')
        turn_step = self.params.sample(rand, 'turn_step')

        if action == self.actions.move_forward:
            self.move_agent(fwd_step, fwd_drift)

        elif action == self.actions.move_back:
            self.move_agent(-fwd_step, fwd_drift)

        elif action == self.actions.turn_left:
            self.turn_agent(turn_step)

        elif action == self.actions.turn_right:
            self.turn_agent(-turn_step)

        # Pick up an object
        elif action == self.actions.pickup:
            # Position at which we will test for an intersection
            test_pos = self.agent.pos + self.agent.dir_vec * 1.5 * self.agent.radius
            ent = self.intersect(self.agent, test_pos, 1.2 * self.agent.radius)
            if not self.agent.carrying:
                if isinstance(ent, Entity):
                    if not ent.is_static:
                        self.agent.carrying = ent

        # Drop an object being carried
        elif action == self.actions.drop:
            if self.agent.carrying:
                self.agent.carrying.pos[1] = 0
                self.agent.carrying = None

        # If we are carrying an object, update its position as we move
        if self.agent.carrying:
            ent_pos = self._get_carry_pos(self.agent.pos, self.agent.carrying)
            self.agent.carrying.pos = ent_pos
            self.agent.carrying.dir = self.agent.dir

        # Generate the current camera image
        obs = self.render_obs()

        # If the maximum time step count is reached
        if self.step_count >= self.max_episode_steps:
            done = True
            reward = 0
            return obs, reward, done, {}

        reward = 0
        done = False

        return obs, reward, done, {}

    def add_rect_room(
        self,
        min_x,
        max_x,
        min_z,
        max_z,
        **kwargs
    ):
        """
        Create a rectangular room
        """

        # 2D outline coordinates of the room,
        # listed in counter-clockwise order when viewed from the top
        outline = np.array([
            # East wall
            [max_x, max_z],
            # North wall
            [max_x, min_z],
            # West wall
            [min_x, min_z],
            # South wall
            [min_x, max_z],
        ])

        return self.add_room(outline=outline, **kwargs)

    def add_room(self, **kwargs):
        """
        Create a new room
        """

        assert len(self.wall_segs) == 0, "cannot add rooms after static data is generated"

        room = Room(**kwargs)
        self.rooms.append(room)

        return room

    def connect_rooms(
        self,
        room_a,
        room_b,
        min_x=None,
        max_x=None,
        min_z=None,
        max_z=None,
        max_y=None
    ):
        """
        Connect two rooms along facing edges
        """

        def find_facing_edges():
            for idx_a in range(room_a.num_walls):
                norm_a = room_a.edge_norms[idx_a]

                for idx_b in range(room_b.num_walls):
                    norm_b = room_b.edge_norms[idx_b]

                    # Reject edges that are not facing each other
                    if np.dot(norm_a, norm_b) > -0.9:
                        continue

                    dir = room_b.outline[idx_b] - room_a.outline[idx_a]

                    # Reject edges that are not touching
                    if np.dot(norm_a, dir) > 0.05:
                        continue

                    return idx_a, idx_b

            return None, None

        idx_a, idx_b = find_facing_edges()
        assert idx_a != None, "matching edges not found in connect_rooms"

        start_a, end_a = room_a.add_portal(
            edge=idx_a,
            min_x=min_x,
            max_x=max_x,
            min_z=min_z,
            max_z=max_z,
            max_y=max_y
        )

        start_b, end_b = room_b.add_portal(
            edge=idx_b,
            min_x=min_x,
            max_x=max_x,
            min_z=min_z,
            max_z=max_z,
            max_y=max_y
        )

        a = room_a.outline[idx_a] + room_a.edge_dirs[idx_a] * start_a
        b = room_a.outline[idx_a] + room_a.edge_dirs[idx_a] * end_a
        c = room_b.outline[idx_b] + room_b.edge_dirs[idx_b] * start_b
        d = room_b.outline[idx_b] + room_b.edge_dirs[idx_b] * end_b

        # If the portals are directly connected, stop
        if np.linalg.norm(a - d) < 0.001:
            return

        len_a = np.linalg.norm(b - a)
        len_b = np.linalg.norm(d - c)

        # Room outline points must be specified in counter-clockwise order
        outline = np.stack([c, b, a, d])
        outline = np.stack([outline[:, 0], outline[:, 2]], axis=1)

        max_y = max_y if max_y != None else room_a.wall_height

        room = Room(
            outline,
            wall_height=max_y,
            wall_tex=room_a.wall_tex_name,
            floor_tex=room_a.floor_tex_name,
            ceil_tex=room_a.ceil_tex_name,
            no_ceiling=room_a.no_ceiling,
        )

        self.rooms.append(room)

        room.add_portal(1, start_pos=0, end_pos=len_a)
        room.add_portal(3, start_pos=0, end_pos=len_b)

    def place_entity(
        self,
        ent,
        room=None,
        pos=None,
        dir=None,
        min_x=None,
        max_x=None,
        min_z=None,
        max_z=None
    ):
        """
        Place an entity/object in the world.
        Find a position that doesn't intersect with any other object.
        """

        assert len(self.rooms) > 0, "create rooms before calling place_entity"
        assert ent.radius != None, "entity must have physical size defined"

        # Generate collision detection data
        if len(self.wall_segs) == 0:
            self._gen_static_data()

        # If an exact position if specified
        if pos is not None:
            ent.dir = dir if dir != None else self.rand.float(-math.pi, math.pi)
            ent.pos = pos
            self.entities.append(ent)
            return ent

        # Keep retrying until we find a suitable position
        while True:
            # Pick a room, sample rooms proportionally to floor surface area
            r = room if room else self.rand.choice(self.rooms, probs=self.room_probs)

            # Choose a random point within the square bounding box of the room
            lx = r.min_x if min_x == None else min_x
            hx = r.max_x if max_x == None else max_x
            lz = r.min_z if min_z == None else min_z
            hz = r.max_z if max_z == None else max_z
            pos = self.rand.float(
                low =[lx + ent.radius, 0, lz + ent.radius],
                high=[hx - ent.radius, 0, hz - ent.radius]
            )

            # Make sure the position is within the room's outline
            if not r.point_inside(pos):
                continue

            # Make sure the position doesn't intersect with any walls
            if self.intersect(ent, pos, ent.radius):
                continue

            # Pick a direction
            d = dir if dir != None else self.rand.float(-math.pi, math.pi)

            ent.pos = pos
            ent.dir = d
            break

        self.entities.append(ent)

        return ent

    def place_agent(
        self,
        room=None,
        dir=None,
        min_x=None,
        max_x=None,
        min_z=None,
        max_z=None
    ):
        """
        Place the agent in the environment at a random position
        and orientation
        """

        return self.place_entity(
            self.agent,
            room=room,
            dir=dir,
            min_x=min_x,
            max_x=max_x,
            min_z=min_z,
            max_z=max_z
        )

    def intersect(self, ent, pos, radius):
        """
        Check if an entity intersects with the world
        """

        # Ignore the Y position
        px, _, pz = pos
        pos = np.array([px, 0, pz])

        # Check for intersection with walls
        if intersect_circle_segs(pos, radius, self.wall_segs):
            return True

        # Check for entity intersection
        for ent2 in self.entities:
            # Entities can't intersect with themselves
            if ent2 is ent:
                continue

            px, _, pz = ent2.pos
            pos2 = np.array([px, 0, pz])

            d = np.linalg.norm(pos2 - pos)
            if d < radius + ent2.radius:
                return ent2

        return None

    def near(self, ent0, ent1=None):
        """
        Test if the two entities are near each other.
        Used for "go to" or "put next" type tasks
        """

        if ent1 == None:
            ent1 = self.agent

        dist = np.linalg.norm(ent0.pos - ent1.pos)
        return dist < ent0.radius + ent1.radius + 1.1 * self.max_forward_step

    def _load_tex(self, tex_name):
        """
        Load a texture, with or without domain randomization
        """

        rand = self.rand if self.params.sample(self.rand, 'tex_rand') else None
        return Texture.get(tex_name, rand)

    def _gen_static_data(self):
        """
        Generate static data needed for rendering and collision detection
        """

        # Generate the static data for each room
        for room in self.rooms:
            room._gen_static_data(
                self.params,
                self.rand if self.domain_rand else None
            )

        # Concatenate the wall segments
        self.wall_segs = np.concatenate([r.wall_segs for r in self.rooms])

        # Room selection probabilities
        self.room_probs = np.array([r.area for r in self.rooms], dtype=float)
        self.room_probs /= np.sum(self.room_probs)

    def _gen_world(self):
        """
        Generate the world. Derived classes must implement this method.
        """

        raise NotImplementedError

    def _reward(self):
        """
        Default sparse reward computation
        """

        return 1.0 - 0.2 * (self.step_count / self.max_episode_steps)

    def _render_static(self):
        """
        Render the static elements of the scene into a display list.
        Called once at the beginning of each episode.
        """

        # TODO: manage this automatically
        # glIsList
        glDeleteLists(1, 1);
        glNewList(1, GL_COMPILE);

        # Light position
        glLightfv(GL_LIGHT0, GL_POSITION, (GLfloat*4)(*self.light_pos + [1]))

        # Background/minimum light level
        glLightfv(GL_LIGHT0, GL_AMBIENT, (GLfloat*4)(*self.light_ambient))

        # Diffuse light color
        glLightfv(GL_LIGHT0, GL_DIFFUSE, (GLfloat*4)(*self.light_color))

        #glLightf(GL_LIGHT0, GL_SPOT_CUTOFF, 180)
        #glLightf(GL_LIGHT0, GL_SPOT_EXPONENT, 0)
        #glLightf(GL_LIGHT0, GL_CONSTANT_ATTENUATION, 0)
        #glLightf(GL_LIGHT0, GL_LINEAR_ATTENUATION, 0)
        #glLightf(GL_LIGHT0, GL_QUADRATIC_ATTENUATION, 0)

        glEnable(GL_LIGHTING)
        glEnable(GL_LIGHT0)

        glShadeModel(GL_SMOOTH)
        glEnable(GL_COLOR_MATERIAL)
        glColorMaterial(GL_FRONT_AND_BACK, GL_AMBIENT_AND_DIFFUSE)

        # Render the rooms
        glEnable(GL_TEXTURE_2D)
        for room in self.rooms:
            room._render()

        # Render the static entities
        for ent in self.entities:
            if ent.is_static:
                ent.render()

        glEndList()

    def _render_world(
        self,
        frame_buffer,
        render_agent
    ):
        """
        Render the world from a given camera position into a frame buffer,
        and produce a numpy image array as output.
        """

        # Call the display list for the static parts of the environment
        glCallList(1)

        # TODO: keep the non-static entities in a different list for efficiency?
        # Render the non-static entities
        for ent in self.entities:
            if not ent.is_static and ent is not self.agent:
                ent.render()
                #ent.draw_bound()

        if render_agent:
            self.agent.render()

        # Resolve the rendered image into a numpy array
        img = frame_buffer.resolve()

        return img

    def render_top_view(self, frame_buffer=None):
        """
        Render a top view of the whole map (from above)
        """

        if frame_buffer == None:
            frame_buffer = self.obs_fb

        # Switch to the default OpenGL context
        # This is necessary on Linux Nvidia drivers
        self.shadow_window.switch_to()

        # Bind the frame buffer before rendering into it
        frame_buffer.bind()

        # Clear the color and depth buffers
        glClearColor(*self.sky_color, 1.0)
        glClearDepth(1.0)
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

        # Scene extents to render
        min_x = self.min_x - 1
        max_x = self.max_x + 1
        min_z = self.min_z - 1
        max_z = self.max_z + 1

        width = max_x - min_x
        height = max_z - min_z
        aspect = width / height
        fb_aspect = frame_buffer.width / frame_buffer.height

        # Adjust the aspect extents to match the frame buffer aspect
        if aspect > fb_aspect:
            # Want to add to denom, add to height
            new_h = width / fb_aspect
            h_diff = new_h - height
            min_z -= h_diff / 2
            max_z += h_diff / 2
        elif aspect < fb_aspect:
            # Want to add to num, add to width
            new_w = height * fb_aspect
            w_diff = new_w - width
            min_x -= w_diff / 2
            max_x += w_diff / 2

        # Set the projection matrix
        glMatrixMode(GL_PROJECTION)
        glLoadIdentity()
        glOrtho(
            min_x,
            max_x,
            -max_z,
            -min_z,
            -100, 100.0
        )

        # Setup the camera
        # Y maps to +Z, Z maps to +Y
        glMatrixMode(GL_MODELVIEW)
        glLoadIdentity()
        m = [
            1, 0, 0, 0,
            0, 0, 1, 0,
            0, -1, 0, 0,
            0, 0, 0, 1,
        ]
        glLoadMatrixf((GLfloat * len(m))(*m))

        return self._render_world(
            frame_buffer,
            render_agent=True
        )

    def render_obs(self, frame_buffer=None):
        """
        Render an observation from the point of view of the agent
        """

        if frame_buffer == None:
            frame_buffer = self.obs_fb

        # Switch to the default OpenGL context
        # This is necessary on Linux Nvidia drivers
        self.shadow_window.switch_to()

        # Bind the frame buffer before rendering into it
        frame_buffer.bind()

        # Clear the color and depth buffers
        glClearColor(*self.sky_color, 1.0)
        glClearDepth(1.0)
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

        # Set the projection matrix
        glMatrixMode(GL_PROJECTION)
        glLoadIdentity()
        gluPerspective(
            self.agent.cam_fov_y,
            frame_buffer.width / float(frame_buffer.height),
            0.04,
            100.0
        )

        # Setup the camera
        glMatrixMode(GL_MODELVIEW)
        glLoadIdentity()
        gluLookAt(
            # Eye position
            *self.agent.cam_pos,
            # Target
            *(self.agent.cam_pos + self.agent.cam_dir),
            # Up vector
            0, 1.0, 0.0
        )

        return self._render_world(
            frame_buffer,
            render_agent=False
        )

    def render_depth(self, frame_buffer=None):
        """
        Produce a depth map
        Values are floating-point, map shape is (H,W,1)
        Distances are in meters from the observer
        """

        if frame_buffer == None:
            frame_buffer = self.obs_fb

        # Render the world
        self.render_obs(frame_buffer)

        return frame_buffer.get_depth_map(0.04, 100.0)

    def get_visible_ents(self):
        """
        Get a list of visible entities.
        Uses OpenGL occlusion queries to approximate visibility.
        :return: set of objects visible to the agent
        """

        # Allocate the occlusion query ids
        num_ents = len(self.entities)
        query_ids = (GLuint * num_ents)()
        glGenQueries(num_ents, query_ids)

        # Switch to the default OpenGL context
        # This is necessary on Linux Nvidia drivers
        self.shadow_window.switch_to()

        # Use the small observation frame buffer
        frame_buffer = self.obs_fb

        # Bind the frame buffer before rendering into it
        frame_buffer.bind()

        # Clear the color and depth buffers
        glClearColor(*self.sky_color, 1.0)
        glClearDepth(1.0)
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

        # Set the projection matrix
        glMatrixMode(GL_PROJECTION)
        glLoadIdentity()
        gluPerspective(
            self.agent.cam_fov_y,
            frame_buffer.width / float(frame_buffer.height),
            0.04,
            100.0
        )

        # Setup the cameravisible objects
        glMatrixMode(GL_MODELVIEW)
        glLoadIdentity()
        gluLookAt(
            # Eye position
            *self.agent.cam_pos,
            # Target
            *(self.agent.cam_pos + self.agent.cam_dir),
            # Up vector
            0, 1.0, 0.0
        )

        # Render the rooms, without texturing
        glDisable(GL_TEXTURE_2D)
        for room in self.rooms:
            room._render()

        # For each entity
        for ent_idx, ent in enumerate(self.entities):
            if ent is self.agent:
                continue

            glBeginQuery(GL_ANY_SAMPLES_PASSED, query_ids[ent_idx])
            pos = ent.pos

            #glColor3f(1, 0, 0)
            drawBox(
                x_min=pos[0] - 0.1,
                x_max=pos[0] + 0.1,
                y_min=pos[1],
                y_max=pos[1] + 0.2,
                z_min=pos[2] - 0.1,
                z_max=pos[2] + 0.1
            )

            glEndQuery(GL_ANY_SAMPLES_PASSED)

        vis_objs = set()

        # Get query results
        for ent_idx, ent in enumerate(self.entities):
            if ent is self.agent:
                continue

            visible = (GLuint*1)(1)
            glGetQueryObjectuiv(query_ids[ent_idx], GL_QUERY_RESULT, visible);

            if visible[0] != 0:
                vis_objs.add(ent)

        # Free the occlusion query ids
        glDeleteQueries(1, query_ids)

        #img = frame_buffer.resolve()
        #return img

        return vis_objs

    def render(self, mode='human', close=False, view='agent'):
        """
        Render the environment for human viewing
        """

        if close:
            if self.window:
                self.window.close()
            return

        # Render the human-view image
        assert view in ['agent', 'top']
        if view == 'agent':
            img = self.render_obs(self.vis_fb)
        else:
            img = self.render_top_view(self.vis_fb)
        img_width = img.shape[1]
        img_height = img.shape[0]

        if mode == 'rgb_array':
            return img

        # Render the agent's view
        obs = self.render_obs()
        obs_width = obs.shape[1]
        obs_height = obs.shape[0]

        window_width = img_width + self.obs_disp_width
        window_height = img_height

        if self.window is None:
            config = pyglet.gl.Config(double_buffer=True)
            self.window = pyglet.window.Window(
                width=window_width,
                height=window_height,
                resizable=False,
                config=config
            )

        self.window.clear()
        self.window.switch_to()

        # Bind the default frame buffer
        glBindFramebuffer(GL_FRAMEBUFFER, 0);

        # Clear the color and depth buffers
        glClearColor(0, 0, 0, 1.0)
        glClearDepth(1.0)
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

        # Setup orghogonal projection
        glMatrixMode(GL_PROJECTION)
        glLoadIdentity()
        glMatrixMode(GL_MODELVIEW)
        glLoadIdentity()
        glOrtho(0, window_width, 0, window_height, 0, 10)

        # Draw the human render to the rendering window
        img_flip = np.ascontiguousarray(np.flip(img, axis=0))
        img_data = pyglet.image.ImageData(
            img_width,
            img_height,
            'RGB',
            img_flip.ctypes.data_as(POINTER(GLubyte)),
            pitch=img_width * 3,
        )
        img_data.blit(
            0,
            0,
            0,
            width=img_width,
            height=img_height
        )

        # Draw the observation
        obs = np.ascontiguousarray(np.flip(obs, axis=0))
        obs_data = pyglet.image.ImageData(
            obs_width,
            obs_height,
            'RGB',
            obs.ctypes.data_as(POINTER(GLubyte)),
            pitch=obs_width * 3,
        )
        obs_data.blit(
            img_width,
            img_height - self.obs_disp_height,
            0,
            width=self.obs_disp_width,
            height=self.obs_disp_height
        )

        # Draw the text label in the window
        self.text_label.text = "pos: (%.2f, %.2f, %.2f)\nangle: %d\nsteps: %d" % (
            *self.agent.pos,
            int(self.agent.dir * 180 / math.pi) % 360,
            self.step_count
        )
        self.text_label.draw()

        # Force execution of queued commands
        glFlush()

        # If we are not running the Pyglet event loop,
        # we have to manually flip the buffers and dispatch events
        if mode == 'human':
            self.window.flip()
            self.window.dispatch_events()

        return img
