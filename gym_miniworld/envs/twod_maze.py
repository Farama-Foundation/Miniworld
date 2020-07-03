import numpy as np
import math
from gym import spaces
from ..miniworld import MiniWorldEnv, Room
from ..entity import Box, ImageFrame
from ..params import DEFAULT_PARAMS
from scipy import ndimage

def find_grid(fplan): #2D floor plan for the maze, have to be binary, walls are 1, hallways are 0
    labeled, nr_objects = ndimage.label(fplan) # find connexted
    x_breaks=[0] #break points/grid in the x-axis
    z_breaks=[0] #breaks points/grid in the z-axis
    # all intermediate points
    for l in range(nr_objects):
        x_breaks.append(np.min(np.argwhere(labeled==l+1)[:,1]))
        x_breaks.append(np.max(np.argwhere(labeled==l+1)[:,1])+1)
        z_breaks.append(np.min(np.argwhere(labeled==l+1)[:,0]))
        z_breaks.append(np.max(np.argwhere(labeled==l+1)[:,0])+1)
    # end points
    end_points=np.shape(fplan)
    x_breaks.append(end_points[1]+1)
    z_breaks.append(end_points[0]+1)
    x_breaks=np.unique(np.array(x_breaks)) # in case there is duplicates of 0 or end
    z_breaks=np.unique(np.array(z_breaks)) # in case there is duplicates of 0 or end
    return x_breaks,z_breaks

class TwoDMaze(MiniWorldEnv):
    """
    A maze environment in which the agent has to reach a red box,
    but it takes in 2D floor plan and make it into a 3D environment:
    If you provide a 2D floor plan of size M x N, it will construct the maze on a grid of size (M+1)x(N+1),
        where each entry (m,n) in the 2D floor plan coresponds to a brick enclosed by (m,n),(m,n+1),(m+1,n+1),(m+1,n+1).
        The construction process works by first getting all the horizontal/vertical grids that are "broken" by walls.
        And then the overall way to layout inherits the usual miniworld maze: contruct rooms and visit them and try to connect.
        Walls are essentially unconnected rooms.

    Example use:
        # in python:
        fplan=np.zeros((8,8))
        fplan[0,5]=1
        agent_pos=np.array([1,0,1])
        reward_pos=np.array([150,0,150])
        testmaze=gym_miniworld.envs.twod_maze.TwoDMaze(fplan,agent_pos,reward_pos)

        # The name of the environment is:
        "MiniWorld-TwoDMaze-v0"
    """

    def __init__(
        self,
        fplan, #2D floor plan for the maze, have to be binary, walls are 1, hallways are 0
        start_pos, #start position, tuple or numpy array, 1x3
        rewards_pos, #end position, tuple or numpy array, 1x3
        max_episode_steps=None,
        **kwargs
    ):  
        self.fplan = fplan
        self.start_pos=start_pos,
        self.reward_pos=rewards_pos,
        self.x_breaks,self.z_breaks=find_grid(fplan)
        self.num_rows = len(self.z_breaks)-1
        self.num_cols = len(self.x_breaks)-1


        super().__init__(
            max_episode_steps = max_episode_steps) #or self.num_rows * self.num_cols * 24,
        #    **kwargs
        #)

        # Allow only the movement actions
        self.action_space = spaces.Discrete(self.actions.move_forward+1)
        


    def _gen_world(self):
        x_breaks, z_breaks= self.x_breaks, self.z_breaks
        bool_val=np.zeros((self.num_rows,self.num_cols)) #if 1 it means this section is wall
        rows = []

        # For each row
        for j in range(self.num_rows):
            row = []

            # For each column
            for i in range(self.num_cols):

                min_x = x_breaks[i]
                max_x = x_breaks[i+1]

                min_z = z_breaks[j]
                max_z = z_breaks[j+1]


                # wall or not
                
                mid_x,mid_z = np.floor(0.5*(min_x+max_x)).astype('int'),np.floor(0.5*(min_z+max_z)).astype('int')
                bool_val[j,i]=self.fplan[mid_z,mid_x]

                
                if bool_val[j,i]:
                    #print('wall')
                    room = self.add_rect_room(
                        min_x=min_x,
                        max_x=max_x,
                        min_z=min_z,
                        max_z=max_z,
                        wall_tex='brick_wall',
                        floor_tex='asphalt')
                else:
                    #print('hallway')
                    room = self.add_rect_room(
                        min_x=min_x,
                        max_x=max_x,
                        min_z=min_z,
                        max_z=max_z,
                        wall_tex='brick_wall',  
                )
                row.append(room)
            rows.append(row)

        visited = set()

        def visit(i, j):
            """
            Explore the maze in depth first search from the (i,j)'s spot break walls
            """
            room = rows[j][i]

            visited.add(room)
            allrooms_visit_record[j,i]=1

            # Reorder the neighbors to visit in a random order
            neighbors = self.rand.subset([(0,1), (0,-1), (-1,0), (1,0)], 4)
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
                if bool_val[nj,ni]>0: # is wall
                    continue

                if di == 0:
                    self.connect_rooms(room, neighbor, min_x=room.min_x, max_x=room.max_x)
                elif dj == 0:
                    self.connect_rooms(room, neighbor, min_z=room.min_z, max_z=room.max_z)

                visit(ni, nj)

        # Generate the maze starting from the top-left corner
        allrooms_visit_record=bool_val.copy()
        while np.sum(allrooms_visit_record==0)>0:
            row_id=np.argwhere(allrooms_visit_record==0)[0,0]
            col_id=np.argwhere(allrooms_visit_record==0)[0,1]
            visit(row_id, col_id)
            

        self.box = self.place_entity(Box(color='red'),pos=self.reward_pos)
        self.place_entity(self.agent,pos=self.start_pos)

    def step(self, action):
        obs, reward, done, info = super().step(action)

        if self.near(self.box):
            reward += self._reward()
            done = True

        return obs, reward, done, info
