from ..miniworld import MiniWorldEnv, Room

class HallwayEnv(MiniWorldEnv):
    def __init__(self):
        super().__init__()

    def _gen_world(self):

        room = self.create_rect_room(
            0, 0,
            2, 10
        )
