from multiprocessing import Process, Pipe, set_start_method
import cloudpickle
import gym

def worker(conn, make_env, seed):
    print('Creating env, seed={}'.format(seed))

    make_env = cloudpickle.loads(make_env)
    env = make_env(seed)

    while True:
        cmd, data = conn.recv()
        if cmd == "step":
            obs, reward, done, info = env.step(data)
            if done:
                obs = env.reset()
            conn.send((obs, reward, done, info))
        elif cmd == "reset":
            obs = env.reset()
            conn.send(obs)
        else:
            raise NotImplementedError

class ParallelEnv(gym.Env):
    """A concurrent execution of environments in multiple processes."""

    def __init__(self, make_env, num_procs, seed):
        assert num_procs >= 1
        self.num_procs = num_procs

        self.env0 = make_env(seed + 10000*0)
        self.observation_space = self.env0.observation_space
        self.action_space = self.env0.action_space



        set_start_method('forkserver')

        make_env = cloudpickle.dumps(make_env)



        self.locals = []
        for proc_idx in range(1, num_procs):
            seed = seed + 10000*proc_idx
            local, remote = Pipe()
            self.locals.append(local)
            p = Process(target=worker, args=(remote, make_env, seed))
            p.daemon = True
            p.start()
            remote.close()

    def reset(self):
        for local in self.locals:
            local.send(("reset", None))
        results = [self.env0.reset()] + [local.recv() for local in self.locals]
        return results

    def step(self, actions):
        for local, action in zip(self.locals, actions[1:]):
            local.send(("step", action))
        obs, reward, done, info = self.env0.step(actions[0])
        if done:
            obs = self.env0.reset()
        results = zip(*[(obs, reward, done, info)] + [local.recv() for local in self.locals])
        return results

    def render(self):
        raise NotImplementedError
