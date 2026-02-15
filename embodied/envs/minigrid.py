import embodied
import gym
import numpy as np
from PIL import Image
import gym_minigrid
from gym_minigrid.wrappers import RGBImgObsWrapper, ImgObsWrapper
from . import from_gym


class ResizeObservation(gym.ObservationWrapper):
    """Resize observation images to target size."""
    def __init__(self, env, size=(64, 64)):
        super().__init__(env)
        self.size = size
        obs_shape = (size[0], size[1], 3)
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=obs_shape, dtype=np.uint8
        )
    
    def observation(self, obs):
        # Use PIL instead of cv2 for better multiprocessing stability
        img = Image.fromarray(obs)
        img = img.resize(self.size, Image.BILINEAR)
        return np.array(img)


class Minigrid(from_gym.FromGym):
    LOCK = None

    def __init__(self, task, seed=None, size=(32, 32), **kwargs):
        # task is the environment id, e.g. 'MiniGrid-Empty-5x5-v0'
        if self.LOCK is None:
            import multiprocessing as mp
            mp = mp.get_context("spawn")
            Minigrid.LOCK = mp.Lock()

        self._seed = seed
        with self.LOCK:
            env = gym.make(task)
        if seed is not None:
            env.seed(seed)
        # RGBImgObsWrapper converts the symbolic grid to RGB pixels
        # ImgObsWrapper then extracts just the 'image' key from the dict obs
        env = RGBImgObsWrapper(env)
        env = ImgObsWrapper(env)
        # Resize to expected size (32x32 seems to be expected by the model)
        env = ResizeObservation(env, size=size)
        # Don't pass kwargs to FromGym since it asserts no kwargs when env is passed
        super().__init__(env)

