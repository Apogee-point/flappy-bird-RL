from typing import Any, Dict, List, SupportsFloat, Tuple
from gymnasium.core import ActType, ObsType, RenderFrame

import functools

import gymnasium as gym
import numpy as np
import pygame

from gymnasium.spaces import Box, Discrete

from .background import Background
from .base import Base
from .bird import Bird
from .pipe import Pipe


class FlappyBirdEnv(gym.Env):
    action_space = Discrete(2)

    observation_space = Box(low=-np.inf,high=np.inf, shape=(4,),
                            dtype=np.float32)
    # observation_space = Box(low=-np.inf,high=np.inf, shape=(6,),
    #                         dtype=np.float32)

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 60}

    def __init__(self, render_mode: str | None = None):
        self.render_mode = render_mode

        self._background = None
        self._pipes = None
        self._base = None
        self._bird = None
        self._surface = None
        self._clock = None
        if self.render_mode == "human":
            self._clock = pygame.time.Clock()

        self._last_action = 0
        self._score = 0

    @property
    def observation(self) -> ObsType:
        next_pipe = None
        for pipe in self._pipes:
            if pipe.x + pipe.pipe_top.get_width() > self._bird.x:
                next_pipe = pipe
                break
        if next_pipe is None:
            return np.array([self._bird.y,self._bird.velocity,0, 0])
        else:
            vertical_distance = next_pipe.height + 200 - self._bird.y
            horizontal_distance = next_pipe.x - self._bird.x
            gapCenter = next_pipe.height + 100
            return np.array([self._bird.y,self._bird.velocity,vertical_distance, horizontal_distance])
        
            # return np.array([self._bird.x,self._bird.y, vertical_distance, horizontal_distance,self._bird.velocity,gap])
            # self._bird.y - self._base.y, next_pipe.top, next_pipe.bottom])

            
    @property
    def reward(self) -> SupportsFloat:
        if any([not pipe.passed and pipe.x < self._bird.x
                for pipe in self._pipes]): #* For passing a pipe
            return 1000
        elif not self.terminated: #* For staying alive in the game
            return 0
        # elif self.terminated and self._bird.y + self._bird.image.get_height() >= 730 or self._bird.y < 0:
        #     return -10000 #* For hitting the ground or the ceiling
        else:
            return -100000 #* For dying

    @property
    def terminated(self) -> bool:
        return any([*[pipe.collide(self._bird) for pipe in self._pipes],
                    self._bird.y + self._bird.image.get_height() >= 730,
                    self._bird.y < 0])

    @property
    def truncated(self) -> bool:
        return False

    @property
    def info(self) -> Dict[str, Any]:
        return {
            "background": {
                "upper_left": (0, 0)
            },
            "pipes": [{
                "x": pipe.x,
                "height": pipe.height,
                "top": pipe.top,
                "bottom": pipe.bottom
            } for pipe in self._pipes],
            "base": {
                "x1": self._base.x1,
                "x2": self._base.x2,
                "y": self._base.y
            },
            "bird": {
                "x": self._bird.x,
                "y": self._bird.y
            },
            "last_action": self._last_action,
            "score": self._score
        }

    def step(self, action: ActType) -> \
            tuple[ObsType, SupportsFloat, bool, bool, Dict[str, Any]]:

        if action == 1:
            self._bird.jump()

        add_pipe = False
        self._bird.move()

        to_be_removed = []
        for pipe in self._pipes:
            if pipe.x + pipe.pipe_top.get_width() < 0:
                to_be_removed.append(pipe)

            if not pipe.passed and pipe.x < self._bird.x:
                self._score += 1
                pipe.passed = True
                add_pipe = True

            pipe.move()

        if add_pipe:
            self._pipes.append(Pipe(700, self.np_random))

        for pipe in to_be_removed:
            self._pipes.remove(pipe)

        self._base.move()

        if self.render_mode == "human":
            self.render()

        return self.observation, self.reward, self.terminated,self.truncated, self.info

    def reset(self, *, seed: int | None = None,
              options: Dict[str, Any] | None = None) \
            -> Tuple[ObsType, Dict[str, Any]]:
        
        super().reset(seed=seed)

        self._background = Background()
        self._pipes = [Pipe(700, self.np_random)]
        self._base = Base(700)
        self._bird = Bird(222, 376)

        self._surface = None

        self._last_action = 0
        self._score = 0

        if self.render_mode is not None:
            self.render()

        return self.observation, self.info

    def render(self) -> RenderFrame | List[RenderFrame] | None:
        
        if self._surface is None:
            pygame.init()

            if self.render_mode == "human":
                pygame.display.init()
                pygame.display.set_caption("Flappy Bird")
                self._surface = pygame.display.set_mode(self._shape)
            elif self.render_mode == "rgb_array":
                self._surface = pygame.Surface(self._shape)
                return self.observation

        assert self._surface is not None, \
            "Something went wrong"

        self._background.draw(self._surface)
        for pipe in self._pipes:
            pipe.draw(self._surface)
        self._base.draw(self._surface)
        self._bird.draw(self._surface)
        font = pygame.font.Font(None, 36)
        score_text = font.render(f"Score: {self._score}", True, (255, 255, 255))
        self._surface.blit(score_text, (10, 10))

        if self.render_mode == "human":
            pygame.event.pump()
            pygame.display.update()
            self._clock.tick(FlappyBirdEnv.metadata["render_fps"])
        elif self.render_mode == "rgb_array":
            return self.observation

    @property
    @functools.cache
    def _width(self) -> int:
        return 576

    @property
    @functools.cache
    def _height(self) -> int:
        return 800

    @property
    @functools.cache
    def _shape(self) -> Tuple[int, int]:
        return self._width, self._height

    def close(self) -> None:
        """
        After the user has finished using the environment, close contains the
        code necessary to "clean up" the environment.
        """

        if self._surface is not None:
            pygame.display.quit()
            pygame.quit()