import argparse
import copy
import os 

import imageio
import skimage
        
import numpy as np 
from typing import List

from hive import agents as agent_lib
from hive import envs
from hive.runners.single_agent_loop import SingleAgentRunner
from hive.runners.utils import TransitionInfo, load_config
from hive.utils import experiment, loggers, schedule, utils
from hive.utils.registry import get_parsed_args

from hive.agents.agent import Agent
from hive.envs.base import BaseEnv
from hive.runners import Runner
from hive.runners.utils import TransitionInfo
from hive.utils import utils
from hive.runners.utils import Metrics
from hive.utils.experiment import Experiment
from hive.utils.loggers import CompositeLogger, NullLogger, ScheduledLogger

import matplotlib.pyplot as plt
import matplotlib as mpl 
import scipy.stats

os.environ['MUJOCO_GL'] = 'egl'

class SingleAgentRunnerEval(SingleAgentRunner):
    """Runner class used to implement a sinle-agent training loop."""
    def __init__(
        self,
        environment: BaseEnv,
        agent: Agent,
        loggers: List[ScheduledLogger],
        experiment_manager: Experiment,
        train_steps: int,
        eval_environment: BaseEnv = None,
        test_frequency: int = -1,
        test_episodes: int = 1,
        stack_size: int = 1,
        max_steps_per_episode: int = 1e9,
        plot_frequency: int = 100,
        plot_vis: bool = False,
        seed: int = None,
        name: str = "experiment",
        logdir: str = 'logs/test',
    ):
        super().__init__(
            environment,
            agent,
            loggers,
            experiment_manager,
            train_steps,
            eval_environment,
            test_frequency,
            test_episodes,
            stack_size,
            max_steps_per_episode,
        )

        self._plot_frequency = plot_frequency
        self._plot_vis = plot_vis
        self._iter = 0
        self.logdir = logdir
        
        try:
            os.makedirs(self.logdir)
        except:
            pass

    def create_episode_metrics(self):
        """Create the metrics used during the loop.""" 
        return Metrics(
            self._agents,
            [("reward", 0), ("episode_length", 0), ("entropy", 0.0), ("visited", 0.0)],
            [("full_episode_length", 0)],
        )
    def run_testing(self):
        """Run a testing phase."""
        if self._eval_environment is None:
            return
        self.train_mode(False)

        aggregated_episode_metrics = self.create_episode_metrics().get_flat_dict()
        episode_metrics, agent_state = self.run_episode(self._eval_environment)
        for metric, value in episode_metrics.get_flat_dict().items():
            aggregated_episode_metrics[metric] += value / self._test_episodes
        
        self._logger.update_step("test")
        self._logger.log_metrics(aggregated_episode_metrics, "test")
        self._run_testing = False
        self.train_mode(True)

    def evaluate_search_time(self):
        """Run a testing phase."""
        if self._eval_environment is None:
            return

        self.train_mode(False)

        self._eval_environment._env.set_explore(True)
        env_timeout = self._eval_environment._env.timeout
        self._eval_environment._env.set_timeout(1000)
        episode_metrics, _ = self.run_episode(self._eval_environment)
        self._eval_environment._env.set_explore(False)
        self._logger.log_scalar('Search Time', episode_metrics[0]['episode_length'], 'test')
        self._eval_environment._env.set_timeout(env_timeout)
        self.train_mode(True)

    def run_training(self):
        """Run the training loop."""
        self.train_mode(True)
        while self._train_schedule.get_value():
            if not self._training:
                self.train_mode(True)
            
            episode_metrics, agent_state = self.run_episode(self._train_environment)
            
            if self._logger.should_log("train"):
                episode_metrics = episode_metrics.get_flat_dict()
                self._logger.log_metrics(episode_metrics, "train")

            # Save experiment state
            if self._save_experiment:
                self._experiment_manager.save()
                self._save_experiment = False
            
            if self._iter % 10 == 0:
                self.evaluate_search_time()

            self._iter += 1
            
        # Run a final test episode and save the experiment.
        self.run_testing()
        self._experiment_manager.save()

    def run_episode(self, environment, render=False):
        """Run a single episode of the environment."""
        episode_metrics = self.create_episode_metrics()
        terminated, truncated = False, False
        observation, _ = environment.reset()
        transition_info = TransitionInfo(self._agents, self._stack_size)
        transition_info.start_agent(self._agents[0])
        agent_state = None
        steps = 0

        runner_episode_log = {'vis_onehot': np.zeros(environment._env.total_states),
                              'traj_onehot': []}
        if render:
            img_gifs = []
        
        # Run the loop until the episode ends or times out
        while (
            not (terminated or truncated)
            and steps < self._max_steps_per_episode
            and (not self._training or self._train_schedule.get_value())
        ):

            if render:
                img_gifs.append(environment._env.render())

            terminated, truncated, observation, agent_state = self.run_one_step(
                environment, observation, episode_metrics, transition_info, agent_state
            )
            self.update_step()
            
            runner_episode_log['vis_onehot'] += observation['output']
            steps += 1
        
            agent = self._agents[0]
            episode_metrics[agent.id].update(runner_episode_log)
            state_dist = np.copy(runner_episode_log['vis_onehot']) / steps
            agent = self._agents[0]
            episode_metrics[agent.id]['entropy'] = scipy.stats.entropy(state_dist)
            episode_metrics[agent.id]['visited'] = np.count_nonzero(state_dist)

        if not (terminated or truncated):
            self.run_end_step(
                environment,
                observation,
                episode_metrics,
                transition_info,
                agent_state,
            )
            self.update_step()
        
        if render:
            return episode_metrics, agent_state, img_gifs
        else:
            return episode_metrics, agent_state
