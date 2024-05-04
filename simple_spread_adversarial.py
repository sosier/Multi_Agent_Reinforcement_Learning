"""
Modified version of the following environments:
 - https://pettingzoo.farama.org/_modules/pettingzoo/mpe/simple_spread/simple_spread/
 - https://pettingzoo.farama.org/_modules/pettingzoo/mpe/simple_adversary/simple_adversary/
"""
import numpy as np
from gymnasium.utils import EzPickle

from pettingzoo.mpe._mpe_utils.core import Agent, Landmark, World
from pettingzoo.mpe._mpe_utils.scenario import BaseScenario
from pettingzoo.mpe._mpe_utils.simple_env import SimpleEnv, make_env
from pettingzoo.utils.conversions import parallel_wrapper_fn

class raw_env(SimpleEnv, EzPickle):
    def __init__(
        self,
        n_agents=2,
        n_adversaries=2,
        n_landmarks=2,
        max_cycles=25,
        continuous_actions=False,
        render_mode=None,
    ):
        assert n_agents > 0 and n_landmarks > 0
        EzPickle.__init__(
            self,
            n_agents=n_agents,
            n_adversaries=n_adversaries,
            n_landmarks=n_landmarks,
            max_cycles=max_cycles,
            continuous_actions=continuous_actions,
            render_mode=render_mode,
        )
        scenario = Scenario()
        world = scenario.make_world(n_agents, n_adversaries, n_landmarks)
        SimpleEnv.__init__(
            self,
            scenario=scenario,
            world=world,
            render_mode=render_mode,
            max_cycles=max_cycles,
            continuous_actions=continuous_actions,
        )
        self.metadata["name"] = "simple_spread_v3_adversarial"

env = make_env(raw_env)
parallel_env = parallel_wrapper_fn(env)

class Scenario(BaseScenario):
    def make_world(self, n_agents=2, n_adversaries=2, n_landmarks=2):
        world = World()
        # set any world properties first
        num_agents = n_agents + n_adversaries
        num_adversaries = n_adversaries
        num_landmarks = n_landmarks
        world.dim_c = 2
        world.num_agents = num_agents
        world.num_good_agents = n_agents
        world.num_adversaries = num_adversaries
        world.num_landmarks = num_landmarks

        # add agents
        world.agents = [Agent() for i in range(num_agents)]
        for i, agent in enumerate(world.agents):
            agent.adversary = True if i < num_adversaries else False
            base_name = "adversary" if agent.adversary else "agent"
            base_index = i if i < num_adversaries else i - num_adversaries
            agent.name = f"{base_name}_{base_index}"
            agent.collide = True
            agent.silent = True
            agent.size = 0.15

        # add landmarks
        world.landmarks = [Landmark() for i in range(num_landmarks)]
        for i, landmark in enumerate(world.landmarks):
            landmark.name = "landmark %d" % i
            landmark.collide = False
            landmark.movable = False

        return world

    def reset_world(self, world, np_random):
        # random properties for agents
        for i in range(world.num_adversaries):
            world.agents[i].color = np.array([0.85, 0.35, 0.35])
        for i in range(world.num_adversaries, world.num_agents):
            world.agents[i].color = np.array([0.35, 0.35, 0.85])

        # random properties for landmarks
        for i, landmark in enumerate(world.landmarks):
            landmark.color = np.array([0.25, 0.25, 0.25])

        # set random initial states
        for agent in world.agents:
            agent.state.p_pos = np_random.uniform(-1, +1, world.dim_p)
            agent.state.p_vel = np.zeros(world.dim_p)
            agent.state.c = np.zeros(world.dim_c)
        
        for i, landmark in enumerate(world.landmarks):
            landmark.state.p_pos = np_random.uniform(-1, +1, world.dim_p)
            landmark.state.p_vel = np.zeros(world.dim_p)

    def benchmark_data(self, agent, world):
        rew = 0
        collisions = 0
        occupied_landmarks = 0
        min_dists = 0
        for lm in world.landmarks:
            dists = [
                np.sqrt(np.sum(np.square(a.state.p_pos - lm.state.p_pos)))
                for a in world.agents
            ]
            min_dists += min(dists)
            rew -= min(dists)
            if min(dists) < 0.1:
                occupied_landmarks += 1
        if agent.collide:
            for a in world.agents:
                if self.is_collision(a, agent):
                    rew -= 1
                    collisions += 1
        return (rew, collisions, min_dists, occupied_landmarks)

    # return all agents that are not adversaries
    def good_agents(self, world):
        return [agent for agent in world.agents if not agent.adversary]

    # return all adversarial agents
    def adversaries(self, world):
        return [agent for agent in world.agents if agent.adversary]

    def reward(self, agent, world):
        # Good agents are rewarded based on minimum agent distance to each
        # landmark, penalized for any collisions
        # 
        # Adversaries are rewarded for collisions with good agents and penalized
        # for collisions with other adveraries
        return (
            self.adversary_reward(agent, world)
            if agent.adversary
            else self.agent_reward(agent, world)
        )

    def is_collision(self, agent1, agent2):
        delta_pos = agent1.state.p_pos - agent2.state.p_pos
        dist = np.sqrt(np.sum(np.square(delta_pos)))
        dist_min = agent1.size + agent2.size
        return True if dist < dist_min else False
    
    def agent_reward(self, agent, world):
        rew = 0

        # Distance to landmark reward:
        for lm in world.landmarks:
            dists = [
                np.sqrt(np.sum(np.square(a.state.p_pos - lm.state.p_pos)))
                for a in world.agents
                if not a.adversary
            ]
            rew -= min(dists)
        
        # Collision penalty:
        if agent.collide:
            for a in world.agents:
                rew -= 1.0 * (self.is_collision(a, agent) and a != agent)

        return rew

    def adversary_reward(self, agent, world):
        rew = 0

        # Collision penalty / reward:
        if agent.collide:
            for a in world.agents:
                if a.adversary:  # Penalize
                    rew -= 1.0 * (self.is_collision(a, agent) and a != agent)
                else:  # Reward
                    rew += 1.0 * (self.is_collision(a, agent) and a != agent)

        return rew

    def observation(self, agent, world):
        # get positions of all landmarks relative to agent
        landmark_pos = []
        for entity in world.landmarks:
            landmark_pos.append(entity.state.p_pos - agent.state.p_pos)

        # get positions of all other agents relative to agent
        other_pos = []
        for other in world.agents:
            if other is agent:
                continue
            
            # Is other agent an adversary?
            other_pos.append(np.array([int(other.adversary)]))
            # Other agent position
            other_pos.append(other.state.p_pos - agent.state.p_pos)

        return np.concatenate(
            [np.array([int(agent.adversary)])]  # Is self an adversary?
            + [agent.state.p_vel]
            + [agent.state.p_pos]
            + landmark_pos
            + other_pos
        )
