import os

from gymnasium.envs.registration import register
from types import SimpleNamespace

from jaxl.constants import *
from jaxl.envs.wrappers import DefaultGymWrapper


def get_environment(env_config: SimpleNamespace) -> DefaultGymWrapper:
    """
    Gets an environment.

    :param env_config: the environment configration file
    :type env_config: SimpleNamespace
    :return: the environment
    :rtype: DefaultGymWrapper

    """
    assert (
        env_config.env_type in VALID_ENV_TYPE
    ), f"{env_config.env_type} is not supported (one of {VALID_ENV_TYPE})"

    if env_config.env_type == CONST_GYM:
        import gymnasium as gym

        env = gym.make(env_config.env_name, **vars(env_config.env_kwargs))
    else:
        raise NotImplementedError

    env = DefaultGymWrapper(env)

    return env


register(
    id="ParameterizedPendulum-v0",
    entry_point="jaxl.envs.classic_control.pendulum:ParameterizedPendulumEnv",
    max_episode_steps=200,
    kwargs={},
)
register(
    id="DMCCheetah-v0",
    entry_point="jaxl.envs.dmc.cheetah:CheetahEnv",
    kwargs={
        "parameter_config_path": os.path.join(
            os.path.dirname(__file__), "dmc/configs/cheetah.json"
        )
    },
)
register(
    id="DMCWalker-v0",
    entry_point="jaxl.envs.dmc.walker:WalkerEnv",
    kwargs={
        "parameter_config_path": os.path.join(
            os.path.dirname(__file__), "dmc/configs/walker.json"
        )
    },
)
register(
    id="DMCCartpole-v0",
    entry_point="jaxl.envs.dmc.cartpole:CartpoleEnv",
    kwargs={
        "parameter_config_path": os.path.join(
            os.path.dirname(__file__), "dmc/configs/cartpole.json"
        )
    },
)
register(
    id="ParameterizedFrozenLake-v0",
    entry_point="jaxl.envs.toy_text.frozen_lake:FrozenLakeEnv",
    max_episode_steps=200,
    kwargs={},
)
