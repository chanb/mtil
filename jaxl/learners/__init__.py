from types import SimpleNamespace

from jaxl.constants import *
from jaxl.learners.learner import Learner
from jaxl.learners.bc import BC
from jaxl.learners.mtbc import MTBC
from jaxl.learners.ppo import PPO


def get_rl_learner(
    learner_config: SimpleNamespace,
    model_config: SimpleNamespace,
    optimizer_config: SimpleNamespace,
) -> Learner:
    """
    Gets reinforcement learning learner.

    :param learner_config: the learner configuration
    :param model_config: the model configuration
    :param optimizer_config: the optimizer configuration
    :type learner_config: SimpleNamespace
    :type model_config: SimpleNamespace
    :type optimizer_config: SimpleNamespace
    :return: the reinforcement learning learner
    :rtype: Learner

    """
    assert (
        learner_config.learner in VALID_RL_LEARNER
    ), f"{learner_config.learner} is not supported (one of {VALID_RL_LEARNER})"
    if learner_config.learner == CONST_PPO:
        learner_constructor = PPO
    else:
        raise NotImplementedError

    return learner_constructor(learner_config, model_config, optimizer_config)


def get_il_learner(
    learner_config: SimpleNamespace,
    model_config: SimpleNamespace,
    optimizer_config: SimpleNamespace,
) -> Learner:
    """
    Gets imitation learning learner.

    :param learner_config: the learner configuration
    :param model_config: the model configuration
    :param optimizer_config: the optimizer configuration
    :type learner_config: SimpleNamespace
    :type model_config: SimpleNamespace
    :type optimizer_config: SimpleNamespace
    :return: the imitation learning learner
    :rtype: Learner

    """
    assert (
        learner_config.learner in VALID_IL_LEARNER
    ), f"{learner_config.learner} is not supported (one of {VALID_IL_LEARNER})"
    if learner_config.learner == CONST_BC:
        learner_constructor = BC
    elif learner_config.learner == CONST_MTBC:
        learner_constructor = MTBC
    else:
        raise NotImplementedError

    return learner_constructor(learner_config, model_config, optimizer_config)
