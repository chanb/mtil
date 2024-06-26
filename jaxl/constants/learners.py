CONST_MLE = "mle"

VALID_REGRESSION_LEARNER = [CONST_MLE]

CONST_PGD = "pgd"

VALID_TRAIN_STEP_WRAPPER = [CONST_PGD]

CONST_PRE_PARAM_NORM = "pre_param_norm"
CONST_POST_PARAM_NORM = "post_param_norm"

CONST_PPO = "ppo"

VALID_RL_LEARNER = [CONST_PPO]

CONST_BC = "bc"
CONST_MTBC = "mtbc"
VALID_IL_LEARNER = [CONST_BC, CONST_MTBC]

CONST_PRE_PARAM_NORM = "pre_param_norm"
CONST_POST_PARAM_NORM = "post_param_norm"

CONST_REGULARIZATION = "regularization"
CONST_PARAM_NORM = "param_norm"
CONST_GRAD_NORM = "grad_norm"
CONST_STOP_UPDATE = "stop_update"

CONST_INPUT_RMS = "input_rms"
CONST_OBS_RMS = "obs_rms"
CONST_VALUE_RMS = "value_rms"

CONST_UPDATE_TIME = "update_time"
CONST_ROLLOUT_TIME = "rollout_time"

CONST_PI_LOSS_SETTING = "pi_loss_setting"
CONST_VF_LOSS_SETTING = "vf_loss_setting"

CONST_NUM_CLIPPED = "num_clipped"

CONST_REVERSE_KL = "reverse_kl"
CONST_CLIP = "clip"
VALID_PPO_OBJECTIVE = [CONST_CLIP, CONST_REVERSE_KL]
