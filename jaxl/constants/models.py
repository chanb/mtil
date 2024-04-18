CONST_ENCODER_PREDICTOR = "encoder_predictor"
CONST_ENSEMBLE = "ensemble"
CONST_MLP = "mlp"

VALID_ARCHITECTURE = [CONST_ENCODER_PREDICTOR, CONST_ENSEMBLE, CONST_MLP]


CONST_DETERMINISTIC = "deterministic"
CONST_GAUSSIAN = "gaussian"
CONST_SOFTMAX = "softmax"
VALID_POLICY_DISTRIBUTION = [
    CONST_DETERMINISTIC,
    CONST_GAUSSIAN,
    CONST_SOFTMAX,
]

CONST_ENCODER = "encoder"
CONST_MODEL = "model"
CONST_POLICY = "policy"
CONST_PREDICTOR = "predictor"
CONST_REPRESENTATION = "representation"
CONST_VF = "vf"

CONST_MIN_STD = "min_std"
CONST_TEMPERATURE = "temperature"

DEFAULT_MIN_STD = 1e-7
DEFAULT_TEMPERATURE = 1.0

CONST_IDENTITY = "identity"
CONST_RELU = "relu"
CONST_TANH = "tanh"
VALID_ACTIVATION = [CONST_IDENTITY, CONST_RELU, CONST_TANH]
