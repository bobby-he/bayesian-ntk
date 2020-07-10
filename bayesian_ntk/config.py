"""
Default hyperparameter values
"""
from functools import partial

NOISE_SCALE = 1e-1
ENSEMBLE_SIZE = 20

_model_configs = {
    "default": dict(
        W_std = 1.5,                # Weight standard deviation
        b_std = 0.05,               # Bias standard deviation
        width = 512,                # Hidden layer width
        depth = 2,                  # Number of hidden layers
        activation = 'erf'          # Activation function
    )
}

_train_configs = {
    "default": dict(
        learning_rate = 1e-3,       # Learning rate
        training_steps = 50000,     # Number of gradient updates
        noise_scale = NOISE_SCALE,  # Observation noise standard deviation
        **_model_configs["default"]
    )
}

_data_configs = {
    "default": dict(
        train_points = 20,          # Training set size
        test_points = 50            # Test set size
    )
}

def get_model_config(name, _cfg_dct=_model_configs):
    try:
        return _cfg_dct[name.lower()]
    except KeyError:
        raise ValueError(
            f"Could not find config {name} in config.py."
            f"Available configs are: {list(_cfg_dct.keys())}"
        )

get_train_config = partial(get_model_config, _cfg_dct=_train_configs)
get_data_config = partial(get_model_config, _cfg_dct=_data_configs)
