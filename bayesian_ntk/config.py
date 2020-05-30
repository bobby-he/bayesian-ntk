from functools import partial
import collections
from types import SimpleNamespace

NOISE_SCALE = 1e-1
ENSEMBLE_SIZE = 20

Gaussian = collections.namedtuple('Gaussian', 'mean standard_deviation')

_model_configs = {
    "default": dict(
        W_std = 1.5,
        b_std = 0.05,
        width = 512,
        depth = 2,
        activation = 'erf'
    )
}

_train_configs = {
    "default": dict(
        learning_rate = 1e-3,
        training_steps = 50000,
        noise_scale = NOISE_SCALE,
        **_model_configs["default"]
    )
}

_data_configs = {
    "default": dict(
        train_points = 20,
        test_points = 50
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

method_input_dict = {
    "Deep ensemble": 'deep_ensemble',
    "RP-param": 'rand_prior_param',
    "RP-fn": 'rand_prior_fn',
    "NTKGP-param": 'ntkgp_param',
    "NTKGP-fn": 'ntkgp_fn',
    "NTKGP-lin": 'ntkgp_lin'
}
