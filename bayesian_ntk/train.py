import jax.numpy as np
from jax import random
from jax import jit, grad
from jax.experimental import optimizers
from .models import homoscedastic_model
from .train_utils import fetch_new_predict_fn, fetch_regularisation_fn

def train_model(
    key,
    train_method,
    train_data,
    test_data,
    activation,
    parameterization,
    learning_rate,
    training_steps,
    noise_scale,
    W_std,
    b_std,
    width,
    depth,
):
    """Train a single baselearner model and calculate test predictions.

    Args:
        key: jax.random.PRNGKey instance
        train_method (str): Ensemble method
        train_data: Tuple of training inputs and targets
        test_data: Tuple of test inputs and targets
        activation (str): Activation function
        parameterization (str): Parameterization
        learning_rate (float): Learning rate
        training_steps (int): Number of gradient updates
        noise_scale (float): output noise standard deviation
        W_std (float): Weight standard deviation
        b_std (float): Bias standard deviation
        width (int): Hidden layer width
        depth (int): Number of hidden layers

    Returns:
        Model predictions on `test_data`
    """

    opt_init, opt_update, get_params = optimizers.sgd(learning_rate)
    opt_update = jit(opt_update)

    # get model
    init_fn, predict_fn, _ = homoscedastic_model(
         W_std,
         b_std,
         width,
         depth,
         activation,
         parameterization
     )

    # initialise initial parameters
    _, init_params = init_fn(key, (-1, 1))

    # initialise auxiliary (non-trainable) parameters for JVPs in NTKGP methods
    # or extra forward pass in RP-fn method
    key, subkey = random.split(key)
    _, aux_params = init_fn(subkey, (-1, 1))

    # define `train_method` dependent modified forward pass and regularisation
    new_predict_fn = fetch_new_predict_fn(
        predict_fn,
        train_method,
        init_params,
        aux_params
    )
    new_predict_fn = jit(new_predict_fn)

    reg_fn = fetch_regularisation_fn(
        train_method,
        init_params,
        parameterization,
        W_std,
        b_std
    )

    # define loss function
    def mse_loss(params, x, y):
        preds = new_predict_fn(params, x)
        return np.mean((preds - y) ** 2)

    train_size = len(train_data.inputs)
    reg_coef = noise_scale**2 / train_size

    @jit
    def loss(params, x, y):
        return 0.5 * mse_loss(params, x, y) + 0.5 * reg_coef * reg_fn(params)

    @jit
    def grad_loss(state, x, y):
        params = get_params(state)
        return grad(loss)(params, x, y)

    opt_state = opt_init(init_params)

    for i in range(training_steps):
        opt_state = opt_update(i, grad_loss(opt_state, *train_data), opt_state)

    final_params = get_params(opt_state)
    fx_final_test = new_predict_fn(final_params, test_data.inputs)

    return fx_final_test
