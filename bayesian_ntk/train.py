# imports
import jax.numpy as np
from train_util import new_predict_fn, regularisation_fn
from jax import random
from jax import jit, grad
from jax.experimental import optimizers
from models import homoscedastic_model

def train_model(
    key,
    train_method,
    train,
    test,
    parameterization,
    learning_rate,
    training_steps,
    noise_scale,
    W_std,
    b_std,
    width,
    depth,
    ):

    train_losses = []
    test_losses = []
    init_fn, apply_fn, _ = homoscedastic_model(W_std, b_std, width, depth, parameterization)
    opt_init, opt_update, get_params = optimizers.sgd(learning_rate)
    opt_update = jit(opt_update)


    _, params_0 = init_fn(key, (-1, 1))
    key, subkey = random.split(key)
    _, params_1 = init_fn(subkey, (-1, 1))

    apply_new_fn = new_predict_fn(apply_fn, train_method, params_0, params_1)
    apply_new_fn = jit(apply_new_fn)
    opt_state = opt_init(params_0)
    init_params = get_params(opt_state)
    pre_reg_loss = lambda params, x, y: 0.5 * np.mean((apply_new_fn(params, x) - y) ** 2)
    regularisation = regularisation_fn(train_method, params_0, parameterization, W_std, b_std)

    train_size = len(train.inputs)
    loss = lambda params, x, y: pre_reg_loss(params, x, y) + 0.5 * noise_scale**2 * regularisation(params) / train_size
    loss = jit(loss)
    grad_loss = jit(lambda state, x, y: grad(loss)(get_params(state), x, y))

    for i in range(training_steps):
        opt_state = opt_update(i, grad_loss(opt_state, *train), opt_state)

    final_params = get_params(opt_state)
    fx_final_test = apply_new_fn(final_params, test.inputs)

    return fx_final_test
