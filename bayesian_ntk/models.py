from neural_tangents import stax
from neural_tangents.stax import Dense
from jax import jit

def homoscedastic_model(
    W_std,
    b_std,
    width,
    depth,
    parameterization
):

    act = stax.Erf
    layers_list = [Dense(width, W_std, b_std, parameterization)]

    def layer_block():
        return stax.serial(act(), Dense(width, W_std, b_std, parameterization))

    for i in range(depth - 1):
        layers_list += [layer_block()]

    layers_list += [act()]
    layers_list += [Dense(1, W_std, b_std, parameterization)]

    init_fn, apply_fn, kernel_fn = stax.serial(*layers_list)

    apply_fn = jit(apply_fn)

    return init_fn, apply_fn, kernel_fn
