from jax import jvp
import jax.numpy as np
from jax.tree_util import tree_flatten, tree_unflatten

method_input_dct = {
    "Deep ensemble": 'deep_ensemble',
    "RP-param": 'rand_prior_param',
    "RP-fn": 'rand_prior_fn',
    "NTKGP-param": 'ntkgp_param',
    "NTKGP-fn": 'ntkgp_fn',
    "NTKGP-lin": 'ntkgp_lin'
}

def new_predict_fn(predict,
                   train_method,
                   params_0,
                   params_1,
                   reweight_cutoff = 2, # Should be 2 to cutoff the final layer weight and bias parameters
                   ):
    jvp_fn_0 = jvp_fn(predict, params_0)
    jvp_fn_1 = jvp_fn(predict, params_1)

    if train_method in ['deep_ensemble', 'rand_prior_param']:
        new_predict = predict

    elif train_method == 'ntkgp_lin':
        new_predict = lambda params, x: jvp_fn_1(params, x)

    elif train_method == 'ntkgp_param':
        params_1_zeroed = reweight_params(params_1, reweight_cutoff, before_factor = 1., after_factor = 0.)
        new_predict = lambda params, x: predict(params, x) + jvp_fn_0(params_1_zeroed, x)

    elif train_method == 'ntkgp_fn':
        reweighted_params_1 = reweight_params(params_1, reweight_cutoff, before_factor = np.sqrt(2), after_factor = 1.)
        new_predict = lambda params, x: predict(params, x) + jvp_fn_0(reweighted_params_1, x)

    elif train_method == 'rand_prior_fn':
        new_predict = lambda params, x: predict(params, x) + predict(params_1, x)

    else:
        raise ValueError('Train method {} not found.'.format(train_method))

    return new_predict


def regularisation_fn(train_method, init_params, parameterization, W_std, b_std):

    weight_decay = lambda params: l2_norm_sq(params, parameterization, W_std, b_std)
    init_params_decay = lambda params: l2_distance_sq(params, init_params, parameterization, W_std, b_std)

    if train_method in ['ntkgp_lin', 'ntkgp_param', 'rand_prior_param']:
        regularisation = init_params_decay
    elif train_method in ['ntkgp_fn', 'rand_prior_fn']:
        regularisation = weight_decay
    elif train_method == 'deep_ensemble':
        regularisation = lambda params: 0.
    else:
        raise ValueError('Train method {} not found.'.format(train_method))
    return regularisation

def jvp_fn(f, params):
    def f_lin(p, x):
        f_params_x, proj = jvp(lambda param: f(param, x), (params,), (p,))
        return proj
    return f_lin

def reweight_params(params, reweight_cutoff = 0, before_factor = 1., after_factor = 0.):
    # multiplies all but last reweight_cutoff params by before_factor,
    # and the rest by after_factor
    param_leaves, param_def = tree_flatten(params)
    len_leaves = len(param_leaves)
    for i in range(len_leaves - reweight_cutoff):
        param_leaves[i] *= before_factor
    for i in range(len_leaves - reweight_cutoff, len_leaves):
        param_leaves[i] *= after_factor
    return tree_unflatten(param_def, param_leaves)


def l2_norm_sq(tree, parameterization = 'ntk', W_std = 1., b_std = 0.05):
    leaves, _ = tree_flatten(tree)
    if parameterization == 'ntk':
        return sum(np.vdot(leaf, leaf) for leaf in leaves)
    elif parameterization =='standard':
        # NB this only works for feedforward rn, because of biases
        reg_list = []
        for leaf in leaves:
            assert leaf.ndim == 1 or leaf.ndim == 2
            if leaf.ndim == 1:
                reg_list.append(1/b_std**2)
            elif leaf.ndim == 2:
                reg_list.append(leaf.shape[0]/W_std**2)
        return sum(reg_factor * np.vdot(leaf, leaf) for reg_factor, leaf in zip(reg_list, leaves))

def l2_distance_sq(tree_1, tree_2, parameterization = 'ntk', W_std = 1., b_std = 0.05):
    leaves_1, _ = tree_flatten(tree_1)
    leaves_2, _ = tree_flatten(tree_2)
    if parameterization == 'ntk':
        return sum(np.vdot(leaf_1 - leaf_2, leaf_1 - leaf_2) \
                for leaf_1, leaf_2 in zip(leaves_1, leaves_2))
    elif parameterization =='standard':
        # NB this only works for feedforward rn, because of biases
        reg_list = []
        for leaf in leaves_1:
            assert leaf.ndim == 1 or leaf.ndim == 2
            if leaf.ndim == 1:
                reg_list.append(1/b_std**2)
            elif leaf.ndim == 2:
                reg_list.append(leaf.shape[0]/W_std**2)
        return sum(reg_factor * np.vdot(leaf_1 - leaf_2, leaf_1 - leaf_2) \
                for reg_factor, leaf_1, leaf_2 in zip(reg_list, leaves_1, leaves_2))
