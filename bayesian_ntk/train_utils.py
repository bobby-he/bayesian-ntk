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

def fetch_new_predict_fn(
     predict_fn,
     train_method,
     init_params,
     aux_params,
     reweight_cutoff = 2
 ):
    """Fetch modified forward pass function for different ensemble methods

    Args:
        predict_fn: Standard NN forward pass function
        train_method (str): Ensemble training method
        init_params (pytree): Initialised NN parameters
        aux_params (pytree): Auxiliary initialised NN parameters, for e.g. JVPs
        reweight_cutoff (int): Cutoff int for reweighting leaves of parameters
                               pytree (default 2)

            Default `reweight_cutoff = 2` because the last 2 leaves of params
            pytree are final layer weight and bias parameters in a homoscedastic
            model with readout final layer.

    Returns:
        `new_predict_fn`: Modified forward pass function
    """

    # define jvp functions expanded around init_params and aux_params
    init_jvp_fn = jvp_fn(predict_fn, init_params)
    aux_jvp_fn = jvp_fn(predict_fn, aux_params)

    if train_method in ['deep_ensemble', 'rand_prior_param']:
        new_predict_fn = predict_fn

    elif train_method == 'ntkgp_lin':
        new_predict_fn = lambda params, x: aux_jvp_fn(params, x)

    elif train_method == 'ntkgp_param':
        # aux_params_zeroed sets final layer parameters to zero
        aux_params_zeroed = reweight_params(
            aux_params,
            reweight_cutoff,
            before_cutoff_coef = 1.,
            after_cutoff_coef = 0.
        )
        new_predict_fn = lambda params, x: predict_fn(params, x) + init_jvp_fn(aux_params_zeroed, x)

    elif train_method == 'ntkgp_fn':
        # reweighted_aux_params multiplies all but last layer parameters by sqrt(2)
        reweighted_aux_params = reweight_params(
            aux_params,
            reweight_cutoff,
            before_cutoff_coef = np.sqrt(2),
            after_cutoff_coef = 1.
        )
        new_predict_fn = lambda params, x: predict_fn(params, x) + init_jvp_fn(reweighted_aux_params, x)

    elif train_method == 'rand_prior_fn':
        new_predict_fn = lambda params, x: predict_fn(params, x) + predict_fn(aux_params, x)

    else:
        raise ValueError('Train method {} not found.'.format(train_method))

    return new_predict_fn


def fetch_regularisation_fn(
    train_method,
    init_params,
    parameterization,
    W_std,
    b_std
):
    """Fetch regularisation function for different ensemble methods

    Args:
        train_method (str): Ensemble training method
        init_params (pytree): Initialised NN parameters
        parameterization (str): 'ntk' or 'standard'
        W_std (float): Weight standard deviation
        b_std (float): Bias standard deviation

    Returns:
        `regularisation_fn`: `Train_method` dependent regularisation function
    """

    weight_decay = lambda params: l2_norm_sq(params, parameterization, W_std, b_std)
    init_params_decay = lambda params: l2_distance_sq(params, init_params, parameterization, W_std, b_std)

    if train_method in ['ntkgp_lin', 'ntkgp_param', 'rand_prior_param']:
        regularisation_fn = init_params_decay
    elif train_method in ['ntkgp_fn', 'rand_prior_fn']:
        regularisation_fn = weight_decay
    elif train_method == 'deep_ensemble':
        # don't regularise standard deep ensembles (optional)
        regularisation_fn = lambda params: 0.
    else:
        raise ValueError('Train method {} not found.'.format(train_method))
    return regularisation_fn

def jvp_fn(f, primal_params):
    """JVP of `f` centered at `primal_params`

    Args:
        f: Standard forward pass functions
        primal_params (pytree): Parameters about which to take JVP

    Returns:
        `f_lin`: Function with inputs (tangent_params, x) and outputs JVP of f
                 at primals `primal_params` with tangents `tangent_params`
    """
    def f_lin(tangent_params, x):
        f_params_x, proj = jvp(lambda param: f(param, x), (primal_params,), (tangent_params,))
        return proj
    return f_lin

def reweight_params(
    params,
    reweight_cutoff = 0,
    before_cutoff_coef = 1.,
    after_cutoff_coef = 0.
):
    """Reweight layers of `params`

    Args:
        params (pytree): Parameter set to reweight
        reweight_cutoff (int):      Cutoff int for reweighting leaves of `params`
                                    pytree (default 0)
        before_cutoff_coef (float): Coefficient to multiply all `params` pytree
                                    leaves apart from the last `reweight_cutoff`
                                    leaves (default 1.)
        after_cutoff_coef (float):  Coefficient to multiply last `reweight_cutoff`
                                    leaves in `params` pytree by. (default 0.)

    Returns:
        reweighted_params (pytree): Reweighted version of `params`
    """
    param_leaves, param_def = tree_flatten(params)
    len_leaves = len(param_leaves)
    for i in range(len_leaves - reweight_cutoff):
        param_leaves[i] *= before_cutoff_coef
    for i in range(len_leaves - reweight_cutoff, len_leaves):
        param_leaves[i] *= after_cutoff_coef
    return tree_unflatten(param_def, param_leaves)


def l2_norm_sq(
    params_tree,
    parameterization = 'ntk',
    W_std = 1.,
    b_std = 0.05
):
    """Parameterization dependent reweighted L2 norm of `params_tree`

    Following Lemma 3 of https://arxiv.org/abs/1806.03335, we reweight weight
    and bias parameters according to their initialisation variances, which are
    `parameterization` dependent.

    Args:
        params_tree (pytree): Pytree of parameters to take L2 norm of
        parameterization (str): 'ntk' or 'standard'
        W_std (float): Weight standard deviation
        b_std (float): Bias standard deviation

    Returns:
        l2_norm (float): Weighted L2 norm of `params_tree`
    """
    leaves, _ = tree_flatten(params_tree)

    # In NTK parameterisation all parameters have variance 1: easy reweighting
    if parameterization == 'ntk':
        return sum(np.vdot(leaf, leaf) for leaf in leaves)

    # In standard parameterization, need to multiply by `N/W_std^2` for weights
    # and `1/b_std^2` for biases, where N is width.
    # NB this only works for stax.Dense MLPs right now.
    # This is extremely ugly and precarious to the arbitrary choice between the weight 
    # matrix and its transpose. 
    elif parameterization =='standard':
        reg_list = []
        for leaf in leaves:
            assert leaf.ndim == 1 or leaf.ndim == 2
            if leaf.ndim == 1:
                reg_list.append(1/b_std**2)
            elif leaf.ndim == 2:
                reg_list.append(leaf.shape[0]/W_std**2)
        return sum(reg_coef * np.vdot(leaf, leaf)
                for reg_coef, leaf in zip(reg_list, leaves))

def l2_distance_sq(
    params_tree_1,
    params_tree_2,
    parameterization = 'ntk',
    W_std = 1.,
    b_std = 0.05
):
    """Parameterization dependent reweighted L2 distance between `params_tree_1`
        and `params_tree_2`

    Args:
        params_tree_1 (pytree): Pytree of parameters
        params_tree_2 (pytree): Pytree of parameters
        parameterization (str): 'ntk' or 'standard'
        W_std (float): Weight standard deviation
        b_std (float): Bias standard deviation

    Returns:
        l2_distance (float): Weighted L2 distance between `params_tree_1` and
                             `params_tree_2`
    """
    leaves_1, _ = tree_flatten(params_tree_1)
    leaves_2, _ = tree_flatten(params_tree_2)

    # In NTK parameterisation all parameters have variance 1: easy reweighting
    if parameterization == 'ntk':
        return sum(np.vdot(leaf_1 - leaf_2, leaf_1 - leaf_2) \
                for leaf_1, leaf_2 in zip(leaves_1, leaves_2))

    # In standard parameterization, need to multiply by `N/W_std^2` for weights
    # and `1/b_std^2` for biases, where N is width.
    # NB this only works for stax.Dense MLPs right now.
    # This is extremely ugly and precarious to the arbitrary choice between the weight 
    # matrix and its transpose. 
    elif parameterization =='standard':
        reg_list = []
        for leaf in leaves_1:
            assert leaf.ndim == 1 or leaf.ndim == 2
            if leaf.ndim == 1:
                reg_list.append(1/b_std**2)
            elif leaf.ndim == 2:
                reg_list.append(leaf.shape[0]/W_std**2)
        return sum(reg_coef * np.vdot(leaf_1 - leaf_2, leaf_1 - leaf_2)
                for reg_coef, leaf_1, leaf_2 in zip(reg_list, leaves_1, leaves_2))
