"""Functions to calculate analytic NNGP and NTKGP posterior predictive

This library is largely a stripped down version of `neural_tangents.predict`,
that only calculates analytic GP posteriors for NNGP and NTK prior kernels. The
key differences to neural_tangents.predict are (as of 31st May 2020):

1)  `get="ntk"` corresponds to the NTKGP posterior, not Eq.(16) in
    https://arxiv.org/abs/1902.06720. `get="nngp"` still corresponds to NNGP
    posterior.

2)  The arg `diag_reg` is directly the regularization strength for
    `predict.__add_diagonal_regularizer`. This allows modelling of output noise.

3) `predict.gp_inference` returns test predictive standard deviations instead of
    covariance matrix. Mean and standard deviation outputs are flattened.
"""

from jax.api import jit
import jax.numpy as np
import jax.scipy as sp

from jax.tree_util import tree_all
from jax.tree_util import tree_map

from neural_tangents.utils.utils import canonicalize_get
from neural_tangents.utils.utils import get_namedtuple

import collections

Gaussian = collections.namedtuple('Gaussian', 'mean standard_deviation')

def _get_matrices(kernel_fn, x_train, x_test, get, compute_cov):
  get = _get_dependency(get, compute_cov)
  kdd = kernel_fn(x_train, None, get)
  ktd = kernel_fn(x_test, x_train, get)
  if compute_cov:
    ktt = kernel_fn(x_test, x_test, get)
  else:
    ktt = None
  return kdd, ktd, ktt

## Utility functions
def _get_dependency(get, compute_cov):
  """Figure out dependency for get."""
  _, get = canonicalize_get(get)
  for g in get:
    if g not in ['nngp', 'ntk']:
      raise NotImplementedError(
          'Can only get either "nngp" or "ntk" predictions, got %s.' % g)
  get_dependency = ()
  if 'nngp' in get or ('ntk' in get and compute_cov):
    get_dependency += ('nngp',)
  if 'ntk' in get:
    get_dependency += ('ntk',)
  return get_dependency

def _add_diagonal_regularizer(covariance, diag_reg=0.):
  dimension = covariance.shape[0]
  return covariance + diag_reg * np.eye(dimension)

def _inv_operator(g_dd, diag_reg=0.0):
  g_dd_plus_reg = _add_diagonal_regularizer(g_dd, diag_reg)
  return lambda vec: sp.linalg.solve(g_dd_plus_reg, vec, sym_pos=True)


def _make_flatten_uflatten(g_td, y_train):
  """Create the flatten and unflatten utilities."""
  output_dimension = y_train.shape[-1]

  def fl(fx):
    """Flatten outputs."""
    return np.reshape(fx, (-1,))

  def ufl(fx):
    """Unflatten outputs."""
    return np.reshape(fx, (-1, output_dimension))

  if y_train.size > g_td.shape[-1]:
    out_dim, ragged = divmod(y_train.size, g_td.shape[-1])
    if ragged or out_dim != output_dimension:
      raise ValueError('The batch size of `y_train` must be the same as the'
                       ' last dimension of `g_td`')
    fl = lambda x: x
    ufl = lambda x: x
  return fl, ufl

def _mean_prediction(op, g_td, y_train):
  """Compute the mean prediction of a Gaussian process.
  Args:
    op: Some vector operator that projects the data along the relevant
      directions, op(vec, dt) = M^{-1} @ vec
    g_td: A kernel relating training data with test data. The kernel should be
      an `np.ndarray` of shape [n_test * output_dim, n_train * output_dim] or
      [n_test, n_train].
    y_train: An `np.ndarray` of shape [n_train, output_dim] of targets for the
      training data.
  Returns:
    The mean prediction of the GP. `g_td @ op @ y_train`.
  """
  fl, ufl = _make_flatten_uflatten(g_td, y_train)

  mean_pred = op(fl(y_train))
  mean_pred = np.dot(g_td, mean_pred)
  return ufl(mean_pred)

def _posterior_std(op, g_td, g_tt, output_noise_var):
  """Computes the test posterior standard deviation (with output noise) for nngp
     or ntkgp.
  """
  # op(vec) = (K + diag_reg * I)^{-1} @ vec for nngp or
  # op(vec) = (\Theta + diag_reg * I)^{-1} @ vec for ntkgp
  cov = op(np.transpose(g_td))
  pred_var = np.diag(g_tt - np.dot(g_td, cov)) + output_noise_var
  return np.sqrt(pred_var)

def _arr_is_on_cpu(x):
  # Utility function from neural_tangents
  if hasattr(x, 'device_buffer'):
    return 'CPU' in str(x.device_buffer.device())

  if isinstance(x, np.ndarray):
    return True

  raise NotImplementedError(type(x))


def _is_on_cpu(x):
  # Utility function from neural_tangents
  return tree_all(tree_map(_arr_is_on_cpu, x))

def gp_inference(kernel_fn,
                 x_train,
                 y_train,
                 x_test,
                 get,
                 diag_reg=0.,
                 compute_cov=True):
  """Compute the mean and standard deviation of the `posterior` of NNGP & NTKGP.

  Args:
    kernel_fn: A kernel function that computes NNGP and NTK.
    x_train: A `np.ndarray`, representing the training data.
    y_train: A `np.ndarray`, representing the labels of the training data.
    x_test: A `np.ndarray`, representing the test data.
    get: string, the mode of the Gaussian process, either "nngp" or "ntk", or a
      tuple, or None. If `None` then both `nngp` and `ntk` predictions are
      returned.
    diag_reg: A float, representing the output noise variance.
    compute_cov: A boolean. If `True` computing both `mean` and `variance` and
      only `mean` otherwise.
  Returns:
    Either a Gaussian(`mean`, `standard deviation`) namedtuple or `mean` of the
    GP posterior.
  """
  if get is None:
    get = ('nngp', 'ntk')
  kdd, ktd, ktt = _get_matrices(kernel_fn, x_train, x_test, get, compute_cov)
  gp_inference_mat = (_gp_inference_mat_jit_cpu if _is_on_cpu(kdd) else
                      _gp_inference_mat_jit)
  return gp_inference_mat(kdd, ktd, ktt, y_train, get, diag_reg)

@get_namedtuple('Gaussians')
def _gp_inference_mat(kdd,
                      ktd,
                      ktt,
                      y_train,
                      get,
                      diag_reg=0.):
  """Compute the mean and standard deviation of the `posterior` of NNGP & NTKGP.

  Args:
    kdd: A train-train `Kernel` namedtuple.
    ktd: A test-train `Kernel` namedtuple.
    ktt: A test-test `Kernel` namedtuple.
    y_train: A `np.ndarray`, representing the train targets.
    get: string, the mode of the Gaussian process, either "nngp" or "ntk", or a
      tuple, or `None`. If `None` then both `nngp` and `ntk` predictions are
      returned.
    diag_reg: A float, representing the strength of the regularization.

  Returns:
    Either a Gaussian(`mean`, `standard deviation`) namedtuple or `mean` of the
    GP posterior.
  """
  out = {}
  if get is None:
    get = ('nngp', 'ntk')
  if 'nngp' in get:
    op = _inv_operator(kdd.nngp, diag_reg)
    pred_mean = _mean_prediction(op, ktd.nngp, y_train)
    pred_mean = pred_mean.reshape(-1,)
    if ktt is not None:
      pred_std = _posterior_std(op, ktd.nngp, ktt.nngp, diag_reg)
    out['nngp'] = (
        Gaussian(pred_mean, pred_std) if ktt is not None else pred_mean)

  if 'ntk' in get:
    op = _inv_operator(kdd.ntk, diag_reg)
    pred_mean = _mean_prediction(op, ktd.ntk, y_train)
    pred_mean = pred_mean.reshape(-1,)
    if ktt is not None:
      pred_std = _posterior_std(op, ktd.ntk, ktt.ntk, diag_reg)
    out['ntk'] = (Gaussian(pred_mean, pred_std) if ktt is not None else pred_mean)

  return out

_gp_inference_mat_jit = jit(_gp_inference_mat, static_argnums=(4,))

_gp_inference_mat_jit_cpu = jit(_gp_inference_mat, static_argnums=(4,),
                                backend='cpu')
