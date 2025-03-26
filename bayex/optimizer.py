from functools import partial
from typing import Dict, NamedTuple, Union

import jax
import jax.numpy as jnp
import jax.random as jr
import jax.tree as jt
import numpy as np

import bayex.acq as boacq
from bayex.gp import GPParams, GPState, gp_optimize_mll


class OptimizerState(NamedTuple):
    params_dict: dict
    ys: Union[jax.Array, np.ndarray]
    best_score: float
    best_params: dict
    mask: jax.Array
    gp_state: GPState


ParamsDict = Dict[str, jax.Array]


class Optimizer:
    """
    A Bayesian optimization class for optimizing expensive-to-evaluate functions.

    Attributes
    ----------
    domain : dict
        A dictionary defining the domain of the parameters to optimize. Each entry specifies the type and domain of a parameter.
    acq : str
        The acquisition function to use. Supported values are 'EI' for Expected Improvement, 'PI' for Probability of Improvement, 'UCB' for Upper Confidence Bound, and 'LCB' for Lower Confidence Bound.
    maximize : bool
        If True, the optimizer seeks to maximize the objective function. If False, it minimizes the function.

    Methods
    -------
    init(self, ys, params):
        Initializes the optimizer state with initial observations and corresponding parameters.

    sample(self, key, opt_state, size=1000, has_prior=False):
        Samples new parameters based on the current optimizer state and acquisition function.

    fit(self, opt_state, y, new_params):
        Updates the optimizer state with a new observation.
    """

    def __init__(self, domain, acq="EI", maximize=False):
        self.domain = domain
        best_fn = jnp.max if maximize else jnp.min
        self.initial = -jnp.inf if maximize else jnp.inf
        self.best_fn = best_fn
        self.best_params_fn = jnp.argmax if maximize else jnp.argmin

        if acq == "EI":
            self.acq = jax.jit(boacq.expected_improvement)
        elif acq == "PI":
            self.acq = jax.jit(boacq.probability_improvement)
        elif acq == "UCB":
            self.acq = jax.jit(boacq.upper_confidence_bounds)
        elif acq == "LCB":
            self.acq = jax.jit(boacq.lower_confidence_bounds)
        else:
            raise ValueError(f"Acquisition function {acq} is not implemented")

    def init(self, ys, params_dict: ParamsDict) -> OptimizerState:
        """
        Initializes the optimizer state with initial observations and corresponding parameters.

        Parameters
        ----------
        ys : Union[jax.Array, np.ndarray]
            The initial set of objective function values corresponding to the initial parameters.
        params_dict : dict
            A dictionary of the initial parameters. Each key should match a key in the domain, and the value should be an array of parameter values.

        Returns
        -------
        OptimizerState
            The initialized state of the optimizer, including the best score and parameters found so far.
        """
        # Create a padded jax array for each parameter and each score.
        # In order to keep jax compilations at a bay.
        num_entries = len(ys)
        pad_value = int(np.ceil(len(ys) / 10) * 10)

        # Convert to jax arrays if they are not already
        ys = jnp.asarray(ys)
        params_dict = jt.map(lambda x: jnp.asarray(x), params_dict)

        # Define padded arrays for the inputs and the outputs
        mask = jnp.zeros(shape=(pad_value,), dtype=jnp.bool_).at[:num_entries].set(True)
        ys = jnp.zeros(shape=(pad_value,), dtype=ys.dtype).at[:num_entries].set(ys)

        _params = {}
        for key, entries in params_dict.items():
            assert key in self.domain, f"Parameter {key} is not in the domain"

            # Get dtype from the domain and create a padded array
            values = jnp.zeros(shape=(pad_value,), dtype=self.domain[key].dtype)
            values = values.at[:num_entries].set(entries)
            _params[key] = values

        # From the given obs, find the best and return the initial optimizer state.
        best_score = float(self.best_fn(ys[mask]))
        best_params_idx = self.best_params_fn(ys[mask])
        best_params = jt.map(lambda p: p[mask][best_params_idx], _params)

        # Initialize the GP state: kernel params and sgd state
        gpparams = GPParams(
            noise=jnp.full((1, 1), -5.0),
            amplitude=jnp.zeros((1, 1)),
            lengthscale=jnp.zeros((1, len(_params))),
        )
        momentums = jt.map(jnp.zeros_like, gpparams)
        scales = jt.map(jnp.ones_like, gpparams)
        gp_state = GPState(gpparams, momentums, scales)

        # Fit to the current observations
        xs = jnp.stack(
            [self.domain[key].transform(_params[key]) for key in _params],
            axis=1,
        )
        gp_state = gp_optimize_mll(xs, ys, mask=mask, state=gp_state)

        opt_state = OptimizerState(
            params_dict=_params,
            ys=ys,
            best_score=best_score,
            best_params=best_params,
            mask=mask,
            gp_state=gp_state,
        )

        return opt_state

    def sample(self, key, opt_state, size=1000, has_prior=False) -> ParamsDict:
        """
        Samples new parameters based on the acquisition function and current state of the optimizer.

        Parameters
        ----------
        key : jax.random.PRNGKey
            A PRNGKey used for random number generation in JAX.
        opt_state : OptimizerState
            The current state of the optimizer.
        size : int, optional
            The number of samples to generate. Defaults to 1000.
        has_prior : bool, optional
            If True, includes prior mean and standard deviation in the return values. Defaults to False.

        Returns
        -------
        dict
            A dictionary of sampled parameters that potentially improve the objective function.
        tuple, optional
            A tuple of arrays (means, stds) representing the prior mean and standard deviation of
            the sampled parameters. Only returned if has_prior is True.
        """
        p_names = opt_state.params_dict.keys()
        keys = jr.split(key, len(p_names))

        # x_star for GP conditioning: sample over entire domain, eval acq at each point
        samples_dict = {
            p_name: self.domain[p_name].sample(key, (size,))
            for p_name, key in zip(p_names, keys)
        }  # {"x": (1000,)}

        xs_star = jnp.stack(list(samples_dict.values()), axis=1)  # (1000,1)

        # current obs that defines GP function prior
        ys = opt_state.ys
        xs = jnp.stack(
            [
                self.domain[p_name].transform(params)
                for p_name, params in opt_state.params_dict.items()
            ],
            axis=1,
        )  #  (10,1)

        mask = opt_state.mask
        gpparams = opt_state.gp_state.params

        # Use the acquisition function to find the best parameters
        zs, (means, stds) = self.acq(xs_star, xs, ys, mask, gpparams)
        idx = jnp.argmax(zs)
        best_params = jt.map(lambda d: d[idx], samples_dict)
        if has_prior:
            return best_params, (xs_star, means, stds)
        return best_params

    def fit(self, opt_state, y_new: float, new_params: ParamsDict) -> OptimizerState:
        """
        Updates the optimizer state with a new observation.

        Parameters
        ----------
        opt_state : OptimizerState
            The current state of the optimizer.
        y : float
            The objective function value for the new observation.
        new_params : dict
            The parameters corresponding to the new observation.

        Returns
        -------
        OptimizerState
            The updated state of the optimizer including the new observation.
        """
        opt_state = self._expand_buffer(opt_state)  # Prompts recompilation
        opt_state = self._fit(opt_state, y_new, new_params)
        return opt_state

    def _expand_buffer(self, opt_state) -> OptimizerState:
        """
        If the buffer is full, double the buffer size. Otherwise, do nothing
        """
        n_points = jnp.sum(opt_state.mask)

        if n_points == len(opt_state.mask):
            pad_value = int(np.ceil(len(opt_state.mask) * 2 / 10) * 10)
            diff = pad_value - len(opt_state.mask)
            mask = jnp.pad(opt_state.mask, (0, diff))
            ys = jnp.pad(opt_state.ys, (0, diff))
            params_dict = {}
            for key in opt_state.params_dict:
                params_dict[key] = jnp.pad(opt_state.params_dict[key], (0, diff))
            print(f"Expanding buffer: {n_points} -> {pad_value}")
        else:
            mask = opt_state.mask
            ys = opt_state.ys
            params_dict = opt_state.params_dict

        opt_state = opt_state._replace(
            mask=mask,
            ys=ys,
            params_dict=params_dict,
        )

        return opt_state

    @partial(jax.jit, static_argnums=(0,))
    def _fit(self, opt_state, y_new: float, new_params: ParamsDict) -> OptimizerState:
        """
        1. To include new obs: update the buffer's pointer, mask, and dataset
        2. Fit the GP to the new dataset
        3. Update the best score and parameters
        """
        last_mask = jnp.arange(len(opt_state.mask)) == jnp.argmin(opt_state.mask)
        mask = jnp.asarray(jnp.where(last_mask, True, opt_state.mask))
        ys = jnp.where(last_mask, y_new, opt_state.ys)
        params_dict = jt.map(
            lambda p_old, p_new: jnp.where(last_mask, p_new, p_old),
            opt_state.params_dict,
            new_params,
        )

        xs = jnp.stack(
            [self.domain[k].transform(v) for k, v in params_dict.items()],
            axis=1,
        )
        gp_state = gp_optimize_mll(xs, ys, mask=mask, state=opt_state.gp_state)

        best_score = self.best_fn(ys, where=mask, initial=self.initial)
        best_params_idx = self.best_params_fn(jnp.where(mask, ys, self.initial))
        best_params_dict = jt.map(lambda x: x[best_params_idx], params_dict)

        opt_state = OptimizerState(
            params_dict=params_dict,
            ys=ys,
            best_score=best_score,
            best_params=best_params_dict,
            mask=mask,
            gp_state=gp_state,
        )
        return opt_state
