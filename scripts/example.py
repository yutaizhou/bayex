import jax
import jax.numpy as jnp
import jax.random as jr
import matplotlib.pyplot as plt

import bayex


def f(x):
    return -(1.4 - 3 * x) * jnp.sin(18 * x)


domain = {"x": bayex.domain.Real(0.0, 2.0)}
optimizer = bayex.Optimizer(domain=domain, maximize=True, acq="PI")

# Define some prior evaluations to initialise the GP.
params_dict = {"x": [0.0, 0.5, 1.0]}
ys = [f(x) for x in params_dict["x"]]
opt_state = optimizer.init(ys, params_dict)  # MLL fit to the initial points

# BayesOpt loop
ori_key = jr.key(42)
for step in range(20):
    key = jr.fold_in(ori_key, step)

    # sample new points by optimizing acq fn
    # evaluate the new points
    # fit the GP surrogate by maximizing MLL on new points
    x_new_dict = optimizer.sample(key, opt_state)
    y_new = f(**x_new_dict)
    opt_state = optimizer.fit(opt_state, y_new, x_new_dict)

    print(opt_state.best_score)

fig, ax = plt.subplots()
x_plt = jnp.linspace(0, 2, 100)
ax.plot(x_plt, f(x_plt))
ax.scatter(opt_state.best_params["x"], opt_state.best_score, s=100, color="red")
plt.show()
