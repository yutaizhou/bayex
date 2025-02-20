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
params = {"x": [0.0, 0.5, 1.0]}
ys = [f(x) for x in params["x"]]
opt_state = optimizer.init(ys, params)

# Sample new points using Jax PRNG approach.
ori_key = jax.random.key(42)
for step in range(20):
    key = jax.random.fold_in(ori_key, step)
    new_params = optimizer.sample(key, opt_state)
    y_new = f(**new_params)
    opt_state = optimizer.fit(opt_state, y_new, new_params)
    print(opt_state.best_score)

fig, ax = plt.subplots()
x_plt = jnp.linspace(0, 2, 100)
ax.plot(x_plt, f(x_plt))
ax.scatter(opt_state.best_params["x"], opt_state.best_score, s=100, color="red")
plt.show()
