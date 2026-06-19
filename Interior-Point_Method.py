import tensorflow as tf

# Define a simple objective function: f(x) = x_1^2 + x_2^2
def objective_fn(x):
    return tf.reduce_sum(tf.square(x))

# Define inequality constraints: g(x) >= 0
# Constraint 1: x_1 + x_2 - 2 >= 0
# Constraint 2: x_1 >= 0.5
def constraints_fn(x):
    g1 = x[0] + x[1] - 2.0
    g2 = x[0] - 0.5
    return tf.stack([g1, g2])

# Interior-Point Barrier Objective
def barrier_objective(x, t):
    obj = objective_fn(x)
    g = constraints_fn(x)
    
    # Penalize if constraints are violated or near-violation
    # tf.math.log handles element-wise calculation
    barrier = -tf.reduce_sum(tf.math.log(g))
    
    return obj + (1.0 / t) * barrier

# Solver Parameters
# Note: Initial point MUST be strictly inside the feasible region (interior)
x = tf.Variable([1.5, 1.5], dtype=tf.float32) 
t = 1.0          # Initial barrier parameter
mu = 2.0         # Scaling factor to increase t
tolerance = 1e-5 # Convergence tolerance
optimizer = tf.optimizers.Adam(learning_rate=0.01)

print("Starting Interior-Point Optimization Loop...\n")

# Outer loop: Sharpens the barrier by increasing t
for outer_step in range(15):
    # Inner loop: Solve the unconstrained barrier problem for current t
    for inner_step in range(200):
        with tf.GradientTape() as tape:
            loss = barrier_objective(x, t)
        
        grads = tape.gradient(loss, [x])
        optimizer.apply_gradients(zip(grads, [x]))
        
    print(f"Outer Step {outer_step:02d} | t = {t:6.1f} | x = [{x[0].numpy():.4f}, {x[1].numpy():.4f}] | Constraints: {constraints_fn(x).numpy()}")
    
    # Check stopping criterion (duality gap proxy)
    if (2.0 / t) < tolerance:
        break
        
    t *= mu # Scale up t for the next sequence

print(f"\nFinal Optimized Solution: {x.numpy()}")
