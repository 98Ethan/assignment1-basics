
Observed Behaviors:

Learning Rate = 1e1 (10.0):

- Steady decay: Loss decreases smoothly from 0.93 → 0.13
- Stable convergence: Each iteration shows consistent improvement
- Good behavior: This learning rate works well for this problem

Learning Rate = 1e2 (100.0):

- Fast convergence: Loss drops dramatically and reaches near zero by iteration 5
- Very effective: Converges faster than lr=10, reaching practically zero loss
- Aggressive but stable: High learning rate but doesn't diverge

Learning Rate = 1e3 (1000.0):

- Divergence: Loss explodes dramatically after iteration 1
- Unstable: Goes from ~1.0 to 90+ trillion in just 10 iterations
- Too aggressive: Learning rate is too high, causing the optimizer to overshoot and become unstable

Deliverable Summary:

With learning rates 1e1, 1e2, and 1e3 over 10 iterations: The loss decays fastest and most stably with lr=1e2, decays steadily with lr=1e1,
but diverges (increases exponentially) with lr=1e3 due to the learning rate being too aggressive for stable optimization.