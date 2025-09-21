"""
The Kullback–Leibler divergence (KL divergence) is a way to measure how different one probability distribution is
from another.
The Kullback–Leibler divergence (often just called KL divergence) is like a measure of how surprised you’d be if you
believed the world followed one story, but in reality it follows another.

Imagine you think a coin is fair (50/50 heads and tails). That’s your model Q.
But in reality, the coin is rigged and lands 90% heads, 10% tails. That’s the true distribution P.
Every time you guess using your fair-coin assumption, you’re “more surprised” when the coin keeps showing heads so
often.

KL divergence quantifies that extra surprise or extra information you need when your assumption doesn’t match reality.

- If your guess matches reality perfectly → KL = 0 (no wasted surprise).
- The worse your guess, the bigger KL gets.
- It’s not symmetric: being wrong in one direction might feel different than being wrong the other way.

KL divergence is a way to measure how inefficient it is to think the world works like Q when it actually works like P.
"""

import numpy as np
from scipy.stats import entropy

# Define two probability distributions (must sum to 1)
P = np.array([0.9, 0.1])   # True distribution
Q = np.array([0.5, 0.5])   # Approximating distribution

# KL divergence: D_KL(P || Q)
kl_pq = entropy(P, Q)   # default is base e (nats)

# KL divergence: D_KL(Q || P)
kl_qp = entropy(Q, P)

"""
D_KL(P || Q) = 0.368
This is the "extra surprise" you get if you assume Q (fair coin) but the world actually follows P (rigged coin).
Since this number is > 0, it means Q is not a great model, but it’s not infinitely bad either.
You’re wasting ~0.368 nats of information per coin flip. (If you want in bits, divide by ln(2) ≈ 0.53 bits.)

D_KL(Q || P) = 0.511
This flips the roles: now we assume the true coin is fair but reality is rigged.
The mismatch feels bigger here, because thinking it’s fair when it’s actually biased causes more wasted “surprise”.

If you’re comparing models to reality, you usually compute D_KL(P || Q) with P = true distribution (from data), Q=model.
Smaller = better model fit.
"""

print("D_KL(P || Q): %.3f" % kl_pq)
print("D_KL(Q || P): %.3f" % kl_qp)
