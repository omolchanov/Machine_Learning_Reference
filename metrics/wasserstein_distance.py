"""
The Wasserstein distance (also called the Earth Mover’s Distance, EMD) is a way to measure how different
two probability distributions are.

Imagine two hills made of sand sitting on a table.
Each hill represents a probability distribution.

Now, suppose you want to reshape the first hill so it looks exactly like the second one. To do this, you’ll need to
move sand around:
If the hills are close together, you won’t need to move the sand very far → small distance.
If the hills are far apart, you’ll need to carry the sand a long way → large distance.

The Wasserstein distance tells you the minimum effort required to do this.
That’s why it’s also nicknamed Earth Mover’s Distance—it measures “how much dirt needs to be moved, and how far.”

It’s a way of saying how different two distributions are, taking into account both the size of the difference
and how far the probability mass has to shift.
"""

import numpy as np
from scipy.stats import wasserstein_distance
import matplotlib.pyplot as plt

# Imagine two sand piles (our "distributions")

# First pile: all the sand is at position 0
pile_A = np.array([-0.2, 0.2, 0.5, -0.4, -0.8])

# Second pile: all the sand is at position 1
pile_B = np.array([1, 1, 1, 1, 1])

# The Wasserstein distance tells us: "How much effort is needed to move pile A so it becomes pile B?"
distance = wasserstein_distance(pile_A, pile_B)
print(f"Wasserstein distance: %.3f" % distance)

plt.hist(pile_A, alpha=0.5, bins=5, label="Pile A (all at 0)")
plt.hist(pile_B, alpha=0.5, bins=5, label="Pile B (all at 1)")
plt.legend()
plt.title("Two sand piles (distributions)")
plt.show()
