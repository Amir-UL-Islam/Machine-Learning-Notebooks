# importing library
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import binom
import statistics

# This makes plots appear in the notebook
get_ipython().run_line_magic("matplotlib", " inline")

# Number of experiments
n = 10

# Probability of success
p = 0.5

# Array of probable outcomes of number of heads
x = range(0,11)

# Get probabilities
prob = binom.pmf(x, n, p)

# Set properties of the plot
fig, binom_plot = plt.subplots(figsize=(10,8))
binom_plot.set_xlabel("Number of Heads",fontsize=16)
binom_plot.set_ylabel("Probability",fontsize=16)
binom_plot.vlines(x, 0, prob, colors='r', lw=5, alpha=0.5)

# Plot the graph
binom_plot.plot(x, prob, 'ro')
plt.show()


binom.rvs(8, 0.5, 1)


binom.pmf(7, 20, 0.5)


binom.cdf(7, 20, 0.5)


import scipy.stats

# Generate 100 random numbers between 0 and 20
Rv = np.linspace(0.0, 20.0, 100)

#calculate the normal distribution
nd = scipy.stats.norm.pdf(Rv,Rv.mean())

# Set properties of the plot
fig, nd_plot = plt.subplots(figsize=(10,8))
nd_plot.set_xlabel("Values of Random Variable", fontsize=16)
nd_plot.set_ylabel("Normal Distribution Values", fontsize=16)

# Plot the graph
nd_plot.plot(Rv,nd)
plt.show()


# Import expon from scipy.stats
from scipy.stats import expon

# Print probability response takes 3-4 hours
print(expon.cdf(4, scale=2.5) - expon.cdf(3, scale=2.5))
