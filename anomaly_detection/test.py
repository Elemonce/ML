import numpy as np

mu = 5
sigma = 2
x = np.array([4, 4])

def gaussian_prob(x, mu, sigma):
    coef = 1 / (np.sqrt(2 * np.pi) * sigma)
    exp = np.exp(-((x - mu)**2) / (2 * sigma**2))
    return coef * exp

# Apply for each element and take the product
probs = gaussian_prob(x, mu, sigma)
p = np.prod(probs)

print("Probability of x:", p)
