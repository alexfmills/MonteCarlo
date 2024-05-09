# basics of random number generation and Monte Carlo

import numpy as np
from numpy.random import seed
from numpy.random import rand
import matplotlib.pyplot as plt

# seed the random number generator
s = 1
n = 1000

seed(s)
unif_rands = rand(n)

print(unif_rands)
plt.hist(unif_rands)
plt.show()

# use the inverse-transform method to convert to exponential(2) random variates
mu = 2
expo_rands = -np.log(unif_rands)/mu
plt.hist(expo_rands)
plt.show()
print(np.mean(expo_rands))

# use the convolution method to produce Erlang(2,2) random variates
unif_rands2 = rand(n)
expo_rands2 = -np.log(unif_rands2)/mu
erlang_rands = expo_rands+expo_rands2
plt.hist(erlang_rands)
plt.show()
print(np.mean(erlang_rands))

# use the acceptance-rejection to simulate from a distribution with pdf
#  f(x) = a + 2(1-a)x , 0 < a < 1, 0 <= x <= 1
a = 0.75
t = 2-a
def f(x):
    return a + 2*(1-a)*x

rand_y = rand(n)
acceptance = rand(n)
accepted_y = rand_y[ acceptance <= f(rand_y)/t ]
plt.hist(accepted_y)
plt.show()
print(np.mean(accepted_y))
print(len(accepted_y))