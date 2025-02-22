import matplotlib.pyplot as plt
import numpy as np
import scipy.optimize as opt

#function we want our data fitted to(2nd order polynomial),
#  x is data and a,b,c are parameters that scipy figures out
def poly_fit(x, a,b,c):
    y = np.empty_like(x)
    for i, x_ in enumerate(x):
        y[i] = a*x_**2 + b*x_ + c

    return y
#example
x = np.linspace(-2,2, 10)
y = np.exp(x)

#get [a,b,c] from optimizer
fit_params = opt.curve_fit(poly_fit, x, y)[0]

print(fit_params)

#Use our polynomial function with our newly fitted parameters to calculate the y-values of our data
y_fit = poly_fit(x, fit_params[0], fit_params[2], fit_params[2])
#plot
fig, ax = plt.subplots()
ax.plot(x, y_fit, label='fitted')
ax.plot(x,y, label = 'original')
ax.legend()
fig.savefig('fitted_exponential.png')
