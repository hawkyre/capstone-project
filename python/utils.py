import numpy as np

def normalize_x(x, y, points = 750):
    x = np.array(x)
    y = np.array(y)

    xn = np.linspace(x.min(), x.max(), points)
    yn = np.interp(xn, x, y)

    return xn, yn