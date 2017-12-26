import numpy as np


def sin(x, t=100):
    return np.sin(2.0 * np.pi * x / t)


def cos(x, t=100):
    return np.cos(2.0 * np.pi * x / t)


def toy_problem(T=100, ampl=0.05):
    x_1 = np.arange(0, 2 * T + 1)
    noise_1 = ampl * np.random.uniform(low=1.0, high=1.0, size=len(x_1))
    x_2 = np.arange(0, 2* T + 1)
    noise_2 = ampl * np.random.uniform(low=1.0, high=1.0, size=len(x_1))
    x_1 = sin(x_1) + noise_1
    x_2 = cos(x_2) + noise_2
    return np.stack([x_1, x_2], axis=1)

if __name__ == '__main__':
    prob = toy_problem(T=100, ampl=0.05)
    ans = toy_problem(T=100, ampl=0)

    np.save('sin-cos-prob', prob)
    np.save('sin-cos-ans', ans)

    print('success')