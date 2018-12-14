import numpy as np
from republic_ml import supervised_models


def main():
    x = np.array(range(1, 11))
    y = np.array(range(2, 22, 2))
    print(f'X = {x}')
    print(f'Y = {y}')
    slr = supervised_models.Regression(x, y)
    slr.fit()
    print('Slope and Intercept calculated - model ready to use')
    f = np.array([12, 13, 14, 15])
    print(f'Independent variables to be predicted: {f}')
    slr.predict(f)
    print('Model fit - y_hat prediction available')
    print(slr.y_hat)


if __name__ == "__main__":
    main()
