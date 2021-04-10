import numpy as np
import argparse
# 糖尿病の進行状況データセット
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF
from sklearn.preprocessing import StandardScaler


def load_data():
    data = load_diabetes()
    X_data = data.data
    Y_data = data.target
    scaler = StandardScaler()
    X_scale = scaler.fit_transform(X_data)
    Y_scale = scaler.fit_transform(Y_data.reshape(-1, 1))
    X_train, X_test, Y_train, Y_test = train_test_split(X_scale, Y_scale, random_state=0, train_size=0.7)
    return X_train, X_test, Y_train, Y_test


def train(scale, alpha, X_train, Y_train):
    kernel = RBF(length_scale=scale)
    gpr = GaussianProcessRegressor(kernel=kernel, alpha=alpha, optimizer=None)
    gpr.fit(X_train, Y_train)
    print(gpr.kernel_)
    return gpr


def test(gpr, X_test, Y_test):
    pred_mu, pred_sigma = gpr.predict(X_test, return_std=True)
    return np.sqrt(np.mean(np.square(pred_mu-Y_test)))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--scale', '-E', default=1.0)
    parser.add_argument('--alpha', '-A', default=0.1)
    args = parser.parse_args()
    print(args)

    X_train, X_test, Y_train, Y_test = load_data()
    gpr = train(args.scale, args.alpha, X_train, Y_train)
    RMSE = test(gpr, X_test, Y_test)
    print(RMSE)
