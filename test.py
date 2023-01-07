import numpy as np
import pandas as pd
import torch

from torch_regressor.abstaract_regressor import AbstaractRegressor

if __name__ == "__main__":
    regressor = AbstaractRegressor(214, weights=torch.randn(214, 1), bias=torch.randn(1), precission="half", device="cuda:0")
    
    train_x, train_y = np.random.randn(1000000, 214) / 10, np.random.randn(1000000, 1) / 10
    test_x, test_y = pd.DataFrame(np.random.randn(10000, 214) / 10), pd.DataFrame(np.random.randn(10000, 1) / 10)
    regressor.fit_configure(train_x, train_y, test_x, test_y, l1_penalty=.3, l2_penalty=.4, learning_rate=0.0001, epochs=1000, val_tol=1e-4)
    print(regressor.info())
    regressor.fit(verbose=True)
