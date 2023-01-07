import numpy as np
import pandas as pd
import torch
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

from torch_regressor.implementations import ElasticRegressor, LogisticRegressor


def relu(x):
    if x > 0:
        return (x)
    else:
        return (x) * 0.125

if __name__ == "__main__":

    print("Search GPU for tests...")
    if torch.cuda.is_available():
        d_count = torch.cuda.device_count()
        print(f"You have {d_count} GPUs with CUDA, use with idx 0")
        device = "cuda:0"
    else:
        print("GPU with CUDA not found, use CPU (slower than sklearn implementation!)")
        device = "cpu"

    print("\n\n\n"+ "=" * 5 + " Test Elastic Regressor on ReLU-like data" + "=" * 5)
    regressor = ElasticRegressor(1, precission="full", device=device, verbose=True)
    
    
    idx = np.linspace(-2, 2, num=5000)
    data = np.array([relu(x) for x in idx])
    data += np.random.randn(len(data)) * 0.2
    X_train, X_test, y_train, y_test = train_test_split(idx, data, test_size=0.2)

    regressor.fit_configure(
        X_train.reshape((-1, 1)), y_train.reshape((-1, 1)), 
        X_test.reshape((-1, 1)), y_test.reshape((-1, 1)), 
        l1_penalty=.3, l2_penalty=.4, learning_rate=0.0001, 
        epochs=1000, val_tol=1e-4
    )
    print("\n")
    print(regressor.info())
    regressor.fit()
    del idx, data, X_train, X_test, y_train, y_test

    print("\n\n\n" + "=" * 5 + " Test Logistic Regressor on `make_classification` from `sklearn`" + "=" * 5)
    regressor = LogisticRegressor(2, threshold=0.5, precission="full", device=device, verbose=True)
    
    X, y = make_classification(n_samples=5000, n_features=2,n_redundant=0, n_informative=2, n_clusters_per_class=1, class_sep=2)
    X += np.random.randn(*X.shape) * 0.15

    X_train, X_test, y_train, y_test = train_test_split(X, y.reshape((-1, 1)), test_size=0.2)

    regressor.fit_configure(X_train, y_train, X_test, y_test, l1_penalty=.3, l2_penalty=.4, learning_rate=0.0001, epochs=1000, val_tol=1e-4)
    print("\n")
    print(regressor.info())
    regressor.fit()
