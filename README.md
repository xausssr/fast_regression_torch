# fast_regression_torch

Realisation of *Elastic Regression* and *Logistic Regression* on `torch` without autograds.

>This repository contains code written to understand the field, there may be inaccuracies.

## Install

use `git clone`, after that `pip install -r requirements.txt`

## Usage

There are 2 main models: *Elastic Regression* and *Logistic Regression* (for 2 classes) and *Grid search* for both models.

### Import

```python
import sys
sys.path.append("<path_to_repo>/fast_regression_torch/")
from torch_regressor.implementations import ElasticRegressor, LogisticRegressor
from torch_regressor.grid_search import GridSearch
```

### Regressors

```python
# init regressor (syntax same for ElasticRegressor and LogisticRegressor)
regressor = ElasticRegressor(num_featires=10, precission="full", device="cuda:0", verbose=True)

# configure regressor to fitting
regressor.fit_configure(
    X_train, y_train, 
    X_test, y_test, 
    l1_penalty=0, 
    l2_penalty=0, 
    learning_rate=0.001,
    epochs=2000
)

# fit
history, best_weights, best_bias, best_iter = regressor.fit()
```

You can fit and evaluate with custom metrics callbacks:

```python
from sklearn.metrics import r2_score

def r2_score_callback(y, y_true):
    y = y.cpu().detach().numpy()
    y_true = y_true.cpu().detach().numpy()
    return r2_score(y, y_true)


history, best_weights, best_bias, best_iter = regressor.fit(callbacks=[r2_score_callback])
evals = regressor.evaluate(X_test, y_test, callbacks=[r2_score_callback])

# evals
{'evaluate': {'r2_score_callback': [0.8018653492281764],
  'optim_cost': [0.09743531048297882]}}
```

Save and load weights:

```python
# save
regressor.save("./weights")

# init regressor and load
regressor = ElasticRegressor(10, precission="full", device="cuda:0")
regressor.load("./weights")
```

Prediction:

```python
# Elastic Regression
preds = regressor.predict(X_test)

# Logistic Regression
preds = regressor.predict(X_test) # classes
preds = regressor.predict_proba(X_test) # before threshold
```

All regressors have 2 precissions (float32 and float16 - `torch half`). `half` precission available only on GPU! You can switch precission on current instance of model:

```python
regressor.switch_precission("half")
preds = regressor.predict(X_test)
```

> Half precission often overflow while fitting!

### Grid Search

Simple implementation of grid search. In this implementation all results save into `pd.DataFrame` and all weight dumps into separate files. `polynom_order` - is feature building with polynoms of input features.

```python
from torch_regressor.grid_search import GridSearch

# define grid

search_params = {
    "learning_rate": np.linspace(1e-5, 1, num=3).tolist(),
    "epochs": [2000],
    "l1_penalty": np.linspace(0, 1, num=3).tolist(),
    "l2_penalty": np.linspace(0, 1, num=3).tolist(),
    "polynom_order": [None, 1, 2, 3, 5, 10, 20]
}

searcher = GridSearch(
    ElasticRegressor, # base class for search
    search_params,
    X_train, y_train, 
    X_val, y_val, 
    save_every = 100, # dump checkpoints
    resume = False, # resume search from last checkpoint
    callbacks = [r2_score_callback], # callbacks
    log_dir = "./grid_runs", # save folder (for csv and model parameters)
    device = "cuda:0", 
    precission = "full"
)

searcher.fit()
```

After fitting you can read result csv or acess to table from `seacher` instance:

```python
data = searcher.parameters_table
# same
data = pd.read_csv("./grid_runs/search_results.csv")
```

> DataFrame contains all filenames for parameters of models, you can just load it into model. If polynom_order (po=) is not None - use grid_search.to_polynom(...) before evaluate / prediction.

## TODO

* add autocast for polynom order
* add fit stop by criteria
* add multiprocessing for gridsearch
