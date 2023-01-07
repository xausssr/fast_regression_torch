import itertools
import os
import pickle
from typing import Any, Callable, Dict, List, Union

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from .abstaract_regressor import custom_print, human_shape, human_size
from .implementations import ElasticRegressor, LogisticRegressor


def to_polynom(x: np.ndarray, order=1) -> np.ndarray:
    """Add polynominal features (base on current features)

    Args:
        x (np.ndarray): input data
        order (int, optional): polynominal order. Defaults to 1.

    Returns:
        np.ndarray: new data with polynominal features
    """
    order_range = range(1, order + 1 ,1)
    x = np.atleast_1d(x)[:]
    out = np.array([])
    for i in order_range:
        out = np.append(out, np.power(x, i))
    return out.reshape(-1, x.size).T.astype(np.float32).copy()


class GridSearch:

    def __init__(
        self, base_class: Union[ElasticRegressor, LogisticRegressor], search_params: Dict[str, List[Any]], 
        train_x: Union[pd.core.frame.DataFrame, np.ndarray], train_y:  Union[pd.core.frame.DataFrame, np.ndarray], 
        test_x: Union[pd.core.frame.DataFrame, np.ndarray], test_y:  Union[pd.core.frame.DataFrame, np.ndarray], 
        save_every: int = 100, resume: bool = False, callbacks: List[Callable[[torch.tensor, torch.tensor], Any]] = [],
        log_dir: str = "./grid_runs", device: str = "cuda:0", precission: str = "full"
    ) -> None:
        """Grid search instance

        Args:
            base_class (Union[ElasticRegressor, LogisticRegressor]): what model was using in grid search
            search_params (Dict[str, List[Any]]): grid parameters, defined by {"param": [x, x, x]}
            train_x (Union[pd.core.frame.DataFrame, np.ndarray]): Train data
            train_y (Union[pd.core.frame.DataFrame, np.ndarray]): Train labels
            test_y (Union[pd.core.frame.DataFrame, np.ndarray]): Val data
            test_x (Union[pd.core.frame.DataFrame, np.ndarray]): Val labels
            save_every (int): Save table with results every N steps. Defaults to 100
            resume (bool): Resume session from last saved step. Defaults to False
            callbacks (List[Callable[[torch.tensor, torch.tensor], Any]], optional):  list with callbacks - any function,
                    that get y_hat and y_true (torch.tensors) as input. Save all results for each fit step. Defaults to [].
            log_dir (str, optional): path to save grid search results. Defaults to "./grid_runs".
            device (str, optional): Device for calculations. Defaults to "cuda:0".
            precission (str, optional): Calculation precission: 'full' - float32, 'half' - float16. Defaults to "full".
        """

        self.__base_class = base_class
        self.__best_classifier = None
        self.__best_order = None
        self.__device = device
        self.__precission = precission
        self.__callbacks_list = callbacks
        self.__log_dir = log_dir
        self.__save_every = save_every
        self.__resume = resume

        self.__validate_data(train_x, train_y, test_x, test_y)

        keys, values = zip(*search_params.items())
        self.__grid_points = [dict(zip(keys, v)) for v in itertools.product(*values)]

        custom_print("Grid Search", f"Have {len(self.__grid_points)} points in hyperparameters space to search")

        if os.path.exists(log_dir):
            if len(os.listdir(log_dir)) > 0 and not self.__resume:
                custom_print("Grid Search", f"WARNING log folder allready exist and not empty! If search parameters is same - files will overwrited!")
        else:
            os.mkdir(log_dir)
            if self.__resume:
                custom_print("Grid Search", f"WARNING not found previus session, set 'resume=False'!")
                self.__resume = False
        
        if "weights" not in os.listdir(log_dir):
            os.mkdir(os.path.join(log_dir, "weights"))

        if self.__resume:
            self.histories = pickle.load(open(os.path.join(self.__log_dir, "fit_histories.pkl", "rb")))
            self.parameters_table = pd.read_csv(os.path.join(self.__log_dir, "search_results.csv"))
        else:
            self.histories = {}
            cols = "filename,learning_rate,epochs,polynom_order,l1_penalty,l2_penalty,best_val_score,best_train_score,best_iter".split(",")
            cols += [x.__name__ for x in self.__callbacks_list]
            self.parameters_table = pd.DataFrame(columns=cols)

    def fit(self):
        point = 0
        for params in tqdm(self.__grid_points):
            if self.__resume and point < len(self.parameters_table):
                    point += 1
                    continue
            else:
                polynom_order = params.pop('polynom_order')
                self._one_round(params, polynom_order)
                point += 1
            
            if point != 0 and point % self.__save_every == 0:
                self.parameters_table.to_csv(os.path.join(self.__log_dir, "search_results.csv"))
                pickle.dump(self.histories, open(os.path.join(self.__log_dir, "fit_histories.pkl"), "wb"))
        
        self.parameters_table.to_csv(os.path.join(self.__log_dir, "search_results.csv"))
        pickle.dump(self.histories, open(os.path.join(self.__log_dir, "fit_histories.pkl"), "wb"))

    def _one_round(self, regression_params, polynom_order):
    # palce params to frame

        filename = "run_po=" + str(polynom_order) + "_" + "_".join([f"{k}={v}" for k, v in regression_params.items()])

        # apply polynominal
        if polynom_order is not None:
            train = to_polynom(self.__train_x[:, 0], polynom_order)
            test = to_polynom(self.__test_x[:, 0], polynom_order)
            for dim in range(1, self.__train_x.shape[1]):
                train = np.hstack([train, to_polynom(self.__train_x[:, dim], polynom_order)])
                test = np.hstack([test, to_polynom(self.__test_x[:, dim], polynom_order)])
        else:
            train = self.__train_x.copy()
            test = self.__test_x.copy()
        
        # construct solver
        temp_reg = self.__base_class(train.shape[1], precission=self.__precission, device=self.__device, verbose=False)
        temp_reg.fit_configure(train, self.__train_y, test, self.__test_y, **regression_params)
        
        # fit and save history
        history, _, _, best_iter = temp_reg.fit(callbacks=[], verbose=False)
        self.histories[filename] = history
        temp_reg.save(os.path.join(self.__log_dir, "weights", filename))
        
        # save best classifier
        best_val_score = history["val"]['optim_cost'][best_iter]

        if self.__best_classifier is None:
            self.__best_classifier = temp_reg
            self.__best_order = polynom_order

        if self.parameters_table["best_val_score"].min() > best_val_score:
            self.__best_classifier = temp_reg
            self.__best_order = polynom_order
        
        # add to table
        add_row = regression_params.copy()
        add_row["polynom_order"] = polynom_order
        add_row["best_train_score"] = history["train"]['optim_cost'][best_iter]
        add_row["best_val_score"] = history["val"]['optim_cost'][best_iter]
        add_row["best_iter"] = best_iter
        add_row["filename"] = filename
        
        #eval callbacks
        if len(self.__callbacks_list) > 0:
            evals = temp_reg.evaluate(test, self.__test_y, callbacks=self.__callbacks_list)
            for k, v in evals['evaluate'].items():
                if k != 'optim_cost':
                    add_row[k] = v[0]
        self.parameters_table = pd.concat([self.parameters_table, pd.DataFrame.from_records([add_row])], ignore_index=True)

    def __validate_data(
        self, train_data: Union[pd.core.frame.DataFrame, np.ndarray], train_labels: Union[pd.core.frame.DataFrame, np.ndarray],
        val_data: Union[pd.core.frame.DataFrame, np.ndarray], val_labels: Union[pd.core.frame.DataFrame, np.ndarray]
    ) -> None:
        """Validate input data

        Args:
            train_data (Union[pd.core.frame.DataFrame, np.ndarray]): Train data
            train_labels (Union[pd.core.frame.DataFrame, np.ndarray]): Train labels
            val_data (Union[pd.core.frame.DataFrame, np.ndarray]): Val data
            val_labels (Union[pd.core.frame.DataFrame, np.ndarray]): Val labels
        """
        assert isinstance(train_data, np.ndarray) or isinstance(train_data, pd.core.frame.DataFrame),\
            f"Non valid input train data type: {type(train_data)}, must be one of [np.ndarray, pd.core.frame.DataFrame]"
        assert isinstance(train_labels, np.ndarray) or isinstance(train_labels, pd.core.frame.DataFrame),\
            f"Non valid input train labels type: {type(train_labels)}, must be one of [np.ndarray, pd.core.frame.DataFrame]"
        assert isinstance(val_data, np.ndarray) or isinstance(train_data, pd.core.frame.DataFrame),\
            f"Non valid input validation data type: {type(val_data)}, must be one of [np.ndarray, pd.core.frame.DataFrame]"
        assert isinstance(val_labels, np.ndarray) or isinstance(val_labels, pd.core.frame.DataFrame),\
            f"Non valid input validation labels type: {type(val_labels)}, must be one of [np.ndarray, pd.core.frame.DataFrame]"

        if isinstance(train_data, np.ndarray):
            assert train_data.ndim == 2, f"Wrong train data shape [{human_shape(train_data)}], need [num_objects x num_features]"
        if isinstance(train_labels, np.ndarray):
            assert train_labels.ndim == 2, f"Wrong train labels shape [{human_shape(train_labels)}], need [num_objects x 1]"
        assert train_data.shape[0] == train_labels.shape[0], f"Non equal lenth of train data and train labels ({train_data.shape[0]}!={train_labels.shape[0]})"

        if isinstance(val_data, np.ndarray):
            assert val_data.ndim == 2, f"Wrong val data shape [{human_shape(val_data)}], need [num_objects x num_features]"
        if isinstance(val_labels, np.ndarray):
            assert val_labels.ndim == 2, f"Wrong val labels shape [{human_shape(val_labels)}], need [num_objects x 1]"
        assert val_data.shape[0] == val_labels.shape[0], f"Non equal lenth of val data and val labels ({val_data.shape[0]}!={val_labels.shape[0]})"

        custom_print(
            "Data validation", 
            f"Train set is correct, train data: [{human_shape(train_data)}] - {human_size(train_data)} "
            f"train labels: [{human_shape(train_labels)}] - {human_size(train_labels)}"
        )
        custom_print(
            "Data validation", 
            f"Val set is correct, val data: [{human_shape(val_data)}] - {human_size(val_data)} "
            f"val labels: [{human_shape(val_labels)}] - {human_size(val_labels)}"
        )

        self.__train_x = train_data.to_numpy().astype(np.float32) if isinstance(train_data, pd.core.frame.DataFrame) else train_data.astype(np.float32)
        self.__train_y = train_labels.to_numpy().astype(np.float32) if isinstance(train_labels, pd.core.frame.DataFrame) else train_labels.astype(np.float32)
        self.__test_x = val_data.to_numpy().astype(np.float32) if isinstance(val_data, pd.core.frame.DataFrame) else val_data.astype(np.float32)
        self.__test_y = val_labels.to_numpy().astype(np.float32) if isinstance(val_labels, pd.core.frame.DataFrame) else val_labels.astype(np.float32)
        if (len(self.__train_x) > 100_000 or len(self.__test_x) > 100_000) and self.__precission == "half":
            custom_print("Data validation", "WARNING Data is large, auto fallback to 'full' precission")
            self.__weights = self.__weights.float()
            self.__bias = self.__bias.float()
            self.__precission = "full"
