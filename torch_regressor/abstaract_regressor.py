import os
import pickle
from datetime import datetime
from typing import Any, Callable, List, Tuple, Union

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from torch_regressor.decorators import profile

SIZES_SUFFIX = ["b", "Kb", "Mb", "Gb", "Tb"]

def custom_print(operation:str, message:str) -> None:
    print(f"{datetime.now().strftime('%d.%m.%Y %H:%M:%S')}\t{operation:20s}\t{message}")

def human_shape(data: Union[torch.Tensor, np.ndarray]) -> str:
    return 'x'.join(map(str, data.shape))

def human_size(tensor: Union[torch.Tensor, pd.core.frame.DataFrame, np.ndarray]) -> str:
    if isinstance(tensor, torch.Tensor):
        bytes = tensor.element_size() * tensor.nelement()
    if isinstance(tensor, np.ndarray):
        bytes = tensor.size * tensor.itemsize
    if isinstance(tensor, pd.core.frame.DataFrame):
        bytes = tensor.values.size * tensor.values.itemsize
    size = bytes
    idx = 0
    while size > 1024:
        size = size / 1024
        idx += 1
    if idx == 0:
        return f"{size:d}{SIZES_SUFFIX[idx]}"
    else:
        return f"{size:.1f}{SIZES_SUFFIX[idx]}"

class AbstaractRegressor:

    def __init__(
        self, num_features: int, weights: Union[None, torch.tensor] = None, bias: Union[None, torch.tensor] = None,
        precission: str = "full", device: str = "cuda:0", verbose: bool = True
    ) -> None:
        """Base regression initialisator

        Args:
            num_features (int): Number of imput features
            weights (Union[None, torch.tensor], optional): Weigths of regression (None - zeros). Defaults to None.
            bias (Union[None, torch.tensor], optional): Biases of regression (None - zeros). Defaults to None.
            precission (str, optional): Calculation precission: 'full' - float32, 'half' - float16. Defaults to "full".
            device (str, optional): Device for calculations. Defaults to "cuda:0".
            verbose (bool): Log information. Defaults to True
        """
        
        self.benchmarks = dict()
        self.regressor_info = ""
        self.verbose = verbose
        
        # verification of device
        assert device.split(":")[0] in ['cpu', 'cuda'], f"Unknown device: {device}, use one of ['cpu', 'cuda']"
        if str.isdigit(device.split(":")[-1]) and device.split(":")[0] == "cuda":
            assert int(device.split(":")[-1]) in range(torch.cuda.device_count()),\
                f"Wrong device idx #{int(device.split(':')[-1])}, available GPU idx: {range(torch.cuda.device_count())}"
        self.device = device

        if self.verbose:
            custom_print("Initialisation", f"Using '{self.device}' for this instance")
        
        # check precission
        assert precission in ["full", "half"], f"Wrong precission, use one of ['full', 'half']"
        if precission == "half":
            assert self.device.split(":")[0] == "cuda", f"Can't use half precission on cpu"
        self.precission = precission

        # verification of weights and bias
        if weights is not None or bias is not None:
            assert weights is not None, f"If init with weights? both of weights and biasses are needed, weights is None"
            assert bias is not None, f"If init with weights? both of weights and biasses are needed, bias is None"
            assert weights.ndim == 2, f"Wrong weights shape: [{human_shape(weights)}], need [num_features x 1]"
            assert bias.ndim == 1, f"Wrong bias shape: [{human_shape(bias)}], need [num_features]"
            assert weights.shape[0] == num_features,\
                f"Wrong weights shape: [{human_shape(weights)}], need [num_features x 1]"
            assert bias.shape[0] == 1,\
                f"Wrong bias shape: [{human_shape(bias)}], need [1]"
            self.weights = weights.to(self.device)
            self.bias = bias.to(self.device)
        else:
            self.weights = torch.zeros(num_features, 1).to(self.device)
            self.bias = torch.zeros(1).to(self.device)
        
        if self.precission == "half":
            self.weights = self.weights.half()
            self.bias = self.bias.half()

        if verbose:
            custom_print(
                "Initialisation", f"Using '{self.precission}' for this instance, amount of memory weights: "
                f"{human_size(self.weights)}, biases: {human_size(self.bias)}"
            )
        self.prepare_to_fit = False
    
    def validate_data(
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
        assert isinstance(val_data, np.ndarray) or isinstance(val_data, pd.core.frame.DataFrame),\
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

        if self.verbose:
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

        self.train_data = train_data.to_numpy().astype(np.float32) if isinstance(train_data, pd.core.frame.DataFrame) else train_data.astype(np.float32)
        self.train_labels = train_labels.to_numpy().astype(np.float32) if isinstance(train_labels, pd.core.frame.DataFrame) else train_labels.astype(np.float32)
        self.val_data = val_data.to_numpy().astype(np.float32) if isinstance(val_data, pd.core.frame.DataFrame) else val_data.astype(np.float32)
        self.val_labels = val_labels.to_numpy().astype(np.float32) if isinstance(val_labels, pd.core.frame.DataFrame) else val_labels.astype(np.float32)
        if (len(self.train_data) > 100_000 or len(self.val_data) > 100_000) and self.precission == "half":
            if self.verbose:
                custom_print("Data validation", "WARNING Data is large, auto fallback to 'full' precission")
                self.weights = self.weights.float()
                self.bias = self.bias.float()
                self.precission = "full"

    def validate_batch(self) -> dict:
        """Validate input data (train and validation)

        Returns:
            dict: if validation is correct return dict with inforamtion about memmory usage
        """
        
        mem_info = {}
        if self.precission == "half":
            self.train_data = torch.tensor(self.train_data).half().to(self.device)
            self.train_labels = torch.tensor(self.train_labels).half().to(self.device)
            self.val_data = torch.tensor(self.val_data).half().to(self.device)
            self.val_labels = torch.tensor(self.val_labels).half().to(self.device)
        else:
            self.train_data = torch.tensor(self.train_data).to(self.device)
            self.train_labels = torch.tensor(self.train_labels).to(self.device)
            self.val_data = torch.tensor(self.val_data).to(self.device)
            self.val_labels = torch.tensor(self.val_labels).to(self.device)
        self.batch_fit(self.train_data, self.train_labels)
        self.batch_val(self.val_data, self.val_labels)

        mem_info["train_batch_input"] = human_size(self.train_data)
        mem_info["train_batch_label"] = human_size(self.train_labels)
        mem_info["val_batch_input"] = human_size(self.val_data)
        mem_info["val_batch_label"] = human_size(self.val_labels)
        mem_info["grad_weights"] = human_size(self.__grad_weight)
        mem_info["grad_bias"] = human_size(self.__grad_bias)
        return mem_info

    def forward(self, x: torch.tensor) -> torch.tensor:
        """Forward pass

        Args:
            x (torch.tensor): input data

        Returns:
            torch.tensor: predictions
        """
        return x @ self.weights + self.bias

    def loss(self, yhat: torch.tensor, y:torch.tensor) -> torch.tensor:
        """General loss for optimisation

        Args:
            yhat (torch.tensor): predictions
            y (torch.tensor): true outputs

        Returns:
            torch.tensor: loss value
        """

        assert self.prepare_to_fit, f"Not prepared to fit, call .fit_configure(...)"
        l1_term = self.l1_penalty * torch.sum(torch.abs(self.weights))
        l2_term = (self.l2_penalty / 2) * torch.sum(torch.square(self.weights))
        return torch.square(yhat - y).mean() + l1_term + l2_term

    def grad_step(self, yhat: torch.tensor, y: torch.tensor, x: torch.tensor) -> torch.tensor:
        """Basic gradient step (for MSE error)

        Args:
            yhat (torch.tensor): prediction
            y (torch.tensor): true output
            x (torch.tensor): input data

        Returns:
            torch.tensor: gradients
        """
        self.__grad_weight = 2 * (x.T @ (y - yhat)) / y.shape[0] 
        self.__grad_bias = (y - yhat).sum() / y.shape[0]
    
    def update(self) -> None:
        """Basic update weights and biases
        """
        l2_term = self.l2_penalty * torch.sum(self.weights)
        self.weights += self.learning_rate * (self.__grad_weight + torch.sign(self.weights) * self.l1_penalty + l2_term)
        self.bias += self.learning_rate * self.__grad_bias

    @profile(operation='optim_step')
    def batch_fit(self, x: torch.tensor, y: torch.tensor) -> Tuple[float, torch.tensor]:
        """Basic method for and update weights

        Args:
            x (torch.tensor): input data
            y (torch.tensor): true labels

        Returns:
            Tuple[float, torch.tensor]: cost value and predictions
        """
        yhat = self.forward(x)
        self.grad_step(yhat,  y, x) 
        self.update()
        return self.loss(yhat, y).mean(), yhat

    @profile(operation='val_step')
    def batch_val(self, x: torch.tensor, y: torch.tensor) -> Tuple[float, torch.tensor]:
        """Basic validation step

        Args:
            x (torch.tensor): input data
            y (torch.tensor): true labels

        Returns:
            Tuple[float, torch.tensor]: cost value and predictions
        """
        yhat = self.forward(x)
        return self.loss(yhat, y).mean(), yhat

    def fit_configure(
        self, train_data: Union[pd.core.frame.DataFrame, np.ndarray], train_labels: Union[pd.core.frame.DataFrame, np.ndarray],
        val_data: Union[pd.core.frame.DataFrame, np.ndarray], val_labels: Union[pd.core.frame.DataFrame, np.ndarray],
        l1_penalty: float = 0.0, l2_penalty: float = 0.0, learning_rate: float = 1e-3, epochs: int = 1000,
        val_tol: Union[None, float] = None
    ) -> None:
        """Configurate model to fit

        Args:
            train_data (Union[pd.core.frame.DataFrame, np.ndarray]): Train data
            train_labels (Union[pd.core.frame.DataFrame, np.ndarray]): Train labels
            val_data (Union[pd.core.frame.DataFrame, np.ndarray]): Val data
            val_labels (Union[pd.core.frame.DataFrame, np.ndarray]): Val labels
            l1_penalty (float, optional): value of L1 penalty. Defaults to 0.0.
            l2_penalty (float, int, float], optional): train batch size, if None - auto set, depends of memory amount. Defaults to None.
            learning_rate (float, optional): learning rate. Defaults to 0.5.
            epochs (int, optional): number of train epochs. Defaults to 1000.
            val_tol (Union[None, float], optional): if set - stop train if validation loss not change (changes > val_tol) for 100 epochs. Defaults to None.
        """

        self.validate_data(train_data, train_labels, val_data, val_labels)

        self.l1_penalty = l1_penalty
        self.l2_penalty = l2_penalty
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.val_tol = val_tol
        self.prepare_to_fit = True

        mem_info = self.validate_batch()

        self.regressor_info += "Fit benchmarks info\nMemory:\n"
        for k, v in mem_info.items():
            self.regressor_info += f"{k:20s}:{v}\n"
        self.regressor_info += f"Time:\n"
        for k, v in self.benchmarks.items():
            self.regressor_info += f"{k:20s}:{v:.4f}s\n"

        if self.verbose:
            custom_print("Configuration", "Regressor configurate to fit.")

    def info(self) -> str:
        """wrapper to get information about regressor, contains memmory usage and time benchmarking

        Returns:
            str: info about regressor instance
        """
        return self.regressor_info

    def fit(
        self, callbacks: List[Callable[[torch.tensor, torch.tensor], Any]] = [], verbose: bool=True
    ) -> Tuple[List[Any], torch.tensor, torch.tensor, int]:
        """Fit cycle implementation

        Args:
            callbacks (List[Callable[[torch.tensor, torch.tensor], Any]], optional): list with callbacks - any function,
                    that get y_hat and y_true (torch.tensors) as input. Save all results for each fit step Defaults to [].
            verbose (bool, optional): if True show progressbar and print some info. Defaults to True.

        Returns:
            Tuple[List[Any], torch.tensor, torch.tensor, int]: list with callbacks results, best weights, 
                    best bias and best fit iteration (depends of cost on validation batches)
        """
        if self.verbose:
            custom_print("Fit", "Start fit model")

        callbacks_holder = {"train": {k.__name__: [] for k in callbacks}, "val": {k.__name__: [] for k in callbacks}}
        callbacks_holder["train"]["optim_cost"] = []
        callbacks_holder["val"]["optim_cost"] = []

        best_weights = torch.clone(self.weights)
        best_bias = torch.clone(self.bias)
        best_iter = 0
        min_loss = None

        pbar = tqdm(range(self.epochs), leave=False, disable=not verbose)
        for epoch in range(self.epochs):
            #break if nan - optimization diverges
            train_loss, pred_train = self.batch_fit(self.train_data, self.train_labels)
            if torch.isinf(train_loss).prod():
                if self.verbose:
                    custom_print("Fit Regressor", "Hit nan, fit regression overflow precission, break fitting")
                break

            val_loss, pred_val = self.batch_val(self.val_data, self.val_labels)
            callbacks_holder["train"]["optim_cost"].append(train_loss.item())
            callbacks_holder["val"]["optim_cost"].append(val_loss.item())

            for callback in callbacks:
                callbacks_holder["train"][callback.__name__].append(callback(pred_train, self.train_labels))
                callbacks_holder["val"][callback.__name__].append(callback(pred_val, self.val_labels))
            
            if min_loss is None:
                min_loss = val_loss.item()
                best_weights = torch.clone(self.weights)
                best_bias = torch.clone(self.bias)
                best_iter = epoch

            if min_loss > val_loss.item():
                min_loss = val_loss.item()
                best_weights = torch.clone(self.weights)
                best_bias = torch.clone(self.bias)
                best_iter = epoch

            pbar.update(1)
            pbar.set_description(f"Train loss: {train_loss:.2f}, val loss: {val_loss:.2f}")

        self.weights.data = torch.clone(best_weights)
        self.bias.data = torch.clone(best_bias)

        if self.verbose:
            custom_print("Fit", f"Fit complete, best val loss: {val_loss:.2f}, best iter: {best_iter}")

        return callbacks_holder, best_weights, best_bias, best_iter

    def predict(self, x: Union[np.ndarray, pd.core.frame.DataFrame]) -> Union[np.ndarray, pd.core.frame.DataFrame]:
        """Predict result by input data. Return in same type.

        Args:
            x (Union[np.ndarray, pd.core.frame.DataFrame]): input data

        Returns:
            Union[np.ndarray, pd.core.frame.DataFrame]: predictions
        """

        return_flag_pd = False
        if isinstance(x, np.ndarray):
            assert x.ndim == 2, f"Wrong input data [{human_shape(x)}], need [num_objects x num_features]"
        assert x.shape[1] == self.weights.shape[0], f"Wrong input data [{human_shape(x)}], need [num_objects x num_features]"
        
        if isinstance(x, pd.core.frame.DataFrame):
            x = x.values
            return_flag_pd = True

        x = torch.tensor(x.astype(np.float32))
        if self.precission == "half":
            x = x.half()
        x = x.to(self.device)
        
        result = (x @ self.weights + self.bias).cpu().detach().numpy().flatten()
        if return_flag_pd:
            result = pd.core.frame.DataFrame({"predictions": result})
        return result

    @profile(operation="evaluate")
    def evaluate(
        self, x: Union[np.ndarray, pd.core.frame.DataFrame], y_true: Union[np.array, pd.core.frame.DataFrame],
        callbacks: List[Callable[[torch.tensor, torch.tensor], Any]] = []
    ) -> List[Tuple[Any]]:
        """Evaluate method

        Args:
            x (Union[np.ndarray, pd.core.frame.DataFrame]): input data
            y_true (Union[np.array, pd.core.frame.DataFrame]): input labels
            callbacks (List[Callable[[torch.tensor, torch.tensor], Any]], optional): list with callbacks - any function,
                    that get y_hat and y_true (torch.tensors) as input. Save all results for each fit step Defaults to [].

        Returns:
            List[Tuple[Any]]: _description_
        """
        
        assert isinstance(x, np.ndarray) or isinstance(x, pd.core.frame.DataFrame),\
            f"Non valid input data type: {type(x)}, must be one of [np.ndarray, pd.core.frame.DataFrame]"
        assert isinstance(y_true, np.ndarray) or isinstance(y_true, pd.core.frame.DataFrame),\
            f"Non valid input labels type: {type(y_true)}, must be one of [np.ndarray, pd.core.frame.DataFrame]"

        if isinstance(x, np.ndarray):
            assert x.ndim == 2, f"Wrong train data [{human_shape(x)}], need [num_objects x num_features]"
        assert x.shape[1] == self.weights.shape[0], f"Wrong data data [{human_shape(x)}], need [num_objects x num_features]"

        if isinstance(y_true, np.ndarray):
            assert y_true.ndim == 2, f"Wrong input labels [{human_shape(y_true)}], need [num_objects x 1]"
        assert y_true.shape[1] == 1, f"Wrong input labels [{human_shape(y_true)}], need [num_objects x 1]"

        if isinstance(x, pd.core.frame.DataFrame):
            x = x.values
        if isinstance(y_true, pd.core.frame.DataFrame):
            y_true = y_true.values

        x = torch.tensor(x.astype(np.float32))
        y_true = torch.tensor(y_true.astype(np.float32))
        if self.precission == "half":
            x = x.half()
            y_true = y_true.half()

        x = x.to(self.device)
        y_true = y_true.to(self.device)
        
        callbacks_holder = {}
        callbacks_holder = {"evaluate": {k.__name__: [] for k in callbacks}}
        callbacks_holder["evaluate"]["optim_cost"] = []

        eval_loss, result = self.batch_val(x, y_true)
        callbacks_holder["evaluate"]["optim_cost"].append(eval_loss.item())
        for callback in callbacks:
            callbacks_holder["evaluate"][callback.__name__].append(callback(result, y_true))

        return callbacks_holder

    def switch_precission(self, precission: str = "half") -> None:
        """Switch precisson of instance

        Args:
            precission (str, optional): Switch to precisson, one of ['full', 'half']. Defaults to "half".
        """

        assert precission in ["full", 'half'], f"Wrong precission ({precission}), choose one of ['full', 'half']"

        if precission == "full":
            if self.precission == "half":
                self.precission = precission
                self.weights = self.weights.float()
                self.bias = self.bias.float()
        else:
            if self.precission == "full":
                self.precission = precission
                self.weights = self.weights.half()
                self.bias = self.bias.half()
        if self.verbose:
            custom_print("Switch precission", f"Sucessfull switch to {precission}")

    def save(self, path: str) -> None:
        """Save weights and bias to file

        Args:
            path (str): path to file
        """

        if os.path.exists(path):
            custom_print("Save weights", f"WARNING File {path} already exist! Overwrite file")

        pickle.dump(
            {
                "w": self.weights.cpu().detach().numpy(), 
                "b": self.bias.cpu().detach().numpy(),
                "l1": self.l1_penalty,
                "l2": self.l2_penalty,
                "lr": self.learning_rate,
                "epochs": self.epochs,
                "val_tol": self.val_tol,
                "prepare_to_fit": self.prepare_to_fit,
                },
            open(path, "wb"))
        if self.verbose:
            custom_print("Save weights", f"Weight save corectly {path}")

    def load(self, path: str) -> None:
        """Load weights and bias to file

        Args:
            path (str): path to file
        """

        assert os.path.exists(path), f"File not exist {path}"
        data = pickle.load(open(path, "rb"))
        self.weights = torch.tensor(data["w"]).float().to(self.device)
        self.bias =  torch.tensor(data["b"]).float().to(self.device)
        if self.precission == "half":
            self.weights = self.weights.half()
            self.bias = self.bias.half()

        self.l1_penalty = data["l1"]
        self.l2_penalty = data["l2"]
        self.learning_rate = data["lr"]
        self.epochs = data["epochs"]
        self.val_tol = data["val_tol"]
        self.prepare_to_fit = data["prepare_to_fit"]
        if self.verbose:
            custom_print("Load weights", f"Weights load correctly")
