from typing import Any, Callable, List, Tuple, Union

import numpy as np
import pandas as pd
import torch
from torch.nn.functional import sigmoid

from torch_regressor.abstaract_regressor import AbstaractRegressor, human_shape


class ElasticRegressor(AbstaractRegressor):
    """Abstaract regressor is elastic regressor, that class is wrapper"""

    def __init__(
        self, num_features: int, weights: Union[None, torch.tensor] = None, bias: Union[None, torch.tensor] = None, 
        precission: str = "full", device: str = "cuda:0", verbose: bool = True
    ) -> None:
        """Linear Regressor regression instance initialisation (with elastic penalty)

        Args:
            num_features (int): Number of imput features.
            weights (Union[None, torch.tensor], optional): Weigths of regression (None - zeros). Defaults to None.
            bias (Union[None, torch.tensor], optional): Biases of regression (None - zeros). Defaults to None.
            precission (str, optional): Calculation precission: 'full' - float32, 'half' - float16. Defaults to "full".
            device (str, optional): Device for calculations. Defaults to "cuda:0".
            verbose (bool): Log information. Defaults to True.
        """
        super(ElasticRegressor).__init__(num_features, weights, bias, precission, device, verbose)


class LogisticRegressor(AbstaractRegressor):
    """Logistic regressor for binary classification problems"""

    def __init__(
        self, num_features: int, threshold: float, weights: Union[None, torch.tensor] = None, 
        bias: Union[None, torch.tensor] = None, precission: str = "full", device: str = "cuda:0", verbose: bool = True
    ) -> None:
        """Logistic Regression instance initialisation (with elastic penalty)

        Args:
            num_features (int): Number of imput features.
            threshold (float): Threshold for class prediction (after sigmoid).
            weights (Union[None, torch.tensor], optional): Weigths of regression (None - zeros). Defaults to None.
            bias (Union[None, torch.tensor], optional): Biases of regression (None - zeros). Defaults to None.
            precission (str, optional): Calculation precission: 'full' - float32, 'half' - float16. Defaults to "full".
            device (str, optional): Device for calculations. Defaults to "cuda:0".
            verbose (bool): Log information. Defaults to True
        """
        super(LogisticRegressor).__init__(num_features, weights, bias, precission, device, verbose)
        
        self.threshold = threshold
        self.__epsilon = 1e-6

    def __loss(self, yhat: torch.tensor, y:torch.tensor) -> torch.tensor:
        """General loss for optimisation

        Args:
            yhat (torch.tensor): predictions
            y (torch.tensor): true outputs

        Returns:
            torch.tensor: loss value
        """

        assert self.__prepare_to_fit, f"Not prepared to fit, call .fit_configure(...)"
        l1_term = self.__l1_penalty * torch.sum(torch.abs(self.__weights))
        l2_term = (self.__l2_penalty / 2) * torch.sum(torch.square(self.__weights))

        main_term = -(y * torch.log(yhat + self.__epsilon) + (1 - y) * torch.log(1 - yhat + self.__epsilon)).mean()
        return main_term + l1_term+ l2_term

    def __forward(self, x: torch.tensor) -> torch.tensor:
        """Forward pass

        Args:
            x (torch.tensor): input data

        Returns:
            torch.tensor: predictions
        """
        return sigmoid(x @ self.__weights + self.__bias)

    def predict(self, x: Union[np.ndarray, pd.core.frame.DataFrame]) -> Union[np.ndarray, pd.core.frame.DataFrame]:
        """Predict class by input data. Return in same type.

        Args:
            x (Union[np.ndarray, pd.core.frame.DataFrame]): input data

        Returns:
            Union[np.ndarray, pd.core.frame.DataFrame]: predictions
        """

        return_flag_pd = False
        if isinstance(x, np.ndarray):
            assert x.ndim == 2, f"Wrong input data [{human_shape(x)}], need [num_objects x num_features]"
        assert x.shape[1] == self.__weights.shape[0], f"Wrong input data [{human_shape(x)}], need [num_objects x num_features]"
        
        if isinstance(x, pd.core.frame.DataFrame):
            x = x.values.astype(np.float32)
            return_flag_pd = True

        x = torch.tensor(x)
        if self.__precission == "half":
            x = x.half()
        x = x.to(self.__device)
        
        result = (sigmoid((x @ self.__weights + self.__bias)).cpu().detach().numpy().flatten() >= self.threshold).astype(np.float32)
        if return_flag_pd:
            result = pd.core.frame.DataFrame({"predictions": result})
        return result

    def predict_proba(self, x: Union[np.ndarray, pd.core.frame.DataFrame]) -> Union[np.ndarray, pd.core.frame.DataFrame]:
        """Predict probability (predictions before threshold) by input data. Return in same type.

        Args:
            x (Union[np.ndarray, pd.core.frame.DataFrame]): input data

        Returns:
            Union[np.ndarray, pd.core.frame.DataFrame]: predictions
        """

        return_flag_pd = False
        if isinstance(x, np.ndarray):
            assert x.ndim == 2, f"Wrong input data [{human_shape(x)}], need [num_objects x num_features]"
        assert x.shape[1] == self.__weights.shape[0], f"Wrong input data [{human_shape(x)}], need [num_objects x num_features]"
        
        if isinstance(x, pd.core.frame.DataFrame):
            x = x.values.astype(np.float32)
            return_flag_pd = True

        x = torch.tensor(x)
        if self.__precission == "half":
            x = x.half()
        x = x.to(self.__device)
        
        result = sigmoid((x @ self.__weights + self.__bias)).cpu().detach().numpy().flatten()
        if return_flag_pd:
            result = pd.core.frame.DataFrame({"predictions": result})
        return result
