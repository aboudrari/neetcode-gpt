import numpy as np
from numpy.typing import NDArray


class Solution:
    def forward(self, x: NDArray[np.float64], w: NDArray[np.float64], b: float, activation: str) -> float:
        
        z = np.dot(x,w) + b
        sigmoid = 1 / (1 + np.exp(-z))
        ReLu = np.maximum(0, z)
        if activation ==  "sigmoid" :
            return np.round(sigmoid, 5)
        elif activation == "relu" :
            return np.round(ReLu, 5 ) 
        else :
            return 0.0