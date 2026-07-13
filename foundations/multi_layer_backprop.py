import numpy as np
from typing import List


class Solution:
    def forward_and_backward(self,
                              x: List[float],
                              W1: List[List[float]], b1: List[float],
                              W2: List[List[float]], b2: List[float],
                              y_true: List[float]) -> dict:
        # Architecture: x -> Linear(W1, b1) -> ReLU -> Linear(W2, b2) -> predictions
        # Loss: MSE = mean((predictions - y_true)^2)
        #
        # Return dict with keys:
        #   'loss':  float (MSE loss, rounded to 4 decimals)
        #   'dW1':   2D list (gradient w.r.t. W1, rounded to 4 decimals)
        #   'db1':   1D list (gradient w.r.t. b1, rounded to 4 decimals)
        #   'dW2':   2D list (gradient w.r.t. W2, rounded to 4 decimals)
        #   'db2':   1D list (gradient w.r.t. b2, rounded to 4 decimals)

        # Forward Pass
        x = np.array(x)
        W1 = np.array(W1)
        b1 = np.array(b1)
        W2 = np.array(W2)
        b2 = np.array(b2)
        y_true = np.array(y_true)

        z1 = x @ W1.T + b1
        a1 = np.maximum(0, z1)
        z2 = a1 @ W2.T + b2 

        # Backword Pass
        n = len(y_true)
        dL = 1
        loss = np.mean((z2 - y_true) ** 2)
        dL_dz2 = 2 * (z2 - y_true) / n
        dL_dW2 = np.outer(dL_dz2, a1)
        dL_db2 = dL_dz2
        dL_da1 = dL_dz2 @ W2 
        dL_dz1 = dL_da1 * (z1 > 0)
        dL_dW1 = np.outer(dL_dz1, x)
        dL_db1 = dL_dz1

        return {
        'loss': round(float(loss), 4),
        'dW1': np.round(dL_dW1, 4).tolist(),
        'db1': np.round(dL_db1, 4).tolist(),
        'dW2': np.round(dL_dW2, 4).tolist(),
        'db2': np.round(dL_db2, 4).tolist()
                }

