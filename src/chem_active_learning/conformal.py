"""Conformal prediction calibration helpers."""

import numpy as np
import torch
class ConformalPrediction:
    def __init__(self, model, train_X, train_Y, alpha=0.1):
        self.model = model
        self.alpha = alpha
        self.q_score = self._calibrate(train_X, train_Y)
        
    def _calibrate(self, X, Y):
        self.model.eval()
        with torch.no_grad():
            posterior = self.model.posterior(X)
            mu = posterior.mean.squeeze(-1)
            sigma = posterior.variance.sqrt().squeeze(-1)
            scores = torch.abs(Y - mu) / (sigma + 1e-6)
            q_val = np.quantile(scores.numpy(), 1 - self.alpha, method='higher')
            return torch.tensor(q_val)
            
    def predict_bound(self, X, direction='upper'):
        self.model.eval()
        with torch.no_grad():
            posterior = self.model.posterior(X)
            mu = posterior.mean.squeeze(-1)
            sigma = posterior.variance.sqrt().squeeze(-1)
            if direction == 'upper': return mu + self.q_score * sigma
            elif direction == 'lower': return mu - self.q_score * sigma


