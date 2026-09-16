"""Model definitions for the DKL ensemble."""

import gpytorch
import torch
from botorch.models.gpytorch import GPyTorchModel
from botorch.posteriors import GPyTorchPosterior
from gpytorch.distributions import MultivariateNormal
from torch import nn
class FeatureExtractor(nn.Sequential):
    def __init__(self, input_dim=2, hidden_dim=4, output_dim=2):
        super(FeatureExtractor, self).__init__(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            # nn.Linear(hidden_dim, hidden_dim),
            # nn.ReLU(),
            nn.Linear(hidden_dim, output_dim) 
        )

class DKLModel(gpytorch.models.ApproximateGP):
    def __init__(self, inducing_points, feature_extractor, kernel_lengthscale_min):
        variational_distribution = gpytorch.variational.CholeskyVariationalDistribution(
            inducing_points.size(0)
        )
        variational_strategy = gpytorch.variational.VariationalStrategy(
            self, inducing_points, variational_distribution, learn_inducing_locations=True
        )
        super(DKLModel, self).__init__(variational_strategy)
        # self.mean_module = gpytorch.means.ConstantMean()
        self.mean_module = gpytorch.means.LinearMean(input_size=2)
        
        # 【修改这里】：将 RBFKernel 替换为 MaternKernel(nu=2.5)
        # self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.MaternKernel(nu=2.5))
        self.covar_module = gpytorch.kernels.ScaleKernel(
            gpytorch.kernels.MaternKernel(
                nu=2.5,
                lengthscale_constraint=gpytorch.constraints.GreaterThan(kernel_lengthscale_min)
            )
        )
        self.feature_extractor = feature_extractor
        self.scale_to_bounds = gpytorch.utils.grid.ScaleToBounds(-1., 1.)

    def forward(self, x):
        projected_x = self.feature_extractor(x)
        projected_x = self.scale_to_bounds(projected_x)
        mean_x = self.mean_module(projected_x)
        covar_x = self.covar_module(projected_x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)

class DeepEnsembleGP(GPyTorchModel):
    num_outputs = 1
    def __init__(self, models, likelihoods):
        super().__init__()
        self.models = nn.ModuleList(models)
        self.likelihoods = nn.ModuleList(likelihoods)
        self.likelihood = likelihoods[0] 
    
    def posterior(self, X, observation_noise=False, **kwargs):
        means, vars_ = [], []
        for model, likelihood in zip(self.models, self.likelihoods):
            model.eval(); likelihood.eval()
            dist = model(X)
            pred_dist = likelihood(dist) if observation_noise else dist
            means.append(pred_dist.mean.unsqueeze(0)) 
            vars_.append(pred_dist.variance.unsqueeze(0))
        
        means = torch.cat(means, dim=0) 
        vars_ = torch.cat(vars_, dim=0) 
        
        ensemble_mean = means.mean(dim=0)
        avg_var = vars_.mean(dim=0)
        var_of_means = means.var(dim=0, unbiased=False)
        ensemble_var = avg_var + var_of_means 
        
        covar = torch.diag_embed(ensemble_var)
        return GPyTorchPosterior(MultivariateNormal(ensemble_mean, covar))

