#
# Copyright (C) 2024 Daniel Ebi
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU Affero General Public License for more details.
#
# You should have received a copy of the GNU Affero General Public License
# along with this program. If not, see <http://www.gnu.org/licenses/>.
#

import cvxpy as cp
import torch as th
from cvxpylayers.torch import CvxpyLayer
from gym import spaces
from stable_baselines3.common.distributions import Distribution, sum_independent_dims
from stable_baselines3.common.policies import BasePolicy
from stable_baselines3.common.preprocessing import get_action_dim
from stable_baselines3.common.preprocessing import get_flattened_obs_dim
from stable_baselines3.common.torch_layers import (
    BaseFeaturesExtractor,
    FlattenExtractor,
)
from stable_baselines3.common.type_aliases import Schedule
from torch import nn
from torch.distributions import MultivariateNormal
from typing import Any, Dict, List, Type, Union
from typing import Optional, Tuple


class BranchedPolicyNetwork(nn.Module):

    def __init__(self, action_dim: int, latent_dim: int):
        super().__init__()

        self.action_dim = action_dim
        self.latent_dim = latent_dim

        self.branches = nn.ModuleList(
            [nn.Sequential(nn.Linear(self.latent_dim, 2), nn.Tanh()) for _ in range(action_dim)])

    def forward(self, features: th.Tensor) -> Tuple[th.Tensor, th.Tensor]:
        output = th.cat([branch(features) for branch in self.branches], dim=len(features.shape) - 1)
        if len(output.shape) > 1:
            return th.Tensor(
                [[output[j, i] for i in range(output.shape[1]) if i % 2 == 0] for j in
                 range(output.shape[0])]), th.Tensor(
                [[output[j, i] for i in range(output.shape[1]) if i % 2 != 0] for j in range(output.shape[0])])
        else:
            return th.Tensor([output[i] for i in range(len(output)) if i % 2 == 0]), th.Tensor(
                [output[i] for i in range(len(output)) if i % 2 != 0])


class ActionDistributionWithProjection(Distribution):

    def __init__(self, action_dim: int, branched_network: bool = True):
        """
        Multi-variate Gaussian distribution with independent dimensions incorporating a projection onto the feasible
        space of actions, for continuous actions.

        :param int action_dim:  Dimension of the action space.
        :param bool branched_network: Whether to use a branched network architecture (default: True).
        """

        super().__init__()
        self.action_dim = action_dim
        self.branched_network = branched_network

        self.mean_actions = None
        self.log_std = None

        # Meta-decision variable
        self.y = None

        # Action projection as optimization problem
        x_proj = cp.Variable(action_dim)
        x_tilde = cp.Parameter(action_dim)
        y = cp.Parameter(1)

        constraints = [x_proj[0] + y * x_proj[1] <= 1,
                       x_proj[1] + x_proj[2] <= y, -x_proj[1] - x_proj[2] <= 1 - y, x_proj[0] <= 1, -x_proj[0] <= 0,
                       x_proj[1:] <= y, -x_proj[1:] <= 1 - y]

        objective = cp.Minimize(cp.sum_squares(x_proj - x_tilde))
        problem = cp.Problem(objective, constraints)

        # Action projection layer
        self.proj_layer = CvxpyLayer(problem, variables=[x_proj],
                                     parameters=[x_tilde, y])

        # Action projection loss
        self.proj_loss = nn.MSELoss()

    def proba_distribution_net(self, latent_dim: int) -> Union[nn.Module, Tuple[nn.Module, nn.Module]]:
        """
        Create the neural networks that represent the distribution:
        one output will be the mean of the Gaussian, the other parameter will be the
        standard deviation (log std in fact to allow negative values)

        :param int latent_dim: Dimension of the last layer of the policy (before the action layer).

        :rytpe: Union[nn.Module, Tuple[nn.Module, nn.Module]]
        :return: Branched policy network or mean action network and log std network.
        """
        if self.branched_network:
            policy_net = BranchedPolicyNetwork(action_dim=self.action_dim, latent_dim=latent_dim)
            return policy_net
        else:
            mean_actions = nn.Sequential(nn.Linear(latent_dim, self.action_dim), nn.Tanh())
            log_std = nn.Sequential(nn.Linear(latent_dim, self.action_dim), nn.Tanh())
            return mean_actions, log_std

    def proba_distribution(
            self, mean_actions: th.Tensor, log_std: th.Tensor, y: th.Tensor
    ) -> Distribution:
        """
        Create the distribution given its parameters (mean, log_std).

        :param th.Tensor mean_actions: Mean actions vector.
        :param th.Tensor log_std: Log std vector.
        :param th.Tensor y: Meta-decision variables

        :rtype: Distribution
        :return: Action distribution with projection.
        """

        action_std = th.ones_like(mean_actions) * log_std.exp()
        self.y = y
        self.distribution = MultivariateNormal(mean_actions, th.diag_embed(action_std))
        return self

    def log_prob(self, actions: th.Tensor) -> th.Tensor:
        """
        Get the log probabilities of actions according to the distribution.
        Note that you must first call the ``proba_distribution()`` method.

        :param th.Tensor actions: Actions vector.

        :rtype: th.Tensor
        :return: The log probabilities.
        """
        log_prob = self.distribution.log_prob(actions)
        return sum_independent_dims(log_prob)

    def entropy(self) -> th.Tensor:
        """
        Return Shannon's entropy of the probability.

        :rtype: th.Tensor
        :return: Shannon's entropy vector.
        """
        return sum_independent_dims(self.distribution.entropy())

    def sample(self) -> th.Tensor:
        """
        Return a sample from the probability distribution.

        :rtype: th.Tensor
        :return: The stochastic action.
        """
        # Reparametrization trick to pass gradients
        return self.distribution.rsample()

    def mode(self) -> th.Tensor:
        """
        Return the most likely action (deterministic output) from the probability distribution.

        :rtype: th.Tensor
        :return: The deterministic action.
        """
        return self.distribution.mean

    def get_actions(self, deterministic: bool = False) -> Tuple[th.Tensor, th.Tensor]:
        """
        Return actions according to the probability distribution and projection onto the space of feasible actions
        along with the corresponding projection loss.

        :param bool deterministic: Whether a deterministic action should be sampled or not (Optional, default is False).

        :rtype: Tuple[th.Tensor, th.Tensor]
        :return: The projected actions and the corresponding projection loss.
        """
        if deterministic:
            action = self.mode()
        else:
            action = self.sample()

        # Project actions using the projection layer onto the space of feasible actions
        action_proj = self.proj_layer(action, self.y)
        action_proj = th.squeeze(th.stack(list(action_proj), dim=0))

        # Compute the projection loss
        proj_loss = self.proj_loss(action, action_proj.reshape((-1, self.action_dim)))

        return action_proj, proj_loss

    def actions_from_params(self, mean_actions: th.Tensor, log_std: th.Tensor,
                            deterministic: bool = False) -> th.Tensor:
        """
        Return (projected) samples from the probability distribution given the mean action and log std vectors.

        :param th.Tensor mean_actions: Mean actions.
        :param th.Tensor log_std: Log standard deviations.
        :param bool deterministic: Whether a deterministic action should be sampled or not
         (Optional, default is False).

        :rtype: th.Tensor
        :return: The (projected) actions.
        """
        # Update the probability distribution
        self.proba_distribution(mean_actions, log_std, self.y)
        actions, _ = self.get_actions(deterministic=deterministic)
        return actions

    def log_prob_from_params(self, mean_actions: th.Tensor, log_std: th.Tensor) -> Tuple[th.Tensor, th.Tensor]:
        """
        Compute the log probability of taking an action given the distribution parameters.

        :param th.Tensor mean_actions: Mean actions.
        :param th.Tensor log_std: Log standard deviations.

        :rtype: Tuple[th.Tensor, th.Tensor]
        :return: The (projected) actions and the corresponding log probabilities.
        """
        actions = self.actions_from_params(mean_actions, log_std)
        log_prob = self.log_prob(actions)
        return actions, log_prob


class MLPExtractor(nn.Module):

    def __init__(self, feature_dim: int, latent_dim_pi: int = 64, latent_dim_vf: int = 32):
        """
        Feature extract using a Multi-Layer Perceptron (MLP).

        :param int feature_dim: Dimension of the feature vector (can also be the output of another network).
        :param int latent_dim_pi: Dimension of the latent layers for the actor network.
        :param int latent_dim_vf: Dimension of the latent layers for the critic network.
        """
        super().__init__()
        self.feature_dim = feature_dim
        self.latent_dim_pi = latent_dim_pi
        self.latent_dim_vf = latent_dim_vf

        # Actor network
        self.pi_net = nn.Sequential(nn.Linear(feature_dim, self.latent_dim_pi), nn.ReLU(),
                                    nn.Linear(self.latent_dim_pi, self.latent_dim_pi))

        # Critic network
        self.vf_net = nn.Sequential(nn.Linear(feature_dim, self.latent_dim_vf), nn.ReLU(),
                                    nn.Linear(self.latent_dim_vf, self.latent_dim_vf), nn.ReLU())

    def forward(self, features: th.Tensor) -> Tuple[th.Tensor, th.Tensor]:
        """
        :return: latent_policy, latent_value of the specified network.
            If all layers are shared, then ``latent_policy == latent_value``
        """
        latent_pi = self.pi_net(features)
        latent_vf = self.vf_net(features)
        return latent_pi, latent_vf


class MicroPPOPolicy(BasePolicy):
    """
        MicroPPO policy based on Stable Baselines 3's BasePolicy class.

        :param observation_space: Observation space
        :param action_space: Action space
        :param lr_schedule: Learning rate schedule (could be constant). Default is 3e-4.
        :param net_arch: The specification of the policy and value networks.
        :param activation_fn: Activation function
        :param ortho_init: Whether to use or not orthogonal initialization
        :param use_sde: Whether to use State Dependent Exploration or not
        :param log_std_init: Initial value for the log standard deviation
        :param full_std: Whether to use (n_features x n_actions) parameters
            for the std instead of only (n_features,) when using gSDE
        :param use_expln: Use ``expln()`` function instead of ``exp()`` to ensure
            a positive standard deviation (cf paper). It allows to keep variance
            above zero and prevent it from growing too fast. In practice, ``exp()`` is usually enough.
        :param squash_output: Whether to squash the output using a tanh function,
            this allows to ensure boundaries when using gSDE.
        :param features_extractor_class: Features extractor to use.
        :param features_extractor_kwargs: Keyword arguments
            to pass to the features extractor.
        :param share_features_extractor: If True, the features extractor is shared between the policy and value networks.
        :param normalize_images: Whether to normalize images or not,
             dividing by 255.0 (True by default)
        :param optimizer_class: The optimizer to use,
            ``th.optim.Adam`` by default
        :param optimizer_kwargs: Additional keyword arguments,
            excluding the learning rate, to pass to the optimizer
        :param branched_pi_net: Whether to use a branched architecture for the policy network (default: True)
        """

    def __init__(
            self,
            observation_space: spaces.Space,
            action_space: spaces.Space,
            lr_schedule: Schedule,
            net_arch: Union[List[int], Dict[str, List[int]], List[Dict[str, List[int]]], None] = None,
            activation_fn: Type[nn.Module] = nn.Tanh,
            ortho_init: bool = True,
            use_sde: bool = False,
            log_std_init: float = 0.0,
            full_std: bool = True,
            use_expln: bool = False,
            squash_output: bool = False,
            features_extractor_class: Type[BaseFeaturesExtractor] = FlattenExtractor,
            features_extractor_kwargs: Optional[Dict[str, Any]] = None,
            share_features_extractor: bool = True,
            normalize_images: bool = True,
            optimizer_class: Type[th.optim.Optimizer] = th.optim.Adam,
            optimizer_kwargs: Optional[Dict[str, Any]] = None,
            branched_pi_net: Optional[bool] = True

    ):
        if optimizer_kwargs is None:
            optimizer_kwargs = {}
            # Small values to avoid NaN in Adam optimizer
            if optimizer_class == th.optim.Adam:
                optimizer_kwargs["eps"] = 1e-5

        super().__init__(
            observation_space,
            action_space,
            features_extractor_class,
            features_extractor_kwargs,
            optimizer_class=optimizer_class,
            optimizer_kwargs=optimizer_kwargs,
            squash_output=squash_output,
            normalize_images=normalize_images,
        )

        self.feature_dim = get_flattened_obs_dim(observation_space)
        self.ortho_init = ortho_init

        # Action distribution
        self.branched_pi_net = branched_pi_net
        self.action_dist = ActionDistributionWithProjection(action_dim=get_action_dim(action_space),
                                                            branched_network=self.branched_pi_net)

        self._build(lr_schedule)

    def _build(self, lr_schedule: Schedule) -> None:
        """
        Create the networks and the optimizer.

        :param Schedule lr_schedule: Learning rate schedule.
            lr_schedule(1) is the initial learning rate.
        """
        # Feature extractor
        self.mlp_extractor = MLPExtractor(self.feature_dim)

        latent_dim_pi = self.mlp_extractor.latent_dim_pi

        # Actor network -> probability distribution
        if self.branched_pi_net:
            self.policy_net = self.action_dist.proba_distribution_net(latent_dim=latent_dim_pi)
        else:
            self.action_net, self.log_std_net = self.action_dist.proba_distribution_net(latent_dim=latent_dim_pi)

        # Critic network
        self.value_net = nn.Linear(self.mlp_extractor.latent_dim_vf, 1)

        # Network branch for meta-decision variables
        self.meta_decision_net = nn.Sequential(nn.Linear(latent_dim_pi, 1), nn.Sigmoid())

        # Setup optimizer with initial learning rate
        self.optimizer = self.optimizer_class(self.parameters(), lr=lr_schedule(1), **self.optimizer_kwargs)

    def forward(self, obs: th.Tensor, deterministic: bool = False) -> Tuple[th.Tensor, th.Tensor, th.Tensor, th.Tensor]:
        """
        Forward pass in all the networks (actor and critic)

        :param th.Tensor obs: Observation.
        :param bool deterministic: Whether to sample or use deterministic actions.

        :rtype: Tuple[th.Tensor, th.Tensor, th.Tensor, th.Tensor]
        :return: The (projected) action, value, log probability of the action, and the projection loss.
        """
        # Extract latent code from the observation input
        latent_pi, latent_vf = self.mlp_extractor(obs)

        # Evaluate the values given the latent code for the critic
        values = self.value_net(latent_vf)

        # Derive the probability distribution from the latent code for the actor
        distribution = self._get_action_dist_from_latent(latent_pi)

        # Sample the (projected) actions from the probability distribution and compute the projection loss.
        actions, proj_loss = distribution.get_actions(deterministic=deterministic)

        # Compute the log probabilities
        log_prob = distribution.log_prob(actions)
        actions = actions.reshape((-1,) + self.action_space.shape)
        return actions, values, log_prob, proj_loss

    def _get_action_dist_from_latent(self, latent_pi: th.Tensor) -> Distribution:
        """
        Retrieve the action distribution given the latent codes.

        :param th.Tensor latent_pi: Latent code for the actor.

        :rtype: Distribution
        :return: The action distribution.
        """
        if self.branched_pi_net:
            mean_actions, log_std = self.policy_net(latent_pi)
        else:
            mean_actions = self.action_net(latent_pi)
            log_std = self.log_std_net(latent_pi)

        # Meta-decision variable: battery_mode (indicating whether to charge or discharge the battery)
        battery_mode = self.meta_decision_net(latent_pi)

        return self.action_dist.proba_distribution(mean_actions, log_std, y=th.round(battery_mode))

    def _predict(self, observation: th.Tensor, deterministic: bool = False) -> th.Tensor:
        """
        Get the action according to the policy for a given observation.

        :param th.Tensor observation: Observation.
        :param bool deterministic: Whether to use stochastic or deterministic actions.

        :rtype: th.Tensor
        :return: Taken (projected) action according to the policy.
        """
        actions, _ = self.get_distribution(observation).get_actions(deterministic=deterministic)
        return actions

    def evaluate_actions(self, obs: th.Tensor, actions: th.Tensor) -> Tuple[
        th.Tensor, th.Tensor, Optional[th.Tensor], Optional[th.Tensor]]:
        """
        Evaluate actions according to the current policy, given the observations.

        :param th.Tensor obs: Observation.
        :param th.Tensor actions: Actions.

        :rtype: Tuple[th.Tensor, th.Tensor, Optional[th.Tensor], Optional[th.Tensor]]
        :return: Estimated value, log likelihood of taking those actions, entropy of the action distribution
        and the projection loss.
        """
        # Extract latent code from the observation input
        latent_pi, latent_vf = self.mlp_extractor(obs)

        # Derive the probability distribution from the latent code for the actor
        distribution = self._get_action_dist_from_latent(latent_pi)

        # Compute the projection loss.
        _, proj_loss = distribution.get_actions()

        # Compute the log probabilities, values and entropy.
        log_prob = distribution.log_prob(actions)
        values = self.value_net(latent_vf)
        entropy = distribution.entropy()
        return values, log_prob, entropy, proj_loss

    def get_distribution(self, obs: th.Tensor) -> Distribution:
        """
        Get the current policy distribution given the observations.

        :param th.Tensor obs: Observation.

        :rtype: Distribution
        :return: The action distribution.
        """
        latent_pi, _ = self.mlp_extractor(obs)
        return self._get_action_dist_from_latent(latent_pi)

    def predict_values(self, obs: th.Tensor) -> th.Tensor:
        """
        Get the estimated values according to the current policy given the observations.

        :param th.Tensor obs: Observation.

        :rtype: th.Tensor
        :return: The estimated values.
        """
        _, latent_vf = self.mlp_extractor(obs)
        return self.value_net(latent_vf)
