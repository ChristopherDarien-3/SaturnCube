"""saturncube: one fit/predict/evaluate interface across many model families."""
from .base import BaseModel, REGISTRY, available, create, register
from .vector import (MLP, DeepFeedforward, ResNet, DenseNet, Highway, SqueezeExcitation,
                     MixtureOfExperts, GatedMixtureOfExperts, Hypernetwork, ImplicitNetwork,
                     DeepEquilibrium, KAN, RBF, ExtremeLearningMachine, RandomNeuralNetwork)

from .rules import FeedbackAlignment, DirectFeedbackAlignment
from .optim import OPTIMIZERS, Lion, LAMB, make_optimizer

from .unsupervised import (Autoencoder, DenoisingAutoencoder, VAE, MaskedPrediction, Contrastive,
                           BYOL, DeepClustering, HebbianLearner)
from .strategies import EWC, LwF, Replay, Distillation, distill, query
from .rl import (GridWorld, PotentialShaping, QLearning, SARSA, ExpectedSARSA, MonteCarloControl,
                 BernoulliBandit, EpsilonGreedyBandit, UCB1, ThompsonBandit, LinUCB, run_bandit)

__version__ = "0.3.0"
