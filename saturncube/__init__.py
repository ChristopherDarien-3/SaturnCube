"""saturncube: one fit/predict/evaluate interface across many model families."""
from .base import BaseModel, REGISTRY, available, create, register
from .vector import (MLP, DeepFeedforward, ResNet, DenseNet, Highway, SqueezeExcitation,
                     MixtureOfExperts, GatedMixtureOfExperts, Hypernetwork, ImplicitNetwork,
                     DeepEquilibrium, KAN, RBF, ExtremeLearningMachine, RandomNeuralNetwork)

__version__ = "0.1.0"
