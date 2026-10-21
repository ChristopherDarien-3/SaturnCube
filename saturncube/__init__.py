"""saturncube: one fit/predict/evaluate interface across many model families."""
from .base import BaseModel, REGISTRY, available, create, register
from .vector import (MLP, DeepFeedforward, ResNet, DenseNet, Highway, SqueezeExcitation,
                     MixtureOfExperts, GatedMixtureOfExperts, Hypernetwork, ImplicitNetwork,
                     DeepEquilibrium, KAN, RBF, ExtremeLearningMachine, RandomNeuralNetwork)

from .rules import (FeedbackAlignment, DirectFeedbackAlignment, ForwardGradient, LocalErrorLearning,
                    SyntheticGradients, EquilibriumPropagation, ContrastiveHebbian, PredictiveCoding)
from .optim import OPTIMIZERS, Lion, LAMB, make_optimizer

from .unsupervised import (Autoencoder, DenoisingAutoencoder, VAE, MaskedPrediction, Contrastive,
                           BYOL, DeepClustering, HebbianLearner, TemporalContrastive, LatentPrediction,
                           EnergyBasedModel)
from .strategies import EWC, LwF, Replay, Distillation, distill, query
from .rl import (GridWorld, PotentialShaping, QLearning, SARSA, ExpectedSARSA, MonteCarloControl,
                 BernoulliBandit, EpsilonGreedyBandit, UCB1, ThompsonBandit, LinUCB, run_bandit,
                 OneHotObs, PointMass, SARSALambda, DynaQ, MaxEntIRL)
from .deeprl import (DQN, DoubleDQN, DuelingDQN, PrioritizedDQN, REINFORCE, ActorCritic, A2C, PPO, TRPO, SAC,
                     ModelBasedMPC, PreferenceRewardModel, LearnedRewardEnv, ReplayBuffer)

from .sequence import (SimpleRNN, ElmanNetwork, LSTM, GRU, BidirectionalRNN, BidirectionalLSTM,
                       BidirectionalGRU, ContinuousTimeRNN, NeuralStateSpace, S4, Mamba, PlasticRNN,
                       JordanNetwork, EchoStateNetwork, LiquidStateMachine, RTRLRNN)

from .attention import (MultiHeadAttention, LinearAttention, TransformerLayer, TransformerEncoder,
                        TransformerDecoder, LinearAttentionTransformer, LocalAttentionTransformer,
                        SparseAttentionTransformer, MultiQueryTransformer, GroupedQueryTransformer,
                        UniversalTransformer, MemoryAugmentedTransformer, Perceiver, SSMAttentionHybrid,
                        RecurrentAttention, RetrievalAugmented, EncoderDecoderTransformer)

from .memory import (HopfieldNetwork, ModernHopfield, AssociativeMemory, ContentAddressableMemory,
                     ProductKeyMemory, ProductKeyMemoryNet, MemoryNetwork, KeyValueMemoryNetwork,
                     NeuralTuringMachine, DifferentiableNeuralComputer)

from .metric import (Siamese, TripletNetwork, TwinTower, ContrastiveEmbedding, PrototypicalNetwork,
                     RelationNetwork, MatchingNetwork)
from .meta import (Task, SineTasks, GaussianTasks, MAML, Reptile, transfer, fit_curriculum, fit_self_paced,
                   GenerativeReplay)

from .graph import (Graph, Tree, GraphConvolutionalNetwork, GraphAttentionNetwork, GraphSAGE,
                    GraphIsomorphismNetwork, MessagePassingNetwork, RelationalGCN, GraphTransformer,
                    GatedGraphNetwork, RecurrentGraphNetwork, HypergraphNeuralNetwork, ProgramGraphNetwork,
                    TreeLSTM, TreeStructuredNetwork, PointerNetwork, PointerGenerator)

__version__ = "0.4.0"
