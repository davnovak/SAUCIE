from .model import SAUCIE
from .utils import *
from .loader import Loader
import tensorflow as tf

class DimRed:
    def __init__(
        self,
        input_dim,
        lambda_b=0,
        lambda_c=0,
        layer_c=0,
        lambda_d=0,
        layers=[512,256,128,2],
        activation=lrelu,
        learning_rate=.001
    ):
        """SAUCIE model for dimensionality reduction

        Args:
            :param input_dim: the dimensionality of the data
            :param lambda_b: the coefficient for the MMD regularization
            :param lambda_c: the coefficient for the ID regularization
            :param layer_c: the index of layer_dimensions that ID regularization should be applied to (usually len(layer_dimensions)-2)
            :param lambda_d: the coefficient for the intracluster distance regularization
            :param activation: the nonlinearity to use in the hidden layers
            :param learning_rate: the learning_rate to use while training
        """

        self.model = SAUCIE(
            input_dim=input_dim,
            lambda_b=lambda_b,
            lambda_c=lambda_c,
            layer_c=layer_c,
            lambda_d=lambda_d,
            layers=layers,
            activation=activation,
            learning_rate=learning_rate,
            restore_folder='',
            save_folder='',
            limit_gpu_fraction=.3,
            no_gpu=False
        )
    
    def fit_transform(self, X, steps=1000):
        train_loader = Loader(X, shuffle=True)
        self.model.train(train_loader, steps=steps)
        eval_loader = Loader(X, shuffle=False)
        embedding = self.model.get_embedding(eval_loader)
        return embedding

