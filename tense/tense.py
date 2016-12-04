from typing import List, Callable, Tuple

import tensorflow as tf


class ThinkinMuscle:
    """
    util class that simplifies building a network.
    """
    def __init__(self):
        self.Ws = []  # type: List[tf.Variable]
        """ list of weight matrices in each layer """
        self.bs = []  # type: List[tf.Variable]
        """ list of bias vectors in each layer """
        self.ys = []  # type: List[tf.Tensor]
        """ list of outputs of each layer. these are tensor operations that need
        to be executed to get the actual output values """

    @classmethod
    def make_weights(cls, shape: List[int]) -> tf.Variable:
        init = tf.truncated_normal(shape, stddev=0.1)
        return tf.Variable(init)

    @classmethod
    def make_bias(cls, shape: List[int]) -> tf.Variable:
        init = tf.constant(0.1, shape=shape)
        return tf.Variable(init)

    def add_layer(
            self,
            weights_shape: List[int],
            activation: Callable[[tf.Tensor, tf.Variable, tf.Variable], tf.Tensor],
            input_override: tf.Variable = None
    ) -> Tuple[tf.Variable, tf.Variable, tf.Tensor]:
        """
        adds a layer to the neural net.

        TODO check compatibility with the previous layer's shape.

        TODO provide way to customize how weights and biases are init. currently
        they're truncated normals

        :param weights_shape: shape of the weights. the last element is used as the
            shape of the bias vector
        :param activation: function that will generate/provide a tensor op that will
            be used as the activation function for the nodes in this layer. it gets
            provided the input tensor and previous weights and biases
        :param input_override: if given, will be passed as input to the activation
            function, instead of the previous layer's output.
        :return: latest weights, bias, and output
        """
        self.Ws.append(ThinkinMuscle.make_weights(weights_shape))
        self.bs.append(ThinkinMuscle.make_bias([weights_shape[-1]]))

        input_tensor = self.ys[-1] if input_override is None else input_override
        self.ys.append(activation(input_tensor, self.Ws[-1], self.bs[-1]))

        activation(input_tensor, self.Ws[-1], self.bs[-1])

        return self.Ws[-1], self.bs[-1], self.ys[-1]

    def output(self) -> tf.Tensor:
        """
        :return: the last tensor layer in the network
        """
        return self.ys[-1]
