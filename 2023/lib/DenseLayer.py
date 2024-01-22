# **************************************************************************** #
#                                                                              #
#                                                         :::      ::::::::    #
#    DenseLayer.py                                      :+:      :+:    :+:    #
#                                                     +:+ +:+         +:+      #
#    By: smodesto <smodesto@student.42.fr>          +#+  +:+       +#+         #
#                                                 +#+#+#+#+#+   +#+            #
#    Created: 2023/08/24 23:24:23 by smodesto          #+#    #+#              #
#    Updated: 2023/08/25 00:12:53 by smodesto         ###   ########.fr        #
#                                                                              #
# **************************************************************************** #

from Layer import Layer
import numpy as np

""" @brief: Dense layers implementation.
            Here, each neuron is completely connected to every other neuron in
            previous layer.
    @params:         
        - input_dim: number of input neurons   
        - input_dim: number of output neurons   
"""
class DenseLayer(Layer):
    def __init__(self, input_dim, output_dim):
        self.weights = np.random.rand(input_dim, output_dim) - 0.5
        self.bias = np.random.rand(1, output_dim) - 0.5
    
    def forward_propagation(self, input_data):
        self.input = input_data
        self.output = np.dot(self.input, self.weights) + self.bias # Y = WX + B
        return self.output
    
    def backward_propagation(self, output_error, learning_rate):
        input_error = np.dot(output_error, self.weights.T)
        weights_error = np.dot(self.input.T, output_error)
        self.weights -= learning_rate * weights_error
        self.bias -= learning_rate * output_error
        return input_error
    