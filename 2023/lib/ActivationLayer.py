# **************************************************************************** #
#                                                                              #
#                                                         :::      ::::::::    #
#    ActivationLayer.py                                 :+:      :+:    :+:    #
#                                                     +:+ +:+         +:+      #
#    By: smodesto <smodesto@student.42.fr>          +#+  +:+       +#+         #
#                                                 +#+#+#+#+#+   +#+            #
#    Created: 2023/08/25 00:08:20 by smodesto          #+#    #+#              #
#    Updated: 2023/08/25 00:22:52 by smodesto         ###   ########.fr        #
#                                                                              #
# **************************************************************************** #

from Layer import Layer

"""
    @brief: Activation function implementation through a layer.
            Here, non linearity is added to enable learning complex process.
"""
class ActivationLayer(Layer):
    def __init__(self, activation, activation_derivative):
        self.activation = activation
        self.activation_derivative = activation_derivative
    
    def forward_propagation(self, input_data):
        self.input = input_data
        self.output = self.activation(self.input)
        return (self.output)
    
    def backward_propagation(self, output_error, learning_rate):
        return (self.activation_derivative(self.input) * output_error)