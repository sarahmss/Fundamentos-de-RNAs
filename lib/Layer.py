# **************************************************************************** #
#                                                                              #
#                                                         :::      ::::::::    #
#    Layer.py                                           :+:      :+:    :+:    #
#                                                     +:+ +:+         +:+      #
#    By: smodesto <smodesto@student.42.fr>          +#+  +:+       +#+         #
#                                                 +#+#+#+#+#+   +#+            #
#    Created: 2023/08/24 23:19:48 by smodesto          #+#    #+#              #
#    Updated: 2023/08/25 00:13:49 by smodesto         ###   ########.fr        #
#                                                                              #
# **************************************************************************** #

""" @brief: Basic class to handle input, output,
            and forward/backward propagation  """
class Layer: 
    def __init__(self):
        self.input = None
        self.output = None
    def forward_propagation(self, input_data):
        raise NotImplementedError
    def backward_propagation(self, input_data, learning_rate):
        raise NotImplementedError
