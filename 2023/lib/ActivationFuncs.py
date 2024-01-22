# **************************************************************************** #
#                                                                              #
#                                                         :::      ::::::::    #
#    ActivationFuncs.py                                 :+:      :+:    :+:    #
#                                                     +:+ +:+         +:+      #
#    By: smodesto <smodesto@student.42.fr>          +#+  +:+       +#+         #
#                                                 +#+#+#+#+#+   +#+            #
#    Created: 2023/08/25 00:17:47 by smodesto          #+#    #+#              #
#    Updated: 2023/08/25 00:28:46 by smodesto         ###   ########.fr        #
#                                                                              #
# **************************************************************************** #

import numpy as np

def tanh(x):
    return np.tanh(x)

def tanh_derivative(x):
    return 1 - np.tanh(x) ** 2
