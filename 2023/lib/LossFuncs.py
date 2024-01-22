# **************************************************************************** #
#                                                                              #
#                                                         :::      ::::::::    #
#    LossFuncs.py                                       :+:      :+:    :+:    #
#                                                     +:+ +:+         +:+      #
#    By: smodesto <smodesto@student.42.fr>          +#+  +:+       +#+         #
#                                                 +#+#+#+#+#+   +#+            #
#    Created: 2023/08/25 00:29:08 by smodesto          #+#    #+#              #
#    Updated: 2023/08/25 00:29:46 by smodesto         ###   ########.fr        #
#                                                                              #
# **************************************************************************** #

import numpy as np

def mse(y_true, y_pred):
    return np.mean(np.power(y_true-y_pred, 2));

def mse_derivative(y_true, y_pred):
    return 2*(y_pred-y_true)/y_true.size;
