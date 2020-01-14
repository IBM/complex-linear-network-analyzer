# +-----------------------------------------------------------------------------+
# |  Copyright 2019 IBM Research - Zurich                                       |
# |                                                                             |
# |  Licensed under the Apache License, Version 2.0 (the "License");            |
# |  you may not use this file except in compliance with the License.           |
# |  You may obtain a copy of the License at                                    |
# |                                                                             |
# |      http://www.apache.org/licenses/LICENSE-2.0                             |
# |                                                                             |
# |  Unless required by applicable law or agreed to in writing, software        |
# |  distributed under the License is distributed on an "AS IS" BASIS,          |
# |  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.   |
# |  See the License for the specific language governing permissions and        |
# |  limitations under the License.                                             |
# +-----------------------------------------------------------------------------+
# |  Authors: Lorenz K. Mueller, Pascal Stark                                   |
# +-----------------------------------------------------------------------------+

""" Convenience module that contains functionality do convert optical propagation loss between different units. """

from math import e
import numpy as np

def dBcm_to_loss_per_m(dBcm):
    """
    Converts dB/cm to loss per meter (alpha)

    :param dBcm: Loss in dB/cm
    :type dBcm: float
    :returns: Loss per meter (alpha)
    :rtype: float
    """
    return 10*dBcm/np.log10(e)

def loss_per_m_to_dBcm(loss_per_m):
    """
    Converts loss per meter (alpha) to dB/cm

    :param loss_per_m: Loss per meter (alpha)
    :type loss_per_m: float
    :retursn: Loss in dB/cm
    :rtype: float
    """
    return np.log10(e)*loss_per_m/10.0

def attenuation_to_loss_per_meter(attenuation, length):
    """
    Converts attenuation to loss per meter

    :param attenuation: Attenuation
    :type attenuation: float
    :param length: propagation distance in meters
    :type length: float
    :returns: loss per meter
    :rtype: float
    """

    return -np.log(attenuation) * 2.0 / length

def loss_per_meter_to_attenuation(loss_per_m, length):
    """
    Converts loss per meter to attenuation

    :param loss_per_m: Loss per meter (alpha)
    :type loss_per_m: float
    :param length: propagation distance in meters
    :type length: float
    :returns: Attenuation value
    :rtype: float
    """

    return np.exp(-loss_per_m / 2.0 * length)


def dBcm_to_attenuation(dBcm, length):
    """
    Converts dB/cm to attenuation

    :param dBcm: Loss in dB/cm
    :type dBcm: float
    :param length: propagation distance in meters
    :type length: float
    :returns: Attenuation value
    :rtype: float
    """

    return np.exp(-dBcm_to_loss_per_m(dBcm) / 2.0 * length)



def attenuation_to_dBcm(attenuation, length):
    """
    Converts attenuation to dB/cm

    :param attenuation: Attenuation
    :type attenuation: float
    :param length: propagation distance in meters
    :type length: float
    :returns: loss in dB/cm
    :rtype: float
    """
    return loss_per_m_to_dBcm(attenuation_to_loss_per_meter(attenuation, length))
