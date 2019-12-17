""" Convenience module that contains functionality do convert optical propagation loss between different units. """

from math import e
import numpy as np


def dBcm_to_loss_per_m(dBcm):
    """
    Converts dB/cm loss value to loss per meter (alpha)

    :param dBcm: Loss in dB/cm
    :type dBcm: Numeric
    :return: Loss per meter (alpha)
    """
    return 10*dBcm/np.log10(e)

def loss_per_m_to_dBcm(loss_per_m):
    """
    Converts loss per meter (alpha) in dB/cm

    :param loss_per_m: Loss per meter (alpha)
    :type loss_per_m: Numeric
    :return: Loss in dB/cm
    """
    return np.log10(e)*loss_per_m/10.0

def attenuation_to_loss_per_meter(attenuation, length):
    """
    Converts attenuation to loss per meter

    :param attenuation: Attenuation
    :type attenuation: Numeric
    :param length: propagation distance in meters
    :return: loss per meter
    """

    return -np.log(attenuation) * 2.0 / length

def loss_per_meter_to_attenuation(loss_per_m, length):
    """
    Converts loss per meter to attenuation

    :param loss_per_m: Loss per meter (alpha)
    :param length: propagation distance in meters
    """

    return np.exp(-loss_per_m / 2.0 * length)


def dBcm_to_attenuation(dBcm, length):
    """
    Converts dB/cm to attenuation

    :param dBcm: Loss in dB/cm
    :param length: propagation distance in meters
    """

    return np.exp(-dBcm_to_loss_per_m(dBcm) / 2.0 * length)



def attenuation_to_dBcm(attenuation, length):
    """
    Converts attenuation to dB/cm

    :param attenuation:
    :param length: propagation distance in meters
    """
    return loss_per_m_to_dBcm(attenuation_to_loss_per_meter(attenuation, length))
