import unittest
import numpy as np
from colna.lossconversion import dBcm_to_loss_per_m, loss_per_m_to_dBcm, attenuation_to_dBcm, attenuation_to_loss_per_meter, loss_per_meter_to_attenuation, dBcm_to_attenuation

class TestUnitconverter(unittest.TestCase):

    def test_loss_conversion(self):
        self.assertEqual(dBcm_to_loss_per_m(1), 100/(10*np.log10(np.exp(1))))
        self.assertEqual(attenuation_to_dBcm(1,1), 0)

        self.assertEqual(loss_per_m_to_dBcm(dBcm_to_loss_per_m(1)), 1 )
        self.assertEqual(dBcm_to_attenuation(attenuation_to_dBcm(1,2),2), 1)
        self.assertEqual(loss_per_meter_to_attenuation(attenuation_to_loss_per_meter(1,2),2), 1)


