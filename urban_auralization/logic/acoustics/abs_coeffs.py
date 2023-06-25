import numpy as np
from urban_auralization.logic.acoustics import constants
from scipy.interpolate import interp1d

class AbsorptionCoeff():
    def __init__(self, abs_coeff : dict = None):
        self.abs_coeff = abs_coeff
        self.f_third = constants.f_third

    def extrapolate(self):
        """
        Extrapolates abs. coefficients, return the abs coeffs in the third octave bands
        Returns
        -------

        """
        f_existing = np.array(list(self.abs_coeff.keys()))
        alpha_existing = np.array(list(self.abs_coeff.values()))
        f = interp1d(f_existing, alpha_existing, fill_value=(alpha_existing[0], alpha_existing[-1]), bounds_error=False)
        alpha_third = f(self.f_third)

        return alpha_third

if __name__ == "__main__":
    rdm_filter = {125 : 0.10,
             250 : 0.40,
             500: 0.62,
             1000 : 0.70,
             2000 : 0.63,
             4000 : 0.888}


    abs_coeff = AbsorptionCoeff(rdm_filter)
    abs_coeff.extrapolate()