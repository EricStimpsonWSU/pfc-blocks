"""
Compatibility layer for a standard (non-log) PFC2D model.

This provides the legacy-style API used in older notebooks while mapping to the
refactored pfc_general components. The standard model is implemented by using
LogPFCModel2D with Hln=0 and Hng=0, which removes the logarithmic and vacancy
terms while preserving the same linear and polynomial terms.
"""

from .pfc2d_vacancy_compat import PFC2D_Vacancy, PFC2D_Vacancy_Parms


class PFC2D_Standard_Parms(PFC2D_Vacancy_Parms):
    """
    Parameter container matching standard PFC2D interface.
    """
    def __init__(self):
        super().__init__()
        # Standard PFC does not use log or vacancy terms.
        self.Hln = 0.0
        self.Hng = 0.0
        self.a = 0.0


class PFC2D_Standard(PFC2D_Vacancy):
    """
    Standard (non-log) PFC2D compatibility wrapper.

    This class preserves the same API as legacy PFC2D models but forces
    logarithmic and vacancy terms to zero.
    """

    def __init__(self):
        super().__init__()
        self.parms = PFC2D_Standard_Parms()

    def InitParms(self):
        # Force standard (non-log) parameters.
        self.parms.Hln = 0.0
        self.parms.Hng = 0.0
        self.parms.a = 0.0
        super().InitParms()
