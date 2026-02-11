import cupy as cp

from .initialize_system import initialize_parms, initialize_fields
from .simulate import simulate
from .report_state import report_state

class pfcParms:
    def __init__(self):
        # system geometry and discretization
        self.lx = 180
        self.ly = 180
        # refactor to initialization routine
        self.nu = 10
        self.mx = 10
        self.my = 18
        self.theta = 0.0
        self.dy = 0.7516098215
        self.dx = 0.7232368879
        self.qx0 = 0.9652877340
        self.qy0 = 0.8359637045

        # energy parameters
        self.epsA = 0.3
        self.epsB = 0.3
        self.alphaAB = 0.5
        self.betaAB = 0.0
        self.betaA = 1.0
        self.betaB = 1.0
        self.gA = 0.5
        self.gB = 0.5
        self.omega = 0.3
        self.mu = 0.0

        # order parameter parameters
        self.disordered = -0.44330000386
        self.ordered = -0.2680999637
        self.ordered_amplitude = 0.226007814838

        # time parameters
        self.dt = 0.01
        self.nstep = 1000

        pass

class pfcData:
    def __init__(self):
        # order parameters
        #   real space
        self.phiA = None
        self.phiB = None
        #   reciprocal space
        self.phiqA = None
        self.phiqB = None
        pass

class pfcModel:
    def __init__(self):
        self.parms = pfcParms()
        self.data = pfcData()
        self.modelName = 'pfc-binary-full'
        self.version = '1.0'

        self._initialize_parms = initialize_parms
        self._initialize_fields = initialize_fields
        self._simulate = simulate
        self._report_state = report_state

    def initialize_parms(self, parm_dict):
        self._initialize_parms(self.parms, parm_dict)

    def initialize_fields(self):
        self._initialize_fields(self.data, self.parms)
    
    # def initialize(self):
    #     self.initialize_system(self.parms, self.data)

    # def simulate(self):
    #     self.simulate(self.parms, self.data)

    # def report_state(self):
    #     self.report_state(self.parms, self.data)