import numpy as np
import cupy as cp
import math
import os

# variables from Program mambl1D_AB
class pfcParms:
    def __init__(self):
        self.name = None

        # variables from module global_variables
        self.L = 2024

        # common variables from Program mambl1D_AB
        #common/converge
        self.TOL = 1e-6
        self.err_s = 0
        self.nmbr_eval = 0

        #common/q12
        self.G_0 = None
        self.deltaj0 = None
        self.qA2 = None
        self.qB2 = None
        self.qAB2 = None

        #common/suf
        self.du = None
        self.epsA = None
        self.epsB = None
        self.beta_A = None
        self.beta_B = None
        self.beta_AB = None
        self.alpha_AB = None
        self.width_per = None

        #common/interf
        self.m_gap_nA = None
        self.m_gap_nB = None
        self.width_nA = None
        self.width_nB = None
        self.width_A = None
        self.width_B = None
        self.gamma_nA = None
        self.gamma_nB = None
        self.Es = None

        #common/interf_coef
        self.dchi_j = None
        self.chi_j2 = None
        self.qu = None

        # local variables from Program mambl1D_AB
        self.q0 = None
        self.q02 = None
        self.qA = None
        self.qB = None

        self.delta_x = None
        self.delta_y = None
        self.eps_m = None

        self.ntype = 1 # ntype=1: set initial configuration; ntype=2: input from file
        self.alpha_11 = None
        self.alpha_22 = None
        self.sigma1 = None
        self.sigma2 = None
        self.sig1_dt = None
        self.sig2_dt = None
        self.delta_2 = None
        self.aq = None
        self.b = None
        self.c = None
        self.sigmaj1 = None
        self.sigmaj2 = None
        self.dt = None
        self.dti = None
        self.dtime = None
        self.iter0 = None
        self.ntype = None
        self.iter = None
        self.idum = None
        self.n_0 = None
        self.n_i = None
        self.nmbr_eval = None

        self.iTheta = 0

        # default values taken from theta=0
        self.nAo_s = -0.2682240335169304e0
        self.nAo_l = -0.4435164100577305e0
        self.nBo_s = -0.2682240335169304e0
        self.nBo_l = -0.4435164100577305e0
        self.A1o = CMPLX(-8.7833886739079561e-2,-0.2081951877739002e0)
        self.A2o = CMPLX(-0.2197898142019807e0,-5.2463915154257496e-2)
        self.A3o = CMPLX(0.1063354260412832e0,-0.1993810548686150e0)
        self.B1o = CMPLX(-9.4460100511589082e-2,0.2052737617858650e0)
        self.B2o = CMPLX(0.1095166345985139e0,0.1976515407502470e0)
        self.B3o = CMPLX(-0.2189209317958431e0,5.5979045079558511e-2)
        # self.time0 = 0
        # self.iter0 = 1

    def __repr__(self) -> str:
        return f'pfcParms: {self.__dict__}'

# variables from module global_variables
class pfcData:
    def __init__(self):
        # self.L = 2024
        self.A0 = None
        self.A0q = None
        self.B0 = None
        self.B0q = None
        self.nA0 = None
        self.nA0q = None
        self.nB0 = None
        self.nB0q = None
        self.exp_j11 = None
        self.exp_j12 = None
        self.exp_j21 = None
        self.exp_j22 = None
        self.cf_j1 = None
        self.cf2_j1 = None
        self.cf_j2 = None
        self.cf2_j2 = None
        self.exp_11 = None
        self.exp_12 = None
        self.exp_21 = None
        self.exp_22 = None
        self.cf_1 = None
        self.cf2_1 = None
        self.cf_2 = None
        self.cf2_2 = None
        self.sigma12 = None
        self.sig1_alpha = None
        self.sig2_alpha = None
        self.alpha_12 = None
        self.alpha_21 = None
        self.sigmaj12 = None
        self.sigj1_alpha = None
        self.sigj2_alpha = None
        self.alpha_j12 = None
        self.alpha_j21 = None
        self.vA = None
        self.vB = None
        self.gA = None
        self.gB = None
        self.w = None
        self.u = None
        self.mB = None
        self.q02_mB = None
        self.q02 = None
        self.qu2 = None
        self.isignamj = None
        self.isignam = None
        self.scale1d_b = None

        # variables from Program mambl1D_AB
        self.time0 = 0
        self.iter0 = 1


    def __repr__(self) -> str:
        return f'pfcData: {self.__dict__}'


class pfc1d:
    def __init__(self):
        self.pfcParms = pfcParms()
        self.pfcData = pfcData()
        self.runHistory = []

    def __repr__(self) -> str:
        return f'pfc2d: {self.__dict__}'
     
    def initParms(self, name=None, L=None, iTheta=None, intfTanh=True, outputDir=None, outputRoot=None):
        self.pfcParms.name = name
        if L is not None:
            self.pfcParms.L= L
        if iTheta is not None:
            self.pfcParms.iTheta = iTheta
        else:
            iTheta = 0
        self._setTheta()
         
        self.outputDir = outputDir
        self.outputRoot = outputRoot
       
        # initialize
        self.pfcParms.q0 = 1
        self.pfcParms.q02 = self.pfcParms.q0**2
        self.pfcParms.qA = self.pfcParms.q0
        self.pfcParms.qB = self.pfcParms.q0
        self.pfcParms.qA2 = self.pfcParms.qA**2
        self.pfcParms.qB2 = self.pfcParms.qB**2
        self.pfcData.vA = 1.
        self.pfcData.vB = 1.
        self.pfcData.gA = 0.5
        self.pfcData.gB = 0.5
        self.pfcData.w = 0.3
        self.pfcData.u = 0.3
        
        self.pfcParms.epsA = 0.3
        self.pfcParms.epsB = 0.3
        self.pfcParms.beta_A = 1
        self.pfcParms.beta_B = 1
        self.pfcParms.beta_AB = 0.
        self.pfcParms.alpha_AB = 0.5

        self.pfcData.mB = 1 # mB=M_B/M_A; mA=1

        self.pfcParms.eps_m = 0
        self.pfcParms.width_per = 0.85 # interface range: 85% of misc. gap

    def printParms(self):
        # pretty print the parms, 1 vaule per line
        print('pfcParms:')
        for key, value in self.pfcParms.__dict__.items():
            print(f'{key}: {value}')
    
    def _setTheta(self):
        iTheta = self.pfcParms.iTheta

        # set the angle theta
        self.pfcParms.theta = iTheta * math.pi / 180

        if iTheta == 0:
            # ?? why does orientation affect density nAo_s, nAo_l, nBo_s, nBo_l ??
            self.pfcParms.nAo_s = -0.2682240335169304
            self.pfcParms.nAo_l = -0.4435164100577305
            self.pfcParms.A1o = CMPLX(-0.087833886739079561,-0.2081951877739002)
            self.pfcParms.A2o = CMPLX(-0.2197898142019807,-0.052463915154257496)
            self.pfcParms.A3o = CMPLX(0.1063354260412832,-0.1993810548686150)
            self.pfcParms.B1o = CMPLX(-0.094460100511589082,0.2052737617858650)
            self.pfcParms.B2o = CMPLX(0.1095166345985139,0.1976515407502470)
            self.pfcParms.B3o = CMPLX(-0.2189209317958431,0.055979045079558511)
            # ?? do these need to change if L changes: ??
            self.pfcParms.wid_N = 6.4953616398e0/2.5123056e0 # width/(2*atanh(width_per)), width_per=0.85; use width_nA(1)
            self.pfcParms.wid_A = 7.4711083227e0/2.5123056e0 # use width_A(1) ! at tmax=10^7, L=2048
        elif iTheta == 90:
            self.pfcParms.nAo_s = -0.2682240752088968
            self.pfcParms.nAo_l = -0.4435164435660391
            self.pfcParms.A1o = CMPLX(-0.087833881982373824,-0.2081951765861624)
            self.pfcParms.A2o = CMPLX(-0.2197898115052828,-0.052463895116867998)
            self.pfcParms.A3o = CMPLX(0.1063354123209136,-0.1993810340288277)
            self.pfcParms.B1o = CMPLX(-0.094460100515079734,0.2052737484152882)
            self.pfcParms.B2o = CMPLX(0.1095166182875313,0.1976515414844259)
            self.pfcParms.B3o = CMPLX(-0.2189209079654175,0.055979038035728526)
            # ?? do these need to change if L changes: ??
            self.pfcParms.wid_N = 6.5237979053e0/2.5123056e0 # width/(2*atanh(width_per)), width_per=0.85; use width_nA(1)
            self.pfcParms.wid_A = 1.0817029688e1/2.5123056e0 # use width_A(1) ! at tmax=10^7, L=2048
        # ?? what about iTheta=30, 19, 41, etc.) ??
        
        self.pfcParms.nBo_s = self.pfcParms.nAo_s
        self.pfcParms.nBo_l = self.pfcParms.nAo_l



        
def CMPLX(real, imag):
    return real + 1j*imag