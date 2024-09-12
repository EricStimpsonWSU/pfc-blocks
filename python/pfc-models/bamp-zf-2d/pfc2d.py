import numpy as np
import cupy as cp
import math
import os

class pfcParms:
    def __init__(self):
        self.name = None
        self.Lx = 400
        self.Ly = 400

        self.iTheta = 0

        self.nAo_s = None
        self.nAo_l = -0.4435164100577305
        self.A1o = None
        self.A2o = None
        self.A3o = None
        self.B1o = None
        self.B2o = None
        self.B3o = None

    def __repr__(self) -> str:
        return f'pfcParms: {self.__dict__}'

class pfcData:
    def __init__(self):
        # self.fn=(3.67704032591281e-02+2.58282403054995e-01)/2.0
        # self.fn=(0.258303076+0.03768087551)/2.0
        self.ffac = 2.74
        self.ao = 2.51

    def __repr__(self) -> str:
        return f'pfcData: {self.__dict__}'

class pfc2d:
    def __init__(self):
        self.pfcParms = pfcParms()
        self.pfcData = pfcData()
        self.runHistory = []

    def __repr__(self) -> str:
        return f'pfc2d: {self.__dict__}'
     
    def initParms(self, name=None, Lx=None, Ly=None, iTheta=None, ):
        self.pfcParms.name = name
        if Lx is not None:
            self.pfcParms.Lx = Lx
        if Ly is not None:
            self.pfcParms.Ly = Ly
        if iTheta is not None:
            self.pfcParms.iTheta = iTheta

        if iTheta == 0:
            self.pfcParms.nAo_s = -0.2682240335169304
            self.pfcParms.nAo_l = -0.4435164100577305
            self.pfcParms.A1o = CMPLX(-0.087833886739079561,-0.2081951877739002)
            self.pfcParms.A2o = CMPLX(-0.2197898142019807,-0.052463915154257496)
            self.pfcParms.A3o = CMPLX(0.1063354260412832,-0.1993810548686150)
            self.pfcParms.B1o = CMPLX(-0.094460100511589082,0.2052737617858650)
            self.pfcParms.B2o = CMPLX(0.1095166345985139,0.1976515407502470)
            self.pfcParms.B3o = CMPLX(-0.2189209317958431,0.055979045079558511)
        elif iTheta == 90:
            self.pfcParms.nAo_s = -0.2682240752088968
            self.pfcParms.nAo_l = -0.4435164435660391
            self.pfcParms.A1o = CMPLX(-0.087833881982373824,-0.2081951765861624)
            self.pfcParms.A2o = CMPLX(-0.2197898115052828,-0.052463895116867998)
            self.pfcParms.A3o = CMPLX(0.1063354123209136,-0.1993810340288277)
            self.pfcParms.B1o = CMPLX(-0.094460100515079734,0.2052737484152882)
            self.pfcParms.B2o = CMPLX(0.1095166182875313,0.1976515414844259)
            self.pfcParms.B3o = CMPLX(-0.2189209079654175,0.055979038035728526)

    def printParms(self):
        # pretty print the parms, 1 vaule per line
        print('pfcParms:')
        for key, value in self.pfcParms.__dict__.items():
            print(f'{key}: {value}')
        
        
def CMPLX(real, imag):
    return real + 1j*imag