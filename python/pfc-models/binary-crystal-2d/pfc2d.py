import numpy as np
import cupy as cp
import math
import os

class pfcParms:
    def __init__(self):
        self.name = None
        self.Lx = None
        self.Ly = None
        self.dx = None
        self.dy = None
        self.dt = None
        self.alpha = None
        self.Vo = None
        self.ns = None
        self.nl = None
        self.epsA = None
        self.epsB = None
        self.vA = None
        self.vB = None
        self.gA = None
        self.gB = None
        self.betaB = None
        self.alphaAB = None
        self.omega = None
        self.mu = None
        self.nend = None
        self.nout = None
        self.neng = None
        self.neng2 = None
        self.ntype = None
        self.itheta = None
        self.A1o = None
        self.A2o = None
        self.A3o = None
        self.B1o = None
        self.B2o = None
        self.B3o = None
        self.noise = None


class pfcData:
    def __init__(self):
        # self.fn=(3.67704032591281e-02+2.58282403054995e-01)/2.0
        # self.fn=(0.258303076+0.03768087551)/2.0
        self.fn = 0.1479919553
        self.ffac = 2.74
        self.ao = 2.51
        self.A1sq = None
        self.A2sq = None
        self.A3sq = None
        self.B1sq = None
        self.B2sq = None
        self.B3sq = None
        self.Ssq = None
        self.Tsq = None
        self.nA = None
        self.nAk = None
        self.nAn = None
        self.A1 = None
        self.A2 = None
        self.A3 = None
        self.Ak1 = None
        self.Ak2 = None
        self.Ak3 = None
        self.An1 = None
        self.An2 = None
        self.An3 = None
        self.nB = None
        self.nBk = None
        self.nBn = None
        self.B1 = None
        self.B2 = None
        self.B3 = None
        self.Bk1 = None
        self.Bk2 = None
        self.Bk3 = None
        self.Bn1 = None
        self.Bn2 = None
        self.Bn3 = None
        self.fx = None
        self.fy = None
        self.theta = None
        self.G1xo = None
        self.G1yo = None
        self.G2xo = None
        self.G2yo = None
        self.G3xo = None
        self.G3yo = None
        self.G1x = None
        self.G1y = None
        self.G2x = None
        self.G2y = None
        self.G3x = None
        self.G3y = None
        self.kA1l = None
        self.kA2l = None
        self.kA3l = None
        self.kB1l = None
        self.kB2l = None
        self.kB3l = None
        self.kAl = None
        self.kBl = None
        self.kA1n = None
        self.kA2n = None
        self.kA3n = None
        self.kB1n = None
        self.kB2n = None
        self.kB3n = None
        self.kAn = None
        self.kBn = None
        self.t = 0

class pfc2d:
    def __init__(self):
        self.pfcParms = pfcParms()
        self.pfcData = pfcData()
        self.runHistory = []

    def __repr__(self) -> str:
        return f'pfc2d: {self.__dict__}'
     
    def initParms(self, parms):
        # constants
        self.pfcParms.name = parms.name
        self.pfcParms.Lx = parms.Lx
        self.pfcParms.Ly = parms.Ly
        self.pfcParms.dx = parms.dx
        self.pfcParms.dy = parms.dy
        self.pfcParms.dt = parms.dt
        self.pfcParms.alpha = parms.alpha
        self.pfcParms.Vo = parms.Vo
        self.pfcParms.ns = parms.ns
        self.pfcParms.nl = parms.nl
        self.pfcParms.epsA = parms.epsA
        self.pfcParms.epsB = parms.epsB
        self.pfcParms.vA = parms.vA
        self.pfcParms.vB = parms.vB
        self.pfcParms.gA = parms.gA
        self.pfcParms.gB = parms.gB
        self.pfcParms.betaB = parms.betaB
        self.pfcParms.alphaAB = parms.alphaAB
        self.pfcParms.omega = parms.omega
        self.pfcParms.mu = parms.mu
        self.pfcParms.nend = parms.nend
        self.pfcParms.nout = parms.nout
        self.pfcParms.neng = parms.neng
        self.pfcParms.neng2 = parms.neng2
        self.pfcParms.ntype = parms.ntype
        self.pfcParms.itheta = parms.itheta
        self.pfcParms.A1o = parms.A1o
        self.pfcParms.A2o = parms.A2o
        self.pfcParms.A3o = parms.A3o
        self.pfcParms.B1o = parms.B1o
        self.pfcParms.B2o = parms.B2o
        self.pfcParms.B3o = parms.B3o
        self.pfcParms.noise = parms.noise

        # static scalars
        self.pfcData.A1sq = self.pfcParms.A1o * cp.conj(self.pfcParms.A1o)
        self.pfcData.A2sq = self.pfcParms.A2o * cp.conj(self.pfcParms.A2o)
        self.pfcData.A3sq = self.pfcParms.A3o * cp.conj(self.pfcParms.A3o)
        self.pfcData.B1sq = self.pfcParms.B1o * cp.conj(self.pfcParms.B1o)
        self.pfcData.B2sq = self.pfcParms.B2o * cp.conj(self.pfcParms.B2o)
        self.pfcData.B3sq = self.pfcParms.B3o * cp.conj(self.pfcParms.B3o)
        self.pfcData.Ssq = 2*(self.pfcData.A1sq + self.pfcData.A2sq + self.pfcData.A3sq)
        self.pfcData.Tsq = 2*(self.pfcData.B1sq + self.pfcData.B2sq + self.pfcData.B3sq)
        self.pfcData.theta = self.pfcParms.itheta * cp.pi / 180.0
        self.pfcData.G1xo = -cp.sqrt(3)/2
        self.pfcData.G1yo = -1./2
        self.pfcData.G2xo = 0.
        self.pfcData.G2yo = 1.
        self.pfcData.G3xo = cp.sqrt(3)/2
        self.pfcData.G3yo = -1./2
        self.pfcData.G1x = self.pfcData.G1xo * cp.cos(self.pfcData.theta) - self.pfcData.G1yo * cp.sin(self.pfcData.theta)
        self.pfcData.G1y = self.pfcData.G1xo * cp.sin(self.pfcData.theta) + self.pfcData.G1yo * cp.cos(self.pfcData.theta)
        self.pfcData.G2x = self.pfcData.G2xo * cp.cos(self.pfcData.theta) - self.pfcData.G2yo * cp.sin(self.pfcData.theta)
        self.pfcData.G2y = self.pfcData.G2xo * cp.sin(self.pfcData.theta) + self.pfcData.G2yo * cp.cos(self.pfcData.theta)
        self.pfcData.G3x = self.pfcData.G3xo * cp.cos(self.pfcData.theta) - self.pfcData.G3yo * cp.sin(self.pfcData.theta)
        self.pfcData.G3y = self.pfcData.G3xo * cp.sin(self.pfcData.theta) + self.pfcData.G3yo * cp.cos(self.pfcData.theta)
        dkx = 2*cp.pi/(self.pfcParms.dx*self.pfcParms.Lx)
        dky = 2*cp.pi/(self.pfcParms.dy*self.pfcParms.Ly)
 
        # static arrays
        kx = cp.concatenate((cp.arange(0, self.pfcParms.Lx//2, dtype=cp.float64), cp.arange(-self.pfcParms.Lx//2, 0, dtype=cp.float64)))*dkx
        ky = cp.concatenate((cp.arange(0, self.pfcParms.Ly//2, dtype=cp.float64), cp.arange(-self.pfcParms.Ly//2, 0, dtype=cp.float64)))*dky
        self.KX, self.KY = cp.meshgrid(kx, ky, indexing='ij')
        self.ksq = self.KX**2 + self.KY**2

        # initialize complex fields on gpu
        self._initFields()
        self._setDt(self.pfcParms.dt)
        self._eng()
        self._gamma()

    def _initFields(self):
        # initialize fields
        self.pfcData.n = 0
        self.pfcData.t = 0

        # copy scalars locally for convenience
        Lx = self.pfcParms.Lx
        Ly = self.pfcParms.Ly
        sf = 1. / (Lx * Ly)
        A1o = self.pfcParms.A1o
        A2o = self.pfcParms.A2o
        A3o = self.pfcParms.A3o
        B1o = self.pfcParms.B1o
        B2o = self.pfcParms.B2o
        B3o = self.pfcParms.B3o

        if self.pfcParms.ntype == 1:
            pass
        elif self.pfcParms.ntype == 2:
            pass
        elif self.pfcParms.ntype == 3:
            pass
        elif self.pfcParms.ntype == 4:
            pass
        elif self.pfcParms.ntype == 34:
            self.pfcData.A1 = A1o * cp.ones((Lx, Ly), dtype=cp.complex128)
            self.pfcData.A2 = A2o * cp.ones((Lx, Ly), dtype=cp.complex128)
            self.pfcData.A3 = A3o * cp.ones((Lx, Ly), dtype=cp.complex128)
            self.pfcData.B1 = B1o * cp.ones((Lx, Ly), dtype=cp.complex128)
            self.pfcData.B2 = B2o * cp.ones((Lx, Ly), dtype=cp.complex128)
            self.pfcData.B3 = B3o * cp.ones((Lx, Ly), dtype=cp.complex128)
            self.pfcData.nA = -0.26822407 * cp.ones((Lx, Ly), dtype=cp.complex128)

            self.pfcData.A1[Lx//2:, :] = 0.
            self.pfcData.A2[Lx//2:, :] = 0.
            self.pfcData.A3[Lx//2:, :] = 0.
            self.pfcData.B1[Lx//2:, :] = 0.
            self.pfcData.B2[Lx//2:, :] = 0.
            self.pfcData.B3[Lx//2:, :] = 0.
            self.pfcData.nA[Lx//2:, :] = -0.443516463
            self.pfcData.nB = self.pfcData.nA.copy()

        self.pfcData.A1 += self.pfcParms.noise * (cp.real(A1o) * (cp.random.rand(Lx, Ly) * 2 - 1) + 1.j * cp.imag(A1o) * (cp.random.rand(Lx, Ly) * 2 - 1))
        self.pfcData.A2 += self.pfcParms.noise * (cp.real(A2o) * (cp.random.rand(Lx, Ly) * 2 - 1) + 1.j * cp.imag(A2o) * (cp.random.rand(Lx, Ly) * 2 - 1))
        self.pfcData.A3 += self.pfcParms.noise * (cp.real(A3o) * (cp.random.rand(Lx, Ly) * 2 - 1) + 1.j * cp.imag(A3o) * (cp.random.rand(Lx, Ly) * 2 - 1))
        self.pfcData.B1 += self.pfcParms.noise * (cp.real(B1o) * (cp.random.rand(Lx, Ly) * 2 - 1) + 1.j * cp.imag(B1o) * (cp.random.rand(Lx, Ly) * 2 - 1))
        self.pfcData.B2 += self.pfcParms.noise * (cp.real(B2o) * (cp.random.rand(Lx, Ly) * 2 - 1) + 1.j * cp.imag(B2o) * (cp.random.rand(Lx, Ly) * 2 - 1))
        self.pfcData.B3 += self.pfcParms.noise * (cp.real(B3o) * (cp.random.rand(Lx, Ly) * 2 - 1) + 1.j * cp.imag(B3o) * (cp.random.rand(Lx, Ly) * 2 - 1))
        self.pfcData.nA += self.pfcParms.noise * (cp.abs(self.pfcData.nA) * (cp.random.rand(Lx, Ly) * 2 - 1) + 1.j * cp.abs(self.pfcData.nA) * (cp.random.rand(Lx, Ly) * 2 - 1))
        self.pfcData.nB += self.pfcParms.noise * (cp.abs(self.pfcData.nB) * (cp.random.rand(Lx, Ly) * 2 - 1) + 1.j * cp.abs(self.pfcData.nB) * (cp.random.rand(Lx, Ly) * 2 - 1))

        # self.pfcData.An1 = self.pfcData.A1.copy()
        # self.pfcData.An2 = self.pfcData.A2.copy()
        # self.pfcData.An3 = self.pfcData.A3.copy()
        # self.pfcData.Bn1 = self.pfcData.B1.copy()
        # self.pfcData.Bn2 = self.pfcData.B2.copy()
        # self.pfcData.Bn3 = self.pfcData.B3.copy()

        self.pfcData.nAk = cp.fft.fft2(self.pfcData.nA) * sf
        self.pfcData.nBk = cp.fft.fft2(self.pfcData.nB) * sf
        self.pfcData.An1 = cp.fft.fft2(self.pfcData.A1)
        self.pfcData.An2 = cp.fft.fft2(self.pfcData.A2)
        self.pfcData.An3 = cp.fft.fft2(self.pfcData.A3)
        self.pfcData.Bn1 = cp.fft.fft2(self.pfcData.B1)
        self.pfcData.Bn2 = cp.fft.fft2(self.pfcData.B2)
        self.pfcData.Bn3 = cp.fft.fft2(self.pfcData.B3)
        self.pfcData.Ak1 = self.pfcData.An1.copy() * sf
        self.pfcData.Ak2 = self.pfcData.An2.copy() * sf
        self.pfcData.Ak3 = self.pfcData.An3.copy() * sf
        self.pfcData.Bk1 = self.pfcData.Bn1.copy() * sf
        self.pfcData.Bk2 = self.pfcData.Bn2.copy() * sf
        self.pfcData.Bk3 = self.pfcData.Bn3.copy() * sf

    def _setDt(self, dt):
        self.pfcParms.dt = dt

        # copy scalars/fields locally for convenience
        Lx = self.pfcParms.Lx
        Ly = self.pfcParms.Ly
        sf = 1. / (Lx * Ly)
        KX = self.KX
        KY = self.KY
        ksq = self.ksq
        epsA = self.pfcParms.epsA
        epsB = self.pfcParms.epsB
        alpha = self.pfcParms.alpha
        betaB = self.pfcParms.betaB
        G1x = self.pfcData.G1x
        G1y = self.pfcData.G1y
        G2x = self.pfcData.G2x
        G2y = self.pfcData.G2y
        G3x = self.pfcData.G3x
        G3y = self.pfcData.G3y

        # calculate linear exponents
        kA1f = -(-epsA + (1. - ksq - 2. * alpha * (G1x * KX + G1y * KY) - alpha**2)**2)
        kA2f = -(-epsA + (1. - ksq - 2. * alpha * (G2x * KX + G2y * KY) - alpha**2)**2)
        kA3f = -(-epsA + (1. - ksq - 2. * alpha * (G3x * KX + G3y * KY) - alpha**2)**2)
        kB1f = -(-epsB + betaB * (1. - ksq - 2. * alpha * (G1x * KX + G1y * KY) - alpha**2)**2)
        kB2f = -(-epsB + betaB * (1. - ksq - 2. * alpha * (G2x * KX + G2y * KY) - alpha**2)**2)
        kB3f = -(-epsB + betaB * (1. - ksq - 2. * alpha * (G3x * KX + G3y * KY) - alpha**2)**2)

        kAl = cp.exp(-ksq * (1. - epsA) * dt)
        kA1l = cp.exp(kA1f * dt)
        kA2l = cp.exp(kA2f * dt)
        kA3l = cp.exp(kA3f * dt)
        kBl = cp.exp(-ksq * (betaB - epsA) * dt)
        kB1l = cp.exp(kB1f * dt)
        kB2l = cp.exp(kB2f * dt)
        kB3l = cp.exp(kB3f * dt)

        ksq_iszero = cp.abs(ksq) < 1e-10
        kA1f_iszero = cp.abs(kA1f) < 1e-10
        kA2f_iszero = cp.abs(kA2f) < 1e-10
        kA3f_iszero = cp.abs(kA3f) < 1e-10
        kB1f_iszero = cp.abs(kB1f) < 1e-10
        kB2f_iszero = cp.abs(kB2f) < 1e-10
        kB3f_iszero = cp.abs(kB3f) < 1e-10

        kAn = cp.zeros((Lx, Ly), dtype=cp.float64)
        kA1n = -dt * sf * cp.ones((Lx, Ly), dtype=cp.float64)
        kA2n = -dt * sf * cp.ones((Lx, Ly), dtype=cp.float64)
        kA3n = -dt * sf * cp.ones((Lx, Ly), dtype=cp.float64)
        kBn = cp.zeros((Lx, Ly), dtype=cp.float64)
        kB1n = -dt * sf * cp.ones((Lx, Ly), dtype=cp.float64)
        kB2n = -dt * sf * cp.ones((Lx, Ly), dtype=cp.float64)
        kB3n = -dt * sf * cp.ones((Lx, Ly), dtype=cp.float64)

        kAn[~ksq_iszero] = (cp.exp(-ksq[~ksq_iszero] * (-epsA + 1) * dt) - 1) / (-epsA + 1) * sf
        kA1n[~kA1f_iszero] = (1. - kA1l[~kA1f_iszero]) / kA1f[~kA1f_iszero] * sf
        kA2n[~kA2f_iszero] = (1. - kA2l[~kA2f_iszero]) / kA2f[~kA2f_iszero] * sf
        kA3n[~kA3f_iszero] = (1. - kA3l[~kA3f_iszero]) / kA3f[~kA3f_iszero] * sf
        kBn[~ksq_iszero] = (cp.exp(-ksq[~ksq_iszero] * (-epsB + betaB) * dt) - 1) / (-epsB + betaB) * sf
        kB1n[~kB1f_iszero] = (1. - kB1l[~kB1f_iszero]) / kB1f[~kB1f_iszero] * sf
        kB2n[~kB2f_iszero] = (1. - kB2l[~kB2f_iszero]) / kB2f[~kB2f_iszero] * sf
        kB3n[~kB3f_iszero] = (1. - kB3l[~kB3f_iszero]) / kB3f[~kB3f_iszero] * sf

        self.pfcData.kAl = kAl
        self.pfcData.kA1l = kA1l
        self.pfcData.kA2l = kA2l
        self.pfcData.kA3l = kA3l
        self.pfcData.kBl = kBl
        self.pfcData.kB1l = kB1l
        self.pfcData.kB2l = kB2l
        self.pfcData.kB3l = kB3l

        self.pfcData.kAn = kAn
        self.pfcData.kA1n = kA1n
        self.pfcData.kA2n = kA2n
        self.pfcData.kA3n = kA3n
        self.pfcData.kBn = kBn
        self.pfcData.kB1n = kB1n
        self.pfcData.kB2n = kB2n
        self.pfcData.kB3n = kB3n

        # # calculate linear exponents
        # kA1f = -(-epsA + (1. - ksq - 2. * alpha * (G1x * KX + G1y * KY) - alpha**2)**2)
        # kA2f = -(-epsA + (1. - ksq - 2. * alpha * (G2x * KX + G2y * KY) - alpha**2)**2)
        # kA3f = -(-epsA + (1. - ksq - 2. * alpha * (G3x * KX + G3y * KY) - alpha**2)**2)
        # kB1f = -(-epsB + betaB * (1. - ksq - 2. * alpha * (G1x * KX + G1y * KY) - alpha**2)**2)
        # kB2f = -(-epsB + betaB * (1. - ksq - 2. * alpha * (G2x * KX + G2y * KY) - alpha**2)**2)
        # kB3f = -(-epsB + betaB * (1. - ksq - 2. * alpha * (G3x * KX + G3y * KY) - alpha**2)**2)

        # kAl = cp.exp(-ksq * (1. - epsA) * dt)
        # kA1l = cp.exp(kA1f * dt)
        # kA2l = cp.exp(kA2f * dt)
        # kA3l = cp.exp(kA3f * dt)
        # kBl = cp.exp(-ksq * (betaB - epsA) * dt)
        # kB1l = cp.exp(kB1f * dt)
        # kB2l = cp.exp(kB2f * dt)
        # kB3l = cp.exp(kB3f * dt)

        # ksq_iszero = cp.abs(ksq) < 1e-10
        # kA1f_iszero = cp.abs(kA1f) < 1e-10
        # kA2f_iszero = cp.abs(kA2f) < 1e-10
        # kA3f_iszero = cp.abs(kA3f) < 1e-10
        # kB1f_iszero = cp.abs(kB1f) < 1e-10
        # kB2f_iszero = cp.abs(kB2f) < 1e-10
        # kB3f_iszero = cp.abs(kB3f) < 1e-10

        # self.pfcData.kAl = kAl
        # self.pfcData.kA1l = kA1l
        # self.pfcData.kA2l = kA2l
        # self.pfcData.kA3l = kA3l
        # self.pfcData.kBl = kBl
        # self.pfcData.kB1l = kB1l
        # self.pfcData.kB2l = kB2l
        # self.pfcData.kB3l = kB3l

        # self.pfcData.kAn = cp.zeros((Lx, Ly), dtype=cp.float64)
        # self.pfcData.kA1n = -dt * sf * cp.ones((Lx, Ly), dtype=cp.float64)
        # self.pfcData.kA2n = -dt * sf * cp.ones((Lx, Ly), dtype=cp.float64)
        # self.pfcData.kA3n = -dt * sf * cp.ones((Lx, Ly), dtype=cp.float64)
        # self.pfcData.kBn = cp.zeros((Lx, Ly), dtype=cp.float64)
        # self.pfcData.kB1n = -dt * sf * cp.ones((Lx, Ly), dtype=cp.float64)
        # self.pfcData.kB2n = -dt * sf * cp.ones((Lx, Ly), dtype=cp.float64)
        # self.pfcData.kB3n = -dt * sf * cp.ones((Lx, Ly), dtype=cp.float64)

        # self.pfcData.kAn[~ksq_iszero] = (cp.exp(-ksq[~ksq_iszero] * (-epsA + 1) * dt) - 1) / (-epsA + 1) * sf
        # self.pfcData.kA1n[~kA1f_iszero] = (1. - self.pfcData.kA1l[~kA1f_iszero]) / kA1f[~kA1f_iszero] * sf
        # self.pfcData.kA2n[~kA2f_iszero] = (1. - self.pfcData.kA2l[~kA2f_iszero]) / kA2f[~kA2f_iszero] * sf
        # self.pfcData.kA3n[~kA3f_iszero] = (1. - self.pfcData.kA3l[~kA3f_iszero]) / kA3f[~kA3f_iszero] * sf
        # self.pfcData.kBn[~ksq_iszero] = (cp.exp(-ksq[~ksq_iszero] * (-epsB + betaB) * dt) - 1) / (-epsB + betaB) * sf
        # self.pfcData.kB1n[~kB1f_iszero] = (1. - self.pfcData.kB1l[~kB1f_iszero]) / kB1f[~kB1f_iszero] * sf
        # self.pfcData.kB2n[~kB2f_iszero] = (1. - self.pfcData.kB2l[~kB2f_iszero]) / kB2f[~kB2f_iszero] * sf
        # self.pfcData.kB3n[~kB3f_iszero] = (1. - self.pfcData.kB3l[~kB3f_iszero]) / kB3f[~kB3f_iszero] * sf

    def step(self):
        self._step(self)

    @cp.fuse()
    def _step(self):
        fft = cp.fft.fft2
        ifft = cp.fft.ifft2

        # get local references to arrays for convenience
        A1 = self.pfcData.A1
        A2 = self.pfcData.A2
        A3 = self.pfcData.A3
        B1 = self.pfcData.B1
        B2 = self.pfcData.B2
        B3 = self.pfcData.B3
        nA = self.pfcData.nA
        nB = self.pfcData.nB

        # get local copies of the scalars
        Lx = self.pfcParms.Lx
        Ly = self.pfcParms.Ly
        vA = self.pfcParms.vA
        vB = self.pfcParms.vB
        Vo = self.pfcParms.Vo
        gA = self.pfcParms.gA
        gB = self.pfcParms.gB
        alphaAB = self.pfcParms.alphaAB
        omega = self.pfcParms.omega
        mu = self.pfcParms.mu

        # grid / block dims
        griddim = (math.ceil(Lx/32), math.ceil(Ly/32))
        blockdim = (32, 32)

        An1 = cp.empty((Lx, Ly), dtype=cp.complex128)
        An2 = cp.empty((Lx, Ly), dtype=cp.complex128)
        An3 = cp.empty((Lx, Ly), dtype=cp.complex128)
        Bn1 = cp.empty((Lx, Ly), dtype=cp.complex128)
        Bn2 = cp.empty((Lx, Ly), dtype=cp.complex128)
        Bn3 = cp.empty((Lx, Ly), dtype=cp.complex128)
        nAn = cp.empty((Lx, Ly), dtype=cp.complex128)
        nBn = cp.empty((Lx, Ly), dtype=cp.complex128)

        step_ABn123(griddim, blockdim, (Lx, Ly, vA, vB, gA, gB, omega, mu, alphaAB, Vo, A1, A2, A3, B1, B2, B3, nA, nB, An1, An2, An3, Bn1, Bn2, Bn3))
        step_ABn(griddim, blockdim, (Lx, Ly, vA, vB, gA, gB, omega, mu, alphaAB, Vo, A1, A2, A3, B1, B2, B3, nA, nB, nAn, nBn))

        An1 = fft(An1, norm="backward")
        An2 = fft(An2, norm="backward")
        An3 = fft(An3, norm="backward")
        Bn1 = fft(Bn1, norm="backward")
        Bn2 = fft(Bn2, norm="backward")
        Bn3 = fft(Bn3, norm="backward")
        nAn = fft(nAn, norm="backward")
        nBn = fft(nBn, norm="backward")

        Ak1 = self.pfcData.Ak1
        Ak2 = self.pfcData.Ak2
        Ak3 = self.pfcData.Ak3
        Bk1 = self.pfcData.Bk1
        Bk2 = self.pfcData.Bk2
        Bk3 = self.pfcData.Bk3
        nAk = self.pfcData.nAk
        nBk = self.pfcData.nBk

        kA1l = self.pfcData.kA1l
        kA2l = self.pfcData.kA2l
        kA3l = self.pfcData.kA3l
        kB1l = self.pfcData.kB1l
        kB2l = self.pfcData.kB2l
        kB3l = self.pfcData.kB3l
        kAl = self.pfcData.kAl
        kBl = self.pfcData.kBl

        kA1n = self.pfcData.kA1n
        kA2n = self.pfcData.kA2n
        kA3n = self.pfcData.kA3n
        kB1n = self.pfcData.kB1n
        kB2n = self.pfcData.kB2n
        kB3n = self.pfcData.kB3n
        kAn = self.pfcData.kAn
        kBn = self.pfcData.kBn

        self.pfcData.Ak1 = kA1l * Ak1 + kA1n * An1
        self.pfcData.Ak2 = kA2l * Ak2 + kA2n * An2
        self.pfcData.Ak3 = kA3l * Ak3 + kA3n * An3
        self.pfcData.Bk1 = kB1l * Bk1 + kB1n * Bn1
        self.pfcData.Bk2 = kB2l * Bk2 + kB2n * Bn2
        self.pfcData.Bk3 = kB3l * Bk3 + kB3n * Bn3
        self.pfcData.nAk = kAl * nAk + kAn * nAn
        self.pfcData.nBk = kBl * nBk + kBn * nBn
            
        self.pfcData.A1 = ifft(self.pfcData.Ak1, norm="forward")
        self.pfcData.A2 = ifft(self.pfcData.Ak2, norm="forward")
        self.pfcData.A3 = ifft(self.pfcData.Ak3, norm="forward")
        self.pfcData.B1 = ifft(self.pfcData.Bk1, norm="forward")
        self.pfcData.B2 = ifft(self.pfcData.Bk2, norm="forward")
        self.pfcData.B3 = ifft(self.pfcData.Bk3, norm="forward")
        self.pfcData.nA = ifft(self.pfcData.nAk, norm="forward")
        self.pfcData.nB = ifft(self.pfcData.nBk, norm="forward")

    def _eng(self):
        fft = cp.fft.fft2
        ifft = cp.fft.ifft2
        conj = cp.conj

        # get local references to arrays for convenience
        A1 = self.pfcData.A1
        A2 = self.pfcData.A2
        A3 = self.pfcData.A3
        B1 = self.pfcData.B1
        B2 = self.pfcData.B2
        B3 = self.pfcData.B3
        nA = self.pfcData.nA
        nB = self.pfcData.nB
        Ak1 = self.pfcData.Ak1
        Ak2 = self.pfcData.Ak2
        Ak3 = self.pfcData.Ak3
        Bk1 = self.pfcData.Bk1
        Bk2 = self.pfcData.Bk2
        Bk3 = self.pfcData.Bk3

        KX = self.KX
        KY = self.KY
        ksq = self.ksq

        # get local copies of the scalars
        Lx = self.pfcParms.Lx
        Ly = self.pfcParms.Ly
        sf = 1. / (Lx * Ly)
        vA = self.pfcParms.vA
        vB = self.pfcParms.vB
        Vo = self.pfcParms.Vo
        gA = self.pfcParms.gA
        gB = self.pfcParms.gB
        epsA = self.pfcParms.epsA
        epsB = self.pfcParms.epsB
        alpha = self.pfcParms.alpha
        betaB = self.pfcParms.betaB
        alphaAB = self.pfcParms.alphaAB
        omega = self.pfcParms.omega
        mu = self.pfcParms.mu

        G1x = self.pfcData.G1x
        G1y = self.pfcData.G1y
        G2x = self.pfcData.G2x
        G2y = self.pfcData.G2y
        G3x = self.pfcData.G3x
        G3y = self.pfcData.G3y

        kA1f = -ksq - 2. * (G1x * KX + G1y * KY) * alpha + 1. - alpha**2
        kA2f = -ksq - 2. * (G2x * KX + G2y * KY) * alpha + 1. - alpha**2
        kA3f = -ksq - 2. * (G3x * KX + G3y * KY) * alpha + 1. - alpha**2
        kB1f = kA1f.copy()
        kB2f = kA2f.copy()
        kB3f = kA3f.copy()

        An1 = A1
        An2 = A2
        An3 = A3
        Bn1 = B1
        Bn2 = B2
        Bn3 = B3
        Ak1 = kA1f * Ak1
        Ak2 = kA2f * Ak2
        Ak3 = kA3f * Ak3
        Bk1 = kB1f * Bk1
        Bk2 = kB2f * Bk2
        Bk3 = kB3f * Bk3

        A1 = ifft(Ak1, norm="forward")
        A2 = ifft(Ak2, norm="forward")
        A3 = ifft(Ak3, norm="forward")
        B1 = ifft(Bk1, norm="forward")
        B2 = ifft(Bk2, norm="forward")
        B3 = ifft(Bk3, norm="forward")

        eng = cp.abs(A1 * conj(A1) + A2 * conj(A2) + A3 * conj(A3) + B1 * conj(B1) + B2 * conj(B2) + B3 * conj(B3))

        A1 = An1
        A2 = An2
        A3 = An3
        B1 = Bn1
        B2 = Bn2
        B3 = Bn3

        An1 = fft(An1)
        An2 = fft(An2)
        An3 = fft(An3)
        Bn1 = fft(Bn1)
        Bn2 = fft(Bn2)
        Bn3 = fft(Bn3)

        Ak1 = An1 * sf
        Ak2 = An2 * sf
        Ak3 = An3 * sf
        Bk1 = Bn1 * sf
        Bk2 = Bn2 * sf
        Bk3 = Bn3 * sf

        Ssq = A1 * conj(A1) + A2 * conj(A2) + A3 * conj(A3)
        Tsq = B1 * conj(B1) + B2 * conj(B2) + B3 * conj(B3)

        gnAo = 2.0 * (3.0 * nA * vA - gA)
        gnBo = 2.0 * (3.0 * nB * vB - gB)
        
        aonm = alphaAB + omega * nA + mu * nB
        
        Asqcoeff = -epsA + nA * (3.0 * vA * nA - 2.0 * gA) + omega * nB
        Bsqcoeff = -epsB + nB * (3.0 * vB * nB - 2.0 * gB) + mu * nA

        eng += cp.abs(Asqcoeff * Ssq \
            + Bsqcoeff * Tsq \
            + gnAo * (A1 * A2 * A3 + conj(A1 * A2 * A3)) \
            + gnBo * (B1 * B2 * B3 + conj(B1 * B2 * B3)) \
            + 3 * vA * Ssq * Ssq + 3 * vB * Tsq * Tsq \
            - 1.5 * vA * (A1 * conj(A1) * A1 * conj(A1) + A2 * conj(A2) * A2 * conj(A2) + A3 * conj(A3) * A3 * conj(A3)) \
            - 1.5 * vB * (B1 * conj(B1) * B1 * conj(B1) + B2 * conj(B2) * B2 * conj(B2) + B3 * conj(B3) * B3 * conj(B3)) \
            + aonm * (A1 * conj(B1) + B1 * conj(A1) + A2 * conj(B2) + B2 * conj(A2) + A3 * conj(B3) + B3 * conj(A3)) \
            + omega * (A1 * A2 * B3 + conj(A1 * A2 * B3) + A1 * A3 * B2 + conj(A1 * A3 * B2) + A2 * A3 * B1 + conj(A2 * A3 * B1)) \
            + mu * (A1 * B2 * B3 + conj(A1 * B2 * B3) + A2 * B1 * B3 + conj(A2 * B1 * B3) + A3 * B1 * B2 + conj(A3 * B1 * B2)) \
            + nA * nA * (0.5 * (-epsA + 1.0) + nA * (-gA / 3. + 0.25 * vA * nA)) \
            + nB * nB * (0.5 * (-epsB + betaB) + nB * (-gB / 3. + 0.25 * vB * nB)) \
            + nA * nB * (alphaAB + 0.5 * (omega * nA + mu * nB)))
        
        self.pfcData.eng = eng

    def _gamma(self):
        Lx = self.pfcParms.Lx
        Ly = self.pfcParms.Ly
        dx = self.pfcParms.dx

        nA = cp.abs(self.pfcData.nA).get()
        # nB = cp.abs(self.pfcData.nB).get()
        eng = self.pfcData.eng.get()

        nA_mean = cp.mean(nA)
        # nB_mean = cp.mean(nB)
        nA_liquid = cp.mean(nA.reshape(Lx,Ly)[280:320,:])
        # nB_liquid = cp.mean(nB.reshape(Lx,Ly)[280:320,:])
        nA_solid = cp.mean(nA.reshape(Lx,Ly)[80:120,:])
        # nB_solid = cp.mean(nB.reshape(Lx,Ly)[80:120,:])

        eng_mean = cp.mean(eng)
        eng_liquid = cp.mean(eng.reshape(Lx,Ly)[280:320,:])
        eng_solid = cp.mean(eng.reshape(Lx,Ly)[80:120,:])

        alpha_solid = (nA_mean - nA_liquid) / (nA_solid - nA_liquid)
        alpha_liquid = (nA_solid - nA_mean) / (nA_solid - nA_liquid)

        self.pfcData.gamma = 2.74 * (4 * np.pi / np.sqrt(3)) / (2.51) * (eng_mean - alpha_solid * eng_solid - alpha_liquid * eng_liquid)*dx*Lx

    def run(self, steps, dt=None, printEvery=None, saveEvery=None, outputFolder="./out", callbackEvery=None, callback=None):
        if dt is not None:
            self._setDt(dt)

        if outputFolder is not None:
            os.makedirs(outputFolder, exist_ok=True)

        for i in range(steps + 1)[1:]:
            self.step()
            
            if printEvery is not None and i % printEvery == 0:
                self._eng()
                self._gamma()
                print(f"Step {i} --> Total time {self.pfcData.t + i * self.pfcParms.dt:.5e} This run time {i * self.pfcParms.dt:.5e} Energy {self.pfcData.eng.mean()} Gamma {self.pfcData.gamma}")
            
            if saveEvery is not None and i % saveEvery == 0:
                np.savez_compressed(os.path.join(outputFolder, f"t_{self.pfcData.t + i * self.pfcParms.dt:.5f}.npz"),
                    t = self.pfcData.t + i * self.pfcParms.dt,
                    A1 = self.pfcData.A1.get(),
                    A2 = self.pfcData.A2.get(),
                    A3 = self.pfcData.A3.get(),
                    B1 = self.pfcData.B1.get(),
                    B2 = self.pfcData.B2.get(),
                    B3 = self.pfcData.B3.get(),
                    nA = self.pfcData.nA.get(),
                    nB = self.pfcData.nB.get(),
                    eng = self.pfcData.eng.get())
                
            if callbackEvery is not None and i % callbackEvery == 0:
                # callback(self.pfcData.t + i * self.pfcParms.dt, self.pfcData.A1.get(), self.pfcData.A2.get(), self.pfcData.A3.get(), self.pfcData.B1.get(), self.pfcData.B2.get(), self.pfcData.B3.get(), self.pfcData.nA.get(), self.pfcData.nB.get(), self.pfcData.eng.get(), self.pfcData.gamma)
                callback(i, self.pfcData.t + i * self.pfcParms.dt, self)
        
        self._eng()
        self._gamma()
        self.pfcData.t += steps * self.pfcParms.dt

        self.runHistory.append({
            "t": self.pfcData.t,
            "A1": self.pfcData.A1.get(),
            "A2": self.pfcData.A2.get(),
            "A3": self.pfcData.A3.get(),
            "B1": self.pfcData.B1.get(),
            "B2": self.pfcData.B2.get(),
            "B3": self.pfcData.B3.get(),
            "nA": self.pfcData.nA.get(),
            "nB": self.pfcData.nB.get(),
            "eng": self.pfcData.eng.get()
            })
    

        
step_ABn123 = cp.RawKernel("""
#include <cupy/complex.cuh>
                                  
extern "C" __global__

void step_ABn123(int height, int width, double vA, double vB, double gA, double gB, double omega, double mu, double alphaAB, double Vo, complex<double> *A1, complex<double> *A2, complex<double> *A3, complex<double> *B1, complex<double> *B2, complex<double> *B3, complex<double> *nA, complex<double> *nB, complex<double> *An1, complex<double> *An2, complex<double> *An3, complex<double> *Bn1, complex<double> *Bn2, complex<double> *Bn3) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int index = y * width + x;
    complex<double> j(0, 1);
    
    if (x < width && y < height) {
        double A1sq = (A1[index] * conj(A1[index])).real();
        double A2sq = (A2[index] * conj(A2[index])).real();
        double A3sq = (A3[index] * conj(A3[index])).real();
        double B1sq = (B1[index] * conj(B1[index])).real();
        double B2sq = (B2[index] * conj(B2[index])).real();
        double B3sq = (B3[index] * conj(B3[index])).real();
        double Ssq = 2 * (A1sq + A2sq + A3sq);
        double Tsq = 2 * (B1sq + B2sq + B3sq);
        double gnAo = (2. * (3. * nA[index] - gA)).real();
        double gnBo = (2. * (3. * nB[index] - gB)).real();
        double aonm = (alphaAB + omega * nA[index] + mu * nB[index]).real();
        double naf = (nA[index] * (3. * vA * nA[index] - 2. * gA) + omega * nB[index]).real();
        double nbf = (nB[index] * (3. * vB * nB[index] - 2. * gB) + mu * nA[index]).real();

        An1[index] = (3. * vA * (Ssq - A1sq) + naf) * A1[index] + gnAo * conj(A2[index] * A3[index]) + aonm * B1[index] + omega * (conj(A2[index] * B3[index]) + conj(A3[index] * B2[index])) + mu * conj(B2[index] * B3[index]) + Vo;
        An2[index] = (3. * vA * (Ssq - A2sq) + naf) * A2[index] + gnAo * conj(A3[index] * A1[index]) + aonm * B2[index] + omega * (conj(A3[index] * B1[index]) + conj(A1[index] * B3[index])) + mu * conj(B1[index] * B3[index]) + Vo;
        An3[index] = (3. * vA * (Ssq - A3sq) + naf) * A3[index] + gnAo * conj(A1[index] * A2[index]) + aonm * B3[index] + omega * (conj(A1[index] * B2[index]) + conj(A2[index] * B1[index])) + mu * conj(B1[index] * B2[index]) + Vo;

        Bn1[index] = (3. * vB * (Tsq - B1sq) + nbf) * B1[index] + gnBo * conj(B2[index] * B3[index]) + aonm * A1[index] + mu * (conj(B2[index] * A3[index]) + conj(B3[index] * A2[index])) + omega * conj(A2[index] * A3[index]) + Vo;
        Bn2[index] = (3. * vB * (Tsq - B2sq) + nbf) * B2[index] + gnBo * conj(B3[index] * B1[index]) + aonm * A2[index] + mu * (conj(B3[index] * A1[index]) + conj(B1[index] * A3[index])) + omega * conj(A3[index] * A1[index]) + Vo;
        Bn3[index] = (3. * vB * (Tsq - B3sq) + nbf) * B3[index] + gnBo * conj(B1[index] * B2[index]) + aonm * A3[index] + mu * (conj(B1[index] * A2[index]) + conj(B2[index] * A1[index])) + omega * conj(A1[index] * A2[index]) + Vo;
    }
}
""", 'step_ABn123')

step_ABn = cp.RawKernel("""
#include <cupy/complex.cuh>
                                  
extern "C" __global__

void step_ABn(int height, int width, double vA, double vB, double gA, double gB, double omega, double mu, double alphaAB, double Vo, complex<double> *A1, complex<double> *A2, complex<double> *A3, complex<double> *B1, complex<double> *B2, complex<double> *B3, complex<double> *nA, complex<double> *nB, complex<double> *nAn, complex<double> *nBn) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int index = y * width + x;
    complex<double> j(0, 1);
    
    if (x < width && y < height) {
        double A1sq = (A1[index] * conj(A1[index])).real();
        double A2sq = (A2[index] * conj(A2[index])).real();
        double A3sq = (A3[index] * conj(A3[index])).real();
        double B1sq = (B1[index] * conj(B1[index])).real();
        double B2sq = (B2[index] * conj(B2[index])).real();
        double B3sq = (B3[index] * conj(B3[index])).real();
        double Ssq = 2 * (A1sq + A2sq + A3sq);
        double Tsq = 2 * (B1sq + B2sq + B3sq);
        double gnAo = (2. * (3. * nA[index] - gA)).real();
        double gnBo = (2. * (3. * nB[index] - gB)).real();
        double aonm = (alphaAB + omega * nA[index] + mu * nB[index]).real();

        nAn[index] = nA[index] * nA[index] * (-gA + vA * nA[index]) + 0.5 * (gnAo * Ssq + mu * Tsq) + 6. * vA * (A1[index] * A2[index] * A3[index] + conj(A1[index] * A2[index] * A3[index])) + omega * (A1[index] * conj(B1[index]) + A2[index] * conj(B2[index]) + A3[index] * conj(B3[index]) + B1[index] * conj(A1[index]) + B2[index] * conj(A2[index]) + B3[index] * conj(A3[index])) + nB[index] * (alphaAB + omega * nA[index] + 0.5 * mu * nB[index]);
        nBn[index] = nB[index] * nB[index] * (-gB + vB * nB[index]) + 0.5 * (gnBo * Tsq + omega * Ssq) + 6. * vB * (B1[index] * B2[index] * B3[index] + conj(B1[index] * B2[index] * B3[index])) + mu * (B1[index] * conj(A1[index]) + B2[index] * conj(A2[index]) + B3[index] * conj(A3[index]) + A1[index] * conj(B1[index]) + A2[index] * conj(B2[index]) + A3[index] * conj(B3[index])) + nA[index] * (alphaAB + mu * nB[index] + 0.5 * omega * nA[index]);
    }
}
""", 'step_ABn')

