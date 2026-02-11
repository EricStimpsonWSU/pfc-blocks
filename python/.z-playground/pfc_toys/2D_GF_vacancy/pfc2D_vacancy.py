import numpy as np
import json

class RunParms:
    def __init__(self):
        self.tansient_time = None
        self.transient_dt = None
        self.transient_time_steps = None
        self.total_time = None
        self.dt = None
        self.checkpoint_time = None

    def __repr__(self) -> str:
        return f'RunParm({json.dumps(self.__dict__, indent=4)})'

class OneModeGeometryParms:
    def __init__(self):
        self.LX = None
        self.LY = None
        self.nx = None
        self.ny = None

class Parms:
    def __init__(self):
        self.LX = None
        self.LY = None
        self.dqx = None
        self.dqy = None
        self.q02 = 1
        self.transient_time = None
        self.transient_dt = None
        self.transient_time_steps = None
        self.total_time = None
        self.dt = None
        self.checkpoint_time = None
        self.checkpoint_time_steps = None
        self.total_time_steps = None
        self.nImages = None
        self.tol = 1.e-3
        self.err = 1.e-4
        self.predictor_corrector_iterations = None
        self.eps = 0.02
        self.b_s = 10
        self.b_l = (1. - self.eps) * self.b_s
        self.g = 0.5 * np.sqrt(3 / self.b_s)

    def __repr__(self) -> str:
        return f'Parms({json.dumps(self.__dict__, indent=4)})'

class Checkpoint:
    def __init__(self):
        self.psi = None
        self.psi0 = None
        self.psiq = None
        self.n = None
        self.time = None
        self.dt = None
        self.qx = None
        self.qy = None
        self.q2 = None
        self.exp = None
        self.cf1 = None
        self.cf21 = None
        self.fMean = None
        self.muMean = None

    def __repr__(self) -> str:
        # Convert numpy arrays and complex numbers to list before serialization
        dict_copy = self.__dict__.copy()
        for key, value in dict_copy.items():
            if isinstance(value, np.ndarray):
                # Only include the first few values for long arrays
                value = value.flatten()
                if len(value) > 5:
                    value = value[:5]
                    value = np.append(value, '...')
                if np.iscomplexobj(value):
                    dict_copy[key] = [str(x) for x in value.tolist()]
                else:
                    dict_copy[key] = value.tolist()

        return f'Checkpoint({json.dumps(dict_copy, indent=4)})'

class _temps:
    def __init__(self):
        self.nonlin1 = None
        self.nonlin1q = None

class pfc2D:
    def __init__(self):
        self.parms = Parms()
        self.checkpoint = Checkpoint()
        self._temps = _temps()

    def __repr__(self) -> str:
        return f'pfc2d({self.__dict__})'

    def set_parms(self, parms):
        """
        This method is used to set up the simulation parameters
        """
        self.parms = parms

    def init_parms_homogeneous(self, init_parms):
        """
        This method is used to initialize the simulation parameters for a homogeneous system
        """
        self.parms.LX = init_parms.LX
        self.parms.LY = init_parms.LY
        self.parms.dqx = init_parms.dx
        self.parms.dqy = init_parms.dy
        self.parms.q02 = init_parms.q02
        self.parms.transient_time = init_parms.transient_time
        self.parms.transient_dt = init_parms.transient_dt
        self.parms.transient_time_steps = init_parms.transient_time_steps
        self.parms.total_time = init_parms.total_time
        self.parms.dt = init_parms.dt
        self.parms.checkpoint_time = init_parms.checkpoint_time
        self.parms.checkpoint_time_steps = init_parms.checkpoint_time_steps
        self.parms.total_time_steps = init_parms.total_time_steps
        self.parms.nImages = init_parms.nImages
        self.parms.tol = init_parms.tol
        self.parms.err = init_parms.err
        self.parms.eps = init_parms.eps
        self.parms.b_s = init_parms.b_s
        self.parms.b_l = init_parms.b_l
        self.parms.g = init_parms.g
        self.parms.predictor_corrector_iterations = init_parms.predictor_corrector_iterations

    def setRunParms(self, transient = {'time': 0, 'dt': 0}, total = {'time': 1000, 'dt': 0.5}, checkpoint_time = 50, predictor_corrector_iterations = 100):
        """
        This method is used to set up the simulation parameters

        Args:
        transient: dict
            Dictionary containing the transient "time" and "dt"
            default: {'time': 0, 'dt': 0}
        total: dict
            Dictionary containing the total "time" and "dt"
            default: {'time': 1000, 'dt': 0.5}
        checkpoint_time: float
            Time interval for saving the checkpoint
            default: 50
        predictor_corrector_iterations: int
            Number of iterations for the predictor-corrector method
            default: 100
        """
        self.parms.transient_time = transient['time']
        self.parms.transient_dt = transient['dt']
        self.parms.total_time = total['time']
        self.parms.dt = total['dt']
        self.parms.checkpoint_time = checkpoint_time
        self.parms.checkpoint_time_steps = checkpoint_time // self.parms.dt

        # calculate the number of time steps
        if self.parms.transient_time == 0:
            self.parms.transient_time_steps = 0
        else:
            self.parms.transient_time_steps = int(self.parms.transient_time / self.parms.transient_dt)
        self.parms.total_time_steps = int((self.parms.total_time - self.parms.transient_time) / self.parms.dt) + self.parms.transient_time_steps

        self.parms.predictor_corrector_iterations = predictor_corrector_iterations
 
    def initGeometryOnemode(self, dims = {'LX': None, 'LY': None}, atoms = {'nx': None, 'ny': None}, dq = {'dx': None, 'dy': None}):
        """
        This method is used to initialize the simulation parameters for a homogeneous system

        Args:
        dims: dict (required)
            Dictionary containing the dimensions of the system
            default: {'LX': None, 'LY': None}
        atoms: dict (required)
            Dictionary containing the number of atoms in the system
            default: {'nx': None, 'ny': None}
        dq: dict (optional)
            Dictionary containing the dq values
            default: {'dx': None, 'dy': None}
        """
        # set the dimensions (required)
        LX = dims['LX']
        LY = dims['LY']
        if LX is None or LY is None:
            raise ValueError('LX and LY must be set before running the simulation')

        self.parms.LX = LX
        self.parms.LY = LY

        # set the number of atoms (required)
        nx = atoms['nx']
        ny = atoms['ny']
        if nx is None or ny is None:
            raise ValueError('nx and ny must be set before running the simulation')

        kx = 4 * np.pi * nx / np.sqrt(3)
        ky = 2 * np.pi * ny

        # density
        n0 = -0.02
        scale = 0.1/9
        offset = -6

        # mesh grid for real space
        x = np.arange(0, LX)
        y = np.arange(0, LY)
        X, Y = np.meshgrid(x, y, indexing='ij')

        # initial density
        self.checkpoint.psi = n0 + scale * (2. * (2. * np.cos(np.sqrt(3) / 2 *  kx * X / LX) * np.cos(0.5 * ky * Y / LY) + np.cos(ky * Y / LY)) + offset)
        self.checkpoint.psi0 = self.checkpoint.psi.copy()
        self.checkpoint.psiq = np.fft.rfft2(self.checkpoint.psi, axes=(0, 1))
        self.checkpoint.psiq[0][0] = n0 * LX * LY
        self.checkpoint.psi = np.fft.irfft2(self.checkpoint.psiq, s=(LX, LY), axes=(0, 1))

        # calculate or set the dq values
        if dq['dx'] is None or dq['dy'] is None:
            self.parms.dqx = 4 * np.pi / np.sqrt(3) * nx / LX
            self.parms.dqy = 2 * np.pi * ny / LY
        else:
            self.parms.dqx = dq['dx']
            self.parms.dqy = dq['dy']

    def get_parms(self):
        """
        This method is used to get the current simulation parameters
        """
        return self.parms

    def get_checkpoint(self):
        """
        This method is used to get the current checkpoint
        """
        return self.checkpoint
    
    def runAll(self):
        """
        This method is used to run the simulation for the total time specified in parms
        """
        self._initRuntime()
        # Print time, fMean, muMean
        print(f'time: {self.checkpoint.time}, fMean: {self.checkpoint.fMean}, muMean: {self.checkpoint.muMean}')
        for self.checkpoint.n in range(self.parms.total_time_steps + 1)[1:]:
            if (self.checkpoint.n < self.parms.transient_time_steps):
                if self.checkpoint.n == 1:
                    self.checkpoint.dt == self.parms.transient_dt
                    self._calcTimeStep()
                self.checkpoint.time = self.checkpoint.n * self.parms.transient_dt
            else:
                if self.checkpoint.n == self.parms.transient_time_steps:
                    self.checkpoint.dt = self.parms.dt
                    self._calcTimeStep()
                self.checkpoint.time = (self.checkpoint.n - self.parms.transient_time_steps) * self.checkpoint.dt + self.parms.transient_time
            self._sheq()
            # Checkpoint at interval
            if ((self.checkpoint.n == self.parms.transient_time_steps) |
                ((self.checkpoint.n > self.parms.transient_time_steps) & ((self.checkpoint.n + self.parms.transient_time // self.parms.dt) % self.parms.checkpoint_time_steps == 0))):
                # calculate free energy
                self.calculateFreeEnergy()
                # Print time, fMean, muMean
                print(f'time: {self.checkpoint.time}, fMean: {self.checkpoint.fMean}, muMean: {self.checkpoint.muMean}')
                

                
    
    def run_to_convergence(self):
        """
        This method is used to run the simulation until the energy stabilizes
        """

    def calculateFreeEnergy(self):
        """
        This method is used to calculate the free energy
        """
        LX = self.parms.LX
        LY = self.parms.LY

        psi = self.checkpoint.psi
        psi2 = psi**2
        eps = self.parms.eps
        g = self.parms.g
        f = (0.25 * psi2 - 0.5 * eps - g * psi / 3.) * psi2
        mu = (psi - g) * psi2 - eps * psi

        q02 = self.parms.q02
        q2 = self.checkpoint.q2
        psiq = self.checkpoint.psiq
        d2nq = (q02 - q2)**2 * psiq

        psi2 = np.fft.irfft2(d2nq, s=(LX, LY), axes=(0, 1))

        f += 0.5 * psi2 * psi
        mu *= psi2

        self.checkpoint.fMean = np.mean(f)
        self.checkpoint.muMean = np.mean(mu)
        self.checkpoint.n = 0

    def _initRuntime(self):
        """
        This method is used to initialize the runtime parameters
        """
        # set up the checkpoint to begin running
        self.checkpoint.time = 0
        if self.parms.transient_time_steps == 0:
            self.checkpoint.dt = self.parms.dt
        else:
            self.checkpoint.dt = self.parms.transient_dt

        self._initQ2()

    def _initQ2(self):
        """
        This method is used to initialize the Q2 field (and cf1, cf21)
        """
        LX = self.parms.LX
        LY = self.parms.LY

        # if these are not set then raise error
        if LX is None or LY is None:
            raise ValueError('LX and LY must be set before running the simulation')

        dx = self.parms.dqx
        dy = self.parms.dqy

        # if these are not set then raise error
        if dx is None or dy is None:
            raise ValueError('dx and dy must be set before running the simulation')

        # mesh qx and qy
        ix = np.arange(0, LX)
        iy = np.arange(0, LY // 2 + 1)

        qx = ix * 2. * np.pi / (dx * LX)
        qx[LX//2 + 1:] = qx[LX//2 - ((LX + 1) % 2):0:-1]

        qy = iy * 2. * np.pi / (dy * LY)

        self.checkpoint.qx = qx
        self.checkpoint.qy = qy

        Qx, Qy = np.meshgrid(qx, qy, indexing='ij')

        # calculate q2, alpha_1, dt, alpha_dt, exp_1, cf_1, cf2_1
        q2 = Qx**2 + Qy**2

        # save the results
        self._qx = qx.copy()
        self._qy = qy.copy()
        self.checkpoint.q2 = q2

        self._calcTimeStep()

    def _calcTimeStep(self):
        q2 = self.checkpoint.q2
        eps = self.parms.eps
        q02 = self.parms.q02

        # if these are not set then raise error
        if eps is None or q02 is None:
            raise ValueError('eps and q02 must be set before running the simulation')

        alpha_1 = -q2 * (-eps + (q2 - q02)**2)

        dt = self.checkpoint.dt

        # if this is not set then raise error
        if dt is None:
            raise ValueError('dt must be set before running the simulation')

        alpha_dt = alpha_1 * dt
        exp = np.exp(alpha_dt)

        cf1 = np.zeros_like(alpha_dt)
        cf21 = np.zeros_like(alpha_dt)

        i_s = np.abs(alpha_dt) < 2.0e-5
        i_l = np.abs(alpha_dt) >= 2.0e-5

        cf1[i_s] = dt * (1. + 0.5 * alpha_dt[i_s] * (1. + alpha_dt[i_s] / 3.))
        cf21[i_s] = 0.5 * dt * (1. + alpha_dt[i_s] * (1. + 0.25 * alpha_dt[i_s]) / 3.)
        cf1[i_l] = (exp[i_l] - 1) / alpha_1[i_l]
        cf21[i_l] = (exp[i_l] - (1 + alpha_dt[i_l])) / (alpha_1[i_l] * alpha_dt[i_l])

        # save the results
        self.checkpoint.exp = exp
        self.checkpoint.cf1 = cf1
        self.checkpoint.cf21 = cf21      

    def _nonlin1(self):
        """
        This method is used to calculate the non-linear term to first order
        """
        self._temps.nonlin1 = self.checkpoint.psi**2 * (self.checkpoint.psi - self.parms.g) - 10 * (np.abs(self.checkpoint.psi)**3 - self.checkpoint.psi**3)
        self._temps.nonlin1q = -np.fft.rfft2(self._temps.nonlin1, axes=(0, 1)) * self.checkpoint.q2

    def _sheq(self):
        """
        This method is used to calculate the time evolution of the field
        """
        # calculate the non-linear term to first order
        self._nonlin1()

        psiq0 = self.checkpoint.psiq
        exp0 = self.checkpoint.exp
        nonlin1q0 = self._temps.nonlin1q
        cf10 = self.checkpoint.cf1

        LX = self.parms.LX
        LY = self.parms.LY

        psiq0 = psiq0 * exp0 + cf10 * nonlin1q0
        psiq0[LX // 2 + 1:, 0] = np.conjugate(psiq0[LX // 2 - ((LX + 1) % 2):0:-1, 0])
        psi0 = np.fft.irfft2(psiq0, s=(LX, LY), axes=(0, 1))
        self.checkpoint.psi = psi0
        self.checkpoint.psiq = psiq0

        # debug
        self._exp0 = exp0.copy()
        self._cf10 = cf10.copy()
        self._nonlin10 = self._temps.nonlin1.copy()
        self._nonlin1q0 = nonlin1q0.copy()
        self._psi0 = psi0.copy()
        self._psiq0 = psiq0.copy()

        predictor_corrector_iterations = self.parms.predictor_corrector_iterations
        if predictor_corrector_iterations > 0:
            cf210 = self.checkpoint.cf21

            # debug
            self._cf210 = cf210.copy()

            err = self.parms.err
            tol = self.parms.tol

            psiq0_sigN = -cf210 * nonlin1q0
            psiq0 += psiq0_sigN

            # debug
            self._psiq0_sigN = psiq0_sigN.copy()

            for iCorr in range(predictor_corrector_iterations):
                # calculate the non-linear term to first order
                self._nonlin1()
                nonlin1q0 = self._temps.nonlin1q

                # debug
                if iCorr == 0:
                    self._nonlin1q1_0 = nonlin1q0.copy()
                if iCorr == 1:
                    self._nonlin1q1_1 = nonlin1q0.copy()

                # calculate the time evolution of the field
                self.checkpoint.psiq = psiq0 + nonlin1q0 * cf210
                self.checkpoint.psiq[LX // 2 + 1:, 0] = np.conjugate(self.checkpoint.psiq[LX // 2 - ((LX + 1) % 2):0:-1, 0])

                # update the field
                self.checkpoint.psi = np.fft.irfft2(self.checkpoint.psiq , s=(LX, LY), axes=(0, 1))

                # debug
                if iCorr == 0:
                    self._psiq1_0 = self.checkpoint.psiq.copy()
                    self._psi1_0 = self.checkpoint.psi.copy()
                if iCorr == 1:
                    self._psiq1_1 = self.checkpoint.psiq.copy()
                    self._psi1_1 = self.checkpoint.psi.copy()

                if iCorr > 0:
                    if np.max(np.abs(psi0)) > 1.e5:
                        time = self.checkpoint.time
                        n = self.checkpoint.n
                        print(f'psi diverged at time {time}, step {n}, iteration {iCorr}')
                        raise ValueError('psi diverged')
                    
                    # indices of small values
                    s_i = np.abs(psi0) < err
                    if ~np.any(s_i):
                        if np.max(np.abs((self.checkpoint.psi - psi0)/self.checkpoint.psi)) < tol:
                            return
                    else:
                        if np.max(np.abs(self.checkpoint.psi[s_i] - psi0[s_i])) < tol:
                            if np.max(np.abs((self.checkpoint.psi[~s_i] - psi0[~s_i])/self.checkpoint.psi[~s_i])) < tol:
                                return
                psi0 = self.checkpoint.psi

            # failed to converge
            time = self.checkpoint.time
            n = self.checkpoint.n
            print(f'psi failed to converge at time {time}, step {n}, after {predictor_corrector_iterations} iterations')
            raise ValueError('psi didn\'t converge')
