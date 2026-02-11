import numpy as np
import cupy as cp

class PFC2D_Vacancy_Parms:
  def __init__(self):
    self.epsilon = None                 # corresponds to temperature
    self.a = None
    self.beta = None
    self.g = None
    self.v0 = None
    self.Hln = None
    self.Hng = None
    self.N = None                       # number of pixels in the square region
    self.PPU = None                     # pixels per unit
    self.phi0 = None                    # the average density of the initial field
    self.eta = None                     # the strength of the gaussian noise added to each time step
    self.dt = None                      # time step
    self.seed = 0                       # set for reproducibility
    self.NoiseDynamicsFlag = False      # set to True to add noise to the dynamics
    self.Noise_CutoffK = 0.5            # high frequency threshold for noise
    self.NoiseTimeSmoothFlag = False = False
                                        # set to True to make noise vary consistantly with time
    self.NoiseTimeSmoothingFrames = 30  # number of time steps to smooth the noise
    self.Noise_CutoffOmega = 0.1        # time scale for smoothing the noise

    self.BoundaryPotential = None       # boundary potential
    self.BoundaryDensity = None         # boundary density

class PFC2D_Vacancy:
  def __init__(self):
    self.t = None      # elapsed time
    self.parms = PFC2D_Vacancy_Parms()
    self.minLog = -30

  def InitParms(self):
    # Set seed for reproducibility
    cp.random.seed(cp.uint64(self.parms.seed))
    
    # Initialize geometry
    self.SetGeometry(self.parms.N, self.parms.N, self.parms.PPU, 1)

    # SetDT
    self.SetDT(self.parms.dt)

  def SetGeometry(self, NX, NY, PPUx, scalefactor):
    self.nx = NX
    self.ny = NY
    self.mx = int(np.round(NX/PPUx))
    self.my = int(np.round(NY/(np.sqrt(3)*PPUx)))*2
    self.dx = (4*np.pi*self.mx/np.sqrt(3))/NX
    self.dy = (2*np.pi*self.my)/NY
    self.dy *= scalefactor
    self.dx *= scalefactor
    x = cp.arange(NX) * self.dx
    y = cp.arange(NY) * self.dy
    self.x, self.y = cp.meshgrid(x,y)

    # using self.x, self.y create an array of ordered pairs
    self.r = cp.dstack((self.x, self.y))

    # Create the wave numbers
    kx = cp.fft.fftfreq(self.nx, d=self.dx) * 2 * cp.pi
    ky = cp.fft.fftfreq(self.ny, d=self.dy) * 2 * cp.pi
    self.kx, self.ky = cp.meshgrid(kx, ky)
    self.k2 = self.kx**2 + self.ky**2
    self.k4 = self.k2**2
    self.k6 = self.k2**3
    self.lincoeff = (-self.k2 * (self.parms.epsilon + self.parms.beta) + 2 * self.parms.beta * self.k4 - self.parms.beta * self.k6)
    self.linenergycoeff = self.parms.beta * self.k4 - self.parms.beta * self.k6 / 2

    # Create noise mask.
    self.noiseMask = cp.ones((self.ny, self.nx))

    # SetDT
    self.SetDT(self.parms.dt)

  def SetDT(self, dt):
    self.parms.dt = dt
    self.expcoeff = cp.exp(self.lincoeff*self.parms.dt)

    self.expcoeff_nonlin = cp.ones_like(self.x) * self.parms.dt
    self.expcoeff_nonlin[self.lincoeff != 0] = (self.expcoeff[self.lincoeff != 0] - 1)/self.lincoeff[self.lincoeff != 0]

    self.expcoeff_nonlin2 = cp.zeros_like(self.x)
    self.expcoeff_nonlin2[self.lincoeff != 0] = (self.expcoeff[self.lincoeff != 0] - (1 + self.lincoeff[self.lincoeff != 0]*self.parms.dt))/cp.power(self.lincoeff[self.lincoeff != 0],2)

  def GetEtaNoise(self):
    # noise = self.noiseMask*cp.random.normal(loc=0, scale=self.parms.eta, size=(self.ny, self.nx))
    # noise_fft = 1j*self.kx*cp.fft.fft2(noise)
    # self.noise_fft = noise_fft.copy()
    # noise = self.noiseMask*cp.random.normal(loc=0, scale=self.parms.eta, size=(self.ny, self.nx))
    # noise_fft = 1j*self.ky*cp.fft.fft2(noise)

    # # high frequency cutoff
    # self.noise_fft += noise_fft
    # self.noise_fft[0,0] = 0
    # self.noise_fft[self.k2 > self.parms.noise_high_freq_cutoff**2] = 0
    
    # # inverse fourier to get normalized noise
    # # noise_fft[0,0] = 0
    # self.noise = cp.fft.ifft2(self.noise_fft).real

    # return self.noise
    noisex = cp.random.normal(loc=0, scale=1, size=(self.ny, self.nx))
    noisex_fft = 1j*self.kx*cp.fft.fft2(noisex)
    noisey = cp.random.normal(loc=0, scale=1, size=(self.ny, self.nx))
    noisey_fft = 1j*self.ky*cp.fft.fft2(noisey)
    self.noise_fft = noisex_fft + noisey_fft

    # normalize
    self.noise_fft[0,0] = 0

    # high (spatial domain) frequency cutoff
    self.noise_fft[self.k2 > self.parms.Noise_CutoffK**2] = 0
    self.noise = cp.fft.ifft2(self.noise_fft).real

    # renormalize to eta
    self.noise *= self.parms.eta/cp.sqrt(cp.power(self.noise, 2).mean())
    self.noise_fft = cp.fft.fft2(self.noise)
    return self.noise
    
  def InitializeTimeSmoothNoise(self):
    self.noise_t = cp.zeros((self.ny, self.nx, self.parms.NoiseTimeSmoothingFrames))
    for i in range(self.parms.NoiseTimeSmoothingFrames):
      self.noise_t[:, :, i] = self.GetEtaNoise()
    
    self.omega2 = cp.fft.fftfreq(self.parms.NoiseTimeSmoothingFrames, d=1)**2  # d=dt?
      
  def GenerateTimeSmoothNoise(self):
    self.noise_t = cp.concatenate((self.noise_t[:,:,1:], cp.array(self.GetEtaNoise()).reshape(self.ny, self.nx,1)), axis=2)
    self.noise_t_hat = cp.fft.fft2(self.noise_t)

    # normalize
    self.noise_t_hat[0,0,:] = 0

    # high (temporal domain) frequency cutoff
    self.noise_t_hat[:, :, self.omega2 > self.parms.Noise_CutoffOmega**2] = 0
    self.noise_t = cp.fft.ifft2(self.noise_t_hat).real
    self.noise = self.noise_t[:, :, -1]

    # renormalize to eta
    self.noise *= self.parms.eta/cp.sqrt(cp.power(self.noise, 2).mean())
    self.noise_fft = cp.fft.fft2(self.noise)
    return self.noise

  def InitFieldFlat(self, noisy = True):
    self.phi = self.parms.phi0 * cp.ones((self.nx, self.ny))
    self.phi += self.GetEtaNoise() if noisy & (self.parms.eta != 0) else 0
    self.phi_hat = cp.fft.fft2(self.phi)
    self.phi0 = cp.fft.ifft2(self.phi_hat).real
    self.t = 0

  def InitFieldCrystal(self, A = None, noisy = True):
    if A == None:
      A == self.parms.phi0/6
    
    q1 = cp.array([-np.sqrt(3)*0.5, -0.5])
    q2 = cp.array([0.0, 1.0])
    q3 = cp.array([np.sqrt(3)*0.5, -0.5])

    self.phi = 2 * A * (cp.cos(self.r.dot(q1)) + cp.cos(self.r.dot(q2)) + cp.cos(self.r.dot(q3))) + self.parms.phi0
    self.phi_hat = cp.fft.fft2(self.phi)
    self.phi_hat[0,0] = self.nx*self.ny*self.parms.phi0
    self.phi = cp.fft.ifft2(self.phi_hat).real
    self.phi += self.GetEtaNoise() if noisy & (self.parms.eta != 0) else 0
    self.phi_hat = cp.fft.fft2(self.phi)
    self.phi_hat[0,0] = self.nx*self.ny*self.parms.phi0
    self.phi = cp.fft.ifft2(self.phi_hat).real
    self.phi0 = self.phi.copy()
    self.t = 0

  def AddNoise(self):
    self.phi += self.GetEtaNoise()
    self.phi_hat = cp.fft.fft2(self.phi)
    self.phi0 = cp.fft.ifft2(self.phi_hat).real

  def TimeStep(self):
    # vacancy terms
    self.vac = -6*1500*cp.power(self.phi,2)*(self.phi < 0)
    self.vac_hat = cp.fft.fft2(self.vac)

    # Log terms

    # phi^3 term
    self.phi3 = cp.power(self.phi, 3)
    self.phi3_hat = cp.fft.fft2(self.phi3)

    self.phi_hat = self.phi_hat + self.parms.dt * (self.lincoeff * self.phi_hat - self.k2 * (self.phi3_hat + self.vac_hat))
    self.phi = cp.fft.ifft2(self.phi_hat).real
    self.t += self.parms.dt

  def TimeStepCross(self):
    self.phi_hat = cp.fft.fft2(self.phi)
        
    # if noise_dynamics is true, add noise to phi2 and phi3
    if self.parms.NoiseDynamicsFlag:
      if self.parms.NoiseTimeSmoothFlag == False:
        self.GenerateTimeSmoothNoise()
      else:
        self.GetEtaNoise()

    self.N0_hat = self.Get_N_hat(self.phi)

    self.phi_hat0 = self.expcoeff * self.phi_hat + -self.k2 * self.expcoeff_nonlin * self.N0_hat
    self.phi0 = cp.fft.ifft2(self.phi_hat0).real

    self.N1_hat = (self.Get_N_hat(self.phi0) - self.N0_hat)/self.parms.dt

    phi_hat1 = self.expcoeff * self.phi_hat + -self.k2 * (self.expcoeff_nonlin * self.N0_hat + self.expcoeff_nonlin2 * self.N1_hat)
    phi1 = cp.fft.ifft2(phi_hat1).real

    delta_phi = phi1 - self.phi0
    if delta_phi.max() - delta_phi.min() > 0.01:
      self.phi0 = phi1
      self.N1_hat = (self.Get_N_hat(self.phi0) - self.N0_hat)/self.parms.dt

      delta_phi = phi1 - self.phi0
      if delta_phi.max() - delta_phi.min() > 0.01:
        raise Exception("predictor-corrector failed")

    # throw an exception if phi max > 3
    if cp.max(phi1) > 5:
      raise Exception("phi max > 5")
    
    # # if noise_dynamics is true, add noise to phi2 and phi3
    # if self.parms.noise_dynamics:
    #   self.GetEtaNoise()
    #   phi_hat1 += -self.k2 * self.noise_fft
    #   phi1 = cp.fft.ifft2(phi_hat1).real

    self.phi_hat = phi_hat1
    self.phi = phi1
    self.t += self.parms.dt

  def Get_N_hat(self, phi):
    # phi^2 term
    self.phi2 = self.parms.g * cp.power(phi, 2)

    # phi^3 term
    self.phi3 = self.parms.v0 * cp.power(phi, 3)

    # vacancy energy (stored as member for debugging)
    self.vac = -6*self.parms.Hng*cp.power(phi,2)*(phi < 0) # 3 ϕ (|ϕ| - ϕ) = { -6ϕ^2 where ϕ < 0, or, 0 where ϕ > 0 }

    # nl energy (stored as member for debugging)
    self.ln = self.parms.Hln * cp.where(phi > -self.parms.a + np.exp(self.minLog), cp.log(phi + self.parms.a), self.minLog)

    if self.parms.BoundaryPotential is not None:
      _VbMax = cp.max(self.parms.BoundaryPotential)
      _x = 1 - self.parms.BoundaryPotential / _VbMax
      self.phi2 *= _x
      self.phi2 *= _x
      self.vac *= _x
      self.ln *= _x
      self.boundary = self.parms.BoundaryPotential * (phi - self.parms.BoundaryDensity)
    else:
      self.boundary = 0
    
    # if noise_dynamics is true, add noise to phi2 and phi3
    if self.parms.NoiseDynamicsFlag:
      return cp.fft.fft2(self.phi2 + self.phi3 + self.vac + self.ln + self.boundary) + self.noise_fft
    
    return cp.fft.fft2(self.phi2 + self.phi3 + self.vac + self.ln + self.boundary)

  def CalcEnergyDensity(self):
    self.phi_hat = cp.fft.fft2(self.phi)
    self.energy_lin_1 = cp.fft.ifft2(self.linenergycoeff * self.phi_hat).real
    self.energy_lin = self.phi * self.energy_lin_1

    self.energy_ln = (self.phi + self.parms.a) * cp.where(self.phi > -self.parms.a, cp.log(self.phi + self.parms.a), 0)

    self.energy_poly = 1/2 * (self.parms.epsilon + self.parms.beta) * cp.power(self.phi,2) + 1/3 * self.parms.g * cp.power(self.phi,3) + 1/4 * self.parms.v0 * cp.power(self.phi,4)

    self.energy = (self.energy_lin + self.energy_ln + self.energy_poly)
    self.f = self.energy.sum() / (self.nx * self.dx * self.ny * self.dy)
    return
