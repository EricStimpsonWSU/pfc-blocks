! active PFC (Phase Field Crystal) model, for self-propulsion and self-spinning in 2D
! 2D phase for density field psi (=nA)
! local polarization vector field (Px, Py) (=(nB,nC))

! use a numerical algorithm with exponential propagation procedure for fully coupled PDEs of 3 variables
!     (exact solution for linear terms and approximation for nonlinearities, with initial transient time)
! pseudo-spetral method (using FFTW3)
  
! using OpenMP (To use single-core only: comment out "define open_MP")
! double precision, images output

! note: real data treatment for j=1 (qy=0) in fftw, otherwise divergence could occur at late time
!       or call forward FFT for psi,... --> psiq,... at the end of each time step (although psiq is known)
!       for convergence at large t, due to bugs/inaccuracies from fftw, or aliasing problem

# define ini_homogeneous ! for homogeneous initial condition

# define cavity_boundary_V ! set up boundary condition using an external potential
# define cavity_boundary_psi_flux0 ! both psi=psi_b and zero-flux psi boundary condition; otherwise only psi=psi_b boundary
# define cavity_boundary_VP0 ! additional P=P_b=0 boundary condition

# define cavity_circular ! set up circular/spherical cavity boundary
!# define hypocycloid_cavity ! set up hypocycloid cavity with n cusps
!# define hypocycloid_ring ! set up hypocycloid ring with n0+n1 cusps

# define xy_cshift05 ! x=i-lx/2-0.5, y=j-ly/2-0.5; otherwise x=i-lx/2, y=j-ly/2

# define conserved_psi ! impose the condition of psi conserved (<psi> unchanged, to be exact)
# define save_conf ! save conf files

# define open_MP ! use openMP for multi-threading

module global_variables
! by default all variables in the module are public (if not being set explicitly as private)
  use, intrinsic :: iso_c_binding
  implicit none
  include 'fftw3.f03'
  
  save ! save all the variables in the module

  integer, parameter :: ly=256, lx=256
!  integer, parameter :: ly=512, lx=512
!  integer, parameter :: ly=1024, lx=1024
!  integer, parameter :: ly=2048, lx=2048

  real(C_DOUBLE), pointer :: nA(:,:),nB(:,:),nC(:,:) ! corresponding to psi, Px, Py
  complex(C_DOUBLE_COMPLEX), pointer :: nAq(:,:),nBq(:,:),nCq(:,:)
  real(8) :: qx(lx),qy(ly/2+1)
  real(8) :: eps,q02,g,u,v0,Dr,C1,C4,M,pi
  double precision, allocatable :: q2(:,:)
  double complex, allocatable :: lambda(:,:,:,:),chi(:,:,:,:),delta(:,:,:,:)
  
  real(8) :: TOL,err_s ! for iteration of predictor-corrector
  integer :: nmbr_eval
  
! for FFTW parameters
  type(C_PTR) :: plan2d_f,plan2d_b
  real(8) :: scale2d_b
  
  character(:), allocatable :: file ! to be automatically allocated

# ifdef cavity_boundary_V
  ! set up cavity boundary condition using an external potential
  real(8) :: psi_b
# ifdef cavity_boundary_VP0
  ! additional P=P_b=0 boundary condition
  real(8) :: Px_b,Py_b
# endif
  double precision, allocatable :: Vb(:,:),n_x(:,:),n_y(:,:)
  integer :: i_Vb(ly,lx) ! i_Vb=0: inside, =1: outside the boundary

# ifdef cavity_circular
  ! set up circular/spherical cavity boundary
  real(8) :: r0 ! r0=cavity radius
# endif
# ifdef hypocycloid_cavity
  ! set up hypocycloid cavity with n cusps
  real(8) :: a0,b0,n
# endif
# ifdef hypocycloid_ring
  ! set up hypocycloid ring with n0+n1 cusps
  real(8) :: a0,b0,n0,a1,b1,n1
# endif

# endif
  ! ifdef cavity_boundary_V
  
end module global_variables


Program activePFC2D
  use global_variables
  implicit none

  type(C_PTR) :: p_alloc_psi,p_alloc_psiq ! for allocating aligned memory in fftw
  double precision, allocatable :: f(:,:)
  double complex, allocatable :: alpha_ij(:,:,:,:) ! coefficients of 3 PDEs
  double complex, allocatable :: sigma(:,:,:)
  integer, allocatable :: isigma(:,:)
  real(8) :: a,b,c,R,Q,R2,Q3,theta_s,AA,BB

  real(8) :: F_avg,area
  real(8) :: dt,dti,dtime
  real(8) :: dx,dy,psi0,Px0,Py0
  real(8) :: time,tmax,time0,t_i
  real(8) :: q0,qx0,qy0
  real(8) :: noise,noise0,ran3,y
  character(1) :: image(ly,lx) ! used for hdf image output
  integer :: i,j,ii,jj,iter,idum,ntype,istep
  integer :: nend,nout,nimpc,raout,nout_conf,n_i,n_dx
  integer :: iter0,n_0,i_imag,nt
  character(10) :: cha
  
# ifdef cavity_boundary_V
  ! set up cavity boundary condition using an external potential
  real(8) :: Vb0,Delta_b
# endif

# ifdef open_MP
  ! use multi-threaded fftw with openMP
  integer iret,nthreads,omp_get_max_threads
# endif
  
  character init_cond*(*) ! initial condition input from previous runs
  parameter(init_cond='data/_conf') ! filename of init_cond to be given from the conf file saved previously 

  ntype=1  ! ntype=1: set initial configuration; ntype=2: input from file
  time0=0
  iter0=1
!  ntype=2 ! initial condition input from previous runs
!  time0=30000
!  iter0=60098+1  ! iter0-1 = nend of previous run

! parameters entering the Eqs
!
! for rescaled PFC equations
  eps=-0.98d0 ! for nA=psi

  psi0=0 ! average density; smectic (stripe) phase
  Px0=0
  Py0=0

  v0=0.31d0 ! strength of active drive
  
  M=0 ! for external torque (if M nonzero: self spinning)
  
  g=0
  u=1

  Dr=0.5d0
  C1=0.2d0
  C4=0

  q0=1 ! for nA=psi
  q02=1

  pi=DACOS(-1.0d0)
  qx0=0.5d0*sqrt(3.d0)*q0 ! for hexagonal pattern; not used for smectics
  qy0=q0
! dx and dy fixed according to the unstrained one-mode state (q0)
  n_dx=8 ! n_dx: number of grid points per lattice period

  dy=2*pi/(n_dx*qy0)
  dx=dy

!  idum=-796921 ! random number seed; <0 for ran3()
  idum=-31131
  
# ifdef cavity_circular
  ! set up circular/spherical cavity boundary
  file='smh256c_bVP0_eps098g0Dr05C02C0_v031M0psi0' ! 256x256,v0=0.31,M=0,psi=0,dt=0.05,tmax=10000
# endif
# ifdef hypocycloid_cavity
  ! set up hypocycloid cavity with n cusps (b=a/n, max_r=a, area=(n-1)(n-2)pi*b^2; set max_r=a=lx/2-5*lx/256)
  file='smh256hypo6_bVP0_eps098g0Dr05C02C0_v031M0psi0' ! 256x256,n=6,v0=0.31,M=0,psi0=0,dt=0.05,tmax=10000
# endif
# ifdef hypocycloid_ring
  ! set up hypocycloid cavity with n0+n1 cusps (b=a/n, area=(n-1)(n-2)pi*(b0^2-b1^2); max_r=a0=lx/2-5*lx/256, a1=a0-5*lx/16)
  file='smh256hypo6r_bVP0_eps098g0Dr05C02C0_v031M0psi0' ! 256x256,n0=n1=n=6,v0=0.31,M=0,psi0=0,dt=0.05,tmax=10000
# endif

# ifdef cavity_boundary_V
  ! set up cavity boundary condition using an external potential
  Vb0=1
  Delta_b=0.1d0 ! in units of grid spacing dx; use [1+tanh(rs/Delta)]/2 profile for Vb
  psi_b=psi0
# ifdef cavity_boundary_VP0
  ! additional P=P_b=0 boundary condition
  Px_b=0
  Py_b=Px_b
# endif
  
# ifdef cavity_circular
  ! set up circular/spherical cavity boundary
  r0=3*lx/8 ! cavity size = 2*r0 = 3*lx/4; =lx/2-32=96 if lx=256
  !r0=lx/2-18*lx/256 ! =lx/2-18=110 if lx=256, =lx/2-36=220 if lx=512
  area=pi*r0*r0
# endif
  
# ifdef hypocycloid_cavity
  ! set up hypocycloid cavity with n cusps
  n=6
  a0=lx/2-5*lx/256 ! =lx/2-5 for 256x256; =lx/2-10 for 512x512
  b0=a0/n
  area=(n-1)*(n-2)*pi*b0*b0
# endif

# ifdef hypocycloid_ring
  ! set up hypocycloid ring with n0+n1 cusps
  n0=6
  a0=lx/2-5*lx/256 ! =lx/2-5 for 256x256; =lx/2-10 for 512x512
  if(int(n0)==3) a0=lx/2
  b0=a0/n0
  n1=n0
  a1=a0-5*lx/16 ! for n0>=6
  if(int(n0)==4.or.int(n0)==3) a1=a0-3*lx/8 ! for n0=4 or 3
  b1=a1/n1
  area=pi*((n0-1)*(n0-2)*b0*b0-(n1-1)*(n1-2)*b1*b1)
# endif
  
# endif
  ! ifdef cavity_boundary_V

! discretization of time
!  dt=0.5d0
  dt=0.2d0
# ifdef cavity_boundary_V
  dt=0.05d0 ! for Vb0=1
# endif

  t_i=1 ! time of initial transient; with t_i/dt=integer
  dti=0.01d0 ! used for initial transient up to t=t_i

  ! nmbr_eval: number of iterations used in the predictor-corrector algorithm
  nmbr_eval=1 ! no iteration of corrector
!  nmbr_eval=2 ! for one step of iteration
!  nmbr_eval=100 ! maximum steps of iterations
  TOL=1.d-3
  err_s=TOL/10

  noise=0.1d0 ! initial random noise
  noise0=0.01d0 ! for <>~0
!  noise0=noise
  
  ! tmax : maximum time
  ! nt: time interval to output images; nimpc : # of images 

!  tmax=5000
  tmax=10000
!  nt=200
  nt=500
!  tmax=100000
!  nt=5000
!  tmax=1000000
!  nt=10000
  
  nimpc=NINT((tmax-time0)/nt)
  n_i=NINT(t_i/dti)
  n_0=NINT(t_i/dt)-n_i ! used for output
  nend=NINT((tmax-t_i)/dt)+n_i
  nout=NINT(nt/dt) ! for image output
  
  raout=nout/10 ! for output of intermediate results
  nout_conf=nout*5 ! for saving conf files
  
! fft related items: qx and qy
  do j=1,ly/2+1
     qy(j)=2*pi*(j-1)/(ly*dy)
  enddo
  do i=1,lx
     if (i.le.lx/2+1) then
        qx(i)=2*pi*(i-1)/(lx*dx)
     else
        qx(i)=-2*pi*(lx-i+1)/(lx*dx)
     endif
  enddo

! write parameters in a separate file
!
  open(unit=2,file=file//'_para.dat')
  write(2,*) 'using 2nd order preditor-corrector method for psi, Px, Py active PFC equations'
  write(2,*) 'use an exponential propagation procedure'
  write(2,*) '    exact solution for linear terms and approximation for nonlinearities'
  write(2,*) '    expanding nonlinearities up to 1st order of t'
  write(2,*) 'lx=',lx, '  ly=',ly
  write(2,*) 'eps=',eps
  write(2,*) 'psi0=',psi0
  write(2,*) 'Px0=',Px0, '  Py0=',Py0
  write(2,*)
# ifdef ini_homogeneous
  write(2,*) 'def ini_homogeneous: for homogeneous initial condition'
# endif
# ifdef cavity_boundary_V
  write(2,*) 'def cavity_boundary_V: set up cavity boundary conditions using an external potential'
# ifdef cavity_boundary_psi_flux0
  write(2,*) 'def cavity_boundary_psi_flux0: both psi=psi_b and zero-flux psi boundary condition'
  write(2,*) '    hat{n} cdot nabla{psi} = 0'
# else
  write(2,*) 'ndef cavity_boundary_psi_flux0: only psi=psi_b for psi boundary conditions'
# endif
  write(2,*) 'use [1+tanh(rs/Delta)]/2 profile for Vb'
  write(2,*) 'Vb0=',Vb0, '  Delta_b=',Delta_b
  write(2,*) 'psi_b=',psi_b
# ifdef cavity_boundary_VP0
  write(2,*) 'def cavity_boundary_VP0: additional P=P_b=0 boundary condition'
  write(2,*) 'Px_b=',Px_b, '  Py_b=',Py_b
# endif
# ifdef cavity_circular
  write(2,*) 'def cavity_circular: set up circular/spherical cavity boundary'
  write(2,*) 'r0=',r0, '  area/(dxdy)=',area
# endif
# ifdef hypocycloid_cavity
  write(2,*) 'def hypocycloid_cavity: set up hypocycloid cavity with n cusps'
  write(2,*) 'n=',n
  write(2,*) 'a0=',a0, '  b0=',b0
  write(2,*) 'max_r=',a0, '  area/(dxdy)=',area
# endif
# ifdef hypocycloid_ring
  write(2,*) 'def hypocycloid_ring: set up hypocycloid ring with n0+n1 cusps'
  write(2,*) 'n0=',n0, '  n1=',n1
  write(2,*) 'a0=',a0, '  b0=',b0
  write(2,*) 'a1=',a1, '  b1=',b1
  write(2,*) 'max_r=',a0,a1
  write(2,*) 'width=',a0-a1,b0*sqrt((n0-1)**2+1+2*(n0-1)*cos(n0*pi/4))-b1*sqrt((n1-1)**2+1+2*(n1-1)*cos(n1*pi/4))
  write(2,*) 'area/(dxdy)=',area
# endif
  write(2,*)
# ifdef xy_cshift05
  write(2,*) 'def xy_cshift05: x=i-lx/2-0.5, y=j-ly/2-0.5'
# else
  write(2,*) 'ndef xy_cshift05: x=i-lx/2, y=j-ly/2'
# endif
# endif
  ! ifdef cavity_boundary_V
  
# ifdef conserved_psi
  write(2,*) 'def conserved_psi: impose the condition of psi conserved (<psi> unchanged, to be exact)'
# endif
  write(2,*) 'v0=',v0, '  M=',M
  write(2,*) 'g=',g, '  u=',u
  write(2,*) 'C1=',C1, '  C4=',C4
  write(2,*) 'Dr=',Dr
  write(2,*)
  write(2,*) 'dt=',dt, '  dti=',dti
  write(2,*) 'dx=',dx, '  dy=',dy
  write(2,*) 
  write(2,*) 'n_dx=', n_dx, '  number of lattice period along y=',ly/n_dx
  write(2,*) 'q0=',q0
  write(2,*) 'tmax=',tmax, '  t_transient=',t_i
  write(2,*) 'nend=',nend, '  n_transient=',n_i, '  nt=',nt
  write(2,*) 'nimpc=',nimpc, '   nout=',nout, '  raout=',raout
  write(2,*) '  data output at t=t_transient, and every ',raout*dt
  write(2,*) '  images output at t=0, t_transient, and every ',nout*dt
# ifdef save_conf
  write(2,*) '  save conf files every ',nout_conf*dt
# endif
  write(2,*) 'time0=',time0, '  iter0=',iter0, '  idum=',idum
  write(2,*) 'noise=',noise,' noise0=',noise0
  write(2,*) 'ntype=',ntype
  if(ntype.eq.2) write(2,*) 'init_cond = ',init_cond
  write(2,*)
  write(2,*) 'dynamics without noise (only initial noise)'
  write(2,*) 'For the predictor-corrector method: '
  write(2,*) '    Maximum number of iterations: nmbr_eval=',nmbr_eval
  write(2,*) '    TOL=',TOL,'  err_s=',err_s
# ifdef open_MP 
  write(2,*) 'use openMP for multi-threading'
# else
  write(2,*) 'no openMP'
# endif
  close(2)

  p_alloc_psi=fftw_alloc_real(int(ly * lx, C_SIZE_T))
  call c_f_pointer(p_alloc_psi, nA, [ly,lx])
  p_alloc_psi=fftw_alloc_real(int(ly * lx, C_SIZE_T))
  call c_f_pointer(p_alloc_psi, nB, [ly,lx])
  p_alloc_psi=fftw_alloc_real(int(ly * lx, C_SIZE_T))
  call c_f_pointer(p_alloc_psi, nC, [ly,lx])
  p_alloc_psiq=fftw_alloc_complex(int((ly/2+1) * lx, C_SIZE_T))
  call c_f_pointer(p_alloc_psiq, nAq, [ly/2+1,lx])
  p_alloc_psiq=fftw_alloc_complex(int((ly/2+1) * lx, C_SIZE_T))
  call c_f_pointer(p_alloc_psiq, nBq, [ly/2+1,lx])
  p_alloc_psiq=fftw_alloc_complex(int((ly/2+1) * lx, C_SIZE_T))
  call c_f_pointer(p_alloc_psiq, nCq, [ly/2+1,lx])

  allocate(q2(ly/2+1,lx))
  allocate(lambda(3,3,ly/2+1,lx),chi(3,3,ly/2+1,lx),delta(3,3,ly/2+1,lx))

  allocate(f(ly,lx))
  
# ifdef open_MP
  ! use multi-threaded fftw with openMP
  nthreads=omp_get_max_threads()
  iret=fftw_init_threads()
  if(iret.eq.0) then
     write(*,*) 'iret=0: error during thread initialization'
     stop
  endif
  call fftw_plan_with_nthreads(nthreads)
  open(unit=2,file=file//'_para.dat',position='append')
  write(2,*) 'use multi-threaded fftw; nthreads=',nthreads
  close(2)
# endif
  
  y=ran3(idum) ! initialize random numbers
 
! for FFTW parameters
  scale2d_b=1.d0/(ly*lx) ! for backward FFT

! initialization for FFTW (out of place; create the plans before initializing the inputs)
  plan2d_f=fftw_plan_dft_r2c_2d(lx,ly,nA,nAq,FFTW_MEASURE) ! note the reverse order of dimensions
  plan2d_b=fftw_plan_dft_c2r_2d(lx,ly,nAq,nA,FFTW_MEASURE)

# ifdef cavity_boundary_V
  ! set up cavity boundary condition using an external potential
  allocate(Vb(ly,lx),n_x(ly,lx),n_y(ly,lx))
  call Vb_n(Vb0,Delta_b)
  i_Vb=0
  do i=1,lx
     do j=1,ly
        if(Vb(j,i)/Vb0>0.1d0) i_Vb(j,i)=1 ! outside the boundary
     enddo
  enddo
# endif
  
! set the initial conditions
!
  if(ntype.eq.1) then
    !
    ! set up the initial configuration
    !
# ifdef ini_homogeneous
     ! for homogeneous initial condition
     do i=1,lx
        do j=1,ly
           if(abs(psi0).lt.1.d-10) then
              nA(j,i)=psi0+noise0*(ran3(idum)-0.5d0) ! if psi0 ~ 0
           else
!              nA(j,i)=psi0*(1+noise*(ran3(idum)-0.5d0))
              nA(j,i)=psi0+noise*(ran3(idum)-0.5d0)
           endif
           if(abs(Px0).lt.1.d-10) then
              nB(j,i)=Px0+noise0*(ran3(idum)-0.5d0) ! if Px0 ~ 0
           else
!              nB(j,i)=Px0*(1+noise*(ran3(idum)-0.5d0))
              nB(j,i)=Px0+noise*(ran3(idum)-0.5d0)
           endif
           if(abs(Py0).lt.1.d-10) then
              nC(j,i)=Py0+noise0*(ran3(idum)-0.5d0) ! if Py0 ~ 0
           else
!              nC(j,i)=Py0*(1+noise*(ran3(idum)-0.5d0))
              nC(j,i)=Py0+noise*(ran3(idum)-0.5d0)
           endif
        enddo
     enddo

     call fftw_execute_dft_r2c(plan2d_f,nA,nAq) ! forward FFT
     call fftw_execute_dft_r2c(plan2d_f,nB,nBq) ! forward FFT
     call fftw_execute_dft_r2c(plan2d_f,nC,nCq) ! forward FFT
     nAq(1,1)=psi0*lx*ly ! <nA>=psi0
     nBq(1,1)=Px0*lx*ly ! <nB>=Px0
     nCq(1,1)=Py0*lx*ly ! <nC>=Py0
! backward FFT (note: the input array is overwritten for all c2r transforms)
     call fftw_execute_dft_c2r(plan2d_b,nAq,nA)
     nA=nA*scale2d_b
     call fftw_execute_dft_c2r(plan2d_b,nBq,nB)
     nB=nB*scale2d_b
     call fftw_execute_dft_c2r(plan2d_b,nCq,nC)
     nC=nC*scale2d_b
# endif

# ifdef cavity_boundary_V
  ! set up cavity boundary condition using an external potential
     do i=1,lx
        do j=1,ly
           if(i_Vb(j,i).eq.1) then ! outside the boundary
              if(abs(psi0-psi_b)>1.d-5) nA(j,i)=psi_b
# ifdef cavity_boundary_VP0
  ! additional P=P_b=0 boundary condition
              if(abs(Px0-Px_b)>1.d-5) nB(j,i)=Px_b
              if(abs(Py0-Py_b)>1.d-5) nC(j,i)=Py_b
# endif
           endif
        enddo
     enddo
# endif

  else
    ! 
    ! read initial configuration from file
    !
     open(unit=11,file=init_cond//'.dat',form='unformatted',status='old')
     read(11) nA,nB,nC
     close(11)
  endif

  call fftw_execute_dft_r2c(plan2d_f,nA,nAq) ! forward FFT
  call fftw_execute_dft_r2c(plan2d_f,nB,nBq) ! forward FFT
  call fftw_execute_dft_r2c(plan2d_f,nC,nCq) ! forward FFT

  time=time0

  if(iter0.le.n_i) then
     dtime=dti ! for initial transient
  else
     dtime=dt
  endif
  
  allocate(alpha_ij(3,3,ly/2+1,lx))
  allocate(sigma(3,ly/2+1,lx))
  allocate(isigma(ly/2+1,lx))

  isigma=0
  alpha_ij=0
  sigma=0
  do i=1,lx
     do j=1,ly/2+1
        q2(j,i)=qx(i)*qx(i)+qy(j)*qy(j)
        alpha_ij(1,1,j,i)=-q2(j,i)*((q2(j,i)-q02)**2+eps)
        alpha_ij(2,2,j,i)=-C1*(q2(j,i)+Dr)
        
        alpha_ij(3,3,j,i)=alpha_ij(2,2,j,i)
        alpha_ij(1,2,j,i)=cmplx(0,-qx(i)*v0)
        alpha_ij(2,1,j,i)=alpha_ij(1,2,j,i)
        alpha_ij(1,3,j,i)=cmplx(0,-qy(j)*v0)
        alpha_ij(3,1,j,i)=alpha_ij(1,3,j,i)
        alpha_ij(2,3,j,i)=-M
        alpha_ij(3,2,j,i)=M

! solve the characteristic cubic equation to get sigma1, sigma2, sigma3 (for real coefficients a, b, c and R, Q)
! (the procedure here kept the same for different set of eqs)
! general formulae for a,b,c
!        a=-(alpha_ij(1,1,j,i)+alpha_ij(2,2,j,i)+alpha_ij(3,3,j,i))
!        b=alpha_ij(1,1,j,i)*alpha_ij(2,2,j,i)+alpha_ij(1,1,j,i)*alpha_ij(3,3,j,i)+alpha_ij(2,2,j,i)*alpha_ij(3,3,j,i) &
!         -alpha_ij(1,2,j,i)*alpha_ij(2,1,j,i)-alpha_ij(1,3,j,i)*alpha_ij(3,1,j,i)-alpha_ij(2,3,j,i)*alpha_ij(3,2,j,i)
!        c=alpha_ij(1,1,j,i)*alpha_ij(2,3,j,i)*alpha_ij(3,2,j,i)+alpha_ij(2,2,j,i)*alpha_ij(1,3,j,i)*alpha_ij(3,1,j,i) &
!         +alpha_ij(3,3,j,i)*alpha_ij(1,2,j,i)*alpha_ij(2,1,j,i)-alpha_ij(1,2,j,i)*alpha_ij(2,3,j,i)*alpha_ij(3,1,j,i) &
!         -alpha_ij(1,3,j,i)*alpha_ij(2,1,j,i)*alpha_ij(3,2,j,i)-alpha_ij(1,1,j,i)*alpha_ij(2,2,j,i)*alpha_ij(3,3,j,i)
! for the specific case here
        a=-(alpha_ij(1,1,j,i)+2*alpha_ij(2,2,j,i))
        b=2*alpha_ij(1,1,j,i)*alpha_ij(2,2,j,i)+alpha_ij(2,2,j,i)*alpha_ij(2,2,j,i)+M*M+v0*v0*q2(j,i)
        c=-(M*M+alpha_ij(2,2,j,i)*alpha_ij(2,2,j,i))*alpha_ij(1,1,j,i)-v0*v0*alpha_ij(2,2,j,i)*q2(j,i)

        Q=a*a/9-b/3
        R=a*a*a/27-a*b/6+c/2
        R2=R*R
        Q3=Q*Q*Q
        if(abs(Q).gt.1.d-20) then ! when Q \neq 0 (not good if using abs(Q3) which could be too small)
           !if(abs(R2-Q3).lt.1.d-8) then ! if R^2=Q^3 (note: not accurate when R2,Q3 are very large)
           if(abs(R2/Q3-1).lt.1.d-8) then ! if R^2/Q^3=1
              isigma(j,i)=2 ! 2 equal roots: sigma1 \neq sigma2 = sigma3 (if R^2 = Q^3 \neq 0)
              AA=sign(abs(R)**(1.d0/3),R) ! to calculate R^1/3
              sigma(1,j,i)=-2*AA-a/3
              sigma(2,j,i)=AA-a/3
              sigma(3,j,i)=sigma(2,j,i)
           else
              isigma(j,i)=1 ! 3 distinct roots
           endif
        else ! when Q=0
           if(abs(R2-Q3).lt.1.d-25) then
              isigma(j,i)=3 ! 3 equal roots: sigma1=sigma2=sigma3 (if R^2=Q^3=0)
              sigma(1,j,i)=-a/3
              sigma(2,j,i)=sigma(1,j,i)
              sigma(3,j,i)=sigma(1,j,i)
           else
              isigma(j,i)=1 ! 3 distinct roots
           endif
        endif
        if(isigma(j,i).eq.1) then ! if R^2 \neq Q^3 (3 distinct roots); use the procedure in Numerical Recipes
           if(R2.lt.Q3) then ! 3 distinct real roots
              theta_s=dacos(R/sqrt(Q3))
              AA=2*sqrt(Q)
              BB=a/3
              sigma(1,j,i)=-AA*cos(theta_s/3)-BB
              sigma(2,j,i)=-AA*cos((theta_s+2*pi)/3)-BB
              sigma(3,j,i)=-AA*cos((theta_s-2*pi)/3)-BB
           else ! 1 real and 2 conjugate complex roots
              AA=(abs(R)+sqrt(R2-Q3))**(1.d0/3)
              if(R.gt.0) AA=-AA
              if(abs(AA).gt.1.d-10) then
                 BB=Q/AA
              else
                 BB=0
              endif
              sigma(1,j,i)=AA+BB-a/3
              sigma(2,j,i)=cmplx(-(AA+BB)/2-a/3,(AA-BB)*sqrt(3.d0)/2)
              sigma(3,j,i)=conjg(sigma(2,j,i))
           endif
           ! for sigma approximately equal roots, particularly when R2,Q3 are very large
           if(abs(sigma(2,j,i)-sigma(3,j,i)).lt.1.d-10) then
              if(abs(sigma(1,j,i)-sigma(2,j,i)).lt.1.d-10) then
                 isigma(j,i)=3 ! sigma1=sigma2=sigma3
              else
                 isigma(j,i)=2 ! sigma1 \neq sigma2 = sigma3
              endif
           endif
        endif
     enddo
  enddo
  
  call coefficients_eqs(dtime,sigma,alpha_ij,isigma) ! obtain the coefficients used in numerical solution

  call F_mu(f,F_avg)
  open(unit=1,file=file//'_f.dat',position='append')
  write(1,*) time,F_avg ! for average of F density
  close(1)

  open(unit=2,file=file//'_out.dat',position='append')
  write(2,*) 't=',time
  write(2,*) maxval(nA(1:ly,1:lx)),minval(nA(1:ly,1:lx))
  write(2,*) '  <psi>=',sum(nA)*scale2d_b
  write(2,*) maxval(nB(1:ly,1:lx)),minval(nB(1:ly,1:lx))
  write(2,*) '  <Px>=',sum(nB)*scale2d_b
  write(2,*) maxval(nC(1:ly,1:lx)),minval(nC(1:ly,1:lx))
  write(2,*) '  <Py>=',sum(nC)*scale2d_b
  write(2,*) 'f=',maxval(f),minval(f)
  write(2,*) 'isigma=',minval(isigma),maxval(isigma)
  write(2,*) 'isigma(1,1)=',isigma(1,1)
  write(2,*) 'sigma(q=0)=',sigma(1,1,1),sigma(2,1,1),sigma(3,1,1)
  write(2,*)

! output hdf image files
  call hdf_r8(ly,lx,nA,image,file//'_psi.hdf')
  call hdf_r8(ly,lx,nB,image,file//'_Px.hdf')
  call hdf_r8(ly,lx,nC,image,file//'_Py.hdf')
  !call hdf_r8(ly,lx,f,image,file//'_f.hdf') ! output free energy image
  f=nB*nB+nC*nC ! P^2=Px^2+Py^2
  call hdf_r8(ly,lx,f,image,file//'_P2.hdf')

  i_imag=1
  
  if(iter0.gt.n_i+1) deallocate(alpha_ij,sigma,isigma)
  
! start running

  do iter=iter0,nend

    if(iter.le.n_i) then ! for initial transient only
       time=iter*dti
    else
       time=(iter-n_i)*dt+t_i
       if(iter.eq.n_i+1) then
          dtime=dt
          call coefficients_eqs(dtime,sigma,alpha_ij,isigma) ! obtain the coefficients used in numerical solution
          deallocate(alpha_ij,sigma,isigma)
       endif
    endif
  
    call eqs(time,iter,n_i,istep)
! for late time convergence in fftw (replacing j=1 (qy=0) real data treatment)
!    call fftw_execute_dft_r2c(plan2d_f,nA,nAq) ! forward FFT
!    call fftw_execute_dft_r2c(plan2d_f,nB,nBq) ! forward FFT
!    call fftw_execute_dft_r2c(plan2d_f,nC,nCq) ! forward FFT

    if((iter.eq.n_i).or.(iter.gt.n_i.and.mod(iter+n_0,raout).eq.0).or.(iter.eq.nend)) then 

      call F_mu(f,F_avg)
      open(unit=1,file=file//'_f.dat',position='append')
      write(1,*) time,F_avg ! for average of F
      close(1)

      open(unit=2,file=file//'_out.dat',position='append')
      write(2,*) 't=',time,'  istep=',istep
      write(2,*) maxval(nA(1:ly,1:lx)),minval(nA(1:ly,1:lx))
      write(2,*) '  <psi>=',sum(nA)*scale2d_b
      write(2,*) maxval(nB(1:ly,1:lx)),minval(nB(1:ly,1:lx))
      write(2,*) '  <Px>=',sum(nB)*scale2d_b
      write(2,*) maxval(nC(1:ly,1:lx)),minval(nC(1:ly,1:lx))
      write(2,*) '  <Py>=',sum(nC)*scale2d_b
      write(2,*) 'f=',maxval(f),minval(f)
      write(2,*)
      close(2)

! save images
      if((iter.eq.n_i).or.(iter.gt.n_i.and.mod(iter+n_0,nout).eq.0).or.(iter.eq.nend)) then
         
         call hdf_r8(ly,lx,nA,image,file//'_psi.hdf')
         call hdf_r8(ly,lx,nB,image,file//'_Px.hdf')
         call hdf_r8(ly,lx,nC,image,file//'_Py.hdf')
         !call hdf_r8(ly,lx,f,image,file//'_f.hdf') ! output free energy image
         f=nB*nB+nC*nC ! P^2=Px^2+Py^2
         call hdf_r8(ly,lx,f,image,file//'_P2.hdf')

! output psi,Px,Py data files at t=itime
         if(iter.eq.n_i) then
            cha='_i'
         else
            write(cha,'(i0)') NINT(time)
         endif
         open(unit=10,file=file//'_psiPxPy_t'//trim(cha)//'.dat')
         do i=1,lx
            do j=1,ly
               write(10,*) nA(j,i),nB(j,i),nC(j,i) ! psi(j,i),Px(j,i),Py(j,i) data
            enddo
         enddo
         close(10)
         
         open(unit=2,file=file//'_out.dat',position='append')
         i_imag=i_imag+1
         write(2,*) 't=',time, '  iter=',iter,' for output of image files'
         write(2,*) '  of No.',i_imag,' for nA=psi, nB=Px, nC=Py'
         write(2,*)
         close(2)
         
      endif ! for nout

# ifdef save_conf
    ! save conf files
    if((iter.gt.n_i.and.mod(iter+n_0,nout_conf).eq.0).or.(iter.eq.nend)) then
       open(unit=11,file=file//'_conf.dat',form='unformatted')
       write(11) nA,nB,nC
       close(11)
       open(unit=2,file=file//'_out.dat',position='append')
       write(2,*) 't=',time, '  iter=',iter,' for output of conf files'
       write(2,*)
       close(2)
    endif
# endif

   endif ! for raout

  enddo ! for iter=iter0,nend
    
# ifdef cavity_boundary_V
  deallocate(Vb,n_x,n_y)
# endif
  
  deallocate(q2)
  deallocate(lambda,chi,delta)
  deallocate(f)
  call fftw_free(p_alloc_psi)
  call fftw_free(p_alloc_psiq)

  call fftw_destroy_plan(plan2d_f)
  call fftw_destroy_plan(plan2d_b)

end Program activePFC2D


! obtain the coefficients used in numerical solution (this subroutine kept the same for different set of eqs)
subroutine coefficients_eqs(dtime,sigma,alpha_ij,isigma)
  use global_variables
  implicit none

  real(8) :: dtime
  complex(8) :: sigma(3,ly/2+1,lx),alpha_ij(3,3,ly/2+1,lx)
  integer :: isigma(ly/2+1,lx)
  complex(8) :: LambdaABC(3,3),sigma123,sigma_kl
  complex(8) :: sig_dt,exp_sig(3),cf_1(3),cf2_1(3),cf_2,cf2_2,cf_3,cf2_3
  integer :: i,j,k,l,ii,jj,kk

  lambda=0
  chi=0
  delta=0
  do i=1,lx
     do j=1,ly/2+1
        do ii=1,3
           sig_dt=sigma(ii,j,i)*dtime
           exp_sig(ii)=exp(sig_dt)
           if(abs(sigma(ii,j,i)).gt.1.d-5) then
              cf_1(ii)=(exp_sig(ii)-1)/sigma(ii,j,i)
              cf2_1(ii)=(exp_sig(ii)-(1+sig_dt))/(sigma(ii,j,i)*sig_dt)
           else ! when sigma --> 0
              cf_1(ii)=dtime*(1.d0+sig_dt*(1.d0+sig_dt/3)/2)
              cf2_1(ii)=0.5d0*dtime*(1.d0+sig_dt*(1.d0+sig_dt/4)/3)
           endif
        enddo
        
        if(isigma(j,i).eq.1) then ! 3 distinct roots: sigma1 \neq sigma2 \neq sigma3
           sigma123=(sigma(1,j,i)-sigma(2,j,i))*(sigma(1,j,i)-sigma(3,j,i))*(sigma(2,j,i)-sigma(3,j,i))
           do ii=1,3 ! corresponding to sigma1,sigma2,sigma3
              k=ii+1
              if(k.gt.3) k=k-3
              l=ii+2
              if(l.gt.3) l=l-3
              sigma_kl=sigma(k,j,i)-sigma(l,j,i)

              ! coefficients to get nAq(t+dt)
              LambdaABC(1,ii)=((sigma(ii,j,i)-alpha_ij(2,2,j,i))*(sigma(ii,j,i)-alpha_ij(3,3,j,i)) &
                              -alpha_ij(2,3,j,i)*alpha_ij(3,2,j,i))*sigma_kl
              LambdaABC(2,ii)=(alpha_ij(1,2,j,i)*(sigma(ii,j,i)-alpha_ij(3,3,j,i)) &
                              +alpha_ij(1,3,j,i)*alpha_ij(3,2,j,i))*sigma_kl
              LambdaABC(3,ii)=(alpha_ij(1,3,j,i)*(sigma(ii,j,i)-alpha_ij(2,2,j,i)) &
                              +alpha_ij(1,2,j,i)*alpha_ij(2,3,j,i))*sigma_kl
              do jj=1,3
                 lambda(1,jj,j,i)=lambda(1,jj,j,i)+LambdaABC(jj,ii)*exp_sig(ii)
                 chi(1,jj,j,i)=chi(1,jj,j,i)+LambdaABC(jj,ii)*cf_1(ii)
                 delta(1,jj,j,i)=delta(1,jj,j,i)+LambdaABC(jj,ii)*cf2_1(ii)
              enddo

              ! coefficients to get nBq(t+dt)
              LambdaABC(1,ii)=(alpha_ij(2,1,j,i)*(sigma(ii,j,i)-alpha_ij(3,3,j,i)) &
                              +alpha_ij(2,3,j,i)*alpha_ij(3,1,j,i))*sigma_kl
              LambdaABC(2,ii)=((sigma(ii,j,i)-alpha_ij(1,1,j,i))*(sigma(ii,j,i)-alpha_ij(3,3,j,i)) &
                              -alpha_ij(1,3,j,i)*alpha_ij(3,1,j,i))*sigma_kl
              LambdaABC(3,ii)=(alpha_ij(2,3,j,i)*(sigma(ii,j,i)-alpha_ij(1,1,j,i)) &
                              +alpha_ij(1,3,j,i)*alpha_ij(2,1,j,i))*sigma_kl
              do jj=1,3
                 lambda(2,jj,j,i)=lambda(2,jj,j,i)+LambdaABC(jj,ii)*exp_sig(ii)
                 chi(2,jj,j,i)=chi(2,jj,j,i)+LambdaABC(jj,ii)*cf_1(ii)
                 delta(2,jj,j,i)=delta(2,jj,j,i)+LambdaABC(jj,ii)*cf2_1(ii)
              enddo
              
              ! coefficients to get nCq(t+dt)
              LambdaABC(1,ii)=(alpha_ij(3,1,j,i)*(sigma(ii,j,i)-alpha_ij(2,2,j,i)) &
                              +alpha_ij(2,1,j,i)*alpha_ij(3,2,j,i))*sigma_kl
              LambdaABC(2,ii)=(alpha_ij(3,2,j,i)*(sigma(ii,j,i)-alpha_ij(1,1,j,i)) &
                              +alpha_ij(1,2,j,i)*alpha_ij(3,1,j,i))*sigma_kl
              LambdaABC(3,ii)=((sigma(ii,j,i)-alpha_ij(1,1,j,i))*(sigma(ii,j,i)-alpha_ij(2,2,j,i)) &
                              -alpha_ij(1,2,j,i)*alpha_ij(2,1,j,i))*sigma_kl
              do jj=1,3
                 lambda(3,jj,j,i)=lambda(3,jj,j,i)+LambdaABC(jj,ii)*exp_sig(ii)
                 chi(3,jj,j,i)=chi(3,jj,j,i)+LambdaABC(jj,ii)*cf_1(ii)
                 delta(3,jj,j,i)=delta(3,jj,j,i)+LambdaABC(jj,ii)*cf2_1(ii)
              enddo
           enddo
           
           do kk=1,3
              do jj=1,3
                 lambda(kk,jj,j,i)=lambda(kk,jj,j,i)/sigma123
                 chi(kk,jj,j,i)=chi(kk,jj,j,i)/sigma123
                 delta(kk,jj,j,i)=delta(kk,jj,j,i)/sigma123
              enddo
           enddo
        endif
        
        if(isigma(j,i).eq.2) then ! 2 equal roots: sigma1 \neq sigma2 = sigma3
           sig_dt=sigma(2,j,i)*dtime
           if(abs(sigma(2,j,i)).gt.1.d-5) then
              cf_2=(1+(sig_dt-1)*exp_sig(2))/(sigma(2,j,i)*sigma(2,j,i))
              cf2_2=(2+sig_dt+(sig_dt-2)*exp_sig(2))/(sigma(2,j,i)*sigma(2,j,i)*sig_dt)
           else ! when sigma --> 0
              cf_2=dtime*dtime*(0.5d0+sig_dt/3+sig_dt*sig_dt/8)
              cf2_2=dtime*dtime*(1.d0/6.d0+sig_dt/12+sig_dt*sig_dt/40)
           endif

           sigma123=sigma(1,j,i)-sigma(2,j,i)
           
           do kk=1,3 ! coefficients to get nAq(t+dt),nBq(t+dt),nCq(t+dt)
           do ii=1,2 ! corresponding to sigma1,sigma2=sigma3
              if(kk.eq.1) then ! for nAq(t+dt)
                 LambdaABC(1,ii)=(sigma(ii,j,i)-alpha_ij(2,2,j,i))*(sigma(ii,j,i)-alpha_ij(3,3,j,i)) &
                                 -alpha_ij(2,3,j,i)*alpha_ij(3,2,j,i)
                 LambdaABC(2,ii)=alpha_ij(1,2,j,i)*(sigma(ii,j,i)-alpha_ij(3,3,j,i)) &
                                 +alpha_ij(1,3,j,i)*alpha_ij(3,2,j,i)
                 LambdaABC(3,ii)=alpha_ij(1,3,j,i)*(sigma(ii,j,i)-alpha_ij(2,2,j,i)) &
                                 +alpha_ij(1,2,j,i)*alpha_ij(2,3,j,i)
              endif
              if(kk.eq.2) then ! for nBq(t+dt)
                 LambdaABC(1,ii)=alpha_ij(2,1,j,i)*(sigma(ii,j,i)-alpha_ij(3,3,j,i)) &
                                 +alpha_ij(2,3,j,i)*alpha_ij(3,1,j,i)
                 LambdaABC(2,ii)=(sigma(ii,j,i)-alpha_ij(1,1,j,i))*(sigma(ii,j,i)-alpha_ij(3,3,j,i)) &
                                 -alpha_ij(1,3,j,i)*alpha_ij(3,1,j,i)
                 LambdaABC(3,ii)=alpha_ij(2,3,j,i)*(sigma(ii,j,i)-alpha_ij(1,1,j,i)) &
                                 +alpha_ij(1,3,j,i)*alpha_ij(2,1,j,i)
              endif
              if(kk.eq.3) then ! for nCq(t+dt)
                 LambdaABC(1,ii)=alpha_ij(3,1,j,i)*(sigma(ii,j,i)-alpha_ij(2,2,j,i)) &
                                 +alpha_ij(2,1,j,i)*alpha_ij(3,2,j,i)
                 LambdaABC(2,ii)=alpha_ij(3,2,j,i)*(sigma(ii,j,i)-alpha_ij(1,1,j,i)) &
                                 +alpha_ij(1,2,j,i)*alpha_ij(3,1,j,i)
                 LambdaABC(3,ii)=(sigma(ii,j,i)-alpha_ij(1,1,j,i))*(sigma(ii,j,i)-alpha_ij(2,2,j,i)) &
                                 -alpha_ij(1,2,j,i)*alpha_ij(2,1,j,i)
              endif
           enddo
           do jj=1,3
              if(jj.eq.kk) then
                 sigma_kl=-(sigma(1,j,i)-alpha_ij(kk,kk,j,i))
              else
                 sigma_kl=alpha_ij(kk,jj,j,i)
              endif
              lambda(kk,jj,j,i)=LambdaABC(jj,1)*exp_sig(1)-(sigma123*(sigma_kl+LambdaABC(jj,2)*dtime) &
                                                           +LambdaABC(jj,2))*exp_sig(2)
              chi(kk,jj,j,i)=LambdaABC(jj,1)*cf_1(1)-(LambdaABC(jj,2)+sigma_kl*sigma123)*cf_1(2) &
                            -LambdaABC(jj,2)*sigma123*cf_2
              delta(kk,jj,j,i)=LambdaABC(jj,1)*cf2_1(1)-(LambdaABC(jj,2)+sigma_kl*sigma123)*cf2_1(2) &
                              -LambdaABC(jj,2)*sigma123*cf2_2
           enddo
           enddo ! do kk=1,3

           sigma123=sigma123**2
           do kk=1,3
              do jj=1,3
                 lambda(kk,jj,j,i)=lambda(kk,jj,j,i)/sigma123
                 chi(kk,jj,j,i)=chi(kk,jj,j,i)/sigma123
                 delta(kk,jj,j,i)=delta(kk,jj,j,i)/sigma123
              enddo
           enddo           
        endif
        
        if(isigma(j,i).eq.3) then ! 3 equal roots: sigma1 = sigma2 = sigma3
           sig_dt=sigma(1,j,i)*dtime
           if(abs(sigma(1,j,i)).gt.1.d-5) then
              cf_2=(1+(sig_dt-1)*exp_sig(1))/(sigma(1,j,i)*sigma(1,j,i))
              cf2_2=(2+sig_dt+(sig_dt-2)*exp_sig(1))/(sigma(1,j,i)*sigma(1,j,i)*sig_dt)
              cf_3=(((sig_dt-1)**2+1)*exp_sig(1)-2)/(sigma(1,j,i)**3)
              cf2_3=(((sig_dt-2)**2+2)*exp_sig(1)-2*(sig_dt+3))/(sigma(1,j,i)**3*sig_dt)
           else ! when sigma --> 0
              cf_2=dtime*dtime*(0.5d0+sig_dt/3+sig_dt*sig_dt/8)
              cf2_2=dtime*dtime*(1.d0/6.d0+sig_dt/12+sig_dt*sig_dt/40)
              cf_3=dtime*dtime*dtime*(1.d0/3+sig_dt/4+sig_dt*sig_dt/10)
              cf2_3=dtime*dtime*dtime*(1.d0/12+sig_dt/20+sig_dt*sig_dt/60)
           endif

           ii=1
           do kk=1,3 ! coefficients to get nAq(t+dt),nBq(t+dt),nCq(t+dt)
              if(kk.eq.1) then ! for nAq(t+dt)
                 LambdaABC(1,ii)=(sigma(ii,j,i)-alpha_ij(2,2,j,i))*(sigma(ii,j,i)-alpha_ij(3,3,j,i)) &
                                 -alpha_ij(2,3,j,i)*alpha_ij(3,2,j,i)
                 LambdaABC(2,ii)=alpha_ij(1,2,j,i)*(sigma(ii,j,i)-alpha_ij(3,3,j,i)) &
                                 +alpha_ij(1,3,j,i)*alpha_ij(3,2,j,i)
                 LambdaABC(3,ii)=alpha_ij(1,3,j,i)*(sigma(ii,j,i)-alpha_ij(2,2,j,i)) &
                                 +alpha_ij(1,2,j,i)*alpha_ij(2,3,j,i)
              endif
              if(kk.eq.2) then ! for nBq(t+dt)
                 LambdaABC(1,ii)=alpha_ij(2,1,j,i)*(sigma(ii,j,i)-alpha_ij(3,3,j,i)) &
                                 +alpha_ij(2,3,j,i)*alpha_ij(3,1,j,i)
                 LambdaABC(2,ii)=(sigma(ii,j,i)-alpha_ij(1,1,j,i))*(sigma(ii,j,i)-alpha_ij(3,3,j,i)) &
                                 -alpha_ij(1,3,j,i)*alpha_ij(3,1,j,i)
                 LambdaABC(3,ii)=alpha_ij(2,3,j,i)*(sigma(ii,j,i)-alpha_ij(1,1,j,i)) &
                                 +alpha_ij(1,3,j,i)*alpha_ij(2,1,j,i)
              endif
              if(kk.eq.3) then ! for nCq(t+dt)
                 LambdaABC(1,ii)=alpha_ij(3,1,j,i)*(sigma(ii,j,i)-alpha_ij(2,2,j,i)) &
                                 +alpha_ij(2,1,j,i)*alpha_ij(3,2,j,i)
                 LambdaABC(2,ii)=alpha_ij(3,2,j,i)*(sigma(ii,j,i)-alpha_ij(1,1,j,i)) &
                                 +alpha_ij(1,2,j,i)*alpha_ij(3,1,j,i)
                 LambdaABC(3,ii)=(sigma(ii,j,i)-alpha_ij(1,1,j,i))*(sigma(ii,j,i)-alpha_ij(2,2,j,i)) &
                                 -alpha_ij(1,2,j,i)*alpha_ij(2,1,j,i)
              endif
              do jj=1,3
                 if(jj.eq.kk) then
                    sigma_kl=-(sigma(1,j,i)-alpha_ij(kk,kk,j,i))
                    sig_dt=1
                 else
                    sigma_kl=alpha_ij(kk,jj,j,i)
                    sig_dt=0
                 endif
                 lambda(kk,jj,j,i)=(sig_dt+(sigma_kl+LambdaABC(jj,1)*dtime/2)*dtime)*exp_sig(1)
                 chi(kk,jj,j,i)=sig_dt*cf_1(1)+sigma_kl*cf_2+LambdaABC(jj,1)*cf_3/2
                 delta(kk,jj,j,i)=sig_dt*cf2_1(1)+sigma_kl*cf2_2+LambdaABC(jj,1)*cf2_3/2
              enddo
           enddo ! do kk=1,3
        endif
     enddo
  enddo
  
  open(unit=2,file=file//'_out.dat',position='append')
  write(2,*) 'dtime=',dtime, '  from lambda, chi, delta formulae'
  write(2,*) 'lambda11,12,13 at q=0:',lambda(1,1,1,1),lambda(1,2,1,1),lambda(1,3,1,1)
  write(2,*) 'lambda21,22,23 at q=0:',lambda(2,1,1,1),lambda(2,2,1,1),lambda(2,3,1,1)
  write(2,*) 'lambda31,32,33 at q=0:',lambda(3,1,1,1),lambda(3,2,1,1),lambda(3,3,1,1)
  write(2,*) '|lambda|11,12,13 max:'
  write(2,*) maxval(abs(lambda(1,1,:,:))),maxval(abs(lambda(1,2,:,:))),maxval(abs(lambda(1,3,:,:)))
  write(2,*) '|chi|11,12,13 max:'
  write(2,*) maxval(abs(chi(1,1,:,:))),maxval(abs(chi(1,2,:,:))),maxval(abs(chi(1,3,:,:)))
  write(2,*) '|delta|11,12,13 max:'
  write(2,*) maxval(abs(delta(1,1,:,:))),maxval(abs(delta(1,2,:,:))),maxval(abs(delta(1,3,:,:)))
  write(2,*) '|lambda|21,22,23 max:'
  write(2,*) maxval(abs(lambda(2,1,:,:))),maxval(abs(lambda(2,2,:,:))),maxval(abs(lambda(2,3,:,:)))
  write(2,*) '|chi|21,22,23 max:'
  write(2,*) maxval(abs(chi(2,1,:,:))),maxval(abs(chi(2,2,:,:))),maxval(abs(chi(2,3,:,:)))
  write(2,*) '|delta|21,22,23 max:'
  write(2,*) maxval(abs(delta(2,1,:,:))),maxval(abs(delta(2,2,:,:))),maxval(abs(delta(2,3,:,:)))
  write(2,*) '|lambda|31,32,33 max:'
  write(2,*) maxval(abs(lambda(3,1,:,:))),maxval(abs(lambda(3,2,:,:))),maxval(abs(lambda(3,3,:,:)))
  write(2,*) '|chi|31,32,33 max:'
  write(2,*) maxval(abs(chi(3,1,:,:))),maxval(abs(chi(3,2,:,:))),maxval(abs(chi(3,3,:,:)))
  write(2,*) '|delta|31,32,33 max:'
  write(2,*) maxval(abs(delta(3,1,:,:))),maxval(abs(delta(3,2,:,:))),maxval(abs(delta(3,3,:,:)))
  write(2,*)
  close(2)
# ifdef conserved_psi
  ! impose the condition of psi conserved (<psi> unchanged, to be exact)
  lambda(1,1,1,1)=1
  lambda(1,2,1,1)=0
  lambda(1,3,1,1)=0
# endif
  
end subroutine coefficients_eqs
  

! main routine that solves the mean field eqs (this subroutine kept the same for different set of eqs)
subroutine eqs(time,iter,n_i,istep)
  use global_variables
  implicit none

  real(8) :: time
  integer :: i,j,ly_2,istep,iter,n_i
  real(8) :: max_conv_nA,max_conv_nB,max_conv_nC,conv,abs_psi

  complex(C_DOUBLE_COMPLEX), pointer :: nA0q(:,:),nB0q(:,:),nC0q(:,:),nAq_tmp(:,:)
  complex(C_DOUBLE_COMPLEX), pointer :: NonlinA_q(:,:),NonlinB_q(:,:),NonlinC_q(:,:)
! for allocating aligned memory in fftw (need different pointers for different arrays
! so that the memory can be released at the end of each call (otherwise the used memory will accumulate))
  type(C_PTR) :: p_alloc_nA0q,p_alloc_nAq_tmp,p_alloc_NonlinA_q
  type(C_PTR) :: p_alloc_nB0q,p_alloc_NonlinB_q
  type(C_PTR) :: p_alloc_nC0q,p_alloc_NonlinC_q
  ! to use nonlin(t+dt)-nonlin(t) with nmbr_eval
  real(C_DOUBLE), pointer :: nA0(:,:),nB0(:,:),nC0(:,:)
  complex(C_DOUBLE_COMPLEX), pointer :: sigN_nAq(:,:),sigN_nBq(:,:),sigN_nCq(:,:)
  type(C_PTR) :: p_alloc_nA0,p_alloc_nB0,p_alloc_nC0
  type(C_PTR) :: p_alloc_sigN_nAq,p_alloc_sigN_nBq,p_alloc_sigN_nCq
    
  ly_2=ly/2

  p_alloc_nA0q=fftw_alloc_complex(int((ly_2+1) * lx, C_SIZE_T))
  call c_f_pointer(p_alloc_nA0q, nA0q, [ly_2+1,lx])
  p_alloc_nB0q=fftw_alloc_complex(int((ly_2+1) * lx, C_SIZE_T))
  call c_f_pointer(p_alloc_nB0q, nB0q, [ly_2+1,lx])
  p_alloc_nC0q=fftw_alloc_complex(int((ly_2+1) * lx, C_SIZE_T))
  call c_f_pointer(p_alloc_nC0q, nC0q, [ly_2+1,lx])
  p_alloc_nAq_tmp=fftw_alloc_complex(int((ly_2+1) * lx, C_SIZE_T))
  call c_f_pointer(p_alloc_nAq_tmp, nAq_tmp, [ly_2+1,lx])
  p_alloc_NonlinA_q=fftw_alloc_complex(int((ly_2+1) * lx, C_SIZE_T))
  call c_f_pointer(p_alloc_NonlinA_q, NonlinA_q, [ly_2+1,lx])
  p_alloc_NonlinB_q=fftw_alloc_complex(int((ly_2+1) * lx, C_SIZE_T))
  call c_f_pointer(p_alloc_NonlinB_q, NonlinB_q, [ly_2+1,lx])
  p_alloc_NonlinC_q=fftw_alloc_complex(int((ly_2+1) * lx, C_SIZE_T))
  call c_f_pointer(p_alloc_NonlinC_q, NonlinC_q, [ly_2+1,lx])
  ! to use nonlin(t+dt)-nonlin(t) with nmbr_eval
  p_alloc_nA0=fftw_alloc_real(int(ly * lx, C_SIZE_T))
  call c_f_pointer(p_alloc_nA0, nA0, [ly,lx])
  p_alloc_nB0=fftw_alloc_real(int(ly * lx, C_SIZE_T))
  call c_f_pointer(p_alloc_nB0, nB0, [ly,lx])
  p_alloc_nC0=fftw_alloc_real(int(ly * lx, C_SIZE_T))
  call c_f_pointer(p_alloc_nC0, nC0, [ly,lx])
  p_alloc_sigN_nAq=fftw_alloc_complex(int((ly_2+1) * lx, C_SIZE_T))
  call c_f_pointer(p_alloc_sigN_nAq, sigN_nAq, [ly_2+1,lx])
  p_alloc_sigN_nBq=fftw_alloc_complex(int((ly_2+1) * lx, C_SIZE_T))
  call c_f_pointer(p_alloc_sigN_nBq, sigN_nBq, [ly_2+1,lx])
  p_alloc_sigN_nCq=fftw_alloc_complex(int((ly_2+1) * lx, C_SIZE_T))
  call c_f_pointer(p_alloc_sigN_nCq, sigN_nCq, [ly_2+1,lx])
  
!  calculate the nonlinear terms
  call nonlin_q(NonlinA_q,NonlinB_q,NonlinC_q)

! First step: predictor
! predictor values of nAq, nBq, nCq

!$OMP PARALLEL PRIVATE(i,j)
!$OMP DO
!(Note: DO loop index variable is by default PRIVATE, all other variables are by default SHARED unless declared)
  do i=1,lx
     do j=1,ly_2+1
        nA0q(j,i)=lambda(1,1,j,i)*nAq(j,i)+lambda(1,2,j,i)*nBq(j,i)+lambda(1,3,j,i)*nCq(j,i) &
                 +chi(1,1,j,i)*NonlinA_q(j,i)+chi(1,2,j,i)*NonlinB_q(j,i)+chi(1,3,j,i)*NonlinC_q(j,i)
        nB0q(j,i)=lambda(2,1,j,i)*nAq(j,i)+lambda(2,2,j,i)*nBq(j,i)+lambda(2,3,j,i)*nCq(j,i) &
                 +chi(2,1,j,i)*NonlinA_q(j,i)+chi(2,2,j,i)*NonlinB_q(j,i)+chi(2,3,j,i)*NonlinC_q(j,i)
        nC0q(j,i)=lambda(3,1,j,i)*nAq(j,i)+lambda(3,2,j,i)*nBq(j,i)+lambda(3,3,j,i)*nCq(j,i) &
                 +chi(3,1,j,i)*NonlinA_q(j,i)+chi(3,2,j,i)*NonlinB_q(j,i)+chi(3,3,j,i)*NonlinC_q(j,i)
        if(nmbr_eval.gt.1) then ! for use in iteration
           sigN_nAq(j,i)=delta(1,1,j,i)*NonlinA_q(j,i)+delta(1,2,j,i)*NonlinB_q(j,i)+delta(1,3,j,i)*NonlinC_q(j,i)
           sigN_nBq(j,i)=delta(2,1,j,i)*NonlinA_q(j,i)+delta(2,2,j,i)*NonlinB_q(j,i)+delta(2,3,j,i)*NonlinC_q(j,i)
           sigN_nCq(j,i)=delta(3,1,j,i)*NonlinA_q(j,i)+delta(3,2,j,i)*NonlinB_q(j,i)+delta(3,3,j,i)*NonlinC_q(j,i)
        endif
     enddo
  enddo
!$OMP END DO
!$OMP END PARALLEL

!$OMP PARALLEL DO
  do i=lx/2+2,lx
     nA0q(1,i)=conjg(nA0q(1,lx-i+2)) ! real data treatment
     nB0q(1,i)=conjg(nB0q(1,lx-i+2))
     nC0q(1,i)=conjg(nC0q(1,lx-i+2))
  enddo
!$OMP END PARALLEL DO

  nAq_tmp=nA0q
! backward FFT (note: the input array is overwritten)
  call fftw_execute_dft_c2r(plan2d_b,nAq_tmp,nA)
  nA=nA*scale2d_b
  nAq=nA0q
  
  nAq_tmp=nB0q
! backward FFT (note: the input array is overwritten)
  call fftw_execute_dft_c2r(plan2d_b,nAq_tmp,nB)
  nB=nB*scale2d_b
  nBq=nB0q

  nAq_tmp=nC0q
! backward FFT (note: the input array is overwritten)
  call fftw_execute_dft_c2r(plan2d_b,nAq_tmp,nC)
  nC=nC*scale2d_b
  nCq=nC0q
  
! for use in iteration
  if(nmbr_eval.gt.1) then
     nA0q=nA0q-sigN_nAq
     nB0q=nB0q-sigN_nBq
     nC0q=nC0q-sigN_nCq
     nA0=nA
     nB0=nB
     nC0=nC
  endif

! Second step: corrector
  do istep=2,nmbr_eval ! get the corrector values of nAq,nBq,nCq by functional iteration
                       ! note: nA0q,nB0q,nC0q remain unchanged during the iteration

! for the nonlinear terms (note: here nA,nAq,nB,nBq,nC,nCq are the updated values)
     call nonlin_q(NonlinA_q,NonlinB_q,NonlinC_q)

!$OMP PARALLEL
!$OMP DO
!(Note: DO loop index variable is by default PRIVATE, all other variables are by default SHARED unless declared
!       thus PRIVATE(i,j) is by default and not needed to declare here)
     do i=1,lx
        do j=1,ly_2+1
           nAq(j,i)=nA0q(j,i)+delta(1,1,j,i)*NonlinA_q(j,i)+delta(1,2,j,i)*NonlinB_q(j,i)+delta(1,3,j,i)*NonlinC_q(j,i)
           nBq(j,i)=nB0q(j,i)+delta(2,1,j,i)*NonlinA_q(j,i)+delta(2,2,j,i)*NonlinB_q(j,i)+delta(2,3,j,i)*NonlinC_q(j,i)
           nCq(j,i)=nC0q(j,i)+delta(3,1,j,i)*NonlinA_q(j,i)+delta(3,2,j,i)*NonlinB_q(j,i)+delta(3,3,j,i)*NonlinC_q(j,i)
        enddo
     enddo
!$OMP END DO

!$OMP DO
     do i=lx/2+2,lx
        nAq(1,i)=conjg(nAq(1,lx-i+2)) ! real data treatment
        nBq(1,i)=conjg(nBq(1,lx-i+2))
        nCq(1,i)=conjg(nCq(1,lx-i+2))
     enddo
!$OMP END DO

!$OMP END PARALLEL

     nAq_tmp=nAq
! backward FFT (note: the input array is overwritten)
     call fftw_execute_dft_c2r(plan2d_b,nAq_tmp,nA)
     nA=nA*scale2d_b

     nAq_tmp=nBq
! backward FFT (note: the input array is overwritten)
     call fftw_execute_dft_c2r(plan2d_b,nAq_tmp,nB)
     nB=nB*scale2d_b

     nAq_tmp=nCq
! backward FFT (note: the input array is overwritten)
     call fftw_execute_dft_c2r(plan2d_b,nAq_tmp,nC)
     nC=nC*scale2d_b

! real space (nA,nB,nC) convergence
!     if(istep.gt.2) then

     max_conv_nA=0.d0
     do i=1,lx
        do j=1,ly
           abs_psi=abs(nA(j,i))
           if(abs_psi.gt.1.d5) then
              open(unit=2,file=file//'_out.dat',position='append')
              write(2,*) 'nA diverges at t=',time,' istep=',istep,' j=',j,' i=',i
              write(2,*) 'nA=',nA(j,i),' nB=',nB(j,i),' nC=',nC(j,i), ' for file ',file
              close(2)
              stop
           endif
           if(abs_psi.le.err_s) then
              conv=abs(nA(j,i)-nA0(j,i))
           else
              conv=abs((nA(j,i)-nA0(j,i))/nA(j,i))
           endif
           if(conv.gt.max_conv_nA) max_conv_nA=conv
        enddo
     enddo

     max_conv_nB=0.d0
     do i=1,lx
        do j=1,ly
           abs_psi=abs(nB(j,i))
           if(abs_psi.gt.1.d5) then
              open(unit=2,file=file//'_out.dat',position='append')
              write(2,*) 'nB diverges at t=',time,' istep=',istep,' j=',j,' i=',i
              write(2,*) 'nA=',nA(j,i),' nB=',nB(j,i),' nC=',nC(j,i), ' for file ',file
              close(2)
              stop
           endif
           if(abs_psi.le.err_s) then
              conv=abs(nB(j,i)-nB0(j,i))
           else
              conv=abs((nB(j,i)-nB0(j,i))/nB(j,i))
           endif
           if(conv.gt.max_conv_nB) max_conv_nB=conv
        enddo
     enddo

     max_conv_nC=0.d0
     do i=1,lx
        do j=1,ly
           abs_psi=abs(nC(j,i))
           if(abs_psi.gt.1.d5) then
              open(unit=2,file=file//'_out.dat',position='append')
              write(2,*) 'nC diverges at t=',time,' istep=',istep,' j=',j,' i=',i
              write(2,*) 'nA=',nA(j,i),' nB=',nB(j,i),' nC=',nC(j,i), ' for file ',file
              close(2)
              stop
           endif
           if(abs_psi.le.err_s) then
              conv=abs(nC(j,i)-nC0(j,i))
           else
              conv=abs((nC(j,i)-nC0(j,i))/nC(j,i))
           endif
           if(conv.gt.max_conv_nC) max_conv_nC=conv
        enddo
     enddo

     if((max_conv_nA.lt.TOL).and.(max_conv_nB.lt.TOL).and.(max_conv_nC.lt.TOL)) then
        call fftw_free(p_alloc_nA0)
        call fftw_free(p_alloc_nA0q)
        call fftw_free(p_alloc_nAq_tmp)
        call fftw_free(p_alloc_NonlinA_q)
        call fftw_free(p_alloc_sigN_nAq)
        call fftw_free(p_alloc_nB0)
        call fftw_free(p_alloc_nB0q)
        call fftw_free(p_alloc_NonlinB_q)
        call fftw_free(p_alloc_sigN_nBq)
        call fftw_free(p_alloc_nC0)
        call fftw_free(p_alloc_nC0q)
        call fftw_free(p_alloc_NonlinC_q)
        call fftw_free(p_alloc_sigN_nCq)
        return
     endif

!     endif ! if istep>2

     nA0=nA
     nB0=nB
     nC0=nC

  enddo ! for istep=2,nmbr_eval

  istep=istep-1
  if(nmbr_eval.gt.2) then
     open(unit=2,file=file//'_out.dat',position='append')
     write(2,*)
     write(2,*) 'Exceeding maximum iterations=',nmbr_eval
     write(2,*) 't=',time,'   TOL=',TOL
     write(2,*) 'max_conv_nA=',max_conv_nA
     write(2,*) maxval(nA(1:ly,1:lx)),minval(nA(1:ly,1:lx))
     write(2,*) 'max_conv_nB=',max_conv_nB
     write(2,*) maxval(nB(1:ly,1:lx)),minval(nB(1:ly,1:lx))
     write(2,*) 'max_conv_nC=',max_conv_nC
     write(2,*) maxval(nC(1:ly,1:lx)),minval(nC(1:ly,1:lx))
     write(2,*)
     close(2)
  endif

  call fftw_free(p_alloc_nA0)
  call fftw_free(p_alloc_nB0)
  call fftw_free(p_alloc_nC0)
  call fftw_free(p_alloc_sigN_nAq)
  call fftw_free(p_alloc_sigN_nBq)
  call fftw_free(p_alloc_sigN_nCq)

  call fftw_free(p_alloc_nA0q)
  call fftw_free(p_alloc_nAq_tmp)
  call fftw_free(p_alloc_NonlinA_q)
  call fftw_free(p_alloc_nB0q)
  call fftw_free(p_alloc_NonlinB_q)
  call fftw_free(p_alloc_nC0q)
  call fftw_free(p_alloc_NonlinC_q)
  return
end subroutine eqs


!  calculate the nonlinear terms
subroutine nonlin_q(Nonlin_nAq,Nonlin_nBq,Nonlin_nCq)
  use global_variables
  implicit none

  integer :: i,j
  complex(C_DOUBLE_COMPLEX) :: Nonlin_nAq(ly/2+1,lx),Nonlin_nBq(ly/2+1,lx),Nonlin_nCq(ly/2+1,lx)
  real(C_DOUBLE), pointer :: Nonlin_nA(:,:),Nonlin_nB(:,:),Nonlin_nC(:,:)
  type(C_PTR) :: p_alloc_Nonlin_nA,p_alloc_Nonlin_nB,p_alloc_Nonlin_nC ! for allocating aligned memory in fftw

# ifdef cavity_boundary_V
  complex(C_DOUBLE_COMPLEX), pointer :: dxpsiq(:,:),dypsiq(:,:)
  real(C_DOUBLE), pointer :: dxpsi(:,:),dypsi(:,:)
  type(C_PTR) :: p_alloc_dxpsiq,p_alloc_dypsiq,p_alloc_dxpsi,p_alloc_dypsi
  p_alloc_dxpsiq=fftw_alloc_complex(int((ly/2+1) * lx, C_SIZE_T))
  call c_f_pointer(p_alloc_dxpsiq, dxpsiq, [ly/2+1,lx])
  p_alloc_dypsiq=fftw_alloc_complex(int((ly/2+1) * lx, C_SIZE_T))
  call c_f_pointer(p_alloc_dypsiq, dypsiq, [ly/2+1,lx])
  p_alloc_dxpsi=fftw_alloc_real(int(ly * lx, C_SIZE_T))
  call c_f_pointer(p_alloc_dxpsi, dxpsi, [ly,lx])
  p_alloc_dypsi=fftw_alloc_real(int(ly * lx, C_SIZE_T))
  call c_f_pointer(p_alloc_dypsi, dypsi, [ly,lx])
# endif
  
  p_alloc_Nonlin_nA=fftw_alloc_real(int(ly * lx, C_SIZE_T))
  call c_f_pointer(p_alloc_Nonlin_nA, Nonlin_nA, [ly,lx])
  p_alloc_Nonlin_nB=fftw_alloc_real(int(ly * lx, C_SIZE_T))
  call c_f_pointer(p_alloc_Nonlin_nB, Nonlin_nB, [ly,lx])
  p_alloc_Nonlin_nC=fftw_alloc_real(int(ly * lx, C_SIZE_T))
  call c_f_pointer(p_alloc_Nonlin_nC, Nonlin_nC, [ly,lx])

! nA=psi, nB=Px, nC=Py
  Nonlin_nA=(u*nA-g)*nA*nA
  Nonlin_nB=C4*(nB*nB+nC*nC)
  Nonlin_nC=Nonlin_nB*nC
  Nonlin_nB=Nonlin_nB*nB
  
# ifdef cavity_boundary_V
  ! use an external potential for boundary conditions

# ifdef cavity_boundary_psi_flux0
  ! both psi=psi_b and zero-flux psi boundary condition; otherwise only psi=psi_b boundary
  
! The nonlinear boundary terms for psi
!$OMP PARALLEL DO
  do i=1,lx
     do j=1,ly/2+1
        dxpsiq(j,i)=CMPLX(0.d0,qx(i))*nAq(j,i)
        dypsiq(j,i)=CMPLX(0.d0,qy(j))*nAq(j,i)
     enddo
  enddo
!$OMP END PARALLEL DO
! backward FFT (note: the input array is overwritten)
  call fftw_execute_dft_c2r(plan2d_b,dxpsiq,dxpsi) ! = \partial_x \psi
  dxpsi=dxpsi*scale2d_b
  call fftw_execute_dft_c2r(plan2d_b,dypsiq,dypsi) ! = \partial_y \psi
  dypsi=dypsi*scale2d_b
  dxpsi=(n_x*dxpsi+n_y*dypsi)*Vb ! zero-flux psi boundary condition
  dypsi=n_y*dxpsi
  dxpsi=n_x*dxpsi
  call fftw_execute_dft_r2c(plan2d_f,dxpsi,dxpsiq) ! forward FFT
  call fftw_execute_dft_r2c(plan2d_f,dypsi,dypsiq) ! forward FFT
!$OMP PARALLEL DO
  do i=1,lx
     do j=1,ly/2+1
        dxpsiq(j,i)=CMPLX(0.d0,qx(i))*dxpsiq(j,i)
        dypsiq(j,i)=CMPLX(0.d0,qy(j))*dypsiq(j,i)
     enddo
  enddo
!$OMP END PARALLEL DO
! backward FFT (note: the input array is overwritten)
  call fftw_execute_dft_c2r(plan2d_b,dxpsiq,dxpsi)
  dxpsi=dxpsi*scale2d_b
  call fftw_execute_dft_c2r(plan2d_b,dypsiq,dypsi)
  dypsi=dypsi*scale2d_b
  Nonlin_nA=Nonlin_nA+Vb*(nA-psi_b)-dxpsi-dypsi ! both zero-flux and psi=psi_b boundary conditions
  !Nonlin_nA=Nonlin_nA-dxpsi-dypsi ! only zero-flux boundary condition
# else
  Nonlin_nA=Nonlin_nA+Vb*(nA-psi_b) ! only psi=psi_b boundary condition
# endif
  
# ifdef cavity_boundary_VP0
  ! additional P=P_b=0 boundary condition
  Nonlin_nB=Nonlin_nB+Vb*(nB-Px_b)
  Nonlin_nC=Nonlin_nC+Vb*(nC-Py_b)
# endif
  
# endif
  ! ifdef cavity_boundary_V
  
  call fftw_execute_dft_r2c(plan2d_f,Nonlin_nA,Nonlin_nAq) ! forward FFT
  call fftw_execute_dft_r2c(plan2d_f,Nonlin_nB,Nonlin_nBq) ! forward FFT
  call fftw_execute_dft_r2c(plan2d_f,Nonlin_nC,Nonlin_nCq) ! forward FFT

  ! conserved dynamics
  Nonlin_nAq=-q2*Nonlin_nAq
  Nonlin_nBq=-(q2+Dr)*Nonlin_nBq
  Nonlin_nCq=-(q2+Dr)*Nonlin_nCq

  call fftw_free(p_alloc_Nonlin_nA)
  call fftw_free(p_alloc_Nonlin_nB)
  call fftw_free(p_alloc_Nonlin_nC)
# ifdef cavity_boundary_V
  call fftw_free(p_alloc_dxpsiq)
  call fftw_free(p_alloc_dypsiq)
  call fftw_free(p_alloc_dxpsi)
  call fftw_free(p_alloc_dypsi)
# endif

  return
end subroutine nonlin_q


! output 2D HDF 8-Bit Raster Images
      SUBROUTINE hdf_r8(l,m,r1a,image,name)
      implicit none
      integer l,m
      real(8), DIMENSION(l,m) :: r1a
      character(1) image(l,m)
      integer d8aimg,iret
      character name*(*)
      real(8) r1amin,r1amax,rdiff
      character(1) image_(m,l)
      integer i,j
!
      r1amin=minval(r1a)
      r1amax=maxval(r1a)
      rdiff=r1amax-r1amin
!
      if(abs(rdiff).gt.1.d-20) then
         image=char(nint(255*(r1a-r1amin)/rdiff))
      else
         image=char(0)
      endif
!
      do j=1,l
         do i=1,m
!            image_(i,j)=image(j,i) ! reverse x vs y
            image_(i,l-j+1)=image(j,i) ! reverse x vs y plus flip over y (for showing in hdfview)
         enddo
      enddo
      iret=d8aimg(name,image_,m,l,0)

      if(iret.ne.0) then
        write(*,*) ' Error writing the HDF file ',name
      endif
      return
    end SUBROUTINE hdf_r8

    
! calculate free energy and chemical potential (here f(x,y) is the free energy density distribution)
subroutine F_mu(f,F_avg)
  use global_variables
  implicit none

  real(8) :: f(ly,lx),F_avg
  real(C_DOUBLE), pointer :: nA2(:,:)
  complex(C_DOUBLE_COMPLEX), pointer :: d2nq(:,:)
  type(C_PTR) :: p_alloc_nA2,p_alloc_d2nq ! for allocating aligned memory in fftw
# ifdef cavity_boundary_V
  real(C_DOUBLE), pointer :: dypsi(:,:)
  complex(C_DOUBLE_COMPLEX), pointer :: dypsiq(:,:)
  type(C_PTR) :: p_alloc_dypsi,p_alloc_dypsiq
  integer :: i,j
  p_alloc_dypsi=fftw_alloc_real(int(ly * lx, C_SIZE_T))
  call c_f_pointer(p_alloc_dypsi, dypsi, [ly,lx])
  p_alloc_dypsiq=fftw_alloc_complex(int((ly/2+1) * lx, C_SIZE_T))
  call c_f_pointer(p_alloc_dypsiq, dypsiq, [ly/2+1,lx])
# endif
  
  p_alloc_nA2=fftw_alloc_real(int(ly * lx, C_SIZE_T))
  call c_f_pointer(p_alloc_nA2, nA2, [ly,lx])
  p_alloc_d2nq=fftw_alloc_complex(int((ly/2+1) * lx, C_SIZE_T))
  call c_f_pointer(p_alloc_d2nq, d2nq, [ly/2+1,lx])

  F_avg=0
  
  nA2=nA*nA
  f=(eps/2-nA*g/3+nA2*u/4)*nA2
  
  d2nq=(q02-q2)**2*nAq
! backward FFT (note: the input array is overwritten)
  call fftw_execute_dft_c2r(plan2d_b,d2nq,nA2)
  nA2=nA2*scale2d_b
  f=f+nA2*nA/2 ! =f_PFC

  nA2=nB*nB+nC*nC ! =Px^2+Py^2
  f=f+(C1+C4*nA2/2)*nA2/2 ! =f_PFC+f_P
  
# ifdef cavity_boundary_V
  ! use an external potential for boundary conditions
# ifdef cavity_boundary_psi_flux0
  ! both psi=psi_b and zero-flux psi boundary condition; otherwise only psi=psi_b boundary
!$OMP PARALLEL DO
  do i=1,lx
     do j=1,ly/2+1
        d2nq(j,i)=CMPLX(0.d0,qx(i))*nAq(j,i)
        dypsiq(j,i)=CMPLX(0.d0,qy(j))*nAq(j,i)
     enddo
  enddo
!$OMP END PARALLEL DO
! backward FFT (note: the input array is overwritten)
  call fftw_execute_dft_c2r(plan2d_b,d2nq,nA2) ! = \partial_x \psi
  nA2=nA2*scale2d_b
  call fftw_execute_dft_c2r(plan2d_b,dypsiq,dypsi) ! = \partial_y \psi
  dypsi=dypsi*scale2d_b  
  nA2=n_x*nA2+n_y*dypsi ! zero-flux psi boundary condition
  f=f+Vb*((nA-psi_b)**2+nA2*nA2)/2 ! both psi=psi_b and zero-flux boundary conditions
  !f=f+Vb*nA2*nA2/2 ! only zero-flux boundary condition
# else
  f=f+Vb*(nA-psi_b)**2/2 ! only psi=psi_b boundary condition
# endif
# ifdef cavity_boundary_VP0
  ! additional P=P_b=0 boundary condition
  f=f+Vb*((nB-Px_b)**2+(nC-Py_b)**2)/2
# endif
# endif
  ! ifdef cavity_boundary_V
  
! note: f(x,y) here is free energy density distribution; total F = \int f(x,y) dxdy
  F_avg=sum(f(1:ly,1:lx))*scale2d_b ! average (density) = total free energy / volume

  call fftw_free(p_alloc_nA2)
  call fftw_free(p_alloc_d2nq)
# ifdef cavity_boundary_V
  call fftw_free(p_alloc_dypsi)
  call fftw_free(p_alloc_dypsiq)
# endif
  
  return
end subroutine F_mu

# ifdef cavity_boundary_V
  ! set up cavity boundary condition using an external potential
subroutine Vb_n(Vb0,Delta_b) ! calculate Vb and n_x,n_y
  use global_variables
  implicit none

  real(8) :: Vb0,Delta_b,rs,x,y
  integer :: i,j
  
# ifdef cavity_circular
  ! set up circular/spherical cavity boundary
  real(8) :: rr
# endif
# ifdef hypocycloid_cavity
  ! set up hypocycloid cavity with n cusps
  integer,parameter :: kmax=100000
  real(8) :: xb(0:kmax),yb(0:kmax),tt,dtt,d2,d2_min,xb_min,yb_min,cos_theta
  integer :: k
  !pi=dacos(-1.0d0)
  dtt=2*pi/kmax
  do k=0,kmax-1
     tt=k*dtt
     xb(k)=b0*((n-1)*cos(tt)+cos((n-1)*tt))
     yb(k)=b0*((n-1)*sin(tt)-sin((n-1)*tt))
  enddo
# endif
# ifdef hypocycloid_ring
  ! set up hypocycloid ring with n0+n1 cusps
  integer,parameter :: kmax=100000
  real(8) :: xb0(0:kmax),yb0(0:kmax),xb1(0:kmax),yb1(0:kmax)
  real(8) :: tt,dtt,d2,d2_min,xb_min,yb_min,cos_theta
  integer :: k,i_min
  !pi=dacos(-1.0d0)
  dtt=2*pi/kmax
  do k=0,kmax-1
     tt=k*dtt
     xb0(k)=b0*((n0-1)*cos(tt)+cos((n0-1)*tt))
     yb0(k)=b0*((n0-1)*sin(tt)-sin((n0-1)*tt))
     xb1(k)=b1*((n1-1)*cos(tt)+cos((n1-1)*tt))
     yb1(k)=b1*((n1-1)*sin(tt)-sin((n1-1)*tt))
  enddo
# endif
 
  do i=1,lx
# ifdef xy_cshift05
     x=i-lx/2-0.5d0
# else
     x=i-lx/2
# endif
     do j=1,ly
# ifdef xy_cshift05
        y=j-ly/2-0.5d0
# else
        y=j-ly/2
# endif
       
# ifdef cavity_circular
  ! set up circular/spherical cavity boundary
        rr=sqrt(x*x+y*y)
        rs=rr-r0
        n_x(j,i)=x/rr
        n_y(j,i)=y/rr
# endif
      
# ifdef hypocycloid_cavity
  ! set up hypocycloid cavity with n cusps
        d2_min=1.d10
        do k=0,kmax-1
           d2=(x-xb(k))**2+(y-yb(k))**2
           if(d2<d2_min) then
              d2_min=d2
              xb_min=xb(k)
              yb_min=yb(k)
           endif
        enddo
        d2_min=sqrt(d2_min)
        cos_theta=xb_min*(xb_min-x)+yb_min*(yb_min-y) !/(d2_min*sqrt(xb_min*xb_min+yb_min*yb_min))
        if(cos_theta>=0) then ! inside
           rs=-d2_min
        else ! outside
           rs=d2_min
        endif
        n_x(j,i)=(x-xb_min)/d2_min
        n_y(j,i)=(y-yb_min)/d2_min
# endif
                
# ifdef hypocycloid_ring
  ! set up hypocycloid ring with n0+n1 cusps
        d2_min=1.d10
        do k=0,kmax-1
           d2=(x-xb0(k))**2+(y-yb0(k))**2
           if(d2<d2_min) then
              d2_min=d2
              xb_min=xb0(k)
              yb_min=yb0(k)
           endif
        enddo
        i_min=0 ! outer hypocycloid
        do k=0,kmax-1
           d2=(x-xb1(k))**2+(y-yb1(k))**2
           if(d2<d2_min) then
              d2_min=d2
              xb_min=xb1(k)
              yb_min=yb1(k)
              i_min=1 ! inner hypocycloid
           endif
        enddo
        d2_min=sqrt(d2_min)
        cos_theta=xb_min*(xb_min-x)+yb_min*(yb_min-y)
        if(cos_theta>=0) then
           if(i_min==0) then
              rs=-d2_min ! inside the ring
           else
              rs=d2_min ! outside (ring inner center)
           endif
        else
           if(i_min==0) then
              rs=d2_min ! outside of ring
           else
              rs=-d2_min ! inside the ring
           endif
        endif
        n_x(j,i)=(x-xb_min)/d2_min
        n_y(j,i)=(y-yb_min)/d2_min
# endif

        Vb(j,i)=Vb0/(1+exp(-2*rs/Delta_b))
        !Vb(j,i)=Vb0*(1+tanh(rs/Delta_b))/2
     enddo
  enddo
  
  return
end subroutine Vb_n
# endif
