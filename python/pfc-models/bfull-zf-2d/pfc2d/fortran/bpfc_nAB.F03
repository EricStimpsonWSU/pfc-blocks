! PFC (Phase Field Crystal) model, for binary alloy systems of nA, nB
! 2D hexagonal/triangular lattice structure for each of A, B
! overall honeycomb structure for binary A-B

! note: here phi = nA, psi = nB

! using OpenMP (To use single-core only: comment out "define open_MP")
! using pgi (pgfortran), FFTW3
! double precision, images output

! pseudo-spetral method,
! using a predictor-corrector method for both phi (nA) and psi (nB)
!   exact solution for linear terms and approximation for nonlinearities
!       (with an exponential propagation procedure, similar to Cross et al., Chaos 1994;
!        donot fix the number of iterations;
!        expand around small sigma when it approaches 0)
! with the transient time

! note: real data treatment for j=1 (qy=0) in fftw, otherwise divergence could occur at late time
!       or call forward FFT for psi --> psiq at the end of each time step (although psiq is known)
!       for convergence at large t, due to bugs/inaccuracies from fftw, or aliasing problem

!# define ini_homogeneous ! for homogeneous initial condition
!# define ini_nucleus ! for initial condition of crystal nuclei
!# define ini_theta ! for initial condition of randomly oriented nuclei; def ini_nucleus
# define ini_crystal ! for initial condition of perfect crystal
# define ini_crystal_sl ! for initial condition of perfect crystal with solid-liquid interface
!# define interstitial ! for modeling interstitials or impurities (not working well; need vacancy model)
!# define surface_ordering ! for surface ordering of heteroepitaxial overlayer

!# define ini_epitaxy_lateral ! for lateral epitaxy of strained AB layers/film
!# define ini_epitaxy_xy ! for epitaxy of AB layers/island in x-y plane
!# define flux_const ! constant flux: each step set constant supercool liquid densities (n0A,n0B)
                    ! by hand above certain distance from top film surface; ifdef ini_epitaxy_lateral
!# define two_side_growth ! growth on both sides of the AB layers

# define dxdy_eq ! dx,dy,qx0_eq,qy0_eq obtained from free energy minimization
                 ! (not needed if coexisiting with liquid state)
# define ini_dxdy_file ! read dx,dy from file (if not using dxdy_eq)
# define ini_theta_file ! read theta from file
# define ini_phi_file ! read phi from file
!# define noise_dynamics ! dynamics with conserved Gaussian noise (for large eps low T, or metastable regime)

# define save_conf ! save conf files (*)
# define image_reverse ! x,y reverse for image output (default)

!# define diff_fft ! fft based on finite differences 
                  ! (smaller dx, better agreement with exact results)
! otherwise: fft based on continuum limit

# define open_MP ! use openMP for multi-threading

module global_variables
! by default all variables in the module are public (if not being set explicitly as private)
  use, intrinsic :: iso_c_binding
  implicit none
  include '/wsu/el7/pgi-openmpi/2018-187/fftw/3.3.8/include/fftw3.f03'
  
  save ! save all the variables in the module

! integer, parameter :: ly=18, lx=10
! integer, parameter :: ly=10, lx=18
! integer, parameter :: ly=180, lx=180
integer, parameter :: ly=1440, lx=1440
! integer, parameter :: ly=256, lx=256
! integer, parameter :: ly=512, lx=506
! integer, parameter :: ly=1024, lx=1024
! integer, parameter :: ly=2048, lx=2048
! integer, parameter :: ly=1024, lx=1024

  real(C_DOUBLE), pointer :: phi(:,:),psi(:,:)
  complex(C_DOUBLE_COMPLEX), pointer :: phiq(:,:),psiq(:,:)
  
  real(8) :: qx(lx),qy(ly/2+1)
  real(8) :: epsA,epsB,alpha_AB,beta_AB,beta_B,q02,q1_2,q2_2
  real(8) :: mB,gA,gB,w,u,v
  double precision, allocatable :: q2(:,:),alpha_12(:,:),alpha_21(:,:)
  double complex, allocatable :: exp_11(:,:),exp_12(:,:),exp_21(:,:),exp_22(:,:)
  double complex, allocatable :: cf_1(:,:),cf2_1(:,:),cf_2(:,:),cf2_2(:,:)
  double complex, allocatable :: sigma12(:,:),sig1_alpha(:,:),sig2_alpha(:,:)
  integer, allocatable :: isigma(:,:)
  real(8) :: TOL,err_s ! for iteration of predictor-corrector
  integer :: nmbr_eval

! for FFTW parameters
  type(C_PTR) :: plan2d_f,plan2d_b
  real(8) :: scale2d_b
  type(C_PTR) :: plan1dx_f,plan1dy_f

end module global_variables


Program bpfc_nAB_hon
  use global_variables
  implicit none

  type(C_PTR) :: p_alloc_psi,p_alloc_psiq ! for allocating aligned memory in fftw
  double precision, allocatable :: f(:,:)
  double complex, allocatable :: sigma1(:,:),sigma2(:,:)
  real(8) :: F_avg,muA_avg,muB_avg
  real(8) :: alpha_11,alpha_22
  complex(8) :: sig1_dt,sig2_dt,delta_2,aq,b,c
  real(8) :: dt,dti,dtime
  real(8) :: dx,dy,pi,n0A,n0B
  real(8) :: time,tmax,time0,t_i
  real(8) :: qx0,qy0,qx0_eq,qy0_eq
  real(8) :: q0,q1,qq2
  real(8) :: phi_A1, phi_A2, phi_A3, phi_B1, phi_B2, phi_B3
  real(8) :: noise,noise0,ran3,y
  character(1) :: image(ly,lx)
  integer :: i,j,iter,idum,ntype,num_file,istep
  integer :: nend,nout,nimpc,raout,nout_conf,n_i,n_dx
  integer :: iter0,n_0,i_imag,nt

# ifdef open_MP
  ! use multi-threaded fftw with openMP
  integer iret,nthreads,omp_get_max_threads
# endif
  
# ifdef ini_crystal
  ! for initial condition of perfect crystal
  real(8) :: x,q_1,q_2,q_3,A0a,a1,a2
  real(8) :: theta=0 ! 0 zigzag y-axis, 90 armchair y-axis
# ifdef ini_crystal_sl
  ! for initial condition of perfect crystal with solid-liquid interface
  integer :: j1,j2,j_width,ly_mid
  real(8) :: n0A_s,n0B_s
  real(8) :: n0A_l,n0B_l
  real(8) :: n0A_mean, n0B_mean
# endif
# endif
# ifdef ini_dxdy_file
    real(8) :: du
    integer :: mx, my
# endif

  integer :: n_time, itime(15)
  character(1) :: cha_t
  character(10) :: cha

! for FFTW parameters
  real(8) :: psiqx(lx),psiqy(ly), psix(lx),psiy(ly)

  parameter(num_file=3)
  character(num_file) file
  character(60) init_cond
  init_cond='data/b2048eps03g05w03aAB05bAB002q1_n035s03_m_005f80_conf.dat'

  pi=DACOS(-1.0d0)
  ntype=1  ! ntype=1: set initial configuration; ntype=2: input from file
  time0=0
  iter0=1
!  ntype=2 ! initial condition input from previous runs
!  time0=20000.0d0
!  iter0=100095+1  ! iter0-1 = nend of previous run

!------------------------------------------------------------------------

file='tmp'

  
! parameters entering the Eqs
!
! for rescaled PFC equations;  eps=1-B^l/B^x
!  epsA=0.02d0 ! for nA
  epsA=0.3d0
  epsB=epsA ! for nB

  n0A=-0.2680999637d0

!  n0A=0
!  n0A=-0.1d0
!  n0A=-0.2d0
!  n0A=-0.3d0
!  n0A=-0.4d0
!  n0A=-0.49d0
!  n0A=-0.5d0
!  n0A=-0.55d0
!  n0A=-0.6d0
!  n0A=-0.8d0

!  n0B=0
!  n0B=-0.1d0
!  n0B=-0.2d0
!  n0B=-0.3d0
!  n0B=-0.4d0
!  n0B=-0.49d0
!  n0B=-0.5d0
!  n0B=-0.55d0
!  n0B=-0.6d0
!  n0B=-0.8d0
!  n0B=0.3d0
  n0B=n0A

  # ifdef ini_crystal_sl
    n0A_s = n0A
    n0B_s = n0B

    n0A_l = -0.44330000386d0
    n0B_l = n0A_l
# endif


  q0=4*pi/sqrt(3.0d0)/7.4191 ! =qA for nA
  q02=q0*q0
! qx0=(sqrt(3)/2)*q0, qy0=q0 for unstrained, bulk state of substrate
  q1=q0 ! =qB for nB
  q1_2=q1*q1

!  qq2=0 ! =qAB for AB
!  qq2=sqrt(3.d0)
  qq2=q0
  q2_2=qq2*qq2

!  beta_AB=-0.02d0
!  beta_AB=-0.05d0 ! similar to beta_AB=-0.02
!  beta_AB=-0.2d0 ! different morphology; A,B appear at the same place; +0.2: diverge
!  beta_AB=0.01d0
!  beta_AB=0.02d0 !
!  beta_AB=0.05d0 ! stripes + squares (change of lattice symmetry)?
beta_AB=0 ! also show binary honeycomb
!  beta_AB=1 or -1 ! diverge
!  beta_AB=-0.1d0
!  beta_AB=0.1d0

! mB=M_B/M_A
  mB=1
!  mB=0.001d0

!  gA=0.3d0 ! note: gA,gB=g of A0n0_ampl1D.f90, but =-g2 for A0n0psi0
!  gA=1
!  gA=1.5d0
!  gA=0
  gA=0.5d0
  gB=gA

  alpha_AB=0.5d0
!  alpha_AB=1 ! diverge
!  alpha_AB=0.05d0 ! A,B appear at the same place
  beta_B=1
  v=1
  w=0.3d0
!  w=0.5d0
!  w=1
!  w=0
  u=w
  
  qx0=0.5d0*sqrt(3.d0)*q0
  qy0=q0
  qx0_eq=qx0
  qy0_eq=qy0
! dx and dy fixed according to the unstrained substrate (q0), same as 1D ampl.
  n_dx=8 ! n_dx: number of grid points per lattice period
  dx=2*pi/(n_dx*qx0)
  dy=dx
  !dy=2*pi/(n_dx*qy0)
  !dx=dy

#ifdef ini_theta_file
    ! for reading theta from a file
    open(10,file='theta.in', status='old', action='read')
    read(10,*) theta
    close(10)
#endif

# ifdef ini_dxdy_file
    ! for reading dx,dy from a file
    open(10,file='dxdy.in', status='old', action='read')
    read(10,*) du, n_dx, mx, my
    close(10)

    write(*,*) 'du, n_dx, mx, my'
    write(*,*) du, n_dx, mx, my

    if(theta.eq.0.0d0) then ! armchair
        dy=du
        dx=sqrt(3.d0) * (real(mx, kind=8)/real(my, kind=8)) * du * (real(ly, kind=8)/real(lx, kind=8))
        ! qx0=2*pi/((n_dx/2)*dx)
        ! qy0=2*pi/(n_dx*dy)
        qx0=4*pi/(lx/mx*dx)
        qy0=2*pi/(ly/my*dy)
    elseif(theta.eq.90.0d0) then ! zigzag
        dx=du
        dy=sqrt(3.d0) * (real(my, kind=8)/real(mx, kind=8)) * du * (real(lx, kind=8)/real(ly, kind=8))
        qx0=2*pi/(lx/mx*dx)
        qy0=4*pi/(ly/my*dy)
    else ! undefined
        write(*,*) 'theta is not defined'
        stop
    endif
    
    write(*,*) 'dx, dy, qx0, qy0'
    write(*,*) dx, dy, qx0, qy0
# endif

  dt=0.2d0 ! for large eps or large g (sharp interface)

  idum=-796921 ! random number seed; <0 for ran3()
!  idum=-31131

!   noise=0.0d0
  noise=0.01d0 ! initial random noise (percentage)
!  noise=2
  noise0=0.01d0 ! for <>~0
!  noise0=0

  t_i=1 ! better for t_i/dt=integer
  dti=0.01d0 ! used for initial transient up to t=t_i

!   tmax=30
!   nt=5
!  tmax=200
!  nt=10
!  tmax=2000
!  nt=100
!  tmax=10000
!  nt=500
!  tmax=20000
 tmax=30000
 nt=1000
!   tmax=100000
!   nt=2000
!   nt=5000
!   tmax=1000000
!   nt=100000
  ! 
  nimpc=NINT((tmax-time0)/nt)
  !
  ! tmax : maximum time
  ! t_i: time of initial transient
  ! nimpc : # of images 

  n_i=NINT(t_i/dti)
  n_0=NINT(t_i/dt)-n_i ! for output
  nend=NINT((tmax-t_i)/dt)+n_i
  nout=(NINT(tmax/dt))/(nimpc+NINT(time0/nt))
!  nout=NINT((tmax-time0)/dt)/nimpc
!  nout=(nend-iter0+1)/nimpc

  raout=nout/10
  nout_conf=nout*5

! nmbr_eval: number of iterations used in the predictor-corrector algorithm
!  nmbr_eval=1
  nmbr_eval=100
  TOL=1.d-3
!  err_s=TOL/2
  err_s=TOL/10

! for output of structure factors
  n_time=12
  itime(1)=NINT((tmax/100-t_i)/dt)+n_i
  itime(2)=NINT((tmax/50-t_i)/dt)+n_i
  itime(3)=NINT((tmax/20-t_i)/dt)+n_i
  itime(4)=NINT((tmax/10-t_i)/dt)+n_i
  itime(5)=NINT((tmax/5-t_i)/dt)+n_i
  itime(6)=NINT((3*tmax/10-t_i)/dt)+n_i
  itime(7)=NINT((2*tmax/5-t_i)/dt)+n_i
  itime(8)=NINT((tmax/2-t_i)/dt)+n_i
  itime(9)=NINT((3*tmax/5-t_i)/dt)+n_i
  itime(10)=NINT((7*tmax/10-t_i)/dt)+n_i
  itime(11)=NINT((4*tmax/5-t_i)/dt)+n_i
  itime(12)=NINT((9*tmax/10-t_i)/dt)+n_i
!
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

#ifdef ini_crystal
  phi_A1=0
  phi_A2=0
  phi_A3=0
  phi_B1=2*pi/3
  phi_B2=2*pi/3
  phi_B3=2*pi/3
# ifdef ini_phi_file
  ! read phase offsets from file for crystal initialization
  open(10,file='phi.in', status='old', action='read')
  read(10,*) phi_A1, phi_A2, phi_A3
  read(10,*) phi_B1, phi_B2, phi_B3
  close(10)
# endif
# endif

! write parameters in a separate file
!
  open(unit=2,file=file//'_para.dat')
# ifdef ini_crystal
  write(2,*) 'def ini_crystal: for initial condition of perfect crystal'
  write(2,*) 'theta=',theta
  write(2,*) 'phi_A:',phi_A1,', ',phi_A2,', ',phi_A3
  write(2,*) 'phi_B:',phi_B1,', ',phi_B2,', ',phi_B3
# endif
  write(2,*)
  write(2,*) 'using 2nd order preditor-corrector method for both nA and nB PFC equations'
  write(2,*) 'use an exponential propagation procedure'
  write(2,*) '    exact solution for linear terms and approximation for nonlinearities'
  write(2,*) '    expanding nonlinearities up to 1st order of t'
  write(2,*) 'FFT based on continuum limit'
  write(2,*) 'lx=',lx, '  ly=',ly
  write(2,*) 'epsA=',epsA, '  epsB=',epsB
  write(2,*) 'q0=',q0, '  q1=',q1
  write(2,*)
  write(2,*) 'n0A=',n0A, '  n0B=',n0B
# ifdef ini_crystal_sl
    write(2,*) '   n0A_s=',n0A_s, '  n0B_s=',n0B_s
    write(2,*) '   n0A_l=',n0A_l, '  n0B_l=',n0B_l
# endif
  write(2,*) 'mB=M_B/M_A=',mB
  write(2,*)
  write(2,*) 'alpha_AB=',alpha_AB
  write(2,*) 'gA=',gA, '  gB=',gB
  write(2,*) 'beta_B=',beta_B, '  v=',v
  write(2,*) 'w=',w, '  u=',u
  write(2,*) 'beta_AB=',beta_AB, '  q2=',qq2
  write(2,*)
  write(2,*) 'dt=',dt, '  dti=',dti
  write(2,*) 'dx=',dx, '  dy=',dy
  write(2,*) 'n_dx=', n_dx, '  number of lattice period along y=',ly/n_dx
  write(2,*) 'qx0=',qx0, '  qy0=',qy0
  write(2,*) 'qx0_eq=',qx0_eq, '  qy0_eq=',qy0_eq
# ifdef dxdy_eq
  write(2,*) 'def dxdy_eq: dx,dy,qx0_eq,qy0_eq obtained from free energy minimization'
# endif
  write(2,*)
  write(2,*) 'tmax=',tmax, '  t_transient=',t_i
  write(2,*) 'nend=',nend, '  n_transient=',n_i
  write(2,*) 'nimpc=',nimpc, '   nout=',nout, '  raout=',raout
  write(2,*) '  data output at t=t_transient, and every ',raout*dt
  write(2,*) '  images output at t=0, t_transient, and every ',nout*dt
# ifdef save_conf
  write(2,*) '  save conf files every ',nout_conf*dt
# endif
  write(2,*) 'output of structure factors and nA,nB profiles:'
  write(2,*) '  at t=',(itime(1:n_time)-n_i)*dt+t_i,tmax
  write(2,*) 'time0=',time0, '  iter0=',iter0, '  idum=',idum
!  write(2,*) 'noise (percentage)=',noise,' noise0=',noise0
  write(2,*) 'noise (not percentage)=',noise,' noise0=',noise0
  write(2,*) 'ntype=',ntype
  if(ntype.eq.2) write(2,*) 'init_cond = ',init_cond
  write(2,*)
  write(2,*) 'dynamics without noise (only initial noise)'
  write(2,*)
  write(2,*) 'For the predictor-corrector method: '
  write(2,*) '    Maximum number of iterations:',nmbr_eval
  write(2,*) '    TOL=',TOL,'  err_s=',err_s
# ifdef open_MP 
  write(2,*) 'use openMP for multi-threading'
# else
  write(2,*) 'no openMP'
# endif
  close(2)

  p_alloc_psi=fftw_alloc_real(int(ly * lx, C_SIZE_T))
  call c_f_pointer(p_alloc_psi, phi, [ly,lx])
  p_alloc_psi=fftw_alloc_real(int(ly * lx, C_SIZE_T))
  call c_f_pointer(p_alloc_psi, psi, [ly,lx])
  p_alloc_psiq=fftw_alloc_complex(int((ly/2+1) * lx, C_SIZE_T))
  call c_f_pointer(p_alloc_psiq, phiq, [ly/2+1,lx])
  p_alloc_psiq=fftw_alloc_complex(int((ly/2+1) * lx, C_SIZE_T))
  call c_f_pointer(p_alloc_psiq, psiq, [ly/2+1,lx])

  allocate(q2(ly/2+1,lx),alpha_12(ly/2+1,lx),alpha_21(ly/2+1,lx))
  allocate(exp_11(ly/2+1,lx),exp_12(ly/2+1,lx),exp_21(ly/2+1,lx),exp_22(ly/2+1,lx))
  allocate(cf_1(ly/2+1,lx),cf2_1(ly/2+1,lx),cf_2(ly/2+1,lx),cf2_2(ly/2+1,lx))
  allocate(sigma12(ly/2+1,lx),sig1_alpha(ly/2+1,lx),sig2_alpha(ly/2+1,lx))
  allocate(isigma(ly/2+1,lx))

  allocate(f(ly,lx))
  allocate(sigma1(ly/2+1,lx),sigma2(ly/2+1,lx))
 
# ifdef open_MP
  ! use multi-threaded fftw with open_MP
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

! for FFTW parameters
  scale2d_b=1.d0/(ly*lx) ! for backward FFT

! initialization for FFTW (out of place; create the plans before initializing the inputs)
  plan2d_f=fftw_plan_dft_r2c_2d(lx,ly,psi,psiq,FFTW_MEASURE) ! note the reverse order of dimensions
  plan2d_b=fftw_plan_dft_c2r_2d(lx,ly,psiq,psi,FFTW_MEASURE)

  plan1dx_f=fftw_plan_r2r_1d(lx,psix,psiqx,FFTW_R2HC,FFTW_MEASURE)
  plan1dy_f=fftw_plan_r2r_1d(ly,psiy,psiqy,FFTW_R2HC,FFTW_MEASURE)

  y=ran3(idum)

!
! set the initial conditions
!
  if(ntype.eq.1) then
    !
    ! set up the initial configuration
    !
# ifdef ini_crystal
     ! for initial condition of perfect crystal
!     A0a=0.2d0
! for the case of symmetric A/B (epsA=epsB, gA=gB, w=u, beta_B=1, v=1, n0A=n0B, q0=q1=qq2 (qA=qB=qAB))
# ifdef ini_crystal_sl
    a1=-epsA-2*gA*n0A_s+3*n0A_s*n0A_s+w*n0B_s
    a2=-gA+3*n0A_s
    c=alpha_AB+w*n0A_s+u*n0B_s
# else
    a1=-epsA-2*gA*n0A+3*n0A*n0A+w*n0B
    a2=-gA+3*n0A
    c=alpha_AB+w*n0A+u*n0B
# endif
    ! A0a=(-(a2-0.75d0*w)+sqrt((a2-0.75d0*w)**2-15*(a1-c/2)))/15
    A0a= 0.226007814838    !from Ken's 2D Binary Amplitude (LS2)  (-(a2-0.75d0*w)+sqrt((a2-0.75d0*w)**2-15*(a1-c/2)))/15
    open(unit=2,file=file//'_para.dat',position='append')
    write(2,*) 'initial amplitude A0a=',A0a
    close(2)
# ifdef ini_crystal_sl

    j_width=ly/2
    !  j_width=ly/4
    j1=ly/2-j_width/2
    if(j1.lt.1) j1=1
    j2=ly/2+j_width/2
    if(j2.gt.ly) j2=ly
    ly_mid=(j1+j2)/2

    do i=1,lx
        do j=1,ly
            x=(i-1-j1-n_dx/4)*dx
            y=(j-1)*dy
            if(j.ge.j1.and.j.lt.j2) then ! for solid AB layers
                if(theta.eq.0.0d0) then
                    q_1=-qy0*y-qx0*x/2
                    q_2=qx0*x
                    q_3=qy0*y-qx0*x/2
                    phi(j,i)=n0A_s+2*A0a*(cos(q_1 + phi_A1*pi/180)+cos(q_2 + phi_A2*pi/180)+cos(q_3 + phi_A3*pi/180)) ! reconstruct nA(y,x)
                    psi(j,i)=n0B_s+2*A0a*(cos(q_1 + phi_B1*pi/180)+cos(q_2 + phi_B2*pi/180)+cos(q_3 + phi_B3*pi/180)) ! reconstruct nB(y,x)
                elseif(theta.eq.90.0d0) then
                    q_1=-qx0*x-qy0*y/2
                    q_2=qy0*y
                    q_3=qx0*x-qy0*y/2
                    phi(j,i)=n0A_s+2*A0a*(cos(q_1 + phi_A1*pi/180)+cos(q_2 + phi_A2*pi/180)+cos(q_3 + phi_A3*pi/180)) ! reconstruct nA(y,x)
                    psi(j,i)=n0B_s+2*A0a*(cos(q_1 + phi_B1*pi/180)+cos(q_2 + phi_B2*pi/180)+cos(q_3 + phi_B3*pi/180)) ! reconstruct nB(y,x)
                endif
            else
                phi(j,i)=n0A_l
                psi(j,i)=n0B_l
            endif
        enddo
    enddo
    call fftw_execute_dft_r2c(plan2d_f,phi,phiq) ! forward FFT
    call fftw_execute_dft_r2c(plan2d_f,psi,psiq) ! forward FFT
    n0A_mean=phiq(1,1)
    n0B_mean=psiq(1,1)
    do i=1,lx
        do j=1,ly
            if(j.ge.j1.and.j.lt.j2) then ! for solid AB layers
            else
                phi(j,i)=phi(j,i)+noise*(ran3(idum)-0.5d0)
                psi(j,i)=psi(j,i)+noise*(ran3(idum)-0.5d0)
            endif
        enddo
    enddo
    call fftw_execute_dft_r2c(plan2d_f,phi,phiq) ! forward FFT
    call fftw_execute_dft_r2c(plan2d_f,psi,psiq) ! forward FFT
    phiq(1,1)=n0A_mean
    psiq(1,1)=n0B_mean
    ! backward FFT (note: the input array is overwritten)
    call fftw_execute_dft_c2r(plan2d_b,phiq,phi)
    phi=phi*scale2d_b
    call fftw_execute_dft_c2r(plan2d_b,psiq,psi)
    psi=psi*scale2d_b
# else
    do i=1,lx
        do j=1,ly
            x=(i-1)*dx
            y=(j-1)*dy
            if(theta.eq.0.0d0) then
                q_1=-qy0*y-qx0*x/2
                q_2=qx0*x
                q_3=qy0*y-qx0*x/2
                phi(j,i)=n0A+2*A0a*(cos(q_1)+cos(q_2)+cos(q_3)) ! reconstruct nA(y,x)
                !    phi(j,i)=phi(j,i)+noise*(ran3(idum)-0.5d0)
                x=x+4*pi/(3*qx0)
                q_1=-qy0*y-qx0*x/2
                q_2=qx0*x
                q_3=qy0*y-qx0*x/2
                psi(j,i)=n0B+2*A0a*(cos(q_1)+cos(q_2)+cos(q_3)) ! reconstruct nB(y,x)
            elseif(theta.eq.90.0d0) then
                q_1=-qx0*x-qy0*y/2
                q_2=qy0*y
                q_3=qx0*x-qy0*y/2
                phi(j,i)=n0A+2*A0a*(cos(q_1)+cos(q_2)+cos(q_3)) ! reconstruct nA(y,x)
                !    phi(j,i)=phi(j,i)+noise*(ran3(idum)-0.5d0)
                y=y+4*pi/(3*qy0)
                q_1=-qx0*x-qy0*y/2
                q_2=qy0*y
                q_3=qx0*x-qy0*y/2
                psi(j,i)=n0B+2*A0a*(cos(q_1)+cos(q_2)+cos(q_3)) ! reconstruct nB(y,x)
            endif
        enddo
     enddo
# endif
# endif

  else
    ! 
    ! read initial configuration from file
    !
     open(unit=11,file=init_cond,form='unformatted',status='old')
     read(11) phi,psi
     close(11)
  endif

  call fftw_execute_dft_r2c(plan2d_f,phi,phiq) ! forward FFT
  call fftw_execute_dft_r2c(plan2d_f,psi,psiq) ! forward FFT

  call hdf_r8(ly,lx,phi,image,file//'_nA.hdf')
  call hdf_r8(ly,lx,psi,image,file//'_nB.hdf')
  call hdf_r8(ly,lx,phi+psi,image,file//'_nAB.hdf')
  call hdf_r8(ly,lx,phi-psi,image,file//'_nA_B.hdf')

  i_imag=1
!  i_imag=0
  time=time0

! show the n & psi profiles (at lx/2 or average)
  open(unit=13,file=file//'_nAB_t0.dat')
  do j=1,ly
     write(13,30) j,j*dy,phi(j,lx/2),sum(phi(j,1:lx))/lx,psi(j,lx/2),sum(psi(j,1:lx))/lx, &
                         maxval(phi(j,1:lx)),maxval(psi(j,1:lx))
  enddo
  close(13)
30 format(i6,7(1x,e12.5))

  if(iter0.le.n_i) then
     dtime=dti ! for initial transient
  else
     dtime=dt
  endif

  isigma=0
  do i=1,lx
     do j=1,ly/2+1
        q2(j,i)=qx(i)*qx(i)+qy(j)*qy(j)
        alpha_11=-q2(j,i)*((q2(j,i)-q02)**2-epsA)
        alpha_22=-mB*q2(j,i)*(beta_B*(q2(j,i)-q1_2)**2-epsB)
        alpha_12(j,i)=-q2(j,i)*(alpha_AB+beta_AB*(q2(j,i)-q2_2)**2)
        alpha_21(j,i)=mB*alpha_12(j,i)

        delta_2=sqrt((alpha_11-alpha_22)**2+4*alpha_12(j,i)*alpha_21(j,i))
        if(abs(delta_2).lt.1.d-8) then ! sigma1=sigma2
           isigma(j,i)=1
           sigma1(j,i)=(alpha_11+alpha_22)/2
           sigma2(j,i)=sigma1(j,i)
           sig1_alpha(j,i)=sigma1(j,i)-alpha_11

           sig1_dt=sigma1(j,i)*dtime
           exp_11(j,i)=exp(sig1_dt)*(1-sig1_alpha(j,i)*dtime)
           exp_12(j,i)=exp(sig1_dt)*alpha_12(j,i)*dtime
           exp_21(j,i)=exp(sig1_dt)*alpha_21(j,i)*dtime
           exp_22(j,i)=exp(sig1_dt)*(1+sig1_alpha(j,i)*dtime)
           if(abs(sigma1(j,i)).lt.1.d-4) then
              cf_1(j,i)=dtime*(1.d0+0.5d0*sig1_dt*(1.d0+sig1_dt/3.d0))
              cf2_1(j,i)=0.5d0*dtime*(1.d0+sig1_dt*(1.d0+0.25d0*sig1_dt)/3.d0)
              cf_2(j,i)=dtime*dtime*(0.5d0+sig1_dt/3+sig1_dt*sig1_dt/8)
              cf2_2(j,i)=dtime*dtime*(1.d0/6.d0+sig1_dt/12+sig1_dt*sig1_dt/40)
           else
              cf_1(j,i)=(exp(sig1_dt)-1)/sigma1(j,i)
              cf2_1(j,i)=(exp(sig1_dt)-(1+sig1_dt))/(sigma1(j,i)*sig1_dt)
              cf_2(j,i)=(1+exp(sig1_dt)*(sig1_dt-1))/(sigma1(j,i)*sigma1(j,i))
              cf2_2(j,i)=(2+sig1_dt+(sig1_dt-2)*exp(sig1_dt))/(sigma1(j,i)*sigma1(j,i)*sig1_dt)
           endif

        else
!           sigma1(j,i)=(alpha_11+alpha_22+delta_2)/2
!           sigma2(j,i)=(alpha_11+alpha_22-delta_2)/2
! use the algorithm of Numerical Recipes: Sec. 5.6
           b=-(alpha_11+alpha_22)
           c=alpha_11*alpha_22-alpha_12(j,i)*alpha_21(j,i)
           if(Real(conjg(b)*delta_2).ge.0) then
              aq=-(b+delta_2)/2
           else
              aq=-(b-delta_2)/2
           endif
           sigma1(j,i)=aq
           sigma2(j,i)=c/aq

           sigma12(j,i)=sigma1(j,i)-sigma2(j,i)
           sig1_alpha(j,i)=sigma1(j,i)-alpha_22
           sig2_alpha(j,i)=sigma2(j,i)-alpha_22

           sig1_dt=sigma1(j,i)*dtime
           sig2_dt=sigma2(j,i)*dtime
           exp_11(j,i)=sig1_alpha(j,i)*exp(sig1_dt)-sig2_alpha(j,i)*exp(sig2_dt)
           exp_12(j,i)=alpha_12(j,i)*(exp(sig1_dt)-exp(sig2_dt))
           exp_21(j,i)=alpha_21(j,i)*(exp(sig1_dt)-exp(sig2_dt))
           exp_22(j,i)=sig1_alpha(j,i)*exp(sig2_dt)-sig2_alpha(j,i)*exp(sig1_dt)
           if(abs(sigma1(j,i)).lt.1.d-4) then
              cf_1(j,i)=dtime*(1.d0+0.5d0*sig1_dt*(1.d0+sig1_dt/3.d0))
              cf2_1(j,i)=0.5d0*dtime*(1.d0+sig1_dt*(1.d0+0.25d0*sig1_dt)/3.d0)
           else
              cf_1(j,i)=(exp(sig1_dt)-1)/sigma1(j,i)
              cf2_1(j,i)=(exp(sig1_dt)-(1+sig1_dt))/(sigma1(j,i)*sig1_dt)
           endif
           if(abs(sigma2(j,i)).lt.1.d-4) then
              cf_2(j,i)=dtime*(1.d0+0.5d0*sig2_dt*(1.d0+sig2_dt/3.d0))
              cf2_2(j,i)=0.5d0*dtime*(1.d0+sig2_dt*(1.d0+0.25d0*sig2_dt)/3.d0)
           else
              cf_2(j,i)=(exp(sig2_dt)-1)/sigma2(j,i)
              cf2_2(j,i)=(exp(sig2_dt)-(1+sig2_dt))/(sigma2(j,i)*sig2_dt)
           endif

        endif

     enddo
  enddo

  call F_mu(f,F_avg,muA_avg,muB_avg)
  open(unit=1,file=file//'_F_mu.dat',position='append')
  write(1,10) time,F_avg,muA_avg,muB_avg ! for average of F and mu
  close(1)
  call hdf_r8(ly,lx,f,image,file//'_f.hdf')

  open(unit=2,file=file//'_out.dat',position='append')
  write(2,*) 't=',time
  write(2,*) maxval(phi(1:ly,1:lx)),minval(phi(1:ly,1:lx))
  write(2,*) '  <nA>=',sum(phi)*scale2d_b
  write(2,*) maxval(psi(1:ly,1:lx)),minval(psi(1:ly,1:lx))
  write(2,*) '  <nB>=',sum(psi)*scale2d_b
  write(2,*) 'f=',maxval(f),minval(f)
  write(2,*) 'mu_avg=',muA_avg,muB_avg
  write(2,*)
  close(2)

  ! save the profiles of nA, nB, nAB, nA-nB
  ! write(cha,'(i0)') NINT(time)
  write(cha, '(F0.3)') time
  open(unit=13,file=file//'_nAB_f_t'//trim(cha)//'.dat')
  write(13,*) 'ix=1:lx, iy=1:ly, dx, dy, phi(iy,ix), psi(iy,ix), f(iy,ix)'
  do i=1,lx
    do j=1,ly
      write(13,51) i,j,i*dx,j*dy,phi(j,i),psi(j,i),f(j,i)
    enddo ! for j
  enddo ! for i
  close(13)

! start running
  do iter=iter0,nend

    if(iter.le.n_i) then ! for initial transient only
       time=iter*dti
    else
       time=(iter-n_i)*dt+t_i
       if(iter.eq.n_i+1) then
          dtime=dt
          do i=1,lx
             do j=1,ly/2+1
                if(isigma(j,i).eq.1) then ! sigma1=sigma2
                   sig1_dt=sigma1(j,i)*dtime
                   exp_11(j,i)=exp(sig1_dt)*(1-sig1_alpha(j,i)*dtime)
                   exp_12(j,i)=exp(sig1_dt)*alpha_12(j,i)*dtime
                   exp_21(j,i)=exp(sig1_dt)*alpha_21(j,i)*dtime
                   exp_22(j,i)=exp(sig1_dt)*(1+sig1_alpha(j,i)*dtime)
                   if(abs(sigma1(j,i)).lt.1.d-4) then
                      cf_1(j,i)=dtime*(1.d0+0.5d0*sig1_dt*(1.d0+sig1_dt/3.d0))
                      cf2_1(j,i)=0.5d0*dtime*(1.d0+sig1_dt*(1.d0+0.25d0*sig1_dt)/3.d0)
                      cf_2(j,i)=dtime*dtime*(0.5d0+sig1_dt/3+sig1_dt*sig1_dt/8)
                      cf2_2(j,i)=dtime*dtime*(1.d0/6.d0+sig1_dt/12+sig1_dt*sig1_dt/40)
                   else
                      cf_1(j,i)=(exp(sig1_dt)-1)/sigma1(j,i)
                      cf2_1(j,i)=(exp(sig1_dt)-(1+sig1_dt))/(sigma1(j,i)*sig1_dt)
                      cf_2(j,i)=(1+exp(sig1_dt)*(sig1_dt-1))/(sigma1(j,i)*sigma1(j,i))
                      cf2_2(j,i)=(2+sig1_dt+(sig1_dt-2)*exp(sig1_dt))/(sigma1(j,i)*sigma1(j,i)*sig1_dt)
                   endif                  
                else
                   sig1_dt=sigma1(j,i)*dtime
                   sig2_dt=sigma2(j,i)*dtime
                   exp_11(j,i)=sig1_alpha(j,i)*exp(sig1_dt)-sig2_alpha(j,i)*exp(sig2_dt)
                   exp_12(j,i)=alpha_12(j,i)*(exp(sig1_dt)-exp(sig2_dt))
                   exp_21(j,i)=alpha_21(j,i)*(exp(sig1_dt)-exp(sig2_dt))
                   exp_22(j,i)=sig1_alpha(j,i)*exp(sig2_dt)-sig2_alpha(j,i)*exp(sig1_dt)
                   if(abs(sigma1(j,i)).lt.1.d-4) then
                      cf_1(j,i)=dtime*(1.d0+0.5d0*sig1_dt*(1.d0+sig1_dt/3.d0))
                      cf2_1(j,i)=0.5d0*dtime*(1.d0+sig1_dt*(1.d0+0.25d0*sig1_dt)/3.d0)
                   else
                      cf_1(j,i)=(exp(sig1_dt)-1)/sigma1(j,i)
                      cf2_1(j,i)=(exp(sig1_dt)-(1+sig1_dt))/(sigma1(j,i)*sig1_dt)
                   endif
                   if(abs(sigma2(j,i)).lt.1.d-4) then
                      cf_2(j,i)=dtime*(1.d0+0.5d0*sig2_dt*(1.d0+sig2_dt/3.d0))
                      cf2_2(j,i)=0.5d0*dtime*(1.d0+sig2_dt*(1.d0+0.25d0*sig2_dt)/3.d0)
                   else
                      cf_2(j,i)=(exp(sig2_dt)-1)/sigma2(j,i)
                      cf2_2(j,i)=(exp(sig2_dt)-(1+sig2_dt))/(sigma2(j,i)*sig2_dt)
                   endif
                endif
             enddo
          enddo
       endif
    endif

    call eqs(time,iter,istep,file)

    if((iter.eq.n_i).or.(iter.gt.n_i.and.mod(iter+n_0,raout).eq.0)) then 

! output wavenumbers qx,qy and 1D spectrum at y=ly/2 and x=lx/2, and nA & nB profiles
      call wavenumbers(iter,n_time,itime,dy,time,ly/2,lx/2,file)

      call F_mu(f,F_avg,muA_avg,muB_avg)
      open(unit=1,file=file//'_F_mu.dat',position='append')
      write(1,10) time,F_avg,muA_avg,muB_avg ! for average of F and mu
      close(1)

      open(unit=2,file=file//'_out.dat',position='append')
      write(2,*) 't=',time,'  istep=',istep
      write(2,*) maxval(phi(1:ly,1:lx)),minval(phi(1:ly,1:lx))
      write(2,*) '  <nA>=',sum(phi)*scale2d_b
      write(2,*) maxval(psi(1:ly,1:lx)),minval(psi(1:ly,1:lx))
      write(2,*) '  <nB>=',sum(psi)*scale2d_b
      write(2,*) 'f=',maxval(f),minval(f)
      write(2,*) 'mu_avg=',muA_avg,muB_avg
      write(2,*)
      close(2)

! save images
    if ((iter.eq.n_i).or.(iter.gt.n_i.and.mod(iter+n_0,nout).eq.0)) then
      call hdf_r8(ly,lx,phi,image,file//'_nA.hdf')
      call hdf_r8(ly,lx,psi,image,file//'_nB.hdf')
      call hdf_r8(ly,lx,phi+psi,image,file//'_nAB.hdf')
      call hdf_r8(ly,lx,phi-psi,image,file//'_nA_B.hdf')
      call hdf_r8(ly,lx,f,image,file//'_f.hdf') ! output free energy image


      i_imag=i_imag+1
      open(unit=2,file=file//'_out.dat',position='append')
      write(2,*) 't=',time, '  iter=',iter,' for ouput of image files'
      write(2,*) '  of No.',i_imag,' for nA, nB, f'
      write(2,*)
      close(2)

      ! save the profiles of nA, nB, nAB, nA-nB
      ! write(cha,'(i0)') NINT(time)
      write(cha, '(F0.3)') time
      open(unit=13,file=file//'_nAB_f_t'//trim(cha)//'.dat')
      write(13,*) 'ix=1:lx, iy=1:ly, dx, dy, phi(iy,ix), psi(iy,ix), f(iy,ix)'
      do i=1,lx
        do j=1,ly
          write(13,51) i,j,i*dx,j*dy,phi(j,i),psi(j,i),f(j,i)
        enddo ! for j
      enddo ! for i
      close(13)
51   format(1x,i8,i8,5(1x,es18.10))
    endif

# ifdef save_conf
! save conf files
    if(iter.gt.n_i.and.mod(iter+n_0,nout_conf).eq.0) then
       open(unit=11,file=file//'_conf.dat',form='unformatted')
       write(11) phi,psi
       close(11)
       open(unit=2,file=file//'_out.dat',position='append')
       write(2,*) 't=',time, '  iter=',iter,' for ouput of conf files (temporary)'
       write(2,*)
       close(2)
    endif
# endif

    endif ! for raout
  enddo ! for iter=iter0,nend

# ifdef save_conf
  open(unit=11,file=file//'_conf.dat',form='unformatted')
  write(11) phi,psi
  close(11)
  open(unit=2,file=file//'_out.dat',position='append')
  write(2,*) 't=',time, '  iter=',iter-1,' for ouput of conf files'
  write(2,*)
  close(2)
# endif

10 format(4(1x,e15.8))

  deallocate(q2,alpha_12,alpha_21)
  deallocate(exp_11,exp_12,exp_21,exp_22)
  deallocate(cf_1,cf2_1,cf_2,cf2_2)
  deallocate(sigma12,sig1_alpha,sig2_alpha,isigma)
  deallocate(f,sigma1,sigma2)

  call fftw_free(p_alloc_psi)
  call fftw_free(p_alloc_psiq)

  call fftw_destroy_plan(plan2d_f)
  call fftw_destroy_plan(plan2d_b)

  call fftw_destroy_plan(plan1dx_f)
  call fftw_destroy_plan(plan1dy_f)

end Program bpfc_nAB_hon
!

subroutine eqs(time,iter,istep,file)   
! 
! main routine that solves the mean field eqs
!
  use global_variables
  implicit none

  real(8) :: time
  integer :: i,j,ly_2,istep,iter
  real(8) :: max_conv_phi,max_conv_psi,conv,abs_psi
  complex(8) :: sigN_1,sigN_2
  character file*(*)

  real(C_DOUBLE), pointer :: phi0(:,:),psi0(:,:)
  complex(C_DOUBLE_COMPLEX), pointer :: phi0q(:,:),psi0q(:,:),phiq_tmp(:,:)
  complex(C_DOUBLE_COMPLEX), pointer :: Nonlin1_q(:,:),Nonlin2_q(:,:),sigN_phiq(:,:),sigN_psiq(:,:)
! for allocating aligned memory in fftw (need different pointers for different arrays
! so that the memory can be released at the end of each call (otherwise the used memory will accumulate))
  type(C_PTR) :: p_alloc_phi0,p_alloc_phi0q,p_alloc_phiq_tmp,p_alloc_Nonlin1_q,p_alloc_sigN_phiq
  type(C_PTR) :: p_alloc_psi0,p_alloc_psi0q,p_alloc_Nonlin2_q,p_alloc_sigN_psiq
  
  p_alloc_phi0=fftw_alloc_real(int(ly * lx, C_SIZE_T))
  call c_f_pointer(p_alloc_phi0, phi0, [ly,lx])
  p_alloc_phi0q=fftw_alloc_complex(int((ly/2+1) * lx, C_SIZE_T))
  call c_f_pointer(p_alloc_phi0q, phi0q, [ly/2+1,lx])
  p_alloc_psi0=fftw_alloc_real(int(ly * lx, C_SIZE_T))
  call c_f_pointer(p_alloc_psi0, psi0, [ly,lx])
  p_alloc_psi0q=fftw_alloc_complex(int((ly/2+1) * lx, C_SIZE_T))
  call c_f_pointer(p_alloc_psi0q, psi0q, [ly/2+1,lx])
  p_alloc_phiq_tmp=fftw_alloc_complex(int((ly/2+1) * lx, C_SIZE_T))
  call c_f_pointer(p_alloc_phiq_tmp, phiq_tmp, [ly/2+1,lx])
  p_alloc_Nonlin1_q=fftw_alloc_complex(int((ly/2+1) * lx, C_SIZE_T))
  call c_f_pointer(p_alloc_Nonlin1_q, Nonlin1_q, [ly/2+1,lx])
  p_alloc_Nonlin2_q=fftw_alloc_complex(int((ly/2+1) * lx, C_SIZE_T))
  call c_f_pointer(p_alloc_Nonlin2_q, Nonlin2_q, [ly/2+1,lx])
  p_alloc_sigN_phiq=fftw_alloc_complex(int((ly/2+1) * lx, C_SIZE_T))
  call c_f_pointer(p_alloc_sigN_phiq, sigN_phiq, [ly/2+1,lx])
  p_alloc_sigN_psiq=fftw_alloc_complex(int((ly/2+1) * lx, C_SIZE_T))
  call c_f_pointer(p_alloc_sigN_psiq, sigN_psiq, [ly/2+1,lx])
  
  ly_2=ly/2

!  calculate the nonlinear terms
  call nonlin_q(Nonlin1_q,Nonlin2_q)

! First step: predictor 

! predictor values of phiq and psiq

!$OMP PARALLEL PRIVATE(i,j,sigN_1,sigN_2)
!$OMP DO
!(Note: DO loop index variable is by default PRIVATE, all other variables are by default SHARED unless declared)
  do i=1,lx
     do j=2,ly_2+1
        if(isigma(j,i).eq.1) then ! sigma1=sigma2
           sigN_2=alpha_12(j,i)*Nonlin2_q(j,i)-sig1_alpha(j,i)*Nonlin1_q(j,i)
        phi0q(j,i)=exp_11(j,i)*phiq(j,i)+exp_12(j,i)*psiq(j,i) &
                  +cf_1(j,i)*Nonlin1_q(j,i)+cf_2(j,i)*sigN_2
        if(nmbr_eval.gt.1) sigN_phiq(j,i)=-(sigN_2*cf2_2(j,i)+Nonlin1_q(j,i)*cf2_1(j,i)) ! for use in iteration
           sigN_2=alpha_21(j,i)*Nonlin1_q(j,i)+sig1_alpha(j,i)*Nonlin2_q(j,i)
        psi0q(j,i)=exp_21(j,i)*phiq(j,i)+exp_22(j,i)*psiq(j,i) &
                  +cf_1(j,i)*Nonlin2_q(j,i)+cf_2(j,i)*sigN_2
        if(nmbr_eval.gt.1) sigN_psiq(j,i)=-(sigN_2*cf2_2(j,i)+Nonlin2_q(j,i)*cf2_1(j,i)) ! for use in iteration

        else

           sigN_1=sig1_alpha(j,i)*Nonlin1_q(j,i)+alpha_12(j,i)*Nonlin2_q(j,i)
           sigN_2=sig2_alpha(j,i)*Nonlin1_q(j,i)+alpha_12(j,i)*Nonlin2_q(j,i)
        phi0q(j,i)=(exp_11(j,i)*phiq(j,i)+exp_12(j,i)*psiq(j,i) &
                  +cf_1(j,i)*sigN_1-cf_2(j,i)*sigN_2)/sigma12(j,i)
        if(nmbr_eval.gt.1) sigN_phiq(j,i)=(sigN_2*cf2_2(j,i)-sigN_1*cf2_1(j,i))/sigma12(j,i) ! for use in iteration
           sigN_1=alpha_21(j,i)*Nonlin1_q(j,i)-sig2_alpha(j,i)*Nonlin2_q(j,i)
           sigN_2=alpha_21(j,i)*Nonlin1_q(j,i)-sig1_alpha(j,i)*Nonlin2_q(j,i)
        psi0q(j,i)=(exp_21(j,i)*phiq(j,i)+exp_22(j,i)*psiq(j,i) &
                  +cf_1(j,i)*sigN_1-cf_2(j,i)*sigN_2)/sigma12(j,i)
        if(nmbr_eval.gt.1) sigN_psiq(j,i)=(sigN_2*cf2_2(j,i)-sigN_1*cf2_1(j,i))/sigma12(j,i) ! for use in iteration

        endif
     enddo
  enddo
!$OMP END DO

! for j=1 (qy=0)
  j=1
!$OMP DO
  do i=1,lx/2+1
     if(isigma(j,i).eq.1) then ! sigma1=sigma2
           sigN_2=alpha_12(j,i)*Nonlin2_q(j,i)-sig1_alpha(j,i)*Nonlin1_q(j,i)
        phi0q(j,i)=exp_11(j,i)*phiq(j,i)+exp_12(j,i)*psiq(j,i) &
                  +cf_1(j,i)*Nonlin1_q(j,i)+cf_2(j,i)*sigN_2
        if(nmbr_eval.gt.1) sigN_phiq(j,i)=-(sigN_2*cf2_2(j,i)+Nonlin1_q(j,i)*cf2_1(j,i)) ! for use in iteration
           sigN_2=alpha_21(j,i)*Nonlin1_q(j,i)+sig1_alpha(j,i)*Nonlin2_q(j,i)
        psi0q(j,i)=exp_21(j,i)*phiq(j,i)+exp_22(j,i)*psiq(j,i) &
                  +cf_1(j,i)*Nonlin2_q(j,i)+cf_2(j,i)*sigN_2
        if(nmbr_eval.gt.1) sigN_psiq(j,i)=-(sigN_2*cf2_2(j,i)+Nonlin2_q(j,i)*cf2_1(j,i)) ! for use in iteration

     else

           sigN_1=sig1_alpha(j,i)*Nonlin1_q(j,i)+alpha_12(j,i)*Nonlin2_q(j,i)
           sigN_2=sig2_alpha(j,i)*Nonlin1_q(j,i)+alpha_12(j,i)*Nonlin2_q(j,i)
        phi0q(j,i)=(exp_11(j,i)*phiq(j,i)+exp_12(j,i)*psiq(j,i) &
                  +cf_1(j,i)*sigN_1-cf_2(j,i)*sigN_2)/sigma12(j,i)
        if(nmbr_eval.gt.1) sigN_phiq(j,i)=(sigN_2*cf2_2(j,i)-sigN_1*cf2_1(j,i))/sigma12(j,i) ! for use in iteration
           sigN_1=alpha_21(j,i)*Nonlin1_q(j,i)-sig2_alpha(j,i)*Nonlin2_q(j,i)
           sigN_2=alpha_21(j,i)*Nonlin1_q(j,i)-sig1_alpha(j,i)*Nonlin2_q(j,i)
        psi0q(j,i)=(exp_21(j,i)*phiq(j,i)+exp_22(j,i)*psiq(j,i) &
                  +cf_1(j,i)*sigN_1-cf_2(j,i)*sigN_2)/sigma12(j,i)
        if(nmbr_eval.gt.1) sigN_psiq(j,i)=(sigN_2*cf2_2(j,i)-sigN_1*cf2_1(j,i))/sigma12(j,i) ! for use in iteration

     endif
  enddo
!$OMP END DO

!$OMP DO
  do i=lx/2+2,lx
     phi0q(1,i)=conjg(phi0q(1,lx-i+2)) ! real data treatment
     psi0q(1,i)=conjg(psi0q(1,lx-i+2))
     if(nmbr_eval.gt.1) sigN_phiq(1,i)=conjg(sigN_phiq(1,lx-i+2)) ! for use in iteration
     if(nmbr_eval.gt.1) sigN_psiq(1,i)=conjg(sigN_psiq(1,lx-i+2))
  enddo
!$OMP END DO

!$OMP END PARALLEL

  phiq_tmp=phi0q
! backward FFT (note: the input array is overwritten)
  call fftw_execute_dft_c2r(plan2d_b,phiq_tmp,phi0)
  phi0=phi0*scale2d_b

  phi=phi0
  phiq=phi0q

  phiq_tmp=psi0q
! backward FFT (note: the input array is overwritten)
  call fftw_execute_dft_c2r(plan2d_b,phiq_tmp,psi0)
  psi0=psi0*scale2d_b

  psi=psi0
  psiq=psi0q

! for use in iteration
  if(nmbr_eval.gt.1) then
     phi0q=phi0q+sigN_phiq
     psi0q=psi0q+sigN_psiq
  endif

! Second step: corrector
  do istep=2,nmbr_eval  ! get the corrector values of psiq by functional iteration
                ! note: phi0q,psi0q remain unchanged during the iteration

! for the nonlinear terms (note: here phi,phiq,psi,psiq are the updated values)
     call nonlin_q(Nonlin1_q,Nonlin2_q)

!$OMP PARALLEL
!$OMP DO
!(Note: DO loop index variable is by default PRIVATE, all other variables are by default SHARED unless declared)
!(      thus PRIVATE(i,j) is by default and not needed to declare here)
     do i=1,lx
        do j=2,ly_2+1
        if(isigma(j,i).eq.1) then ! sigma1=sigma2
           phiq(j,i)=phi0q(j,i)+cf2_1(j,i)*Nonlin1_q(j,i) &
                    +cf2_2(j,i)*(alpha_12(j,i)*Nonlin2_q(j,i)-sig1_alpha(j,i)*Nonlin1_q(j,i))
           psiq(j,i)=psi0q(j,i)+cf2_1(j,i)*Nonlin2_q(j,i) &
                    +cf2_2(j,i)*(alpha_21(j,i)*Nonlin1_q(j,i)+sig1_alpha(j,i)*Nonlin2_q(j,i))
        else
           phiq(j,i)=phi0q(j,i)+(cf2_1(j,i)*(sig1_alpha(j,i)*Nonlin1_q(j,i)+alpha_12(j,i)*Nonlin2_q(j,i)) &
                    -cf2_2(j,i)*(sig2_alpha(j,i)*Nonlin1_q(j,i)+alpha_12(j,i)*Nonlin2_q(j,i)))/sigma12(j,i)
           psiq(j,i)=psi0q(j,i)+(cf2_1(j,i)*(alpha_21(j,i)*Nonlin1_q(j,i)-sig2_alpha(j,i)*Nonlin2_q(j,i)) &
                    -cf2_2(j,i)*(alpha_21(j,i)*Nonlin1_q(j,i)-sig1_alpha(j,i)*Nonlin2_q(j,i)))/sigma12(j,i)
        endif
        enddo
     enddo
!$OMP END DO

     ! for j=1 (qy=0)
     j=1
!$OMP DO
     do i=1,lx/2+1
        if(isigma(j,i).eq.1) then ! sigma1=sigma2
           phiq(j,i)=phi0q(j,i)+cf2_1(j,i)*Nonlin1_q(j,i) &
                    +cf2_2(j,i)*(alpha_12(j,i)*Nonlin2_q(j,i)-sig1_alpha(j,i)*Nonlin1_q(j,i))
           psiq(j,i)=psi0q(j,i)+cf2_1(j,i)*Nonlin2_q(j,i) &
                    +cf2_2(j,i)*(alpha_21(j,i)*Nonlin1_q(j,i)+sig1_alpha(j,i)*Nonlin2_q(j,i))
        else
           phiq(j,i)=phi0q(j,i)+(cf2_1(j,i)*(sig1_alpha(j,i)*Nonlin1_q(j,i)+alpha_12(j,i)*Nonlin2_q(j,i)) &
                    -cf2_2(j,i)*(sig2_alpha(j,i)*Nonlin1_q(j,i)+alpha_12(j,i)*Nonlin2_q(j,i)))/sigma12(j,i)
           psiq(j,i)=psi0q(j,i)+(cf2_1(j,i)*(alpha_21(j,i)*Nonlin1_q(j,i)-sig2_alpha(j,i)*Nonlin2_q(j,i)) &
                    -cf2_2(j,i)*(alpha_21(j,i)*Nonlin1_q(j,i)-sig1_alpha(j,i)*Nonlin2_q(j,i)))/sigma12(j,i)
        endif
     enddo
!$OMP END DO

!$OMP DO
     do i=lx/2+2,lx
        phiq(1,i)=conjg(phiq(1,lx-i+2)) ! real data treatment
        psiq(1,i)=conjg(psiq(1,lx-i+2))
     enddo
!$OMP END DO

!$OMP END PARALLEL

     phiq_tmp=phiq
! backward FFT (note: the input array is overwritten)
     call fftw_execute_dft_c2r(plan2d_b,phiq_tmp,phi)
     phi=phi*scale2d_b

     phiq_tmp=psiq
! backward FFT (note: the input array is overwritten)
     call fftw_execute_dft_c2r(plan2d_b,phiq_tmp,psi)
     psi=psi*scale2d_b

! real space (phi,psi) converge
!     if(istep.gt.2) then

     max_conv_phi=0.d0
     do i=1,lx
        do j=1,ly
           abs_psi=abs(phi(j,i))
           if(abs_psi.gt.1.d5) then
              write(*,*) 'phi (nA) diverges at t=',time,' istep=',istep,' j=',j,' i=',i
              write(*,*) 'phi=',phi(j,i),' psi=',psi(j,i), ' for file ',file
              stop
           endif
           if(abs_psi.le.err_s) then
              conv=abs(phi(j,i)-phi0(j,i))
           else
              conv=abs((phi(j,i)-phi0(j,i))/phi(j,i))
           endif
           if(conv.gt.max_conv_phi) max_conv_phi=conv
        enddo
     enddo

     max_conv_psi=0.d0
     do i=1,lx
        do j=1,ly
           abs_psi=abs(psi(j,i))
           if(abs_psi.gt.1.d5) then
              write(*,*) 'psi (nB) diverges at t=',time,' istep=',istep,' j=',j,' i=',i
              write(*,*) 'phi=',phi(j,i),' psi=',psi(j,i), ' for file ',file
              stop
           endif
           if(abs_psi.le.err_s) then
              conv=abs(psi(j,i)-psi0(j,i))
           else
              conv=abs((psi(j,i)-psi0(j,i))/psi(j,i))
           endif
           if(conv.gt.max_conv_psi) max_conv_psi=conv
        enddo
     enddo

     if((max_conv_phi.lt.TOL).and.(max_conv_psi.lt.TOL)) then
        call fftw_free(p_alloc_phi0)
        call fftw_free(p_alloc_phi0q)
        call fftw_free(p_alloc_phiq_tmp)
        call fftw_free(p_alloc_Nonlin1_q)
        call fftw_free(p_alloc_sigN_phiq)
        call fftw_free(p_alloc_psi0)
        call fftw_free(p_alloc_psi0q)
        call fftw_free(p_alloc_Nonlin2_q)
        call fftw_free(p_alloc_sigN_psiq)
        return
     endif

!     endif ! if istep>2

     phi0=phi
     psi0=psi

  enddo ! for istep=2,nmbr_eval

  istep=istep-1
  if(nmbr_eval.gt.2) then
     open(unit=2,file=file//'_out.dat',position='append')
     write(2,*)
     write(2,*) 'Exceeding maximum iterations=',nmbr_eval
     write(2,*) 't=',time,'   TOL=',TOL
     write(2,*) 'max_conv_phi=',max_conv_phi
     write(2,*) maxval(phi(1:ly,1:lx)),minval(phi(1:ly,1:lx))
     write(2,*) 'max_conv_psi=',max_conv_psi
     write(2,*) maxval(psi(1:ly,1:lx)),minval(psi(1:ly,1:lx))
     write(2,*)
     close(2)
  endif

  call fftw_free(p_alloc_phi0)
  call fftw_free(p_alloc_phi0q)
  call fftw_free(p_alloc_phiq_tmp)
  call fftw_free(p_alloc_Nonlin1_q)
  call fftw_free(p_alloc_sigN_phiq)
  call fftw_free(p_alloc_psi0)
  call fftw_free(p_alloc_psi0q)
  call fftw_free(p_alloc_Nonlin2_q)
  call fftw_free(p_alloc_sigN_psiq)
  return
end subroutine eqs


!  calculate the nonlinear terms
subroutine nonlin_q(Nonlin_phiq,Nonlin_psiq)
  use global_variables
  implicit none

  complex(C_DOUBLE_COMPLEX) :: Nonlin_phiq(ly/2+1,lx),Nonlin_psiq(ly/2+1,lx)
  real(C_DOUBLE), pointer :: Nonlin_phi(:,:),Nonlin_psi(:,:)
  type(C_PTR) :: p_alloc_Nonlin_phi,p_alloc_Nonlin_psi ! for allocating aligned memory in fftw
  integer :: i,j

  p_alloc_Nonlin_phi=fftw_alloc_real(int(ly * lx, C_SIZE_T))
  call c_f_pointer(p_alloc_Nonlin_phi, Nonlin_phi, [ly,lx])
  p_alloc_Nonlin_psi=fftw_alloc_real(int(ly * lx, C_SIZE_T))
  call c_f_pointer(p_alloc_Nonlin_psi, Nonlin_psi, [ly,lx])
  
! note: here phi = nA, psi = nB
!!$OMP PARALLEL DO 
! (Note: PRIVATE(i,j) is by default and not needed to declare here)
!  do i=1,lx
!     do j=1,ly
!        Nonlin_phi(j,i)=(-gA+phi(j,i))*phi(j,i)*phi(j,i)+(w*phi(j,i)+u*psi(j,i)/2)*psi(j,i)
!        Nonlin_psi(j,i)=(-gB+v*psi(j,i))*psi(j,i)*psi(j,i)+(u*psi(j,i)+w*phi(j,i)/2)*phi(j,i)
!     enddo
!  enddo
!!$OMP END PARALLEL DO
  Nonlin_phi=(-gA+phi)*phi*phi+(w*phi+u*psi/2)*psi ! faster than above using OMP do loop
  Nonlin_psi=(-gB+v*psi)*psi*psi+(u*psi+w*phi/2)*phi

  call fftw_execute_dft_r2c(plan2d_f,Nonlin_phi,Nonlin_phiq) ! forward FFT
  call fftw_execute_dft_r2c(plan2d_f,Nonlin_psi,Nonlin_psiq) ! forward FFT

  Nonlin_phiq=-q2*Nonlin_phiq
  Nonlin_psiq=-mB*q2*Nonlin_psiq

  call fftw_free(p_alloc_Nonlin_phi)
  call fftw_free(p_alloc_Nonlin_psi)

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
# ifdef image_reverse
      character(1) image_(m,l)
      integer i,j
# endif
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
# ifdef image_reverse
      do j=1,l
         do i=1,m
            image_(i,j)=image(j,i)
         enddo
      enddo
      iret=d8aimg(name,image_,m,l,0)
# else
      iret=d8aimg(name,image,l,m,0)
# endif

      if(iret.ne.0) then
        write(*,*) ' Error writing the HDF file ',name
      endif
      return
    end SUBROUTINE hdf_r8


! output wavenumbers qx,qy and 1D spectrum at y=ly_q and x=lx_q, and nA,nB profiles
subroutine wavenumbers(iter,n_time,itime,dy,time,ly_q,lx_q,file)
  use global_variables
  implicit none

  real(8) :: time,dy
  character file*(*)
  real(8) :: psiqx(lx),psiqy(ly), psix(lx),psiy(ly)
  real(8) :: Sqx((lx+2)/2),Sqy((ly+2)/2)
  integer :: lx_q,ly_q,i,j,jj,index_qx((lx+2)/2),index_qy((ly+2)/2)

  integer :: iter,n_time,itime(15)
  character(1) :: cha

  open(unit=10,file=file//'_qx.dat')
  open(unit=11,file=file//'_qy.dat')
  open(unit=12,file=file//'_qxy.dat',position='append')
  open(unit=13,file=file//'_nAB.dat') ! nA & nB profiles

  do i=1,lx
     psix(i)=psi(ly_q,i)
  enddo
  call fftw_execute_r2r(plan1dx_f,psix,psiqx) ! forward FFT

  do j=1,ly
     psiy(j)=psi(j,lx_q)
  enddo
  call fftw_execute_r2r(plan1dy_f,psiy,psiqy) ! forward FFT

!$OMP PARALLEL DO
  do i=2,lx/2
     Sqx(i)=-sqrt(psiqx(i)*psiqx(i)+psiqx(lx-i+2)*psiqx(lx-i+2))
  enddo
!$OMP END PARALLEL DO
  Sqx(1)=-abs(psiqx(1))
  Sqx(lx/2+1)=-abs(psiqx(lx/2+1))

!$OMP PARALLEL DO
  do j=2,ly/2
     Sqy(j)=-sqrt(psiqy(j)*psiqy(j)+psiqy(ly-j+2)*psiqy(ly-j+2))
  enddo
!$OMP END PARALLEL DO
  Sqy(1)=-abs(psiqy(1))
  Sqy(ly/2+1)=-abs(psiqy(ly/2+1))

  write(10,50) qx(1),-Sqx(1),psiqx(1),psiqx(1)
  do i=2,lx/2+1
     write(10,50) qx(i),-Sqx(i),psiqx(i),psiqx(lx-i+2)
  enddo
  write(11,50) qy(1),-Sqy(1),psiqy(1),psiqy(1)
  do j=2,ly/2+1
     write(11,50) qy(j),-Sqy(j),psiqy(j),psiqy(ly-j+2)
  enddo

! indexing and ranking in ascending order, with Sqx, Sqy unchanged after calling
  call indexx((lx+2)/2,Sqx,index_qx)
  call indexx((ly+2)/2,Sqy,index_qy)

  write(12,*) 'output for nB (=psi) at j_y =',ly_q, ' i_x=',lx_q
  write(12,*) 'with ly=',ly,' lx=',lx,' time=',time
  write(12,10) qy(index_qy(1)), qy(index_qy(2)), qy(index_qy(3)), &
               qx(index_qx(1)), qx(index_qx(2)), qx(index_qx(3))

! show the nA & nB profiles (at lx/2 or average)
  do j=1,ly
     write(13,30) j,j*dy,phi(j,lx/2),sum(phi(j,1:lx))/lx,psi(j,lx/2),sum(psi(j,1:lx))/lx, &
                         maxval(phi(j,1:lx)),maxval(psi(j,1:lx))
  enddo

  do jj=1,n_time
     if(iter.eq.itime(jj)) then
        cha=char(48+jj)
        if(jj.ge.10) cha=char(55+jj)
        open(unit=7,file=file//'_nAB_t'//cha//'.dat') ! nA, nB profiles
        do j=1,ly
           write(7,30) j,j*dy,phi(j,lx/2),sum(phi(j,1:lx))/lx,psi(j,lx/2),sum(psi(j,1:lx))/lx, &
                       maxval(phi(j,1:lx)),maxval(psi(j,1:lx))
        enddo
        close(7)
        open(unit=8,file=file//'_out.dat',position='append')
        write(8,*) 'output of nA, nB profiles at x=lx/2 or average'
        write(8,*) '  at time=',time
        write(8,*)
        close(8)
     endif
  enddo

10 format(6(1x,e12.5))
50 format(4(1x,e12.5))
30 format(i6,7(1x,e12.5))

  close(10)
  close(11)
  close(12)
  close(13)

  return
end subroutine wavenumbers

! calculate free energy and chemical potential (here f(x,y) is the free energy density distribution)
subroutine F_mu(f,F_avg,muA_avg,muB_avg)
  use global_variables
  implicit none

  real(8) :: f(ly,lx),F_avg,muA_avg,muB_avg
  double precision, allocatable :: muA(:,:),muB(:,:),dq2(:,:)
  integer :: i,j

  real(C_DOUBLE), pointer :: phi2(:,:),psi2(:,:)
  complex(C_DOUBLE_COMPLEX), pointer :: d2nq(:,:)
  type(C_PTR) :: p_alloc_phi2,p_alloc_psi2,p_alloc_d2nq ! for allocating aligned memory in fftw

  p_alloc_phi2=fftw_alloc_real(int(ly * lx, C_SIZE_T))
  call c_f_pointer(p_alloc_phi2, phi2, [ly,lx])
  p_alloc_psi2=fftw_alloc_real(int(ly * lx, C_SIZE_T))
  call c_f_pointer(p_alloc_psi2, psi2, [ly,lx])
  p_alloc_d2nq=fftw_alloc_complex(int((ly/2+1) * lx, C_SIZE_T))
  call c_f_pointer(p_alloc_d2nq, d2nq, [ly/2+1,lx])
  allocate(muA(ly,lx),muB(ly,lx),dq2(ly/2+1,lx))

  F_avg=0
  muA_avg=0
  muB_avg=0
  
  phi2=phi*phi
  psi2=psi*psi
  f=(phi2/4-epsA/2-gA*phi/3)*phi2+(psi2*v/4-epsB/2-gB*psi/3)*psi2 &
    +(alpha_AB+w*phi/2+u*psi/2)*phi*psi
  muA=(-gA+phi)*phi2+(w*phi+u*psi/2+alpha_AB)*psi-epsA*phi
  muB=(-gB+v*psi)*psi2+(u*psi+w*phi/2+alpha_AB)*phi-epsB*psi

  d2nq=(q02-q2)**2*phiq
! backward FFT (note: the input array is overwritten)
  call fftw_execute_dft_c2r(plan2d_b,d2nq,phi2)
  phi2=phi2*scale2d_b
  d2nq=beta_B*(q1_2-q2)**2*psiq
! backward FFT (note: the input array is overwritten)
  call fftw_execute_dft_c2r(plan2d_b,d2nq,psi2)
  psi2=psi2*scale2d_b
  f=f+(phi2*phi+psi2*psi)/2
  muA=muA+phi2
  muB=muB+psi2

  dq2=beta_AB*(q2_2-q2)**2
  d2nq=dq2*psiq
! backward FFT (note: the input array is overwritten)
  call fftw_execute_dft_c2r(plan2d_b,d2nq,psi2)
  psi2=psi2*scale2d_b
  f=f+phi*psi2
  muA=muA+psi2
!
  d2nq=dq2*phiq
! backward FFT (note: the input array is overwritten)
  call fftw_execute_dft_c2r(plan2d_b,d2nq,phi2)
  muB=muB+phi2*scale2d_b

! note: f(x,y) here is free energy density distribution; total F = \int f(x,y) dxdy
  F_avg=sum(f(1:ly,1:lx))*scale2d_b ! average (density) = total free energy / volume
  muA_avg=sum(muA(1:ly,1:lx))*scale2d_b ! average chemical potential
  muB_avg=sum(muB(1:ly,1:lx))*scale2d_b

  deallocate(muA,muB,dq2)
  call fftw_free(p_alloc_phi2)
  call fftw_free(p_alloc_psi2)
  call fftw_free(p_alloc_d2nq)
  
  return
end subroutine F_mu
