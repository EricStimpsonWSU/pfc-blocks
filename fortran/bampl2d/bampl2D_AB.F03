! calculate the 0th-order stationary/equilibrium state of binary AB ampl. eqs.
! 2D, Aj^0(u,t), Bj^0(u,t), nA0^0(u,t), and nB0^0(u,t)
! using FFTW3

! use openMP for multi-threading (To use single-core onLy: comment out "define open_MP")

! note: when calculating 0th-order 2D equilibrium/stationary solution, set mA=mB=1
!       also use conserved dynamics for nA0^0,nB0^0, to get mu values (to be used in nonconserved case if needed)
!       use n0_reduced eq, i.e., neglect gradient terms of n0

! pseudo-spectral method
! using a predictor-corrector method for Aj^0, Bj^0, nA0^0, nB0^0
!   exact solution for linear terms and approximation for nonlinearities
!       (with an exponential propagation procedure, similar to Cross et al., Chaos 1994;
!        donot fix the number of iterations;
!        expand around small sigma when it approaches 0)
! with the transient time

!# define debug ! for debugging

# define liq_sol ! for liquid_solid; "l": liquid; "s": solid

! # define ini_tanh ! use initial tanh profile

! for subroutines interface and width_gap
!# define n0s_lt_l ! for n0_s < n0_l

# define save_conf

# define open_MP ! use openMP for multi-threading

module global_variables
! by default all variables in the module are public (if not being set explicitLy as private)
  use, intrinsic :: iso_c_binding
  implicit none
  include '/wsu/el7/pgi-openmpi/2018-187/fftw/3.3.8/include/fftw3.f03'
  
  save ! save all the variables in the module

  ! integer, parameter :: Lx=400,Ly=400
  integer :: Lx, Ly

  real(8), allocatable :: G_0(:,:,:)
  complex(C_DOUBLE_COMPLEX), pointer :: A0(:,:,:),A0q(:,:,:),B0(:,:,:),B0q(:,:,:)
  real(C_DOUBLE), pointer :: nA0(:,:),nB0(:,:)
  complex(C_DOUBLE_COMPLEX), pointer :: nA0q(:,:),nB0q(:,:)
  ! real(C_DOUBLE) :: exp_j11(3,Ly,Lx),exp_j12(3,Ly,Lx),exp_j21(3,Ly,Lx),exp_j22(3,Ly,Lx) ! for Aj and Bj
  ! real(C_DOUBLE) :: cf_j1(3,Ly,Lx),cf2_j1(3,Ly,Lx),cf_j2(3,Ly,Lx),cf2_j2(3,Ly,Lx)
  ! real(C_DOUBLE) :: exp_11(Ly/2+1,Lx),exp_12(Ly/2+1,Lx),exp_21(Ly/2+1,Lx),exp_22(Ly/2+1,Lx) ! for nA0 and nB0
  ! real(C_DOUBLE) :: cf_1(Ly/2+1,Lx),cf2_1(Ly/2+1,Lx),cf_2(Ly/2+1,Lx),cf2_2(Ly/2+1,Lx)
  ! real(C_DOUBLE) :: sigma12(Ly/2+1,Lx),sig1_alpha(Ly/2+1,Lx),sig2_alpha(Ly/2+1,Lx),alpha_12(Ly/2+1,Lx),alpha_21(Ly/2+1,Lx)
  ! real(C_DOUBLE) :: sigmaj12(3,Ly,Lx),sigj1_alpha(3,Ly,Lx),sigj2_alpha(3,Ly,Lx),alpha_j12(3,Ly,Lx),alpha_j21(3,Ly,Lx)
  ! real(8) :: vA,vB,gA,gB,w,u,mB,q02_mB,q02,q2(Ly,Lx),pi
  ! integer :: isigmaj(3,Ly,Lx),isigma(Ly/2+1,Lx)
  real(C_DOUBLE), allocatable :: exp_j11(:,:,:),exp_j12(:,:,:),exp_j21(:,:,:),exp_j22(:,:,:) ! for Aj and Bj
  real(C_DOUBLE), allocatable :: cf_j1(:,:,:),cf2_j1(:,:,:),cf_j2(:,:,:),cf2_j2(:,:,:)
  real(C_DOUBLE), allocatable :: exp_11(:,:),exp_12(:,:),exp_21(:,:),exp_22(:,:) ! for nA0 and nB0
  real(C_DOUBLE), allocatable :: cf_1(:,:),cf2_1(:,:),cf_2(:,:),cf2_2(:,:)
  real(C_DOUBLE), allocatable :: sigma12(:,:),sig1_alpha(:,:),sig2_alpha(:,:),alpha_12(:,:),alpha_21(:,:)
  real(C_DOUBLE), allocatable :: sigmaj12(:,:,:),sigj1_alpha(:,:,:),sigj2_alpha(:,:,:),alpha_j12(:,:,:),alpha_j21(:,:,:)
  real(8) :: vA,vB,gA,gB,w,u,mB,q02_mB,q02,pi
  real(8), allocatable :: q2(:,:)
  integer, allocatable :: isigmaj(:,:,:),isigma(:,:)
    
  ! for FFTW parameters
  type(C_PTR) :: plan2d_r_f,plan2d_r_b,plan2d_c_f,plan2d_c_b
  real(8) :: scale2D_b

end module global_variables

Program bampl2D_AB
  use global_variables
  implicit none
  
  common/converge/TOL,err_s,nmbr_eval
  ! common/q12/G_0(:,:,:),deltaj0(3),qA2,qB2,qAB2
  common/q12/deltaj0(3),qA2,qB2,qAB2
  common/surf/dx,dy,epsA,epsB,beta_A,beta_B,beta_AB,alpha_AB,width_per
  common/interf/m_gap_nA(2),m_gap_nB(2),width_nA(2),width_nB(2),width_A(2),width_B(2),gamma_nA,gamma_nB,Es
  common/interf_coef/dchi_xj(3),dchi_yj(3),chi_xj2(3),sin_theta

  type(C_PTR) :: p_alloc_A,p_alloc_Aq,p_alloc_n,p_alloc_nq ! for allocating aligned memory in fftw
  real(C_DOUBLE) :: alpha_11,alpha_22
  ! real(C_DOUBLE) :: sigma1(Ly/2+1,Lx),sigma2(Ly/2+1,Lx),sig1_dt,sig2_dt,delta_2,aq,b,c
  ! real(C_DOUBLE) :: sigmaj1(3,Ly,Lx),sigmaj2(3,Ly,Lx)
  real(C_DOUBLE) :: sig1_dt,sig2_dt,delta_2,aq,b,c
  real(C_DOUBLE), allocatable :: sigma1(:,:),sigma2(:,:)
  real(C_DOUBLE), allocatable :: sigmaj1(:,:,:),sigmaj2(:,:,:)
  real(8) :: dt,dti,dtime
  integer :: iter0,ntype,iter,idum,n_0,n_i,nmbr_eval,istep
  integer :: i_theta,n1,n2,i_p

  real(8) :: epsA,epsB,q0,dx,dy,noise,noise0
  ! real(8) :: qx(Lx),qy(Ly)
  real(8), allocatable :: qx(:),qy(:)
  real(8) :: qxd,qyd
  real(8) :: qx0,qy0,qu
  real(8) :: qA,qA2,qB,qB2,qAB,qAB2
  real(8) :: beta_A,beta_B,beta_AB,alpha_AB
  real(8) :: sin_theta,cos_theta
  real(8) :: delta_x,delta_y,eps_m
  ! real(8) :: G_0,deltaj0
  real(8) :: deltaj0
  real(8) :: time,tmax,time0,t_i
  real(8) :: yy,ran3
  real(8) :: vb1A,vb2A,ystar1A,ystar2A,ystar1A_0,ystar2A_0
  integer :: n_dx,nend,raout,nout_conf,i,ix,iy,j,nimpc_v,nout_v,i_ystar,nt
! for sharp-interface coefficients
  real(8) :: dchi_xj,dchi_yj,chi_xj2

  real(8) :: nA0_s,nA0_l,nB0_s,nB0_l
  complex(8) :: A1_s,A2_s,A3_s,B1_s,B2_s,B3_s,A_l,B_l
  !real(8) :: phi_A1,phi_A2,phi_A3,phi_B1,phi_B2,phi_B3
  real(8) :: A_top,width_per,wid_n,wid_A
  real(8) :: width_nA,width_nB,width_A,width_B,m_gap_nA,m_gap_nB,gamma_nA,gamma_nB,Es
  integer :: i1,i2,i_width,Lx_mid
  integer :: n_time, itime(15)
  character(10) :: cha
  real(8) :: TOL,err_s ! for iteration of predictor-corrector

  complex(C_DOUBLE_COMPLEX), pointer :: A_tmp(:,:),Aq_tmp(:,:)
  real(C_DOUBLE), pointer :: n_tmp(:,:)

# ifdef open_MP
! use multi-threaded fftw with openMP
  integer iret,nthreads,omp_get_max_threads
# endif
  
  character(:), allocatable :: file ! to be automaticalLy allocated
  character(:), allocatable :: filename ! to be automaticalLy allocated
  !character(5) filename
  character(:), allocatable :: init_cond ! to be automaticalLy allocated

  ntype=1  ! ntype=1: set initial configuration; ntype=2: input from file
!  ntype=2 ! for nonzero misfit: input from config of 0 misfit
  time0=0
  iter0=1
!  ntype=2 ! initial condition input from files of previous calculation
!  time0=10000
!  iter0=10000+1  ! iter0-1 = nend of previous run
!  init_cond='_conf.dat'

! for rescaled PFC equations;  eps=1-B^l/B^x
  epsA=0.3d0 ! for nA
  epsB=epsA ! for nB

  q0=1
  q02=1 ! =q0^2
  qA=q0
  qB=q0
  qA2=qA*qA
  qB2=qB*qB
  
  beta_A=1
  beta_B=1
  gA=0.5d0
  gB=gA
  vA=1
  vB=1
  w=0.3d0
  u=w
  alpha_AB=0.5d0
  
  beta_AB=0
  qAB=q0
  qAB2=qAB*qAB
  mB=1 ! mB=M_B/M_A; mA=1

  eps_m=0 ! overlayer/substrate misfit

  pi=DACOS(-1.0d0)

  width_per=0.85d0 ! interface range: 85% of misc. gap

! interface orientation theta (with respect to the y axis)

  open(unit=10, file="bampl2D.in", status='old', action='read')
  read(10, *) i_theta
  read(10, *) Lx, Ly
  read(10, *) noise, noise0
  read(10, *) dti, t_i
  read(10, *) dt, tmax
  close(10)

  allocate(exp_j11(3,Ly,Lx),exp_j12(3,Ly,Lx),exp_j21(3,Ly,Lx),exp_j22(3,Ly,Lx)) ! for Aj and Bj
  allocate(cf_j1(3,Ly,Lx),cf2_j1(3,Ly,Lx),cf_j2(3,Ly,Lx),cf2_j2(3,Ly,Lx))
  allocate(exp_11(Ly/2+1,Lx),exp_12(Ly/2+1,Lx),exp_21(Ly/2+1,Lx),exp_22(Ly/2+1,Lx)) ! for nA0 and nB0
  allocate(cf_1(Ly/2+1,Lx),cf2_1(Ly/2+1,Lx),cf_2(Ly/2+1,Lx),cf2_2(Ly/2+1,Lx))
  allocate(sigma12(Ly/2+1,Lx),sig1_alpha(Ly/2+1,Lx),sig2_alpha(Ly/2+1,Lx),alpha_12(Ly/2+1,Lx),alpha_21(Ly/2+1,Lx))
  allocate(sigmaj12(3,Ly,Lx),sigj1_alpha(3,Ly,Lx),sigj2_alpha(3,Ly,Lx),alpha_j12(3,Ly,Lx),alpha_j21(3,Ly,Lx))
  allocate(q2(Ly,Lx))
  allocate(isigmaj(3,Ly,Lx),isigma(Ly/2+1,Lx))
  allocate(G_0(3,Ly,Lx))
  allocate(sigma1(Ly/2+1,Lx),sigma2(Ly/2+1,Lx))
  allocate(sigmaj1(3,Ly,Lx),sigmaj2(3,Ly,Lx))
  allocate(qx(Lx),qy(Ly))

! zigzag: theta=0 or +/-60^o (equivalent)
!  i_theta=0 ! direction 1, along y
!  i_theta=60 ! direction 2, along (qx,qy/2)
!  i_theta=-60 ! direction 3, along (-qx,qy/2)

! armchair: theta=90^o or +/-30^o (equivalent)
!  i_theta=90 ! direction 4, along x
!  i_theta=30 ! direction 5, along (qx,3qy/2)
!  i_theta=-30 ! direction 6, along (-qx,3qy/2)
!
!  i_theta=19 ! direction a1, theta=19.1066 degree
!  i_theta=-19 ! direction a2, theta=-19.1066 degree
!  i_theta=41 ! direction a3, theta=40.8934 degree
!  i_theta=-41 ! direction a4, theta=-40.8934 degree
!  i_theta=79 ! direction a5, theta=79.1066 degree
!  i_theta=-79 ! direction a6, theta=-79.1066 degree
!
!  i_theta=11 ! direction b1, theta=10.8934 degree
!  i_theta=-11 ! direction b2, theta=-10.8934 degree
!  i_theta=49 ! direction b3, theta=49.1066 degree
!  i_theta=-49 ! direction b4, theta=-49.1066 degree
!  i_theta=71 ! direction b5, theta=70.8934 degree
!  i_theta=-71 ! direction b6, theta=-70.8934 degree
!
!  i_theta=23 ! direction c1, theta=23.4132 degree
!  i_theta=-23 ! direction c2, theta=-23.4132 degree
!  i_theta=37 ! direction c3, theta=36.5868 degree
!  i_theta=-37 ! direction c4, theta=-36.5868 degree
!  i_theta=83 ! direction c5, theta=83.4132 degree
!  i_theta=-83 ! direction c6, theta=-83.4132 degree
!
!  i_theta=5 ! direction d1, theta=5.2087 degree
  
!  i_theta=14 ! direction e1, theta=13.8979 degree
  
!  i_theta=8 ! direction f1, theta= 7.5891 degree
  
!  i_theta=9 ! direction g1, theta= 8.9483 degree
  
!  i_theta=16 ! direction h1, theta= 16.1021 degree
  
!  i_theta=17 ! direction i1, theta= 17.4802 degree
  
!  i_theta=21 ! direction j1, theta= 20.6330 degree
  
!  i_theta=22 ! direction k1, theta= 21.7868 degree
  
!  i_theta=25 ! direction l1, theta= 25.2850 degree
  
  !file='tem'

# ifdef liq_sol ! "f": liquid; "s": solid(same as substrate)
!  epsA=epsB=0.3,g=0.5,v=1,w=u=0.3,alpha_AB=0.5,beta_AB=0,qA=qB=q0=1,dt=0.1,ini.noise=0
  !file='bampl512eps03g05w03aAB05bAB0noise0' ! _4: i_theta=90; _1: i_theta=0; Lx=512,tmax=10^7, def ini_tanh
                                 ! sharp interface, zeta_D(AB) sensitive on Gibbs surface positions, _4: zeta(1) >0, (2) <0
       ! _1: gamma_nA,B(eV/A)=0.7969663466119443, 0.7969663466969760
       !     sigma(eV/A)=1.347555778200326, 1.347555778194726; sigma_avg(eV/A)=1.347555778197526
       ! _4: gamma_nA,B(eV/A)=0.8026562460117934, 0.8026562460180963
       !     sigma(eV/A)=1.214095473038613, 1.214095473038397; sigma_avg(eV/A)=1.214095473038505
  file='bampl400x400eps03g05w03aAB05bAB0noise0' ! Lx=2048,tmax=10^7; _1,_4: same sign of zeta_DAB for 2 interfaces
       ! _1: gamma_nA,B(eV/A)=0.7969663396406974, 0.7969663541770061
       !     sigma(eV/A)=1.347555773529631, 1.347555773550504; sigma_avg(eV/A)=1.347555773540068
       ! _4: gamma_nA,B(eV/A)=0.8026562457240588, 0.8026562462267081
       !     sigma(eV/A)=1.214095507651008, 1.214095507651016; sigma_avg(eV/A)=1.214095507651012
  ! armchair: theta=90 degree (_4, interface oriented along x)
  ! zigzag: theta=0 degree (_1, interface oriented along y)
  
  nA0_s=-0.2682240335169304d0 ! at tmax=10^7, L=2048
  nA0_l=-0.4435164100577305d0
  A1_s=CMPLX(-8.7833886739079561d-2,-0.2081951877739002d0)
  A2_s=CMPLX(-0.2197898142019807d0,-5.2463915154257496d-2)
  A3_s=CMPLX(0.1063354260412832d0,-0.1993810548686150d0)
  B1_s=CMPLX(-9.4460100511589082d-2,0.2052737617858650d0)
  B2_s=CMPLX(0.1095166345985139d0,0.1976515407502470d0)
  B3_s=CMPLX(-0.2189209317958431d0,5.5979045079558511d-2)

  if(i_theta.eq.0) then ! _1
    !nA0_s=-0.2682240267782519d0 ! at tmax=10^7, Lx=512
    !nA0_l=-0.4435164077168946d0
    !A1_s=CMPLX(-8.7833888186763867d-2,-0.2081951907328163d0)
    !A2_s=CMPLX(-0.2197897860826280d0,-5.2463957420903942d-2)
    !A3_s=CMPLX(0.1063354275974040d0,-0.1993810577661679d0)
    !B1_s=CMPLX(-9.4460101349764639d-2,0.2052737650337667d0)
    !B2_s=CMPLX(0.1095166667429461d0,0.1976515029031947d0)
    !B3_s=CMPLX(-0.2189209348939724d0,5.5979046287830070d-2)
    nA0_s=-0.2682240335169304d0 ! at tmax=10^7, Lx=2048
    nA0_l=-0.4435164100577305d0
    A1_s=CMPLX(-8.7833886739079561d-2,-0.2081951877739002d0)
    A2_s=CMPLX(-0.2197898142019807d0,-5.2463915154257496d-2)
    A3_s=CMPLX(0.1063354260412832d0,-0.1993810548686150d0)
    B1_s=CMPLX(-9.4460100511589082d-2,0.2052737617858650d0)
    B2_s=CMPLX(0.1095166345985139d0,0.1976515407502470d0)
    B3_s=CMPLX(-0.2189209317958431d0,5.5979045079558511d-2)
  !* note: ini amplitudes need to be complex
  !        otherwise ini solid region melts and resolidifies with domains (undulating ampls, n0)
    # ifdef ini_tanh
      !wid_n=6.4945552734d0/2.5123056d0 ! width/(2*atanh(width_per)), width_per=0.85; use width_nA(1)
      !wid_A=7.4713937980d0/2.5123056d0 ! use width_A(1) ! at tmax=10^7, Lx=512
      wid_n=6.4953616398d0/2.5123056d0 ! width/(2*atanh(width_per)), width_per=0.85; use width_nA(1)
      wid_A=7.4711083227d0/2.5123056d0 ! use width_A(1) ! at tmax=10^7, Lx=2048
    # endif
  endif

  if(i_theta.eq.90) then ! _4
    !nA0_s=-0.2682240848947746d0 ! at tmax=10^7, Lx=512
    !nA0_l=-0.4435164451836281d0
    !A1_s=CMPLX(-8.7833887170658681d-2,-0.2081951788275782d0)
    !A2_s=CMPLX(-0.2197898102873447d0,-5.2463896518067529d-2)
    !A3_s=CMPLX(0.1063353980911257d0,-0.1993810034300522d0)
    !B1_s=CMPLX(-9.4460102249663669d-2,0.2052737521105319d0)
    !B2_s=CMPLX(0.1095166163656826d0,0.1976515415670090d0)
    !B3_s=CMPLX(-0.2189208766383217d0,5.5979030531853054d-2)
    nA0_s=-0.2682240752088968d0 ! at tmax=10^7, Lx=2048
    nA0_l=-0.4435164435660391d0
    A1_s=CMPLX(-8.7833881982373824d-2,-0.2081951765861624d0)
    A2_s=CMPLX(-0.2197898115052828d0,-5.2463895116867998d-2)
    A3_s=CMPLX(0.1063354123209136d0,-0.1993810340288277d0)
    B1_s=CMPLX(-9.4460100515079734d-2,0.2052737484152882d0)
    B2_s=CMPLX(0.1095166182875313d0,0.1976515414844259d0)
    B3_s=CMPLX(-0.2189209079654175d0,5.5979038035728526d-2)
    # ifdef ini_tanh
        !wid_n=6.5233310268d0/2.5123056d0 ! width/(2*atanh(width_per)), width_per=0.85; use width_nA(1)
        !wid_A=1.0817743105d1/2.5123056d0 ! use width_A(1) ! at tmax=10^7, Lx=512
        wid_n=6.5237979053d0/2.5123056d0 ! width/(2*atanh(width_per)), width_per=0.85; use width_nA(1)
        wid_A=1.0817029688d1/2.5123056d0 ! use width_A(1) ! at tmax=10^7, Lx=2048
    # endif
  endif
     
  nB0_s=nA0_s
  nB0_l=nA0_l
  A_l=0
  B_l=0
  
  ! Ken's 2D ampl results:
  !nA0_s=-0.2674999833
  !nA0_l=-0.4435000122
  !feq=(0.258303076+0.03768087551)/2.0 equilibrium bulk energy density
  !gamma = 0.80272 or 0.79703 eV/A at theta=90 or 0 (per l/s interface) from 2D (quasi 1D) code
  !gamma = 1.214077 eV/A at theta=90 from anaLytic formula based on 1D ampl

# endif ! ifdef liq_sol
  
  filename=file

  if(1.eq.1) then ! assign n1,n2, sin_theta, cos_theta
    if(i_theta.eq.0) then
      n1=0
      n2=2
      sin_theta=0
      cos_theta=1
    endif
    if(i_theta.eq.60) then
      n1=1
      n2=1
      sin_theta=sqrt(3.d0)/2
      cos_theta=0.5d0
    endif
    if(i_theta.eq.-60) then
      n1=-1
      n2=1
      sin_theta=-sqrt(3.d0)/2
      cos_theta=0.5d0
    endif
    if(i_theta.eq.90) then
      n1=2
      n2=0
      sin_theta=1
      cos_theta=0
    endif
    if(i_theta.eq.30) then
      n1=1
      n2=3
      sin_theta=0.5d0
      cos_theta=sqrt(3.d0)/2
    endif
    if(i_theta.eq.-30) then
      n1=-1
      n2=3
      sin_theta=-0.5d0
      cos_theta=sqrt(3.d0)/2
    endif
    if(i_theta.eq.19) then ! direction a1, theta=19.1066 degree ! for surface tension and kinetic coefficient onLy
      n1=1
      n2=5
      sin_theta=sqrt(3.d0/7)/2
      cos_theta=sqrt(1-sin_theta*sin_theta)
    endif
    if(i_theta.eq.-19) then ! direction a2, theta=-19.1066 degree
      n1=-1
      n2=5
      sin_theta=-sqrt(3.d0/7)/2
      cos_theta=sqrt(1-sin_theta*sin_theta)
    endif
    if(i_theta.eq.41) then ! direction a3, theta=40.8934 degree
      n1=2
      n2=4
      sin_theta=sqrt(3.d0/7)
      cos_theta=sqrt(1-sin_theta*sin_theta)
    endif
    if(i_theta.eq.-41) then ! direction a4, theta=-40.8934 degree
      n1=-2
      n2=4
      sin_theta=-sqrt(3.d0/7)
      cos_theta=sqrt(1-sin_theta*sin_theta)
    endif
    if(i_theta.eq.79) then ! direction a5, theta=79.1066 degree
      n1=3
      n2=1
      sin_theta=1.5d0*sqrt(3.d0/7)
      cos_theta=sqrt(1-sin_theta*sin_theta)
    endif
    if(i_theta.eq.-79) then ! direction a6, theta=-79.1066 degree
      n1=-3
      n2=1
      sin_theta=-1.5d0*sqrt(3.d0/7)
      cos_theta=sqrt(1-sin_theta*sin_theta)
    endif
    if(i_theta.eq.11) then ! direction b1, theta=10.8934 degree
      n1=1
      n2=9
      sin_theta=1/(sqrt(7.d0)*2)
      cos_theta=sqrt(1-sin_theta*sin_theta)
    endif
    if(i_theta.eq.-11) then ! direction b2, theta=-10.8934 degree
      n1=-1
      n2=9
      sin_theta=-1/(sqrt(7.d0)*2)
      cos_theta=sqrt(1-sin_theta*sin_theta)
    endif
    if(i_theta.eq.49) then ! direction b3, theta=49.1066 degree
      n1=4
      n2=6
      sin_theta=2/sqrt(7.d0)
      cos_theta=sqrt(1-sin_theta*sin_theta)
    endif
    if(i_theta.eq.-49) then ! direction b4, theta=-49.1066 degree
      n1=-4
      n2=6
      sin_theta=-2/sqrt(7.d0)
      cos_theta=sqrt(1-sin_theta*sin_theta)
    endif
    if(i_theta.eq.71) then ! direction b5, theta=70.8934 degree
      n1=5
      n2=3
      sin_theta=2.5d0/sqrt(7.d0)
      cos_theta=sqrt(1-sin_theta*sin_theta)
    endif
    if(i_theta.eq.-71) then ! direction b6, theta=-70.8934 degree
      n1=-5
      n2=3
      sin_theta=-2.5d0/sqrt(7.d0)
      cos_theta=sqrt(1-sin_theta*sin_theta)
    endif
    if(i_theta.eq.23) then ! direction c1, theta=23.4132 degree
      n1=2
      n2=8
      sin_theta=sqrt(3.d0/19)
      cos_theta=sqrt(1-sin_theta*sin_theta)
    endif
    if(i_theta.eq.-23) then ! direction c2, theta=-23.4132 degree
      n1=-2
      n2=8
      sin_theta=-sqrt(3.d0/19)
      cos_theta=sqrt(1-sin_theta*sin_theta)
    endif
    if(i_theta.eq.37) then ! direction c3, theta=36.5868 degree
      n1=3
      n2=7
      sin_theta=1.5d0*sqrt(3.d0/19)
      cos_theta=sqrt(1-sin_theta*sin_theta)
    endif
    if(i_theta.eq.-37) then ! direction c4, theta=-36.5868 degree
      n1=-3
      n2=7
      sin_theta=-1.5d0*sqrt(3.d0/19)
      cos_theta=sqrt(1-sin_theta*sin_theta)
    endif
    if(i_theta.eq.83) then ! direction c5, theta=83.4132 degree
      n1=5
      n2=1
      sin_theta=2.5d0*sqrt(3.d0/19)
      cos_theta=sqrt(1-sin_theta*sin_theta)
    endif
    if(i_theta.eq.-83) then ! direction c6, theta=-83.4132 degree
      n1=-5
      n2=1
      sin_theta=-2.5d0*sqrt(3.d0/19)
      cos_theta=sqrt(1-sin_theta*sin_theta)
    endif
    if(i_theta.eq.5) then ! direction d1, theta=5.2087 degree
      n1=1
      n2=19
      sin_theta=0.5d0*sqrt(3.d0/91)
      cos_theta=sqrt(1-sin_theta*sin_theta)
    endif
    if(i_theta.eq.14) then ! direction e1, theta=13.8979 degree
      n1=1
      n2=7
      sin_theta=0.5d0*sqrt(3.d0/13)
      cos_theta=sqrt(1-sin_theta*sin_theta)
    endif
    if(i_theta.eq.8) then ! direction f1, theta=7.5891 degree
      n1=1
      n2=13
      sin_theta=0.5d0*sqrt(3.d0/43)
      cos_theta=sqrt(1-sin_theta*sin_theta)
    endif
    if(i_theta.eq.9) then ! direction g1, theta=8.9483 degree
      n1=1
      n2=11
      sin_theta=0.5d0*sqrt(3.d0/31)
      cos_theta=sqrt(1-sin_theta*sin_theta)
    endif
    if(i_theta.eq.16) then ! direction h1, theta= 16.1021 degree
      n1=2
      n2=12
      sin_theta=1/sqrt(13.d0)
      cos_theta=sqrt(1-sin_theta*sin_theta)
    endif
    if(i_theta.eq.17) then ! direction i1, theta= 17.4802 degree
      n1=4
      n2=22
      sin_theta=2.d0*sqrt(3.d0/133)
      cos_theta=sqrt(1-sin_theta*sin_theta)
    endif
    if(i_theta.eq.21) then ! direction j1, theta= 20.6330 degree
      n1=5
      n2=23
      sin_theta=2.5d0*sqrt(3.d0/151)
      cos_theta=sqrt(1-sin_theta*sin_theta)
    endif
    if(i_theta.eq.22) then ! direction k1, theta= 21.7868 degree
      n1=3
      n2=13
      sin_theta=1.5d0*sqrt(3.d0)/7
      cos_theta=sqrt(1-sin_theta*sin_theta)
    endif
    if(i_theta.eq.25) then ! direction l1, theta= 25.2850 degree
      n1=3
      n2=11
      sin_theta=1.5d0*sqrt(3.d0/37)
      cos_theta=sqrt(1-sin_theta*sin_theta)
    endif
  endif

  if(i_theta.eq.0.or.i_theta.eq.60.or.i_theta.eq.-60) i_p=0
  if(i_theta.eq.90.or.i_theta.eq.30.or.i_theta.eq.-30) i_p=1

  if(abs(i_theta).eq.19.or.abs(i_theta).eq.41.or.abs(i_theta).eq.79) i_p=-1
  if(abs(i_theta).eq.11.or.abs(i_theta).eq.49.or.abs(i_theta).eq.71) i_p=-2
  if(abs(i_theta).eq.23.or.abs(i_theta).eq.37.or.abs(i_theta).eq.83) i_p=-3
  if(abs(i_theta).eq.5.or.abs(i_theta).eq.55.or.abs(i_theta).eq.65) i_p=-4
  if(abs(i_theta).eq.14.or.abs(i_theta).eq.46.or.abs(i_theta).eq.74) i_p=-5
  if(abs(i_theta).eq.8.or.abs(i_theta).eq.52.or.abs(i_theta).eq.68) i_p=-6
  if(abs(i_theta).eq.9.or.abs(i_theta).eq.51.or.abs(i_theta).eq.69) i_p=-7
  if(abs(i_theta).eq.16.or.abs(i_theta).eq.44.or.abs(i_theta).eq.76) i_p=-8
  if(abs(i_theta).eq.17.or.abs(i_theta).eq.43.or.abs(i_theta).eq.77) i_p=-9
  if(abs(i_theta).eq.21.or.abs(i_theta).eq.39.or.abs(i_theta).eq.81) i_p=-10
  if(abs(i_theta).eq.22.or.abs(i_theta).eq.38.or.abs(i_theta).eq.82) i_p=-11
  if(abs(i_theta).eq.25.or.abs(i_theta).eq.35.or.abs(i_theta).eq.85) i_p=-12

  if(i_theta.eq.0) filename=file//'_1' ! direction 1, along y; width_per=0.85
  if(i_theta.eq.90) filename=file//'_4' ! direction 4, along x; width_per=0.85
  if(i_theta.eq.19) filename=file//'_a1' ! direction a1, theta=19.1066 degree
  if(i_theta.eq.11) filename=file//'_b1' ! direction b1, theta=10.8934 degree
  if(i_theta.eq.23) filename=file//'_c1' ! direction c1, theta=23.4132 degree
  if(i_theta.eq.5) filename=file//'_d1' ! direction d1, theta=5.2087 degree
  if(i_theta.eq.14) filename=file//'_e1' ! direction e1, theta=13.8979 degree
  if(i_theta.eq.8) filename=file//'_f1' ! direction f1, theta=7.5891 degree
  if(i_theta.eq.9) filename=file//'_g1' ! direction g1, theta= 8.9483 degree
  if(i_theta.eq.16) filename=file//'_h1' ! direction h1, theta= 16.1021 degree
  if(i_theta.eq.17) filename=file//'_i1' ! direction i1, theta= 17.4802 degree
  if(i_theta.eq.21) filename=file//'_j1' ! direction j1, theta= 20.6330 degree
  if(i_theta.eq.22) filename=file//'_k1' ! direction k1, theta= 21.7868 degree
  if(i_theta.eq.25) filename=file//'_l1' ! direction l1, theta= 25.2850 degree

! discretization of time & space

  qx0=0.5d0*sqrt(3.d0)*q0
  qy0=q0

  delta_x=qx0*eps_m ! for overlayer/substrate
  delta_y=q0*eps_m
!  delta_y=-q0*eps_m/3 ! for nonzero misfit eps_m in solid; Poisson relaxation (Poisson ratio=1/3)
!  delta_y=0 ! for test of y growth (theta=0); cause nonzero phase in the solid bulk

  deltaj0(2)=delta_y*(delta_y+2*q0)
  deltaj0(1)=(sqrt(3.d0)*q0+delta_x)*delta_x+deltaj0(2)/4
  deltaj0(3)=deltaj0(1)

  qxd=qx0+delta_x
  qyd=qy0+delta_y
  dx=1 ! if smaller dx, need smaller dt to converge
  dy=1

  chi_xj2(1)=(-2*qxd*cos_theta+qyd*sin_theta)**2/2 ! chi_xj^2/2 or beta_j^2/2 in the notes
  chi_xj2(2)=4*qyd*qyd*sin_theta*sin_theta/2
  chi_xj2(3)=(2*qxd*cos_theta+qyd*sin_theta)**2/2
  
  dchi_xj(1)= 2*qxd*sin_theta+qyd*cos_theta ! dchi_xj/dtheta
  dchi_yj(1)=-2*qxd*cos_theta+qyd*sin_theta
  
  dchi_xj(2)=-2*qyd*cos_theta
  dchi_yj(2)=-2*qyd*sin_theta
  
  dchi_xj(3)=-2*qxd*sin_theta+qyd*cos_theta
  dchi_yj(3)= 2*qxd*cos_theta+qyd*sin_theta

!  noise=0.1d0 ! noise percentage
!  noise0=0.01d0 ! for 0 values
  ! noise=0
  ! noise0=0

  idum=-796921

!  dt=0.01d0
!  dt=0.02d0
!  dt=0.05d0
  ! dt=0.1d0
!  dt=0.2d0 ! diverge for eps=0.3

  ! t_i=1 ! better for t_i/dt=integer
  ! dti=0.01d0 ! used for initial transient up to t=t_i

  ! tmax=1000
  nt=10
!  tmax=10000
!  nt=100
!  tmax=1.d5
!  nt=1000
!  tmax=1.d6
!  nt=10000
  ! tmax=1.d7
  ! nt=10000

  nimpc_v=NINT((tmax-time0)/nt)

  n_i=NINT(t_i/dti)
  n_0=NINT(t_i/dt)-n_i ! for output
  nend=NINT((tmax-t_i)/dt)+n_i
!3  nout_v=(nend-iter0+1)/nimpc_v
3  nout_v=NINT((tmax-time0)/dt)/nimpc_v
  if(nout_v.le.1) then
     nimpc_v=nimpc_v/2
     goto 3
  endif
!  raout=nout_v/10
  raout=nout_v
!  raout=nout_v*5
  nout_conf=raout*10

! for initial film configuration
  i_width=Lx/2
!  i_width=Lx/4

  i1=Lx/2-i_width/2
  i2=Lx/2+i_width/2
  Lx_mid=(i1+i2)/2

!  A_top=A_l+(1-width_per)*(A1_s-A_l)/2
  A_top=abs(A_l)+(1-width_per)*(abs(A1_s)-abs(A_l))

! nmbr_eval: number of iterations used in the predictor-corrector algorithm
  nmbr_eval=1 ! no predictor-corrector iteration
!  nmbr_eval=100
  TOL=1.d-3
!  err_s=TOL/2
  err_s=TOL/10

! for output of amplitudes and phases
  itime=0
  n_time=5
  itime(1)=nt
  itime(2)=NINT(tmax/4) ! to make sure as integer of nt (nout)
  itime(3)=NINT(tmax/2)
  !itime(4)=NINT(tmax*0.7d0)
  itime(4)=NINT(tmax*0.75d0)
  itime(5)=NINT(tmax*0.9d0)
  !itime(5)=tmax-nt
  itime=NINT((itime-t_i)/dt)+n_i
  do i=1,15
     if(itime(i).eq.nend) itime(i)=nend-nout_v ! to avoid itime overlap with tmax
  enddo

  ! parameters output
  open(unit=2,file=filename//'_para.dat')
  write(2,*) 'For base state: 1D amplitude equations of Aj^0(u,t),Bj^0(u,t),nA0^0(u,t),nB0^0(u,t)'
  write(2,*) '    n0_reduced: neglect gradient terms of n0'
# ifdef liq_sol
  write(2,*) 'def liq_sol: liquid+solid configuration'
# endif
  write(2,*)
  write(2,*) 'Lx=',Lx,'  Ly=',Ly
  write(2,*) 'epsA=',epsA, '  epsB=',epsB
  write(2,*) 'q0=',q0
  write(2,*) 'qA=',qA, '  qB=',qB
  write(2,*) 'misfit: eps_m=',eps_m
  write(2,*)
  write(2,*) 'alpha_AB=',alpha_AB
  write(2,*) 'beta_A=',beta_A, '  beta_B=',beta_B
  write(2,*) 'gA=',gA, '  gB=',gB
  write(2,*) 'vA=',vA, '  vB=',vB
  write(2,*) 'w=',w, '  u=',u
  write(2,*) 'beta_AB=',beta_AB, '  qAB=',qAB
  write(2,*) 'mA=1,  mB=M_B/M_A=',mB
  write(2,*)
  write(2,*) 'dt=',dt, '  dx=',dx, '  dy=',dy
  write(2,*) 'interface orientation theta=',dasin(sin_theta)*180/pi
  write(2,*) '                    i_theta=',i_theta
  write(2,*) 'n1=',n1,  '  n2=',n2
  write(2,*)
  write(2,*) 'qx0=',qx0, '   qy0=', qy0
  write(2,*) 'qxd=',qxd, '   qyd=', qyd
  write(2,*) 'delta_x=',delta_x, '  delta_y=',delta_y
  write(2,*)
  write(2,*) 'tmax=',tmax, 'nend=',nend
  write(2,*) 't_transient=',t_i, '  dti=',dti
  write(2,*) 'nimpc_v=',nimpc_v, '  nout_v=',nout_v, '  raout=',raout
  write(2,*) '  data output at t=t_transient, and every ',raout*dt
  write(2,*) 'time0=',time0, '  iter0=',iter0, '  idum=',idum
  write(2,*) 'for interface position and velocity: A_top=',A_top
  write(2,*) 'output of amplitudes and phases:'
  write(2,*) '  at t=', (itime(1:n_time)-n_i)*dt+t_i,tmax
# ifdef save_conf
  write(2,*) '  save conf files every ',nout_conf*dt
# else
  write(2,*) '  ndef save_conf'
# endif
  write(2,*)
  write(2,*) 'Initial conditions:'
  write(2,*) 'nA0_s=',nA0_s, '  nA0_l=',nA0_l
  write(2,*) 'nB0_s=',nB0_s, '  nB0_l=',nB0_l
  write(2,*) 'A1_s=',A1_s
  write(2,*) 'A2_s=',A2_s
  write(2,*) 'A3_s=',A3_s
  write(2,*) 'B1_s=',B1_s
  write(2,*) 'B2_s=',B2_s
  write(2,*) 'B3_s=',B3_s
  write(2,*) 'A_l=',A_l
  write(2,*) 'B_l=',B_l
# ifdef ini_tanh 
  if(ntype.eq.1) then
     write(2,*) 'def ini_tanh: use initial tanh profile'
     write(2,*) 'wid_n=',wid_n, '  wid_A=',wid_A
  endif
# else
  write(2,*) 'ndef ini_tanh'
# endif
  write(2,*)
  write(2,*) 'initial solid/substrate from u=',i1,'  to',i2
  write(2,*) '   with i_width=',i_width
  write(2,*) 'interface range: width_per=',width_per
  write(2,*) 'noise=',noise,'  (if n0=0) noise0=',noise0
  write(2,*)
  write(2,*) 'For the predictor-corrector method:'
  write(2,*) '    Maximum number of iterations:   nmbr_eval=',nmbr_eval
  write(2,*) '    TOL=',TOL,'  err_s=',err_s
  write(2,*) 'ntype=',ntype
  if(ntype.eq.2) write(2,*) 'init_cond = ',init_cond
# ifdef open_MP 
  write(2,*) 'use openMP for multi-threading'
# else
  write(2,*) 'no openMP'
# endif
  close(2)

  !complex(8) :: A0(3,Ly,Lx),A0q(3,Ly,Lx),B0(3,Ly,Lx),B0q(3,Ly,Lx)
  p_alloc_A=fftw_alloc_complex(int(3*Ly*Lx, C_SIZE_T))
  call c_f_pointer(p_alloc_A, A0, [3,Ly,Lx])
  p_alloc_Aq=fftw_alloc_complex(int(3*Ly*Lx, C_SIZE_T))
  call c_f_pointer(p_alloc_Aq, A0q, [3,Ly,Lx])
  p_alloc_A=fftw_alloc_complex(int(3*Ly*Lx, C_SIZE_T))
  call c_f_pointer(p_alloc_A, B0, [3,Ly,Lx])
  p_alloc_Aq=fftw_alloc_complex(int(3*Ly*Lx, C_SIZE_T))
  call c_f_pointer(p_alloc_Aq, B0q, [3,Ly,Lx])
  !complex(8) :: A_tmp(Ly,Lx),Aq_tmp(Ly,Lx)
  p_alloc_A=fftw_alloc_complex(int(Ly*Lx, C_SIZE_T))
  call c_f_pointer(p_alloc_A, A_tmp, [Ly,Lx])
  p_alloc_Aq=fftw_alloc_complex(int(Ly*Lx, C_SIZE_T))
  call c_f_pointer(p_alloc_Aq, Aq_tmp, [Ly,Lx])
  
  !real(8) :: nA0(Ly,Lx),nA0q(Ly,Lx),nB0(Ly,Lx),nB0q(Ly,Lx)
  p_alloc_n=fftw_alloc_real(int(Ly*Lx, C_SIZE_T))
  call c_f_pointer(p_alloc_n, nA0, [Ly,Lx])
  p_alloc_nq=fftw_alloc_complex(int((Ly/2+1)*Lx, C_SIZE_T))
  call c_f_pointer(p_alloc_nq, nA0q, [Ly/2+1,Lx])
  p_alloc_n=fftw_alloc_real(int(Ly*Lx, C_SIZE_T))
  call c_f_pointer(p_alloc_n, nB0, [Ly,Lx])
  p_alloc_nq=fftw_alloc_complex(int((Ly/2+1)*Lx, C_SIZE_T))
  call c_f_pointer(p_alloc_nq, nB0q, [Ly/2+1,Lx])
  !real(8) :: n_tmp(Ly,Lx)
  p_alloc_n=fftw_alloc_real(int(Ly*Lx, C_SIZE_T))
  call c_f_pointer(p_alloc_n, n_tmp, [Ly,Lx])
  
! for 1D FFTW
  scale2D_b=1.d0/Ly/Lx ! for backward FFT

  do ix=1,Lx
    if (ix.le.Lx/2+1) then
      qx(ix)=2*pi*(ix-1)/(Lx*dx)
    else
      qx(ix)=-2*pi*(Lx-ix+1)/(Lx*dx)
    endif
  enddo
  do iy=1,Ly
    if (iy.le.Ly/2+1) then
       qy(iy)=2*pi*(iy-1)/(Ly*dy)
    else
       qy(iy)=-2*pi*(Ly-iy+1)/(Ly*dy)
    endif
  enddo
   
# ifdef open_MP
  ! use multi-threaded fftw with openMP
  nthreads=omp_get_max_threads()
  iret=fftw_init_threads()
  if(iret.eq.0) then
     write(*,*) 'iret=0: error during thread initialization'
     stop
  endif
  call fftw_plan_with_nthreads(nthreads)
  open(unit=2,file=filename//'_para.dat',position='append')
  write(2,*) 'use multi-threaded fftw; nthreads=',nthreads
  close(2)
# endif
  
! initialization for FFTW (out of place; create the plans before initializing the inputs)
  plan2d_c_f=fftw_plan_dft_2d(Lx,Ly,A_tmp,Aq_tmp,FFTW_FORWARD,FFTW_MEASURE)
  plan2d_c_b=fftw_plan_dft_2d(Lx,Ly,Aq_tmp,A_tmp,FFTW_BACKWARD,FFTW_MEASURE)
  plan2d_r_f=fftw_plan_dft_r2c_2d(Lx,Ly,nA0,nA0q,FFTW_MEASURE)
  plan2d_r_b=fftw_plan_dft_c2r_2d(Lx,Ly,nA0q,nA0,FFTW_MEASURE)

  yy=ran3(idum)
! set the initial conditions
  
  if(ntype.eq.1) then
    !
    ! put the initial configuration by hand 
# ifdef ini_tanh ! use initial tanh profile
  do ix=1,Lx/2
    do iy=1,Ly
      A0(1,iy,ix)=((A1_s+A_l)/2+(A1_s-A_l)/2*tanh((ix-i1)*dx/wid_A))*(1+noise*(ran3(idum)-0.5d0))
      A0(2,iy,ix)=((A2_s+A_l)/2+(A2_s-A_l)/2*tanh((ix-i1)*dx/wid_A))*(1+noise*(ran3(idum)-0.5d0))
      A0(3,iy,ix)=((A3_s+A_l)/2+(A3_s-A_l)/2*tanh((ix-i1)*dx/wid_A))*(1+noise*(ran3(idum)-0.5d0))
      B0(1,iy,ix)=((B1_s+B_l)/2+(B1_s-B_l)/2*tanh((ix-i1)*dx/wid_A))*(1+noise*(ran3(idum)-0.5d0))
      B0(2,iy,ix)=((B2_s+B_l)/2+(B2_s-B_l)/2*tanh((ix-i1)*dx/wid_A))*(1+noise*(ran3(idum)-0.5d0))
      B0(3,iy,ix)=((B3_s+B_l)/2+(B3_s-B_l)/2*tanh((ix-i1)*dx/wid_A))*(1+noise*(ran3(idum)-0.5d0))
      nA0(iy,ix)=((nA0_s+nA0_l)/2+(nA0_s-nA0_l)/2*tanh((ix-i1)*dx/wid_n))*(1+noise*(ran3(idum)-0.5d0))
      nB0(iy,ix)=((nB0_s+nB0_l)/2+(nB0_s-nB0_l)/2*tanh((ix-i1)*dx/wid_n))*(1+noise*(ran3(idum)-0.5d0))
    enddo ! do iy=1,Ly
  enddo ! do ix=1,Lx/2
  do ix=Lx/2+1,Lx
    do iy=1,Ly
      A0(1,iy,ix)=((A1_s+A_l)/2+(A_l-A1_s)/2*tanh((ix-i2)*dx/wid_A))*(1+noise*(ran3(idum)-0.5d0))
      A0(2,iy,ix)=((A2_s+A_l)/2+(A_l-A2_s)/2*tanh((ix-i2)*dx/wid_A))*(1+noise*(ran3(idum)-0.5d0))
      A0(3,iy,ix)=((A3_s+A_l)/2+(A_l-A3_s)/2*tanh((ix-i2)*dx/wid_A))*(1+noise*(ran3(idum)-0.5d0))
      B0(1,iy,ix)=((B1_s+B_l)/2+(B_l-B1_s)/2*tanh((ix-i2)*dx/wid_A))*(1+noise*(ran3(idum)-0.5d0))
      B0(2,iy,ix)=((B2_s+B_l)/2+(B_l-B2_s)/2*tanh((ix-i2)*dx/wid_A))*(1+noise*(ran3(idum)-0.5d0))
      B0(3,iy,ix)=((B3_s+B_l)/2+(B_l-B3_s)/2*tanh((ix-i2)*dx/wid_A))*(1+noise*(ran3(idum)-0.5d0))
      nA0(iy,ix)=((nA0_s+nA0_l)/2+(nA0_l-nA0_s)/2*tanh((ix-i2)*dx/wid_n))*(1+noise*(ran3(idum)-0.5d0))
      nB0(iy,ix)=((nB0_s+nB0_l)/2+(nB0_l-nB0_s)/2*tanh((ix-i2)*dx/wid_n))*(1+noise*(ran3(idum)-0.5d0))
    enddo ! do iy=1,Ly
  enddo ! do ix=Lx/2+1,Lx
# else
  do ix=1,Lx
!        u=(ix-1)*dx
# ifdef liq_sol
      ! for liquid_solid
    if(ix.ge.i1.and.ix.le.i2) then ! solid region
# endif
      do iy=1,Ly
        A0(1,iy,ix)=A1_s*(1+noise*(ran3(idum)-0.5d0))
        A0(2,iy,ix)=A2_s*(1+noise*(ran3(idum)-0.5d0))
        A0(3,iy,ix)=A3_s*(1+noise*(ran3(idum)-0.5d0))
        B0(1,iy,ix)=B1_s*(1+noise*(ran3(idum)-0.5d0))
        B0(2,iy,ix)=B2_s*(1+noise*(ran3(idum)-0.5d0))
        B0(3,iy,ix)=B3_s*(1+noise*(ran3(idum)-0.5d0))
        if(abs(nA0_s).lt.1.d-10) then
          nA0(iy,ix)=nA0_s+noise0*(ran3(idum)-0.5d0) ! if nA0_s=0
        else
          !nA0(iy,ix)=nA0_s+noise*(ran3(idum)-0.5d0)
          nA0(iy,ix)=nA0_s*(1+noise*(ran3(idum)-0.5d0))
        endif
        if(abs(nB0_s).lt.1.d-10) then
          nB0(iy,ix)=nB0_s+noise0*(ran3(idum)-0.5d0) ! if nB0_s=0
        else
          !nB0(iy,ix)=nB0_s+noise*(ran3(idum)-0.5d0)
          nB0(iy,ix)=nB0_s*(1+noise*(ran3(idum)-0.5d0))
        endif
      enddo
# ifdef liq_sol
      ! for liquid_solid
    else ! liquid region
      do iy=1,Ly
        if(abs(nA0_l).lt.1.d-10) then
          nA0(iy,ix)=nA0_l+noise0*(ran3(idum)-0.5d0) ! if nA0_l=0
        else
          !nA0(iy,ix)=nA0_l+noise*(ran3(idum)-0.5d0)
          nA0(iy,ix)=nA0_l*(1+noise*(ran3(idum)-0.5d0))
        endif
        if(abs(nB0_l).lt.1.d-10) then
          nB0(iy,ix)=nB0_l+noise0*(ran3(idum)-0.5d0) ! if nB0_l=0
        else
          !nB0(iy,ix)=nB0_l+noise*(ran3(idum)-0.5d0)
          nB0(iy,ix)=nB0_l*(1+noise*(ran3(idum)-0.5d0))
        endif
        !if(abs(A_l).lt.1.d-10) then ! liquid film
        do j=1,3
          !A0(j,iy,ix)=CMPLX(A_l+noise0*(ran3(idum)-0.5d0),noise0*(ran3(idum)-0.5d0))
          !B0(j,iy,ix)=CMPLX(B_l+noise0*(ran3(idum)-0.5d0),noise0*(ran3(idum)-0.5d0))
          A0(j,iy,ix)=A_l+noise0*(ran3(idum)-0.5d0)
          B0(j,iy,ix)=B_l+noise0*(ran3(idum)-0.5d0)
        enddo
      enddo
    endif
# endif
    enddo
# endif ! ifdef ini_tanh
     
  else
     ! read configuration from file
     open(unit=1,file=init_cond,form='unformatted',status='old')
     read(1) A0,B0,nA0,nB0
     close(1)
  endif

  do j=1,3
    A_tmp=A0(j,1:Ly,1:Lx)
    call fftw_execute_dft(plan2d_c_f,A_tmp,Aq_tmp) ! forward FFT (complex)
    do ix=1,Lx
      do iy=1,Ly
        A0q(j,iy,ix)=Aq_tmp(iy,ix)
      enddo
    enddo
    A_tmp=B0(j,1:Ly,1:Lx)
    call fftw_execute_dft(plan2d_c_f,A_tmp,Aq_tmp) ! forward FFT (complex)
    do ix=1,Lx
      do iy=1,Ly
        B0q(j,iy,ix)=Aq_tmp(iy,ix)
      enddo
    enddo
  enddo
  n_tmp=nA0
  call fftw_execute_dft_r2c(plan2d_r_f,n_tmp,nA0q) ! forward FFT (real)
  n_tmp=nB0
  call fftw_execute_dft_r2c(plan2d_r_f,n_tmp,nB0q) ! forward FFT (real)
  if(ntype.eq.1) then
     if(abs(nA0_s-nB0_s).lt.1.d-10.and.abs(nA0_l-nB0_l).lt.1.d-10) then
        nB0q(1:Ly,1)=nA0q(1:Ly,1) ! fix me !
        call fftw_execute_dft_c2r(plan2d_r_b,nB0q,nB0) ! backward FFT (real)
        nB0=nB0*scale2D_b
        n_tmp=nB0
        call fftw_execute_dft_r2c(plan2d_r_f,n_tmp,nB0q) ! forward FFT (real)
     endif
  endif

  time=time0
  call interface(A_top,ystar1A,ystar2A,Lx_mid)
  ystar1A_0=ystar1A
  ystar2A_0=ystar2A
  open(unit=7,file=filename//'_ystar.dat',position='append')
  write(7,*) time, ystar1A, ystar2A
  close(7)
  call phase_output(filename//'_phase_t0.dat',filename//'_A0B0n0_t0.dat')

  open(unit=9,file=filename//'_out.dat',position='append')
  write(9,*) 't=',time
  write(9,*) '<nA0>=',sum(nA0)*scale2d_b,REAL(nA0q(1,1))*scale2d_b
  write(9,*) '<nB0>=',sum(nB0)*scale2d_b,REAL(nB0q(1,1))*scale2d_b
  write(9,*) 'A0=',maxval(abs(A0(1:3,1:Ly,1:Lx))),minval(abs(A0(1:3,1:Ly,1:Lx)))
  write(9,*) 'B0=',maxval(abs(B0(1:3,1:Ly,1:Lx))),minval(abs(B0(1:3,1:Ly,1:Lx)))
  write(9,*) 'nA0=', maxval(nA0(1:Ly,1:Lx)),minval(nA0(1:Ly,1:Lx))
  write(9,*) 'nB0=', maxval(nB0(1:Ly,1:Lx)),minval(nB0(1:Ly,1:Lx))
  close(9)

  if(iter0.le.n_i) then
     dtime=dti ! for initial transient
  else
     dtime=dt
  endif

  q02_mB=q02*mB
  isigma=0
  isigmaj=0
  do ix=1,Lx
    do iy=1,Ly
      q2(iy,ix)=qx(ix)**2+qy(iy)**2

      ! eq(13)
      !G_0(1,iy,ix)=-q2(iy,ix)+(2*qxd*sin_theta+qyd*cos_theta)*qx(ix)+(-2*qxd*cos_theta+qyd*sin_theta)*qy(iy)
      G_0(1,iy,ix)=-q2(iy,ix)+dchi_xj(1)*qx(ix)+dchi_yj(1)*qy(iy)
      !G_0(2,iy,ix)=-q2(iy,ix)+(-2*qyd*cos_theta)*qx(ix)+(-2*qyd*sin_theta)*qy(iy)
      G_0(2,iy,ix)=-q2(iy,ix)+dchi_xj(2)*qx(ix)+dchi_yj(2)*qy(iy)
      !G_0(3,iy,ix)=-q2(iy,ix)+(-2*qxd*sin_theta+qyd*cos_theta)*qx(ix)+(2*qxd*cos_theta+qyd*sin_theta)*qy(iy)
      G_0(3,iy,ix)=-q2(iy,ix)+dchi_xj(3)*qx(ix)+dchi_yj(3)*qy(iy)
    
! for A_j, B_j
      do j=1,3
        alpha_11=-q02*(beta_A*(G_0(j,iy,ix)+qA2-q02)**2-epsA)
        alpha_22=-q02_mB*(beta_B*(G_0(j,iy,ix)+qB2-q02)**2-epsB)
        alpha_j12(j,iy,ix)=-q02*(alpha_AB+beta_AB*(G_0(j,iy,ix)+qAB2-q02)**2)
        alpha_j21(j,iy,ix)=mB*alpha_j12(j,iy,ix)
        
        b=(alpha_11-alpha_22)**2+4*alpha_j12(j,iy,ix)*alpha_j21(j,iy,ix)
        if(b.lt.0) then
           open(unit=9,file=filename//'_out.dat',position='append')
           write(9,*) 'error: \Delta(Aj,Bj) < 0 at ix=',ix,'  iy=',iy,'  qx(ix)=',qx(ix),'  qy(iy)=',qy(iy),'  for j=',j
           close(9)
           stop
        endif
        delta_2=sqrt(b) ! note: delta_2 is real here
        if(abs(delta_2).lt.1.d-8) then ! sigma1=sigma2
           isigmaj(j,iy,ix)=1
           sigmaj1(j,iy,ix)=(alpha_11+alpha_22)/2
           sigmaj2(j,iy,ix)=sigmaj1(j,iy,ix)
           sigj1_alpha(j,iy,ix)=sigmaj1(j,iy,ix)-alpha_11

           sig1_dt=sigmaj1(j,iy,ix)*dtime
           exp_j11(j,iy,ix)=exp(sig1_dt)*(1-sigj1_alpha(j,iy,ix)*dtime)
           exp_j12(j,iy,ix)=exp(sig1_dt)*alpha_j12(j,iy,ix)*dtime
           exp_j21(j,iy,ix)=exp(sig1_dt)*alpha_j21(j,iy,ix)*dtime
           exp_j22(j,iy,ix)=exp(sig1_dt)*(1+sigj1_alpha(j,iy,ix)*dtime)
           if(abs(sigmaj1(j,iy,ix)).lt.1.d-4) then
              cf_j1(j,iy,ix)=dtime*(1.d0+0.5d0*sig1_dt*(1.d0+sig1_dt/3.d0))
              cf2_j1(j,iy,ix)=0.5d0*dtime*(1.d0+sig1_dt*(1.d0+0.25d0*sig1_dt)/3.d0)
              cf_j2(j,iy,ix)=dtime*dtime*(0.5d0+sig1_dt/3+sig1_dt*sig1_dt/8)
              cf2_j2(j,iy,ix)=dtime*dtime*(1.d0/6.d0+sig1_dt/12+sig1_dt*sig1_dt/40)
           else
              cf_j1(j,iy,ix)=(exp(sig1_dt)-1)/sigmaj1(j,iy,ix)
              cf2_j1(j,iy,ix)=(exp(sig1_dt)-(1+sig1_dt))/(sigmaj1(j,iy,ix)*sig1_dt)
              cf_j2(j,iy,ix)=(1+exp(sig1_dt)*(sig1_dt-1))/(sigmaj1(j,iy,ix)*sigmaj1(j,iy,ix))
              cf2_j2(j,iy,ix)=(2+sig1_dt+(sig1_dt-2)*exp(sig1_dt))/(sigmaj1(j,iy,ix)*sigmaj1(j,iy,ix)*sig1_dt)
           endif

        else !sigma1!=sigma2
          ! use the algorithm of Numerical Recipes: Sec. 5.6
          b=-(alpha_11+alpha_22)
          c=alpha_11*alpha_22-alpha_j12(j,iy,ix)*alpha_j21(j,iy,ix)
          if(b.ge.0) then ! note: here b is real in this case
              aq=-(b+delta_2)/2
          else
              aq=-(b-delta_2)/2
          endif
          sigmaj1(j,iy,ix)=aq
          sigmaj2(j,iy,ix)=c/aq

          sigmaj12(j,iy,ix)=sigmaj1(j,iy,ix)-sigmaj2(j,iy,ix)
          sigj1_alpha(j,iy,ix)=sigmaj1(j,iy,ix)-alpha_22
          sigj2_alpha(j,iy,ix)=sigmaj2(j,iy,ix)-alpha_22

          sig1_dt=sigmaj1(j,iy,ix)*dtime
          sig2_dt=sigmaj2(j,iy,ix)*dtime
          exp_j11(j,iy,ix)=sigj1_alpha(j,iy,ix)*exp(sig1_dt)-sigj2_alpha(j,iy,ix)*exp(sig2_dt)
          exp_j12(j,iy,ix)=alpha_j12(j,iy,ix)*(exp(sig1_dt)-exp(sig2_dt))
          exp_j21(j,iy,ix)=alpha_j21(j,iy,ix)*(exp(sig1_dt)-exp(sig2_dt))
          exp_j22(j,iy,ix)=sigj1_alpha(j,iy,ix)*exp(sig2_dt)-sigj2_alpha(j,iy,ix)*exp(sig1_dt)
          if(abs(sigmaj1(j,iy,ix)).lt.1.d-4) then
            cf_j1(j,iy,ix)=dtime*(1.d0+0.5d0*sig1_dt*(1.d0+sig1_dt/3.d0))
            cf2_j1(j,iy,ix)=0.5d0*dtime*(1.d0+sig1_dt*(1.d0+0.25d0*sig1_dt)/3.d0)
          else
            cf_j1(j,iy,ix)=(exp(sig1_dt)-1)/sigmaj1(j,iy,ix)
            cf2_j1(j,iy,ix)=(exp(sig1_dt)-(1+sig1_dt))/(sigmaj1(j,iy,ix)*sig1_dt)
          endif
          if(abs(sigmaj2(j,iy,ix)).lt.1.d-4) then
            cf_j2(j,iy,ix)=dtime*(1.d0+0.5d0*sig2_dt*(1.d0+sig2_dt/3.d0))
            cf2_j2(j,iy,ix)=0.5d0*dtime*(1.d0+sig2_dt*(1.d0+0.25d0*sig2_dt)/3.d0)
          else
            cf_j2(j,iy,ix)=(exp(sig2_dt)-1)/sigmaj2(j,iy,ix)
            cf2_j2(j,iy,ix)=(exp(sig2_dt)-(1+sig2_dt))/(sigmaj2(j,iy,ix)*sig2_dt)
          endif
        endif

      enddo ! do j=1,3
    enddo ! do iy=1,Ly
  enddo ! do ix=1,Lx

! for nA0 and nB0 (note: dft_r2c kind, with q2(iy,ix) already matching that of the 2nd half for imaginary part)
  do ix=1,Lx
    do iy=1,Ly/2+1
      alpha_11=-q2(iy,ix)*(beta_A*qA2*qA2-epsA) ! if neglecting gradients in n0; mA=1
      alpha_22=-mB*q2(iy,ix)*(beta_B*qB2*qB2-epsB)
      alpha_12(iy,ix)=-q2(iy,ix)*(alpha_AB+beta_AB*qAB2*qAB2)
      alpha_21(iy,ix)=mB*alpha_12(iy,ix)

      b=(alpha_11-alpha_22)**2+4*alpha_12(iy,ix)*alpha_21(iy,ix)
      if(b.lt.0) then
          open(unit=9,file=filename//'_out.dat',position='append')
          write(9,*) 'error: \Delta(n0) < 0 at ix=',ix, '  iy=',iy, '  qx(ix)=',qx(ix), '  qy(iy)=',qy(iy)
          close(9)
          stop
      endif
      delta_2=sqrt(b) ! note: delta_2 is real here
      if(abs(delta_2).lt.1.d-8) then ! sigma1=sigma2
        isigma(iy,ix)=1
        sigma1(iy,ix)=(alpha_11+alpha_22)/2
        sigma2(iy,ix)=sigma1(iy,ix)
        sig1_alpha(iy,ix)=sigma1(iy,ix)-alpha_11

        sig1_dt=sigma1(iy,ix)*dtime
        exp_11(iy,ix)=exp(sig1_dt)*(1-sig1_alpha(iy,ix)*dtime)
        exp_12(iy,ix)=exp(sig1_dt)*alpha_12(iy,ix)*dtime
        exp_21(iy,ix)=exp(sig1_dt)*alpha_21(iy,ix)*dtime
        exp_22(iy,ix)=exp(sig1_dt)*(1+sig1_alpha(iy,ix)*dtime)
        if(abs(sigma1(iy,ix)).lt.1.d-4) then
          cf_1(iy,ix)=dtime*(1.d0+0.5d0*sig1_dt*(1.d0+sig1_dt/3.d0))
          cf2_1(iy,ix)=0.5d0*dtime*(1.d0+sig1_dt*(1.d0+0.25d0*sig1_dt)/3.d0)
          cf_2(iy,ix)=dtime*dtime*(0.5d0+sig1_dt/3+sig1_dt*sig1_dt/8)
          cf2_2(iy,ix)=dtime*dtime*(1.d0/6.d0+sig1_dt/12+sig1_dt*sig1_dt/40)
        else
          cf_1(iy,ix)=(exp(sig1_dt)-1)/sigma1(iy,ix)
          cf2_1(iy,ix)=(exp(sig1_dt)-(1+sig1_dt))/(sigma1(iy,ix)*sig1_dt)
          cf_2(iy,ix)=(1+exp(sig1_dt)*(sig1_dt-1))/(sigma1(iy,ix)*sigma1(iy,ix))
          cf2_2(iy,ix)=(2+sig1_dt+(sig1_dt-2)*exp(sig1_dt))/(sigma1(iy,ix)*sigma1(iy,ix)*sig1_dt)
        endif

      else  ! use the algorithm of Numerical Recipes: Sec. 5.6

        b=-(alpha_11+alpha_22)
        c=alpha_11*alpha_22-alpha_12(iy,ix)*alpha_21(iy,ix)
        if(b.ge.0) then ! note: here b is real in this case
          aq=-(b+delta_2)/2
        else
          aq=-(b-delta_2)/2
        endif
        sigma1(iy,ix)=aq
        sigma2(iy,ix)=c/aq

        sigma12(iy,ix)=sigma1(iy,ix)-sigma2(iy,ix)
        sig1_alpha(iy,ix)=sigma1(iy,ix)-alpha_22
        sig2_alpha(iy,ix)=sigma2(iy,ix)-alpha_22

        sig1_dt=sigma1(iy,ix)*dtime
        sig2_dt=sigma2(iy,ix)*dtime
        exp_11(iy,ix)=sig1_alpha(iy,ix)*exp(sig1_dt)-sig2_alpha(iy,ix)*exp(sig2_dt)
        exp_12(iy,ix)=alpha_12(iy,ix)*(exp(sig1_dt)-exp(sig2_dt))
        exp_21(iy,ix)=alpha_21(iy,ix)*(exp(sig1_dt)-exp(sig2_dt))
        exp_22(iy,ix)=sig1_alpha(iy,ix)*exp(sig2_dt)-sig2_alpha(iy,ix)*exp(sig1_dt)
        if(abs(sigma1(iy,ix)).lt.1.d-4) then
          cf_1(iy,ix)=dtime*(1.d0+0.5d0*sig1_dt*(1.d0+sig1_dt/3.d0))
          cf2_1(iy,ix)=0.5d0*dtime*(1.d0+sig1_dt*(1.d0+0.25d0*sig1_dt)/3.d0)
        else
          cf_1(iy,ix)=(exp(sig1_dt)-1)/sigma1(iy,ix)
          cf2_1(iy,ix)=(exp(sig1_dt)-(1+sig1_dt))/(sigma1(iy,ix)*sig1_dt)
        endif
        if(abs(sigma2(iy,ix)).lt.1.d-4) then
          cf_2(iy,ix)=dtime*(1.d0+0.5d0*sig2_dt*(1.d0+sig2_dt/3.d0))
          cf2_2(iy,ix)=0.5d0*dtime*(1.d0+sig2_dt*(1.d0+0.25d0*sig2_dt)/3.d0)
        else
          cf_2(iy,ix)=(exp(sig2_dt)-1)/sigma2(iy,ix)
          cf2_2(iy,ix)=(exp(sig2_dt)-(1+sig2_dt))/(sigma2(iy,ix)*sig2_dt)
        endif
      endif

    enddo ! do iy=1,Ly/2+1
  enddo ! do ix=1,Lx

! start running
!
  i_ystar=0 ! for liquid/film interfaces
  do iter=iter0,nend

    if(iter.le.n_i) then ! for initial transient onLy
      time=iter*dti
    else
      time=(iter-n_i)*dt+t_i
      if(iter.eq.n_i+1) then
        dtime=dt
        do ix=1,Lx
          do iy=1,Ly
        ! for A_j and B_j
            do j=1,3
              if(isigmaj(j,iy,ix).eq.1) then ! sigma1=sigma2
                 sig1_dt=sigmaj1(j,iy,ix)*dtime
                 exp_j11(j,iy,ix)=exp(sig1_dt)*(1-sigj1_alpha(j,iy,ix)*dtime)
                 exp_j12(j,iy,ix)=exp(sig1_dt)*alpha_j12(j,iy,ix)*dtime
                 exp_j21(j,iy,ix)=exp(sig1_dt)*alpha_j21(j,iy,ix)*dtime
                 exp_j22(j,iy,ix)=exp(sig1_dt)*(1+sigj1_alpha(j,iy,ix)*dtime)
                 if(abs(sigmaj1(j,iy,ix)).lt.1.d-4) then
                    cf_j1(j,iy,ix)=dtime*(1.d0+0.5d0*sig1_dt*(1.d0+sig1_dt/3.d0))
                    cf2_j1(j,iy,ix)=0.5d0*dtime*(1.d0+sig1_dt*(1.d0+0.25d0*sig1_dt)/3.d0)
                    cf_j2(j,iy,ix)=dtime*dtime*(0.5d0+sig1_dt/3+sig1_dt*sig1_dt/8)
                    cf2_j2(j,iy,ix)=dtime*dtime*(1.d0/6.d0+sig1_dt/12+sig1_dt*sig1_dt/40)
                 else
                    cf_j1(j,iy,ix)=(exp(sig1_dt)-1)/sigmaj1(j,iy,ix)
                    cf2_j1(j,iy,ix)=(exp(sig1_dt)-(1+sig1_dt))/(sigmaj1(j,iy,ix)*sig1_dt)
                    cf_j2(j,iy,ix)=(1+exp(sig1_dt)*(sig1_dt-1))/(sigmaj1(j,iy,ix)*sigmaj1(j,iy,ix))
                    cf2_j2(j,iy,ix)=(2+sig1_dt+(sig1_dt-2)*exp(sig1_dt))/(sigmaj1(j,iy,ix)*sigmaj1(j,iy,ix)*sig1_dt)
                 endif
              else
                 sig1_dt=sigmaj1(j,iy,ix)*dtime
                 sig2_dt=sigmaj2(j,iy,ix)*dtime
                 exp_j11(j,iy,ix)=sigj1_alpha(j,iy,ix)*exp(sig1_dt)-sigj2_alpha(j,iy,ix)*exp(sig2_dt)
                 exp_j12(j,iy,ix)=alpha_j12(j,iy,ix)*(exp(sig1_dt)-exp(sig2_dt))
                 exp_j21(j,iy,ix)=alpha_j21(j,iy,ix)*(exp(sig1_dt)-exp(sig2_dt))
                 exp_j22(j,iy,ix)=sigj1_alpha(j,iy,ix)*exp(sig2_dt)-sigj2_alpha(j,iy,ix)*exp(sig1_dt)
                 if(abs(sigmaj1(j,iy,ix)).lt.1.d-4) then
                    cf_j1(j,iy,ix)=dtime*(1.d0+0.5d0*sig1_dt*(1.d0+sig1_dt/3.d0))
                    cf2_j1(j,iy,ix)=0.5d0*dtime*(1.d0+sig1_dt*(1.d0+0.25d0*sig1_dt)/3.d0)
                 else
                    cf_j1(j,iy,ix)=(exp(sig1_dt)-1)/sigmaj1(j,iy,ix)
                    cf2_j1(j,iy,ix)=(exp(sig1_dt)-(1+sig1_dt))/(sigmaj1(j,iy,ix)*sig1_dt)
                 endif
                 if(abs(sigmaj2(j,iy,ix)).lt.1.d-4) then
                    cf_j2(j,iy,ix)=dtime*(1.d0+0.5d0*sig2_dt*(1.d0+sig2_dt/3.d0))
                    cf2_j2(j,iy,ix)=0.5d0*dtime*(1.d0+sig2_dt*(1.d0+0.25d0*sig2_dt)/3.d0)
                 else
                    cf_j2(j,iy,ix)=(exp(sig2_dt)-1)/sigmaj2(j,iy,ix)
                    cf2_j2(j,iy,ix)=(exp(sig2_dt)-(1+sig2_dt))/(sigmaj2(j,iy,ix)*sig2_dt)
                 endif
              endif
            enddo ! do j=1,3
          enddo ! do iy=1,Ly
        enddo ! do ix=1,Lx
           
! for nA0 and nB0
        do ix=1,Lx
          do iy=1,Ly/2+1
            if(isigma(iy,ix).eq.1) then ! sigma1=sigma2
              sig1_dt=sigma1(iy,ix)*dtime
              exp_11(iy,ix)=exp(sig1_dt)*(1-sig1_alpha(iy,ix)*dtime)
              exp_12(iy,ix)=exp(sig1_dt)*alpha_12(iy,ix)*dtime
              exp_21(iy,ix)=exp(sig1_dt)*alpha_21(iy,ix)*dtime
              exp_22(iy,ix)=exp(sig1_dt)*(1+sig1_alpha(iy,ix)*dtime)
              if(abs(sigma1(iy,ix)).lt.1.d-4) then
                cf_1(iy,ix)=dtime*(1.d0+0.5d0*sig1_dt*(1.d0+sig1_dt/3.d0))
                cf2_1(iy,ix)=0.5d0*dtime*(1.d0+sig1_dt*(1.d0+0.25d0*sig1_dt)/3.d0)
                cf_2(iy,ix)=dtime*dtime*(0.5d0+sig1_dt/3+sig1_dt*sig1_dt/8)
                cf2_2(iy,ix)=dtime*dtime*(1.d0/6.d0+sig1_dt/12+sig1_dt*sig1_dt/40)
              else
                cf_1(iy,ix)=(exp(sig1_dt)-1)/sigma1(iy,ix)
                cf2_1(iy,ix)=(exp(sig1_dt)-(1+sig1_dt))/(sigma1(iy,ix)*sig1_dt)
                cf_2(iy,ix)=(1+exp(sig1_dt)*(sig1_dt-1))/(sigma1(iy,ix)*sigma1(iy,ix))
                cf2_2(iy,ix)=(2+sig1_dt+(sig1_dt-2)*exp(sig1_dt))/(sigma1(iy,ix)*sigma1(iy,ix)*sig1_dt)
              endif
            else
              sig1_dt=sigma1(iy,ix)*dtime
              sig2_dt=sigma2(iy,ix)*dtime
              exp_11(iy,ix)=sig1_alpha(iy,ix)*exp(sig1_dt)-sig2_alpha(iy,ix)*exp(sig2_dt)
              exp_12(iy,ix)=alpha_12(iy,ix)*(exp(sig1_dt)-exp(sig2_dt))
              exp_21(iy,ix)=alpha_21(iy,ix)*(exp(sig1_dt)-exp(sig2_dt))
              exp_22(iy,ix)=sig1_alpha(iy,ix)*exp(sig2_dt)-sig2_alpha(iy,ix)*exp(sig1_dt)
              if(abs(sigma1(iy,ix)).lt.1.d-4) then
                cf_1(iy,ix)=dtime*(1.d0+0.5d0*sig1_dt*(1.d0+sig1_dt/3.d0))
                cf2_1(iy,ix)=0.5d0*dtime*(1.d0+sig1_dt*(1.d0+0.25d0*sig1_dt)/3.d0)
              else
                cf_1(iy,ix)=(exp(sig1_dt)-1)/sigma1(iy,ix)
                cf2_1(iy,ix)=(exp(sig1_dt)-(1+sig1_dt))/(sigma1(iy,ix)*sig1_dt)
              endif
              if(abs(sigma2(iy,ix)).lt.1.d-4) then
                cf_2(iy,ix)=dtime*(1.d0+0.5d0*sig2_dt*(1.d0+sig2_dt/3.d0))
                cf2_2(iy,ix)=0.5d0*dtime*(1.d0+sig2_dt*(1.d0+0.25d0*sig2_dt)/3.d0)
              else
                cf_2(iy,ix)=(exp(sig2_dt)-1)/sigma2(iy,ix)
                cf2_2(iy,ix)=(exp(sig2_dt)-(1+sig2_dt))/(sigma2(iy,ix)*sig2_dt)
              endif
            endif

          enddo ! do iy=1,Ly/2+1
        enddo ! do ix=1,Lx
      endif
    endif

    call An(time,iter,istep,filename)

    # ifdef debug 
    ! output to console "End An: time=xxx iter=xxx"
    write(*,*) 'Debug-End An: time=',time,' iter=',iter
    # endif

    if(((iter.eq.n_i).or.(iter.gt.n_i.and.mod(iter+n_0,nout_v).eq.0)).and.(i_ystar.eq.0)) then
      call interface(A_top,ystar1A,ystar2A,Lx_mid)
      if((ystar1A.le.1.or.ystar1A.ge.Lx_mid).and.(ystar2A.ge.Lx.or.ystar2A.le.Lx_mid)) then
         i_ystar=1
         goto 110
      endif
      vb1A=(ystar1A_0-ystar1A)*dx/dtime
      vb2A=(ystar2A-ystar2A_0)*dx/dtime
      open(unit=10,file=filename//'_vb.dat',position='append')
      write(10,*) time,vb1A,vb2A
      close(10)
      open(unit=7,file=filename//'_ystar.dat',position='append')
      write(7,*) time, ystar1A, ystar2A
      close(7)
    endif
    if(((iter+1.eq.n_i).or.(iter.gt.n_i.and.mod(iter+n_0+1,nout_v).eq.0)).and.(i_ystar.eq.0)) then 
      call interface(A_top,ystar1A_0,ystar2A_0,Lx_mid)
    endif
!
110 continue

    if((iter.lt.10).or.(iter.eq.n_i).or.(iter.gt.n_i.and.mod(iter+n_0,raout).eq.0)) then
      open(unit=9,file=filename//'_out.dat',position='append')
      write(9,*)
      write(9,*) 't=',time,'  istep=',istep
      write(9,*) '<nA0>=',sum(nA0)*scale2d_b,REAL(nA0q(1,1))*scale2d_b
      write(9,*) '<nB0>=',sum(nB0)*scale2d_b,REAL(nB0q(1,1))*scale2d_b
      write(9,*) 'A0=',maxval(abs(A0(1:3,1:Ly,1:Lx))),minval(abs(A0(1:3,1:Ly,1:Lx)))
      write(9,*) 'B0=',maxval(abs(B0(1:3,1:Ly,1:Lx))),minval(abs(B0(1:3,1:Ly,1:Lx)))
      write(9,*) 'nA0=', maxval(nA0(1:Ly,1:Lx)),minval(nA0(1:Ly,1:Lx))
      write(9,*) 'nB0=', maxval(nB0(1:Ly,1:Lx)),minval(nB0(1:Ly,1:Lx))
      close(9)

      if(iter.eq.1) then
        open(unit=8,file=filename//'_width.dat',position='append')
        write(8,*) '# time,epsA,epsB,width_nA(1,2),width_nB(1,2),m_gap_nA(1,2),m_gap_nB(1,2),width_A(1,2),width_B(1,2)'
        close(8)
        open(unit=9,file=filename//'_surf.dat',position='append')
        write(9,*) '# time,epsA,epsB,gamma_nA,gamma_nB,enx,enl,Es,(nA0_avg-nA0_l)*(nB0_s-nB0_avg)-(nB0_avg-nB0_l)*(nA0_s-nA0_avg)'
        close(9)
      endif
      call surface_tension(time,filename) ! calculate surface tension directLy from simulation (not anaLytic eq)

! output amplitudes and phases
      do j=1,n_time
        if(iter.eq.itime(j)) then
          ! write(cha,'(i0)') NINT(time)
          write(cha, '(F0.3)') time
          call phase_output(filename//'_phase_t'//trim(cha)//'.dat',filename//'_A0B0n0_t'//trim(adjustl(cha))//'.dat')
          open(unit=8,file=filename//'_out.dat',position='append')
          write(8,*) 'Output amplitudes and phases at time=',time
          write(8,*)
          close(8)
        endif
    enddo

# ifdef save_conf ! save conf files
      if((iter.gt.n_i.and.mod(iter+n_0,nout_conf).eq.0).or.(iter.eq.nend)) then
        open(unit=11,file=filename//'_conf.dat',form='unformatted')
        write(11) A0,B0,nA0,nB0
        close(11)
        open(unit=9,file=filename//'_out.dat',position='append')
        write(9,*) 't=',time, '  iter=',iter,' for ouput of conf files'
        close(9)
      endif
# endif
    endif

  enddo ! do iter=iter0,nend

  call phase_output(filename//'_phase.dat',filename//'_A0B0n0.dat')
  if(mod(nend+n_0,raout).ne.0) then
  call surface_tension(time,filename) ! calculate surface tension directLy from simulation (not anaLytic eq)
# ifdef save_conf
  open(unit=11,file=filename//'_conf.dat',form='unformatted')
  write(11) A0,B0,nA0,nB0
  close(11)
  open(unit=9,file=filename//'_out.dat',position='append')
  write(9,*) 't=',time, '  iter=',iter-1,' for ouput of conf files'
  write(9,*)
  close(9)
# endif
  endif

  call fftw_destroy_plan(plan2d_r_f)
  call fftw_destroy_plan(plan2d_r_b)
  call fftw_destroy_plan(plan2d_c_f)
  call fftw_destroy_plan(plan2d_c_b)

  call fftw_free(p_alloc_A)
  call fftw_free(p_alloc_Aq)
  call fftw_free(p_alloc_n)
  call fftw_free(p_alloc_nq)

  deallocate(exp_j11,exp_j12,exp_j21,exp_j22) ! for Aj and Bj
  deallocate(cf_j1,cf2_j1,cf_j2,cf2_j2)
  deallocate(exp_11,exp_12,exp_21,exp_22) ! for nA0 and nB0
  deallocate(cf_1,cf2_1,cf_2,cf2_2)
  deallocate(sigma12,sig1_alpha,sig2_alpha,alpha_12,alpha_21)
  deallocate(sigmaj12,sigj1_alpha,sigj2_alpha,alpha_j12,alpha_j21)
  deallocate(q2)
  deallocate(isigmaj,isigma)
  deallocate(G_0)
  deallocate(sigma1,sigma2)
  deallocate(sigmaj1,sigmaj2)
  deallocate(qx,qy)

end Program bampl2D_AB

! main routine that solves the 0th order amplitude eqs
!
subroutine An(time,iter,istep,file)
  use global_variables
  implicit none

  common/converge/TOL,err_s,nmbr_eval

  real(8) :: time
  integer :: iter,ix,iy,j,nmbr_eval,istep
  real(8) :: TOL,err_s,max_conv_A,max_conv_B,max_conv_nA0,max_conv_nB0,conv,abs_psi
  character file*(*)

  complex(C_DOUBLE_COMPLEX) :: Nonlin_A_q(3,Ly,Lx),sigN_A0q(3,Ly,Lx)
  complex(C_DOUBLE_COMPLEX) :: Nonlin_B_q(3,Ly,Lx),sigN_B0q(3,Ly,Lx)
  real(C_DOUBLE) :: Nonlin1_q(Ly/2+1,Lx),Nonlin2_q(Ly/2+1,Lx)
  complex(C_DOUBLE_COMPLEX) :: A0_0(3,Ly,Lx),A0q_0(3,Ly,Lx),B0_0(3,Ly,Lx),B0q_0(3,Ly,Lx)!,Aq_tmp(Ly,Lx),A_tmp(Ly,Lx)
  real(C_DOUBLE) :: sigN_nA0q(Ly,Lx),sigN_nB0q(Ly,Lx),sigN_1,sigN_2
  complex(C_DOUBLE_COMPLEX) :: sigN_j1,sigN_j2
  complex(C_DOUBLE_COMPLEX), pointer :: Aq_tmp(:,:),A_tmp(:,:),nA0q_0(:,:),nB0q_0(:,:),nq_tmp(:,:)
  real(C_DOUBLE), pointer :: nA0_0(:,:),nB0_0(:,:)!,nA0q_0(:,:),nB0q_0(:,:),nq_tmp(:,:)
  type(C_PTR) :: p_alloc_A_tmp,p_alloc_Aq_tmp
  type(C_PTR) :: p_alloc_nA0_0,p_alloc_nA0q_0,p_alloc_nB0_0,p_alloc_nB0q_0,p_alloc_nq_tmp

  p_alloc_A_tmp=fftw_alloc_complex(int(Ly*Lx, C_SIZE_T))
  call c_f_pointer(p_alloc_A_tmp, A_tmp, [Ly,Lx])
  p_alloc_Aq_tmp=fftw_alloc_complex(int(Ly*Lx, C_SIZE_T))
  call c_f_pointer(p_alloc_Aq_tmp, Aq_tmp, [Ly,Lx])

  p_alloc_nA0_0=fftw_alloc_real(int(Ly*Lx, C_SIZE_T))
  call c_f_pointer(p_alloc_nA0_0, nA0_0, [Ly,Lx])
  p_alloc_nB0_0=fftw_alloc_real(int(Ly*Lx, C_SIZE_T))
  call c_f_pointer(p_alloc_nB0_0, nB0_0, [Ly,Lx])
  p_alloc_nA0q_0=fftw_alloc_complex(int((Ly/2+1)*Lx, C_SIZE_T))
  call c_f_pointer(p_alloc_nA0q_0, nA0q_0, [(Ly/2+1),Lx])
  p_alloc_nB0q_0=fftw_alloc_complex(int((Ly/2+1)*Lx, C_SIZE_T))
  call c_f_pointer(p_alloc_nB0q_0, nB0q_0, [(Ly/2+1),Lx])
  p_alloc_nq_tmp=fftw_alloc_complex(int((Ly/2+1)*Lx, C_SIZE_T))
  call c_f_pointer(p_alloc_nq_tmp, nq_tmp, [(Ly/2+1),Lx])

!  calculate the nonlinear terms
  call nonlin_q(Nonlin_A_q,Nonlin_B_q,Nonlin1_q,Nonlin2_q)

! First step: predictor

!$OMP PARALLEL PRIVATE(ix,iy,j,sigN_1,sigN_2,sigN_j1,sigN_j2)

! for Aj,Bj
  do j=1,3
!$OMP DO
!(Note: DO loop index variable is by default PRIVATE, all other variables are by default SHARED unless declared)
    do ix=1,Lx
      do iy=1,Ly
        if(isigmaj(j,iy,ix).eq.1) then ! sigma1=sigma2
          sigN_j2=alpha_j12(j,iy,ix)*Nonlin_B_q(j,iy,ix)-sigj1_alpha(j,iy,ix)*Nonlin_A_q(j,iy,ix)
          A0q_0(j,iy,ix)=exp_j11(j,iy,ix)*A0q(j,iy,ix)+exp_j12(j,iy,ix)*B0q(j,iy,ix) &
                    +cf_j1(j,iy,ix)*Nonlin_A_q(j,iy,ix)+cf_j2(j,iy,ix)*sigN_j2
          if(nmbr_eval.gt.1) sigN_A0q(j,iy,ix)=-(sigN_j2*cf2_j2(j,iy,ix)+Nonlin_A_q(j,iy,ix)*cf2_j1(j,iy,ix)) ! for use in iteration
          sigN_j2=alpha_j21(j,iy,ix)*Nonlin_A_q(j,iy,ix)+sigj1_alpha(j,iy,ix)*Nonlin_B_q(j,iy,ix)
          B0q_0(j,iy,ix)=exp_j21(j,iy,ix)*A0q(j,iy,ix)+exp_j22(j,iy,ix)*B0q(j,iy,ix) &
                    +cf_j1(j,iy,ix)*Nonlin_B_q(j,iy,ix)+cf_j2(j,iy,ix)*sigN_j2
          if(nmbr_eval.gt.1) sigN_B0q(j,iy,ix)=-(sigN_j2*cf2_j2(j,iy,ix)+Nonlin_B_q(j,iy,ix)*cf2_j1(j,iy,ix)) ! for use in iteration
        else
          sigN_j1=sigj1_alpha(j,iy,ix)*Nonlin_A_q(j,iy,ix)+alpha_j12(j,iy,ix)*Nonlin_B_q(j,iy,ix)
          sigN_j2=sigj2_alpha(j,iy,ix)*Nonlin_A_q(j,iy,ix)+alpha_j12(j,iy,ix)*Nonlin_B_q(j,iy,ix)
          A0q_0(j,iy,ix)=(exp_j11(j,iy,ix)*A0q(j,iy,ix)+exp_j12(j,iy,ix)*B0q(j,iy,ix) &
                    +cf_j1(j,iy,ix)*sigN_j1-cf_j2(j,iy,ix)*sigN_j2)/sigmaj12(j,iy,ix)
          if(nmbr_eval.gt.1) sigN_A0q(j,iy,ix)=(sigN_j2*cf2_j2(j,iy,ix)-sigN_j1*cf2_j1(j,iy,ix))/sigmaj12(j,iy,ix) ! for use in iteration
          sigN_j1=alpha_j21(j,iy,ix)*Nonlin_A_q(j,iy,ix)-sigj2_alpha(j,iy,ix)*Nonlin_B_q(j,iy,ix)
          sigN_j2=alpha_j21(j,iy,ix)*Nonlin_A_q(j,iy,ix)-sigj1_alpha(j,iy,ix)*Nonlin_B_q(j,iy,ix)
          B0q_0(j,iy,ix)=(exp_j21(j,iy,ix)*A0q(j,iy,ix)+exp_j22(j,iy,ix)*B0q(j,iy,ix) &
                    +cf_j1(j,iy,ix)*sigN_j1-cf_j2(j,iy,ix)*sigN_j2)/sigmaj12(j,iy,ix)
          if(nmbr_eval.gt.1) sigN_B0q(j,iy,ix)=(sigN_j2*cf2_j2(j,iy,ix)-sigN_j1*cf2_j1(j,iy,ix))/sigmaj12(j,iy,ix) ! for use in iteration
        endif
      enddo ! do iy=1,Ly
    enddo ! do ix=1,Lx
  !$OMP END DO
  enddo ! do j=1,3

! for nA0,nB0
!$OMP DO
  do ix=1,Lx
    do iy=1,Ly/2+1
      if(isigma(iy,ix).eq.1) then ! sigma1=sigma2
        sigN_2=alpha_12(iy,ix)*Nonlin2_q(iy,ix)-sig1_alpha(iy,ix)*Nonlin1_q(iy,ix)
        nA0q_0(iy,ix)=exp_11(iy,ix)*nA0q(iy,ix)+exp_12(iy,ix)*nB0q(iy,ix) &
                  +cf_1(iy,ix)*Nonlin1_q(iy,ix)+cf_2(iy,ix)*sigN_2
        if(nmbr_eval.gt.1) sigN_nA0q(iy,ix)=-(sigN_2*cf2_2(iy,ix)+Nonlin1_q(iy,ix)*cf2_1(iy,ix)) ! for use in iteration
        sigN_2=alpha_21(iy,ix)*Nonlin1_q(iy,ix)+sig1_alpha(iy,ix)*Nonlin2_q(iy,ix)
        nB0q_0(iy,ix)=exp_21(iy,ix)*nA0q(iy,ix)+exp_22(iy,ix)*nB0q(iy,ix) &
                  +cf_1(iy,ix)*Nonlin2_q(iy,ix)+cf_2(iy,ix)*sigN_2
        if(nmbr_eval.gt.1) sigN_nB0q(iy,ix)=-(sigN_2*cf2_2(iy,ix)+Nonlin2_q(iy,ix)*cf2_1(iy,ix)) ! for use in iteration
      else
        sigN_1=sig1_alpha(iy,ix)*Nonlin1_q(iy,ix)+alpha_12(iy,ix)*Nonlin2_q(iy,ix)
        sigN_2=sig2_alpha(iy,ix)*Nonlin1_q(iy,ix)+alpha_12(iy,ix)*Nonlin2_q(iy,ix)
        nA0q_0(iy,ix)=(exp_11(iy,ix)*nA0q(iy,ix)+exp_12(iy,ix)*nB0q(iy,ix) &
                  +cf_1(iy,ix)*sigN_1-cf_2(iy,ix)*sigN_2)/sigma12(iy,ix)
        if(nmbr_eval.gt.1) sigN_nA0q(iy,ix)=(sigN_2*cf2_2(iy,ix)-sigN_1*cf2_1(iy,ix))/sigma12(iy,ix) ! for use in iteration
        sigN_1=alpha_21(iy,ix)*Nonlin1_q(iy,ix)-sig2_alpha(iy,ix)*Nonlin2_q(iy,ix)
        sigN_2=alpha_21(iy,ix)*Nonlin1_q(iy,ix)-sig1_alpha(iy,ix)*Nonlin2_q(iy,ix)
        nB0q_0(iy,ix)=(exp_21(iy,ix)*nA0q(iy,ix)+exp_22(iy,ix)*nB0q(iy,ix) &
                  +cf_1(iy,ix)*sigN_1-cf_2(iy,ix)*sigN_2)/sigma12(iy,ix)
        if(nmbr_eval.gt.1) sigN_nB0q(iy,ix)=(sigN_2*cf2_2(iy,ix)-sigN_1*cf2_1(iy,ix))/sigma12(iy,ix) ! for use in iteration
      endif
    enddo ! do iy=1,Ly/2+1
  enddo ! do ix=1,Lx
!$OMP END DO

!$OMP END PARALLEL

  if(iter.lt.10) then
    open(unit=9,file=file//'_out.dat',position='append')
    write(9,*)
    write(9,*) 'iter=',iter, '  time=',time
    write(9,*) 'Nonlin_A_q=',maxval(abs(Nonlin_A_q(1:3,1:Ly,1:Lx))),minval(abs(Nonlin_A_q(1:3,1:Ly,1:Lx)))
    write(9,*) 'Nonlin_B_q=',maxval(abs(Nonlin_B_q(1:3,1:Ly,1:Lx))),minval(abs(Nonlin_B_q(1:3,1:Ly,1:Lx)))
    write(9,*) 'Nonlin_nA0_q=',maxval(Nonlin1_q(1:Ly/2+1,1:Lx)),minval(Nonlin1_q(1:Ly/2+1,1:Lx))
    write(9,*) 'Nonlin_nB0_q=',maxval(Nonlin2_q(1:Ly/2+1,1:Lx)),minval(Nonlin2_q(1:Ly/2+1,1:Lx))
    write(9,*) 'A0q=',maxval(abs(A0q(1:3,1:Ly,1:Lx))),minval(abs(A0q(1:3,1:Ly,1:Lx)))
    write(9,*) 'B0q=',maxval(abs(B0q(1:3,1:Ly,1:Lx))),minval(abs(B0q(1:3,1:Ly,1:Lx)))
    write(9,*) 'nA0q=',maxval(REAL(nA0q(1:Ly,1:Lx))),minval(REAL(nA0q(1:Ly,1:Lx)))
    write(9,*) 'nB0q=',maxval(REAL(nB0q(1:Ly,1:Lx))),minval(REAL(nB0q(1:Ly,1:Lx)))
    write(9,*) 'A0q_0=',maxval(abs(A0q_0(1:3,1:Ly,1:Lx))),minval(abs(A0q_0(1:3,1:Ly,1:Lx)))
    write(9,*) 'B0q_0=',maxval(abs(B0q_0(1:3,1:Ly,1:Lx))),minval(abs(B0q_0(1:3,1:Ly,1:Lx)))
    write(9,*) 'nA0q_0=',maxval(REAL(nA0q_0(1:Ly/2+1,1:Lx))),minval(REAL(nA0q_0(1:Ly/2+1,1:Lx)))
    write(9,*) 'nB0q_0=',maxval(REAL(nB0q_0(1:Ly/2+1,1:Lx))),minval(REAL(nB0q_0(1:Ly/2+1,1:Lx)))
    write(9,*) 'nA0q_0(1,1)=',REAL(nA0q_0(1,1)), 'nB0q_0(1,1)=',REAL(nB0q_0(1,1))
    write(9,*) 'sig1_alpha=',maxval(sig1_alpha(1:Ly/2+1,1:Lx)),minval(sig1_alpha(1:Ly/2+1,1:Lx))
    write(9,*) 'sig2_alpha=',maxval(sig2_alpha(1:Ly/2+1,1:Lx)),minval(sig2_alpha(1:Ly/2+1,1:Lx))
    write(9,*) 'alpha_12=',maxval(alpha_12(1:Ly/2+1,1:Lx)),minval(alpha_12(1:Ly/2+1,1:Lx))
    write(9,*) 'alpha_21=',maxval(alpha_21(1:Ly/2+1,1:Lx)),minval(alpha_21(1:Ly/2+1,1:Lx))
    write(9,*) 'sigma12(2:Ly,2:Lx)=',maxval(sigma12(2:Ly/2+1,2:Lx)),minval(sigma12(2:Ly/2+1,2:Lx))
    do ix=2,Lx
      do iy=1,Ly/2+1
        if(abs(sigma12(iy,ix)).le.1.d-10) then
           write(9,*) 'ix=',ix, '  iy=',iy, '  sigma12(iy,ix)=',sigma12(iy,ix)
           write(9,*) 'isigma(iy,ix)=',isigma(iy,ix), '  q2(iy,ix)=',q2(iy,ix)
        endif
      enddo ! do iy=1,Ly
    enddo ! do ix=1,Lx
    write(9,*) 'exp_11=',maxval(exp_11(1:Ly/2+1,1:Lx)),minval(exp_11(1:Ly/2+1,1:Lx))
    write(9,*) 'exp_12=',maxval(exp_12(1:Ly/2+1,1:Lx)),minval(exp_12(1:Ly/2+1,1:Lx))
    write(9,*) 'exp_21=',maxval(exp_21(1:Ly/2+1,1:Lx)),minval(exp_21(1:Ly/2+1,1:Lx))
    write(9,*) 'exp_22=',maxval(exp_22(1:Ly/2+1,1:Lx)),minval(exp_22(1:Ly/2+1,1:Lx))
    write(9,*) 'cf_1=',maxval(cf_1(1:Ly/2+1,1:Lx)),minval(cf_1(1:Ly/2+1,1:Lx))
    write(9,*) 'cf_2=',maxval(cf_2(1:Ly/2+1,1:Lx)),minval(cf_2(1:Ly/2+1,1:Lx))
    write(9,*) 'isigma(2:Ly/2+1,2:Lx)=',maxval(isigma(2:Ly/2+1,2:Lx)),minval(isigma(2:Ly/2+1,2:Lx))
    write(9,*) 'isigmaj(1:3,1:Ly,1:Lx)=',maxval(isigmaj(1:3,1:Ly,1:Lx)),minval(isigmaj(1:3,1:Ly,1:Lx))
    close(9)
  endif
  
  # ifdef debug 
  ! output to console "Line 1451: time=xxx iter=xxx"
  write(*,*) 'Debug-Line 1451: time=',time,' iter=',iter
  # endif

  do j=1,3
    Aq_tmp=A0q_0(j,1:Ly,1:Lx)
    call fftw_execute_dft(plan2d_c_b,Aq_tmp,A_tmp) ! backward FFT (complex)
    do ix=1,Lx
      do iy=1,Ly
        A0_0(j,iy,ix)=A_tmp(iy,ix)*scale2D_b
      enddo ! do iy=1,Ly
    enddo ! do ix=1,Lx
    Aq_tmp=B0q_0(j,1:Ly,1:Lx)
    call fftw_execute_dft(plan2d_c_b,Aq_tmp,A_tmp) ! backward FFT (complex)
    do ix=1,Lx
      do iy=1,Ly
        B0_0(j,iy,ix)=A_tmp(iy,ix)*scale2D_b
      enddo ! do iy=1,Ly
    enddo ! do ix=1,Lx
    # ifdef debug 
    ! output to console "End c2c(Aq_tmp,N_tmp): time=xxx iter=xxx j=xxx"
    write(*,*) 'Debug-End c2c(Aq_tmp,N_tmp): time=',time,' iter=',iter,' j=',j
    # endif
  enddo ! do j=1,3
  # ifdef debug 
  ! output to console "Begin nq_tmp=nA0q_0: time=xxx iter=xxx"
  write(*,*) 'Debug-Begin nq_tmp=nA0q_0: time=',time,' iter=',iter
  # endif
  nq_tmp=nA0q_0
  # ifdef debug 
  ! output to console "WIP nq_tmp=nA0q_0: time=xxx iter=xxx"
  write(*,*) 'Debug-WIP nq_tmp=nA0q_0: time=',time,' iter=',iter
  # endif
  call fftw_execute_dft_c2r(plan2d_r_b,nq_tmp,nA0_0) ! backward FFT (real)
  # ifdef debug 
  ! output to console "WIP c2r(nq_tmp,nA0_0): time=xxx iter=xxx"
  write(*,*) 'Debug-WIP c2r(nq_tmp,nA0_0): time=',time,' iter=',iter
  # endif
  nA0_0=nA0_0*scale2D_b
  # ifdef debug 
  ! output to console "WIP c2r(nq_tmp,nA0_0)-rescale: time=xxx iter=xxx"
  write(*,*) 'Debug-WIP c2r(nq_tmp,nA0_0)-rescale: time=',time,' iter=',iter
  # endif
  nq_tmp=nB0q_0
  call fftw_execute_dft_c2r(plan2d_r_b,nq_tmp,nB0_0) ! backward FFT (real)
  nB0_0=nB0_0*scale2D_b
  # ifdef debug 
  ! output to console "End c2r(nq_tmp, nN0_0): time=xxx iter=xxx"
  write(*,*) 'Debug-End c2r(nq_tmp, nN0_0): time=',time,' iter=',iter
  # endif

  A0=A0_0
  A0q=A0q_0

  B0=B0_0
  B0q=B0q_0

  nA0=nA0_0
  nA0q=nA0q_0

  nB0=nB0_0
  nB0q=nB0q_0

! for use in iteration
  if(nmbr_eval.gt.1) then
     A0q_0=A0q_0+sigN_A0q
     B0q_0=B0q_0+sigN_B0q
     nA0q_0=nA0q_0+sigN_nA0q
     nB0q_0=nB0q_0+sigN_nB0q
  endif
  
! Second step: corrector
  do istep=2,nmbr_eval  ! get the corrector values of A0q,n0q,psi0q by functional iteration
                        ! note: A0q_0,B0q_0,nA0q_0,nB0q_0 remain unchanged during the iteration

! for the nonlinear terms (note: here A0,A0q,B0,B0q,nA0,nA0q,nB0,nB0q are the updated values)
    call nonlin_q(Nonlin_A_q,Nonlin_B_q,Nonlin1_q,Nonlin2_q)

!$OMP PARALLEL

! for Aj,Bj
    do j=1,3
!$OMP DO
!(Note: DO loop index variable is by default PRIVATE, all other variables are by default SHARED unless declared)
!(      thus PRIVATE(ix,iy,j) is by default and not needed to declare here)
      do ix=1,Lx
        do iy=1,Ly
          if(isigmaj(j,iy,ix).eq.1) then ! sigma1=sigma2
            A0q(j,iy,ix)=A0q_0(j,iy,ix)+cf2_j1(j,iy,ix)*Nonlin_A_q(j,iy,ix) &
                          +cf2_j2(j,iy,ix)*(alpha_j12(j,iy,ix)*Nonlin_B_q(j,iy,ix)-sigj1_alpha(j,iy,ix)*Nonlin_A_q(j,iy,ix))
            B0q(j,iy,ix)=B0q_0(j,iy,ix)+cf2_j1(j,iy,ix)*Nonlin_B_q(j,iy,ix) &
                          +cf2_j2(j,iy,ix)*(alpha_j21(j,iy,ix)*Nonlin_A_q(j,iy,ix)+sigj1_alpha(j,iy,ix)*Nonlin_B_q(j,iy,ix))
          else
            A0q(j,iy,ix)=A0q_0(j,iy,ix)+(cf2_j1(j,iy,ix)*(sigj1_alpha(j,iy,ix)*Nonlin_A_q(j,iy,ix)+alpha_j12(j,iy,ix)*Nonlin_B_q(j,iy,ix)) &
                          -cf2_j2(j,iy,ix)*(sigj2_alpha(j,iy,ix)*Nonlin_A_q(j,iy,ix)+alpha_j12(j,iy,ix)*Nonlin_B_q(j,iy,ix)))/sigmaj12(j,iy,ix)
            B0q(j,iy,ix)=B0q_0(j,iy,ix)+(cf2_j1(j,iy,ix)*(alpha_j21(j,iy,ix)*Nonlin_A_q(j,iy,ix)-sigj2_alpha(j,iy,ix)*Nonlin_B_q(j,iy,ix)) &
                          -cf2_j2(j,iy,ix)*(alpha_j21(j,iy,ix)*Nonlin_A_q(j,iy,ix)-sigj1_alpha(j,iy,ix)*Nonlin_B_q(j,iy,ix)))/sigmaj12(j,iy,ix)
          endif
        enddo ! do iy=1,Ly
      enddo ! do ix=1,Lx
  !$OMP END DO
    enddo ! do j=1,3

! for nA0,nB0
!$OMP DO
    do ix=1,Lx
      do iy=1,Ly/2+1
        if(isigma(iy,ix).eq.1) then ! sigma1=sigma2
          nA0q(iy,ix)=nA0q_0(iy,ix)+cf2_1(iy,ix)*Nonlin1_q(iy,ix) &
                      +cf2_2(iy,ix)*(alpha_12(iy,ix)*Nonlin2_q(iy,ix)-sig1_alpha(iy,ix)*Nonlin1_q(iy,ix))
          nB0q(iy,ix)=nB0q_0(iy,ix)+cf2_1(iy,ix)*Nonlin2_q(iy,ix) &
                      +cf2_2(iy,ix)*(alpha_21(iy,ix)*Nonlin1_q(iy,ix)+sig1_alpha(iy,ix)*Nonlin2_q(iy,ix))
        else
          nA0q(iy,ix)=nA0q_0(iy,ix)+(cf2_1(iy,ix)*(sig1_alpha(iy,ix)*Nonlin1_q(iy,ix)+alpha_12(iy,ix)*Nonlin2_q(iy,ix)) &
                      -cf2_2(iy,ix)*(sig2_alpha(iy,ix)*Nonlin1_q(iy,ix)+alpha_12(iy,ix)*Nonlin2_q(iy,ix)))/sigma12(iy,ix)
          nB0q(iy,ix)=nB0q_0(iy,ix)+(cf2_1(iy,ix)*(alpha_21(iy,ix)*Nonlin1_q(iy,ix)-sig2_alpha(iy,ix)*Nonlin2_q(iy,ix)) &
                      -cf2_2(iy,ix)*(alpha_21(iy,ix)*Nonlin1_q(iy,ix)-sig1_alpha(iy,ix)*Nonlin2_q(iy,ix)))/sigma12(iy,ix)
        endif
      enddo ! do iy=1,Ly/2+1
    enddo ! do ix=1,Lx
!$OMP END DO

!$OMP END PARALLEL
     
    do j=1,3
      Aq_tmp=A0q(j,1:Ly,1:Lx)
      call fftw_execute_dft(plan2d_c_b,Aq_tmp,A_tmp) ! backward FFT (complex)
      do ix=1,Lx
        do iy=1,Ly
            A0(j,iy,ix)=A_tmp(iy,ix)*scale2D_b
        enddo ! do iy=1,Ly
      enddo ! do ix=1,Lx
      Aq_tmp=B0q(j,1:Ly,1:Lx)
      call fftw_execute_dft(plan2d_c_b,Aq_tmp,A_tmp) ! backward FFT (complex)
      do ix=1,Lx
        do iy=1,Ly
          B0(j,iy,ix)=A_tmp(iy,ix)*scale2D_b
        enddo ! do iy=1,Ly
      enddo ! do ix=1,Lx
    enddo ! do j=1,3
    nq_tmp=nA0q
    call fftw_execute_dft_c2r(plan2d_r_b,nq_tmp,nA0) ! backward FFT (real)
    nA0=nA0*scale2D_b
    nq_tmp=nB0q
    call fftw_execute_dft_c2r(plan2d_r_b,nq_tmp,nB0) ! backward FFT (real)
    nB0=nB0*scale2D_b

! real space (A0,B0,nA0,nB0) convergence

    max_conv_A=0.d0
    do j=1,3
      do ix=1,Lx
        do iy=1,Ly
          abs_psi=abs(A0(j,iy,ix))
          if(abs_psi.gt.1.d5) then
            open(unit=2,file=file//'_out.dat',position='append')
            write(2,*) 'A0(j,iy,ix) diverges at t=',time,' istep=',istep, &
                        '  j=',j,' ix=',ix,' iy=',iy,' A0=',A0(j,iy,ix)
            close(2)
            stop
          endif
          if(abs_psi.le.err_s) then
            conv=abs(A0(j,iy,ix)-A0_0(j,iy,ix))
          else
            conv=abs(A0(j,iy,ix)-A0_0(j,iy,ix))/abs(A0(j,iy,ix))
          endif
          if(conv.gt.max_conv_A) max_conv_A=conv
        enddo ! do iy=1,Ly
      enddo ! do ix=1,Lx
    enddo ! do j=1,3

    max_conv_B=0.d0
    do j=1,3
      do ix=1,Lx
        do iy=1,Ly
          abs_psi=abs(B0(j,iy,ix))
          if(abs_psi.gt.1.d5) then
            open(unit=2,file=file//'_out.dat',position='append')
            write(2,*) 'B0(j,iy,ix) diverges at t=',time,' istep=',istep, &
                        '  j=',j,' ix=',ix,' iy=',iy,' B0=',B0(j,iy,ix)
            close(2)
            stop
          endif
          if(abs_psi.le.err_s) then
            conv=abs(B0(j,iy,ix)-B0_0(j,iy,ix))
          else
            conv=abs(B0(j,iy,ix)-B0_0(j,iy,ix))/abs(B0(j,iy,ix))
          endif
          if(conv.gt.max_conv_B) max_conv_B=conv
        enddo ! do iy=1,Ly
      enddo ! do ix=1,Lx
    enddo ! do j=1,3
     
    max_conv_nA0=0.d0
    do ix=1,Lx
      do iy=1,Ly
      abs_psi=abs(nA0(iy,ix))
        if(abs_psi.gt.1.d5) then
          open(unit=2,file=file//'_out.dat',position='append')
          write(2,*) 'nA0 diverges at t=',time,' istep=',istep,' ix=',ix,' iy=',iy,' nA0(iy,ix)=',nA0(iy,ix)
          close(2)
          stop
        endif
        if(abs_psi.le.err_s) then
          conv=abs(nA0(iy,ix)-nA0_0(iy,ix))
        else
          conv=abs((nA0(iy,ix)-nA0_0(iy,ix))/nA0(iy,ix))
        endif
        if(conv.gt.max_conv_nA0) max_conv_nA0=conv
      enddo ! do ix=1,Lx
    enddo ! do j=1,3

    max_conv_nB0=0.d0
    do ix=1,Lx
      do iy=1,Ly
        abs_psi=abs(nB0(iy,ix))
        if(abs_psi.gt.1.d5) then
          open(unit=2,file=file//'_out.dat',position='append')
          write(2,*) 'nB0 diverges at t=',time,' istep=',istep,' ix=',ix,' iy=',iy,' nB0(iy,ix)=',nB0(iy,ix)
          close(2)
          stop
        endif
        if(abs_psi.le.err_s) then
          conv=abs(nB0(iy,ix)-nB0_0(iy,ix))
        else
          conv=abs((nB0(iy,ix)-nB0_0(iy,ix))/nB0(iy,ix))
        endif
        if(conv.gt.max_conv_nB0) max_conv_nB0=conv
      enddo ! do iy=1,Ly
    enddo ! do ix=1,Lx

    if((max_conv_A.lt.TOL).and.(max_conv_B.lt.TOL).and.(max_conv_nA0.lt.TOL).and.(max_conv_nB0.lt.TOL)) then
      call fftw_free(p_alloc_A_tmp)
      call fftw_free(p_alloc_Aq_tmp)
      call fftw_free(p_alloc_nA0_0)
      call fftw_free(p_alloc_nA0q_0)
      call fftw_free(p_alloc_nB0_0)
      call fftw_free(p_alloc_nB0q_0)
      call fftw_free(p_alloc_nq_tmp)
      return
    endif

    A0_0=A0
    B0_0=B0
    nA0_0=nA0
    nB0_0=nB0

  enddo ! do istep=2,nmbr_eval

  istep=istep-1
  if(nmbr_eval.gt.2) then
    open(unit=2,file=file//'_out.dat',position='append')
    write(2,*)
    write(2,*) 'Exceeding maximum iterations=',nmbr_eval
    write(2,*) 't=',time,'   TOL=',TOL
    write(2,*) 'max_conv_A=',max_conv_A
    write(2,*) 'A0=',maxval(abs(A0(1:3,1:Ly,1:Lx))),minval(abs(A0(1:3,1:Ly,1:Lx)))
    write(2,*) 'max_conv_B=',max_conv_B
    write(2,*) 'B0=',maxval(abs(B0(1:3,1:Ly,1:Lx))),minval(abs(B0(1:3,1:Ly,1:Lx)))
    write(2,*) 'max_conv_nA0=',max_conv_nA0
    write(2,*) 'nA0=',maxval(nA0(1:Ly,1:Lx)),minval(nA0(1:Ly,1:Lx))
    write(2,*) 'max_conv_nB0=',max_conv_nB0
    write(2,*) 'nB0=',maxval(nB0(1:Ly,1:Lx)),minval(nB0(1:Ly,1:Lx))
    write(2,*)
    close(2)
  endif

  call fftw_free(p_alloc_A_tmp)
  call fftw_free(p_alloc_Aq_tmp)
  call fftw_free(p_alloc_nA0_0)
  call fftw_free(p_alloc_nA0q_0)
  call fftw_free(p_alloc_nB0_0)
  call fftw_free(p_alloc_nB0q_0)
  call fftw_free(p_alloc_nq_tmp)
  
  return
end subroutine An


! calculate the nonlinear terms
subroutine nonlin_q(Nonlin_A_q,Nonlin_B_q,Nonlin_nA0_q,Nonlin_nB0_q)
  use global_variables
  implicit none

  complex(C_DOUBLE_COMPLEX) :: Nonlin_A_q(3,Ly,Lx),Nonlin_B_q(3,Ly,Lx)
  real(C_DOUBLE) :: Nonlin_nA0_q(Ly/2+1,Lx),Nonlin_nB0_q(Ly/2+1,Lx),Nonlin_nA0(Ly,Lx),Nonlin_nB0(Ly,Lx)
  complex(C_DOUBLE_COMPLEX) :: A(Ly,Lx),Aq(Ly,Lx),A12,A13,A23,B12,B13,B23,AB12,AB13,AB23
  real(C_DOUBLE) :: n(Ly,Lx)
  complex(C_DOUBLE_COMPLEX) :: nq(Ly/2+1,Lx)
  real(8) :: A_R,A_I,A02(3),B02(3),B_R,ng6A,ng6B,nAB,uw_nAB,A2,B2
  integer :: ix,iy,j

!$OMP PARALLEL PRIVATE(A_R,A_I,A02,B02,B_R,A23,A13,A12,B12,B13,B23,AB12,AB13,AB23,ng6A,ng6B,nAB,uw_nAB,A2,B2)
!$OMP DO
  ! (Note: PRIVATE(ix,iy,j) is by default and not needed to declare here)
  do ix=1,Lx
    do iy=1,Ly
      do j=1,3
        A_R=REAL(A0(j,iy,ix))
        A_I=AIMAG(A0(j,iy,ix))
        A02(j)=A_R*A_R+A_I*A_I
        A_R=REAL(B0(j,iy,ix))
        A_I=AIMAG(B0(j,iy,ix))
        B02(j)=A_R*A_R+A_I*A_I
      enddo ! do j=1,3
      A23=conjg(A0(2,iy,ix)*A0(3,iy,ix))
      A13=conjg(A0(1,iy,ix)*A0(3,iy,ix))
      A12=conjg(A0(1,iy,ix)*A0(2,iy,ix))
      B23=conjg(B0(2,iy,ix)*B0(3,iy,ix))
      B13=conjg(B0(1,iy,ix)*B0(3,iy,ix))
      B12=conjg(B0(1,iy,ix)*B0(2,iy,ix))
      AB23=conjg(A0(2,iy,ix)*B0(3,iy,ix)+A0(3,iy,ix)*B0(2,iy,ix))
      AB13=conjg(A0(1,iy,ix)*B0(3,iy,ix)+A0(3,iy,ix)*B0(1,iy,ix))
      AB12=conjg(A0(1,iy,ix)*B0(2,iy,ix)+A0(2,iy,ix)*B0(1,iy,ix))

      ng6A=6*vA*nA0(iy,ix)-2*gA
      nAB=nA0(iy,ix)*(3*vA*nA0(iy,ix)-2*gA)+w*nB0(iy,ix)
      uw_nAB=w*nA0(iy,ix)+u*nB0(iy,ix)
      Nonlin_A_q(1,iy,ix)=(nAB+3*vA*(A02(1)+2*(A02(2)+A02(3))))*A0(1,iy,ix)+ng6A*A23+uw_nAB*B0(1,iy,ix)+u*B23+w*AB23
      Nonlin_A_q(2,iy,ix)=(nAB+3*vA*(A02(2)+2*(A02(1)+A02(3))))*A0(2,iy,ix)+ng6A*A13+uw_nAB*B0(2,iy,ix)+u*B13+w*AB13
      Nonlin_A_q(3,iy,ix)=(nAB+3*vA*(A02(3)+2*(A02(1)+A02(2))))*A0(3,iy,ix)+ng6A*A12+uw_nAB*B0(3,iy,ix)+u*B12+w*AB12

      ng6B=6*vB*nB0(iy,ix)-2*gB
      nAB=nB0(iy,ix)*(3*vB*nB0(iy,ix)-2*gB)+u*nA0(iy,ix)
      Nonlin_B_q(1,iy,ix)=(nAB+3*vB*(B02(1)+2*(B02(2)+B02(3))))*B0(1,iy,ix)+ng6B*B23+uw_nAB*A0(1,iy,ix)+w*A23+u*AB23
      Nonlin_B_q(2,iy,ix)=(nAB+3*vB*(B02(2)+2*(B02(1)+B02(3))))*B0(2,iy,ix)+ng6B*B13+uw_nAB*A0(2,iy,ix)+w*A13+u*AB13
      Nonlin_B_q(3,iy,ix)=(nAB+3*vB*(B02(3)+2*(B02(1)+B02(2))))*B0(3,iy,ix)+ng6B*B12+uw_nAB*A0(3,iy,ix)+w*A12+u*AB12

      A_R=real(A0(1,iy,ix)*A0(2,iy,ix)*A0(3,iy,ix))
      A2=A02(1)+A02(2)+A02(3)
      B_R=real(B0(1,iy,ix)*B0(2,iy,ix)*B0(3,iy,ix))
      B2=B02(1)+B02(2)+B02(3)
      A_I=real(A0(1,iy,ix)*conjg(B0(1,iy,ix))+A0(2,iy,ix)*conjg(B0(2,iy,ix))+A0(3,iy,ix)*conjg(B0(3,iy,ix)))

      Nonlin_nA0(iy,ix)=nA0(iy,ix)*nA0(iy,ix)*(vA*nA0(iy,ix)-gA)+(w*nA0(iy,ix)+0.5d0*u*nB0(iy,ix))*nB0(iy,ix)+ng6A*A2+12*vA*A_R+u*B2+2*w*A_I
      Nonlin_nB0(iy,ix)=nB0(iy,ix)*nB0(iy,ix)*(vB*nB0(iy,ix)-gB)+(u*nB0(iy,ix)+0.5d0*w*nA0(iy,ix))*nA0(iy,ix)+ng6B*B2+12*vB*B_R+w*A2+2*u*A_I
    enddo ! do iy=1,Ly
  enddo ! do ix=1,Lx
!$OMP END DO
!$OMP END PARALLEL

  do j=1,3
    A=Nonlin_A_q(j,1:Ly,1:Lx)
    call fftw_execute_dft(plan2d_c_f,A,Aq) ! forward FFT (complex)
    do ix=1,Lx
      do iy=1,Ly
        Nonlin_A_q(j,iy,ix)=Aq(iy,ix)
      enddo ! do iy=1,Ly
    enddo ! do ix=1,Lx
    A=Nonlin_B_q(j,1:Ly,1:Lx)
    call fftw_execute_dft(plan2d_c_f,A,Aq) ! forward FFT (complex)
    do ix=1,Lx
      do iy=1,Ly
        Nonlin_B_q(j,iy,ix)=Aq(iy,ix)
      enddo ! do iy=1,Ly
    enddo ! do ix=1,Lx
  enddo ! do j=1,3

  Nonlin_A_q=-q02*Nonlin_A_q
  Nonlin_B_q=-q02_mB*Nonlin_B_q ! q02_mB = mB*q0^2

  n=Nonlin_nA0
  call fftw_execute_dft_r2c(plan2d_r_f,n,nq) ! forward FFT (real)
  Nonlin_nA0_q=-q2(1:Ly/2+1,1:Lx)*nq
  n=Nonlin_nB0
  call fftw_execute_dft_r2c(plan2d_r_f,n,nq) ! forward FFT (real)
  Nonlin_nB0_q=-mB*q2(1:Ly/2+1,1:Lx)*nq

  return
end subroutine nonlin_q


! to determine film/liquid interfaces: ystar1 (bottom one) and ystar2 (top one)
! note: A0_s > A0_l(f)
subroutine interface(A_top,ystar1A,ystar2A,Lx_mid)
  use global_variables
  implicit none

  real(8) :: psij,psijp
  real(8) :: A_top,ystar1A,ystar2A
  integer :: j,Lx_mid

  ! psijp=sum(abs(A0(1:3,1)))/3
  psijp=sum(abs(A0(1:3,1:Ly,1)))/(3*Ly)
! for A_s > A_l
  if(psijp.ge.A_top) then
     ystar1A=1
     goto 11
  endif
  do j=2,Lx_mid
    ! psij=sum(abs(A0(1:3,j)))/3
    psij=sum(abs(A0(1:3,1:Ly,j)))/(3*Ly)
! for A_s > A_l
    if(psij.ge.A_top) then
      ystar1A=j+(A_top-psij)/(psij-psijp) ! psijp=psi(j-1)
      goto 11
    endif
    psijp=psij
  enddo
  ystar1A=Lx_mid
11 continue

  psijp=sum(abs(A0(1:3,1:Ly,Lx)))/(3*Ly)
! for A_s > A_l
  if(psijp.ge.A_top) then
     ystar2A=Lx
     goto 12
  endif
  do j=Lx-1,Lx_mid,-1
    ! psij=sum(abs(A0(1:3,j)))/3
    psij=sum(abs(A0(1:3,1:Ly,j)))/(3*Ly)
! for A_s > A_l
    if(psij.ge.A_top) then
      ystar2A=j+(A_top-psij)/(psijp-psij) ! psijp=psi(j+1)
      goto 12
    endif
    psijp=psij
  enddo
  ystar2A=Lx_mid
12 continue

  return
end subroutine interface


! output amplitudes and phases
subroutine phase_output(phase_t,An_t)
  use global_variables
  implicit none

  character phase_t*(*),An_t*(*)

  real(8) :: phi_A(3,Ly,Lx), phiA_old(3)
  real(8) :: phi_B(3,Ly,Lx), phiB_old(3)
  real(8) :: pi2, A_R, A_I
  integer :: ix,iy,j,iA(3),ii

!  pi=DACOS(-1.0d0)
  pi2=2.d0*pi
!  pi_2=pi/2.d0

!  phiA_old=100.d0
!  iA=Lx/2
  iA=1

  open(unit=16,file=An_t,status='unknown')
  open(unit=15,file=phase_t,status='unknown')
! do ix=Lx/2,Lx
  do iy=1,Ly
    do ix=1,Lx
      do j=1,3
        
        A_R=REAL(A0(j,iy,ix))
        A_I=AIMAG(A0(j,iy,ix))
  !       if((dabs(A_R).lt.1.e-5).and.(dabs(A_I).lt.1.e-5)) then
        if((dabs(A_R).lt.5.e-5).and.(dabs(A_I).lt.5.e-5)) then
          phi_A(j,iy,ix)=0.d0
        else
          phi_A(j,iy,ix)=datan(A_I/A_R)
          if(A_R.lt.0.) phi_A(j,iy,ix)=phi_A(j,iy,ix)+pi
  !          if(phi_A(j,iy,ix).lt.0.) phi_A(j,iy,ix)=phi_A(j,iy,ix)+pi2
        endif

        if(ix.gt.1) then
          if(phi_A(j,iy,ix)-phiA_old(j).gt.pi) then
            do ii=iA(j),ix-1
              phi_A(j,iy,ii)=phi_A(j,iy,ii)+pi2
            enddo
    !          iA(j)=ix
          else
            if(phi_A(j,iy,ix)-phiA_old(j).lt.-pi) then
                do ii=iA(j),ix-1
                  phi_A(j,iy,ii)=phi_A(j,iy,ii)-pi2
                enddo
            endif
          endif
        endif
        phiA_old(j)=phi_A(j,iy,ix)
        
        A_R=REAL(B0(j,iy,ix))
        A_I=AIMAG(B0(j,iy,ix))
  !       if((dabs(A_R).lt.1.e-5).and.(dabs(A_I).lt.1.e-5)) then
        if((dabs(A_R).lt.5.e-5).and.(dabs(A_I).lt.5.e-5)) then
          phi_B(j,iy,ix)=0.d0
        else
          phi_B(j,iy,ix)=datan(A_I/A_R)
          if(A_R.lt.0.) phi_B(j,iy,ix)=phi_B(j,iy,ix)+pi
  !          if(phi_B(j,iy,ix).lt.0.) phi_B(j,iy,ix)=phi_B(j,iy,ix)+pi2
        endif

        if(ix.gt.1) then
          if(phi_B(j,iy,ix)-phiB_old(j).gt.pi) then
            do ii=iA(j),ix-1
              phi_B(j,iy,ii)=phi_B(j,iy,ii)+pi2
            enddo
    !          iA(j)=ix
          else
            if(phi_B(j,iy,ix)-phiB_old(j).lt.-pi) then
              do ii=iA(j),ix-1
                phi_B(j,iy,ii)=phi_B(j,iy,ii)-pi2
              enddo
            endif
          endif
        endif
        phiB_old(j)=phi_B(j,iy,ix)
       
      enddo ! do j=1,3
    enddo ! do ix=1,Lx
  enddo ! do iy=1,Ly

  write(16,*) '# ix=1,Lx, iy=1,Ly, abs(A0(1,iy,ix)),abs(A0(2,iy,ix)),abs(A0(3,iy,ix)),abs(B0(1,iy,ix)),abs(B0(2,iy,ix)),abs(B0(3,iy,ix)),nA0(iy,ix),nB0(iy,ix)'
  write(15,*) '# ix=1,Lx, iy=1,Ly, phi_A(1,iy,ix),phi_A(2,iy,ix),phi_A(3,iy,ix),phi_B(1,iy,ix),phi_B(2,iy,ix),phi_B(3,iy,ix)'

  do ix=1,Lx
    do iy=1,Ly
      write(16,51) ix,iy,abs(A0(1,iy,ix)),abs(A0(2,iy,ix)),abs(A0(3,iy,ix)),abs(B0(1,iy,ix)),abs(B0(2,iy,ix)),abs(B0(3,iy,ix)),nA0(iy,ix),nB0(iy,ix)
      write(15,52) ix,iy,phi_A(1,iy,ix),phi_A(2,iy,ix),phi_A(3,iy,ix),phi_B(1,iy,ix),phi_B(2,iy,ix),phi_B(3,iy,ix)
    enddo ! do iy=1,Ly
  enddo ! do ix=1,Lx
  close(15)
  close(16)

51  format(1x,i8,i8,8(1x,es18.10))
52  format(1x,i8,i8,6(1x,es18.10))

  return
end subroutine phase_output


! note: here use ave(n0) for determing interface position and calculating surface tension directLy from simulation
subroutine surface_tension(time,file)
  use global_variables
  implicit none

  ! common/q12/G_0(:,:,:),deltaj0(3),qA2,qB2,qAB2
  common/q12/deltaj0(3),qA2,qB2,qAB2
  common/surf/dx,dy,epsA,epsB,beta_A,beta_B,beta_AB,alpha_AB,width_per
  common/interf/m_gap_nA(2),m_gap_nB(2),width_nA(2),width_nB(2),width_A(2),width_B(2),gamma_nA,gamma_nB,Es

  complex(8) :: A(Ly,Lx),Aq(Ly,Lx),B(Ly,Lx)
  real(8) :: duA0(Ly,Lx),nq(Ly,Lx)
  ! real(8) :: G_0,deltaj0,qA2,qB2,qAB2
  real(8) :: deltaj0,qA2,qB2,qAB2
  real(8) :: time,dx,dy,epsA,epsB,beta_A,beta_B,beta_AB,alpha_AB
  integer :: ix,iy,j
  real(8) :: A_R,A_I,A02(3),B02(3),ng6A,ng6B,ng3A,ng3B
  real(8) :: en,enj,enl,enx,nA0_avg,nB0_avg,nA0_s,nA0_l,nB0_s,nB0_l
  real(8) :: Es,A02_s,B02_s,dphi,phi_A,phi_B
  real(8) :: gamma_nA,gamma_nB
  real(8) :: width_nA,width_nB,width_A,width_B,m_gap_nA,m_gap_nB,width_per
  character file*(*)

  nA0_avg=sum(nA0(1:Ly,1:Lx))*scale2d_b ! average nA0
  nB0_avg=sum(nB0(1:Ly,1:Lx))*scale2d_b ! average nB0
  
! calculate the derivatives in F formula
  duA0=0
  do j=1,3
    do ix=1,Lx
      do iy=1,Ly
        Aq(iy,ix)=A0q(j,iy,ix)*(G_0(j,iy,ix)+qA2-q02)
      enddo ! do iy=1,Ly
    enddo ! do ix=1,Lx
    call fftw_execute_dft(plan2d_c_b,Aq,A) ! backward FFT (complex)
    A=A*scale2D_b
    do ix=1,Lx
      do iy=1,Ly
        Aq(iy,ix)=B0q(j,iy,ix)*(G_0(j,iy,ix)+qB2-q02)
      enddo ! do iy=1,Ly
    enddo ! do ix=1,Lx
    call fftw_execute_dft(plan2d_c_b,Aq,B) ! backward FFT (complex)
    B=B*scale2D_b
    do ix=1,Lx
      do iy=1,Ly
        A_R=REAL(A(iy,ix))
        A_I=AIMAG(A(iy,ix))
        duA0(iy,ix)=duA0(iy,ix)+beta_A*(A_R*A_R+A_I*A_I) ! beta_A*\sum_j |(G_0-delta_jA^0) Aj0|^2
        A_R=REAL(B(iy,ix))
        A_I=AIMAG(B(iy,ix))
        duA0(iy,ix)=duA0(iy,ix)+beta_B*(A_R*A_R+A_I*A_I) ! beta_B*\sum_j |(G_0-delta_jB^0) Bj0|^2
      enddo ! do iy=1,Ly
    enddo ! do ix=1,Lx
    if(beta_AB.gt.1.e-10) then
      do ix=1,Lx
        do iy=1,Ly
          Aq(iy,ix)=A0q(j,iy,ix)*(G_0(j,iy,ix)+qAB2-q02)
        enddo ! do iy=1,Ly
      enddo ! do ix=1,Lx
      call fftw_execute_dft(plan2d_c_b,Aq,A) ! backward FFT (complex)
      A=A*scale2D_b
      do ix=1,Lx
        do iy=1,Ly
          Aq(iy,ix)=B0q(j,iy,ix)*(G_0(j,iy,ix)+qAB2-q02)
        enddo ! do iy=1,Ly
      enddo ! do ix=1,Lx
      call fftw_execute_dft(plan2d_c_b,Aq,B) ! backward FFT (complex)
      B=B*scale2D_b
      do ix=1,Lx
        do iy=1,Ly
          duA0(iy,ix)=duA0(iy,ix)+beta_AB*2*REAL(A(iy,ix)*conjg(B(iy,ix))) ! beta_AB*\sum_j [(G_0-delta_jAB^0)Aj0][(G_0-delta_jAB^0)Bj0]^*
        enddo ! do iy=1,Ly
      enddo ! do ix=1,Lx
    endif
  enddo ! do j=1,3

  en=0 ! total free energy
  do ix=1,Lx
    do iy=1,Ly
      do j=1,3
        A_R=REAL(A0(j,iy,ix))
        A_I=AIMAG(A0(j,iy,ix))
        A02(j)=A_R*A_R+A_I*A_I
        A_R=REAL(B0(j,iy,ix))
        A_I=AIMAG(B0(j,iy,ix))
        B02(j)=A_R*A_R+A_I*A_I
      enddo ! do j=1,3
    
      ng6A=6*vA*nA0(iy,ix)-2*gA
      ng3A=nA0(iy,ix)*(3*vA*nA0(iy,ix)-2*gA)+w*nB0(iy,ix)
      ng6B=6*vB*nB0(iy,ix)-2*gB
      ng3B=nB0(iy,ix)*(3*vB*nB0(iy,ix)-2*gB)+u*nA0(iy,ix)

      enj=duA0(iy,ix)+(-epsA+ng3A)*(A02(1)+A02(2)+A02(3))+1.5d0*vA*(A02(1)*A02(1)+A02(2)*A02(2)+A02(3)*A02(3)) &
          +2*ng6A*real(A0(1,iy,ix)*A0(2,iy,ix)*A0(3,iy,ix))+6*vA*(A02(1)*(A02(2)+A02(3))+A02(2)*A02(3)) &
          +(-epsB+ng3B)*(B02(1)+B02(2)+B02(3))+1.5d0*vB*(B02(1)*B02(1)+B02(2)*B02(2)+B02(3)*B02(3)) &
          +2*ng6B*real(B0(1,iy,ix)*B0(2,iy,ix)*B0(3,iy,ix))+6*vB*(B02(1)*(B02(2)+B02(3))+B02(2)*B02(3)) &
          +(alpha_AB+w*nA0(iy,ix)+u*nB0(iy,ix))*2*REAL(A0(1,iy,ix)*conjg(B0(1,iy,ix))+A0(2,iy,ix)*conjg(B0(2,iy,ix))+A0(3,iy,ix)*conjg(B0(3,iy,ix))) &
          +2*REAL(u*(A0(1,iy,ix)*B0(2,iy,ix)*B0(3,iy,ix)+A0(2,iy,ix)*B0(1,iy,ix)*B0(3,iy,ix)+A0(3,iy,ix)*B0(1,iy,ix)*B0(2,iy,ix)) &
          +w*(B0(1,iy,ix)*A0(2,iy,ix)*A0(3,iy,ix)+B0(2,iy,ix)*A0(1,iy,ix)*A0(3,iy,ix)+B0(3,iy,ix)*A0(1,iy,ix)*A0(2,iy,ix)))
      enj=enj+nA0(iy,ix)*nA0(iy,ix)*(0.5d0*(-epsA+beta_A*qA2*qA2)-gA*nA0(iy,ix)/3+vA*nA0(iy,ix)*nA0(iy,ix)/4) &
          +nB0(iy,ix)*nB0(iy,ix)*(0.5d0*(-epsB+beta_B*qB2*qB2)-gB*nB0(iy,ix)/3+vB*nB0(iy,ix)*nB0(iy,ix)/4) &
          +nA0(iy,ix)*nB0(iy,ix)*(alpha_AB+beta_AB*qAB2*qAB2+(w*nA0(iy,ix)+u*nB0(iy,ix))/2)
      en=en+enj
      
      if(ix.eq.1) enl=enj ! liquid region
      if(ix.eq.Lx/2) then ! solid region
          enx=enj
          A02_s=A02(1)
          B02_s=B02(1)
      endif
    enddo ! do iy=1,Ly
  enddo ! do ix=1,Lx
  nA0_s=sum(nA0(1:Ly,Lx/2))/Ly
  nA0_l=sum(nA0(1:Ly,1))/Ly
  nB0_s=sum(nB0(1:Ly,Lx/2))/Ly
  nB0_l=sum(nB0(1:Ly,1))/Ly
  
! surface tension (from lever rule; see the notes and also Dantzig's book p72, WuPRB07)
! in equilibrium of liq-sol coexistence, (nA0_avg-nA0_l)/(nB0_avg-nB0_l) = (nA0_s-nA0_avg)/(nB0_s-nB0_avg)
  gamma_nA=dx*(en/Ly+Lx*(enl*(nA0_avg-nA0_s)+enx*(nA0_l-nA0_avg))/(nA0_s-nA0_l))/2
  gamma_nB=dx*(en/Ly+Lx*(enl*(nB0_avg-nB0_s)+enx*(nB0_l-nB0_avg))/(nB0_s-nB0_l))/2

  !pi=DACOS(-1.0d0)
  !pi2=2.d0*pi
  A_R=sum(REAL(A0(1,1:Ly,Lx/2)))/Ly
  A_I=sum(AIMAG(A0(1,1:Ly,Lx/2)))/Ly
!  if((dabs(A_R).lt.1.e-5).and.(dabs(A_I).lt.1.e-5)) then
  if((dabs(A_R).lt.5.e-5).and.(dabs(A_I).lt.5.e-5)) then
     phi_A=0.d0
  else
     phi_A=datan(A_I/A_R)
     if(A_R.lt.0.) phi_A=phi_A+pi
!     if(phi_A.lt.0.) phi_A=phi_A+pi2
  endif
  A_R=sum(REAL(B0(1,1:Ly,Lx/2)))/Ly
  A_I=sum(AIMAG(B0(1,1:Ly,Lx/2)))/Ly
!  if((dabs(A_R).lt.1.e-5).and.(dabs(A_I).lt.1.e-5)) then
  if((dabs(A_R).lt.5.e-5).and.(dabs(A_I).lt.5.e-5)) then
     phi_B=0.d0
  else
     phi_B=datan(A_I/A_R)
     if(A_R.lt.0.) phi_B=phi_B+pi
!     if(phi_B.lt.0.) phi_B=phi_B+pi2
  endif
  dphi=phi_A-phi_B

  Es=8*q02*q02*(beta_A*A02_s+beta_B*B02_s+2*beta_AB*sqrt(A02_s*B02_s)*cos(dphi)) ! Young's modulus

! for interface width and miscibility gap
  call width_gap(width_nA,width_nB,width_A,width_B,m_gap_nA,m_gap_nB,width_per,dx)
  open(unit=8,file=file//'_width.dat',position='append')
  write(8,50) time,epsA,epsB,width_nA(1),width_nA(2),width_nB(1),width_nB(2), &
              m_gap_nA(1),m_gap_nA(2),m_gap_nB(1),m_gap_nB(2),width_A(1),width_A(2),width_B(1),width_B(2)
50 format(es11.5,2(1x,f9.3),12(1x,es17.10))
  close(8)

  open(unit=9,file=file//'_surf.dat',position='append')
  write(9,51) time,epsA,epsB,gamma_nA,gamma_nB,enx,enl,Es,(nA0_avg-nA0_l)*(nB0_s-nB0_avg)-(nB0_avg-nB0_l)*(nA0_s-nA0_avg)
! gamma: surface tension; enx,enl: solid,liquid F density; 
! Es: Young's modulus; in equilibrium (nA0_avg-nA0_l)/(nB0_avg-nB0_l) = (nA0_s-nA0_avg)/(nB0_s-nB0_avg)
51 format(es11.5,2(1x,f9.3),6(1x,es17.10))
  close(9)
  open(unit=10,file=file//'_f.dat',position='append')
  write(10,*) time,en/Lx ! =en*dx/(Lx*dx) total free energy density
  close(10)

  return
end subroutine surface_tension


! for interface width and miscibility gap (determined by n0A or n0B)
subroutine width_gap(width_nA,width_nB,width_A,width_B,m_gap_nA,m_gap_nB,width_per,dx)
  use global_variables
  implicit none

  real(8) :: width_nA(2),width_nB(2),m_gap_nA(2),m_gap_nB(2),width_per,dx
  real(8) :: nA0_s,nA0_l,nB0_s,nB0_l,width_n1,width_n2,width1,width2
  real(8) :: width_A(2),width_B(2),A0_s,A0_l,m_gap_A,B0_s,B0_l,m_gap_B
  integer :: ix

! for first half (1<ix<Lx/2)
  nA0_s=sum(nA0(1:Ly,Lx/2))/Ly
  nA0_l=sum(nA0(1:Ly,1))/Ly
  nB0_s=sum(nB0(1:Ly,Lx/2))/Ly
  nB0_l=sum(nB0(1:Ly,1))/Ly
  A0_s=sum(abs(A0(1,1:Ly,Lx/2)))/Ly
  A0_l=sum(abs(A0(1,1:Ly,1)))/Ly
  B0_s=sum(abs(B0(1,1:Ly,Lx/2)))/Ly
  B0_l=sum(abs(B0(1,1:Ly,1)))/Ly

  m_gap_nA(1)=nA0_s-nA0_l
  m_gap_nB(1)=nB0_s-nB0_l
  m_gap_A=A0_s-A0_l
  m_gap_B=B0_s-B0_l

  if(abs(m_gap_nA(1)).gt.1.e-6) then
  width1=0
  width2=0
  width_n1=nA0_l+(1-width_per)*m_gap_nA(1)/2
  width_n2=nA0_s-(1-width_per)*m_gap_nA(1)/2
  do ix=1,Lx/2
# ifdef n0s_lt_l ! if n0_s < n0_l
     if(sum(nA0(1:Ly,ix))/Ly.le.width_n1) then
# else  ! if n0_s > n0_l(f)
     if(sum(nA0(1:Ly,ix))/Ly.ge.width_n1) then
# endif
        width1=ix+(width_n1-sum(nA0(1:Ly,ix))/Ly)/(sum(nA0(1:Ly,ix))/Ly-sum(nA0(1:Ly,ix-1))/Ly) ! interpolation
        goto 1
     endif
  enddo ! do ix=1,Lx/2

1 continue
  do ix=Lx/2,1,-1
# ifdef n0s_lt_l ! if n0_s < n0_l
     if(sum(nA0(1:Ly,ix))/Ly.ge.width_n2) then
# else ! if n0_s > n0_l(f)
     if(sum(nA0(1:Ly,ix))/Ly.le.width_n2) then
# endif
        width2=ix+(width_n2-sum(nA0(1:Ly,ix))/Ly)/(sum(nA0(1:Ly,ix+1))/Ly-sum(nA0(1:Ly,ix))/Ly) ! interpolation
        goto 2
     endif
  enddo ! do ix=Lx/2,1,-1
2 continue
  width_nA(1)=(width2-width1)*dx

  else
     width_nA(1)=0
  endif

  if(abs(m_gap_nB(1)).gt.1.e-6) then
  width1=0
  width2=0
  width_n1=nB0_l+(1-width_per)*m_gap_nB(1)/2
  width_n2=nB0_s-(1-width_per)*m_gap_nB(1)/2
  do ix=1,Lx/2
# ifdef n0s_lt_l ! if n0_s < n0_l
     if(sum(nB0(1:Ly,ix))/Ly.le.width_n1) then
# else  ! if n0_s > n0_l
     if(sum(nB0(1:Ly,ix))/Ly.ge.width_n1) then
# endif
        width1=ix+(width_n1-sum(nB0(1:Ly,ix))/Ly)/(sum(nB0(1:Ly,ix))/Ly-sum(nB0(1:Ly,ix-1))/Ly) ! interpolation
        goto 3
     endif
  enddo ! do ix=1,Lx/2
3 continue
  do ix=Lx/2,1,-1
# ifdef n0s_lt_l ! if n0_s < n0_l
     if(sum(nB0(1:Ly,ix))/Ly.ge.width_n2) then
# else ! if n0_s > n0_l
     if(sum(nB0(1:Ly,ix))/Ly.le.width_n2) then
# endif
        width2=ix+(width_n2-sum(nB0(1:Ly,ix))/Ly)/(sum(nB0(1:Ly,ix+1))/Ly-sum(nB0(1:Ly,ix))/Ly) ! interpolation
        goto 4
     endif
  enddo ! do ix=Lx/2,1,-1
4 continue
  width_nB(1)=(width2-width1)*dx

  else
     width_nB(1)=0
  endif

  if(abs(m_gap_A).gt.1.e-6) then
  width1=0
  width2=0
  width_n1=A0_l+(1-width_per)*m_gap_A/2
  width_n2=A0_s-(1-width_per)*m_gap_A/2
  do ix=1,Lx/2
     if(sum(abs(A0(1,1:Ly,ix)))/Ly.ge.width_n1) then
        width1=ix+(width_n1-sum(abs(A0(1,1:Ly,ix)))/Ly)/(sum(abs(A0(1,1:Ly,ix)))/Ly-sum(abs(A0(1,1:Ly,ix-1)))/Ly) ! interpolation
        goto 5
     endif
  enddo ! do ix=1,Lx/2
5 continue
  do ix=Lx/2,1,-1
     if(sum(abs(A0(1,1:Ly,ix)))/Ly.le.width_n2) then
        width2=ix+(width_n2-sum(abs(A0(1,1:Ly,ix)))/Ly)/(sum(abs(A0(1,1:Ly,ix+1)))/Ly-sum(abs(A0(1,1:Ly,ix)))/Ly) ! interpolation
        goto 6
     endif
  enddo ! do ix=Lx/2,1,-1
6 continue
  width_A(1)=(width2-width1)*dx

  else
     width_A(1)=0
  endif

  if(abs(m_gap_B).gt.1.e-6) then
  width1=0
  width2=0
  width_n1=B0_l+(1-width_per)*m_gap_B/2
  width_n2=B0_s-(1-width_per)*m_gap_B/2
  do ix=1,Lx/2
     if(sum(abs(B0(1,1:Ly,ix)))/Ly.ge.width_n1) then
        width1=ix+(width_n1-sum(abs(B0(1,1:Ly,ix)))/Ly)/(sum(abs(B0(1,1:Ly,ix)))/Ly-sum(abs(B0(1,1:Ly,ix-1)))/Ly) ! interpolation
        goto 7
     endif
  enddo
7 continue
  do ix=Lx/2,1,-1
     if(sum(abs(B0(1,1:Ly,ix)))/Ly.le.width_n2) then
        width2=ix+(width_n2-sum(abs(B0(1,1:Ly,ix)))/Ly)/(sum(abs(B0(1,1:Ly,ix+1)))/Ly-sum(abs(B0(1,1:Ly,ix)))/Ly) ! interpolation
        goto 8
     endif
  enddo ! do ix=Lx/2,1,-1
8 continue
  width_B(1)=(width2-width1)*dx

  else
     width_B(1)=0
  endif

! for second half (Lx/2<ix<Lx)
  nA0_s=sum(nA0(1:Ly,Lx/2+1))/Ly
  nA0_l=sum(nA0(1:Ly,Lx))/Ly
  nB0_s=sum(nB0(1:Ly,Lx/2+1))/Ly
  nB0_l=sum(nB0(1:Ly,Lx))/Ly
  A0_s=sum(abs(A0(1,1:Ly,Lx/2+1)))/Ly
  A0_l=sum(abs(A0(1,1:Ly,Lx)))/Ly
  B0_s=sum(abs(B0(1,1:Ly,Lx/2+1)))/Ly
  B0_l=sum(abs(B0(1,1:Ly,Lx)))/Ly

  m_gap_nA(2)=nA0_s-nA0_l
  m_gap_nB(2)=nB0_s-nB0_l
  m_gap_A=A0_s-A0_l
  m_gap_B=B0_s-B0_l

  if(abs(m_gap_nA(2)).gt.1.e-6) then
  width1=0
  width2=0
  width_n1=nA0_l+(1-width_per)*m_gap_nA(2)/2
  width_n2=nA0_s-(1-width_per)*m_gap_nA(2)/2
  do ix=Lx,Lx/2+1,-1
# ifdef n0s_lt_l ! if n0_s < n0_l
     if(sum(nA0(1:Ly,ix))/Ly.le.width_n1) then
# else  ! if n0_s > n0_l
     if(sum(nA0(1:Ly,ix))/Ly.ge.width_n1) then
# endif
        width2=ix+(width_n1-sum(nA0(1:Ly,ix))/Ly)/(sum(nA0(1:Ly,ix+1))/Ly-sum(nA0(1:Ly,ix))/Ly) ! interpolation
        goto 11
     endif
  enddo ! do ix=Lx,Lx/2+1,-1
11 continue
  do ix=Lx/2+1,Lx
# ifdef n0s_lt_l ! if n0_s < n0_l
     if(sum(nA0(1:Ly,ix))/Ly.ge.width_n2) then
# else ! if n0_s > n0_l(f)
     if(sum(nA0(1:Ly,ix))/Ly.le.width_n2) then
# endif
        width1=ix+(width_n2-sum(nA0(1:Ly,ix))/Ly)/(sum(nA0(1:Ly,ix))/Ly-sum(nA0(1:Ly,ix-1))/Ly) ! interpolation
        goto 12
     endif
  enddo ! do ix=Lx/2+1,Lx
12 continue
  width_nA(2)=(width2-width1)*dx

  else
     width_nA(2)=0
  endif

  if(abs(m_gap_nB(2)).gt.1.e-6) then
  width1=0
  width2=0
  width_n1=nB0_l+(1-width_per)*m_gap_nB(2)/2
  width_n2=nB0_s-(1-width_per)*m_gap_nB(2)/2
  do ix=Lx,Lx/2+1,-1
# ifdef n0s_lt_l ! if n0_s < n0_l
     if(sum(nB0(1:Ly,ix))/Ly.le.width_n1) then
# else  ! if n0_s > n0_l
     if(sum(nB0(1:Ly,ix))/Ly.ge.width_n1) then
# endif
        width2=ix+(width_n1-sum(nB0(1:Ly,ix))/Ly)/(sum(nB0(1:Ly,ix+1))/Ly-sum(nB0(1:Ly,ix))/Ly) ! interpolation
        goto 13
     endif
  enddo ! do ix=Lx,Lx/2+1,-1
13 continue
  do ix=Lx/2+1,Lx
# ifdef n0s_lt_l ! if n0_s < n0_l
     if(sum(nB0(1:Ly,ix))/Ly.ge.width_n2) then
# else ! if n0_s > n0_l(f)
     if(sum(nB0(1:Ly,ix))/Ly.le.width_n2) then
# endif
        width1=ix+(width_n2-sum(nB0(1:Ly,ix))/Ly)/(sum(nB0(1:Ly,ix))/Ly-sum(nB0(1:Ly,ix-1))/Ly) ! interpolation
        goto 14
     endif
  enddo ! do ix=Lx/2+1,Lx
14 continue
  width_nB(2)=(width2-width1)*dx

  else
     width_nB(2)=0
  endif

  if(abs(m_gap_A).gt.1.e-6) then
  width1=0
  width2=0
  width_n1=A0_l+(1-width_per)*m_gap_A/2
  width_n2=A0_s-(1-width_per)*m_gap_A/2
  do ix=Lx,Lx/2+1,-1
     if(sum(abs(A0(1,1:Ly,ix)))/Ly.ge.width_n1) then
        width2=ix+(width_n1-sum(abs(A0(1,1:Ly,ix)))/Ly)/(sum(abs(A0(1,1:Ly,ix+1)))/Ly-sum(abs(A0(1,1:Ly,ix)))/Ly) ! interpolation
        goto 15
     endif
  enddo ! do ix=Lx,Lx/2+1,-1
15 continue
  do ix=Lx/2+1,Lx
     if(sum(abs(A0(1,1:Ly,ix)))/Ly.le.width_n2) then
        width1=ix+(width_n2-sum(abs(A0(1,1:Ly,ix)))/Ly)/(sum(abs(A0(1,1:Ly,ix)))/Ly-sum(abs(A0(1,1:Ly,ix-1)))/Ly) ! interpolation
        goto 16
     endif
  enddo ! do ix=Lx/2+1,Lx
16 continue
  width_A(2)=(width2-width1)*dx

  else
     width_A(2)=0
  endif

  if(abs(m_gap_B).gt.1.e-6) then
  width1=0
  width2=0
  width_n1=B0_l+(1-width_per)*m_gap_B/2
  width_n2=B0_s-(1-width_per)*m_gap_B/2
  do ix=Lx,Lx/2+1,-1
     if(sum(abs(B0(1,1:Ly,ix)))/Ly.ge.width_n1) then
        width2=ix+(width_n1-sum(abs(B0(1,1:Ly,ix)))/Ly)/(sum(abs(B0(1,1:Ly,ix+1)))/Ly-sum(abs(B0(1,1:Ly,ix)))/Ly) ! interpolation
        goto 17
     endif
  enddo ! do ix=Lx,Lx/2+1,-1
17 continue
  do ix=Lx/2+1,Lx
     if(sum(abs(B0(1,1:Ly,ix)))/Ly.le.width_n2) then
        width1=ix+(width_n2-sum(abs(B0(1,1:Ly,ix)))/Ly)/(sum(abs(B0(1,1:Ly,ix)))/Ly-sum(abs(B0(1,1:Ly,ix-1)))/Ly) ! interpolation
        goto 18
     endif
  enddo ! do ix=Lx/2+1,Lx
18 continue
  width_B(2)=(width2-width1)*dx

  else
     width_B(2)=0
  endif

end subroutine width_gap
