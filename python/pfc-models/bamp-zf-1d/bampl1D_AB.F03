! calculate the 0th-order stationary/equilibrium state of binary AB ampl. eqs.
! 1D, Aj^0(u,t), Bj^0(u,t), nA0^0(u,t), and nB0^0(u,t)
! using FFTW3

! use openMP for multi-threading (To use single-core only: comment out "define open_MP")

! note: when calculating 0th-order 1D equilibrium/stationary solution, set mA=mB=1
!       also use conserved dynamics for nA0^0,nB0^0, to get mu values (to be used in nonconserved case if needed)
!       use n0_reduced eq, i.e., neglect gradient terms of n0

! pseudo-spectral method
! using a predictor-corrector method for Aj^0, Bj^0, nA0^0, nB0^0
!   exact solution for linear terms and approximation for nonlinearities
!       (with an exponential propagation procedure, similar to Cross et al., Chaos 1994;
!        donot fix the number of iterations;
!        expand around small sigma when it approaches 0)
! with the transient time

# define liq_sol ! for liquid_solid; "l": liquid; "s": solid

# define ini_tanh ! use initial tanh profile

! for subroutines interface and width_gap
!# define n0s_lt_l ! for n0_s < n0_l

!# define coeffs1D_only ! only calculate coeffs1D based on conf.dat equilibrium profiles
# define save_conf

# define open_MP ! use openMP for multi-threading

module global_variables
! by default all variables in the module are public (if not being set explicitly as private)
  use, intrinsic :: iso_c_binding
  implicit none
  include '/wsu/el7/pgi-openmpi/2018-187/fftw/3.3.8/include/fftw3.f03'
  
  save ! save all the variables in the module

  !integer, parameter :: L=512
  !integer, parameter :: L=1024
  integer, parameter :: L=2048
  !integer, parameter :: L=32768  ! L=2^15

  complex(C_DOUBLE_COMPLEX), pointer :: A0(:,:),A0q(:,:),B0(:,:),B0q(:,:)
  real(C_DOUBLE), pointer :: nA0(:),nA0q(:),nB0(:),nB0q(:)
  real(C_DOUBLE) :: exp_j11(3,L),exp_j12(3,L),exp_j21(3,L),exp_j22(3,L) ! for Aj and Bj
  real(C_DOUBLE) :: cf_j1(3,L),cf2_j1(3,L),cf_j2(3,L),cf2_j2(3,L)
  real(C_DOUBLE) :: exp_11(L),exp_12(L),exp_21(L),exp_22(L) ! for nA0 and nB0
  real(C_DOUBLE) :: cf_1(L),cf2_1(L),cf_2(L),cf2_2(L)
  real(C_DOUBLE) :: sigma12(L),sig1_alpha(L),sig2_alpha(L),alpha_12(L),alpha_21(L)
  real(C_DOUBLE) :: sigmaj12(3,L),sigj1_alpha(3,L),sigj2_alpha(3,L),alpha_j12(3,L),alpha_j21(3,L)
  real(8) :: vA,vB,gA,gB,w,u,mB,q02_mB,q02,qu2(L),pi
  integer :: isigmaj(3,L),isigma(L)
  
! for FFTW parameters
  type(C_PTR) :: plan1d_r_f,plan1d_r_b,plan1d_c_f,plan1d_c_b
  real(8) :: scale1D_b
  
end module global_variables


Program bampl1D_AB
  use global_variables
  implicit none
  
  common/converge/TOL,err_s,nmbr_eval
  common/q12/G_0(3,L),deltaj0(3),qA2,qB2,qAB2
  common/surf/du,epsA,epsB,beta_A,beta_B,beta_AB,alpha_AB,width_per
  common/interf/m_gap_nA(2),m_gap_nB(2),width_nA(2),width_nB(2),width_A(2),width_B(2),gamma_nA,gamma_nB,Es
  common/interf_coef/dchi_j(3),chi_j2(3),sin_theta,qu(L)

  type(C_PTR) :: p_alloc_A,p_alloc_Aq,p_alloc_n,p_alloc_nq ! for allocating aligned memory in fftw
  real(C_DOUBLE) :: alpha_11,alpha_22
  real(C_DOUBLE) :: sigma1(L),sigma2(L),sig1_dt,sig2_dt,delta_2,aq,b,c
  real(C_DOUBLE) :: sigmaj1(3,L),sigmaj2(3,L)
  real(8) :: dt,dti,dtime
  integer :: iter0,ntype,iter,idum,n_0,n_i,nmbr_eval,istep
  integer :: i_theta,n1,n2,i_p

  real(8) :: epsA,epsB,q0,du,noise,noise0
  real(8) :: qx0,qy0,qx,qy,q,qu
  real(8) :: qA,qA2,qB,qB2,qAB,qAB2
  real(8) :: beta_A,beta_B,beta_AB,alpha_AB
  real(8) :: sin_theta,cos_theta
  real(8) :: delta_x,delta_y,eps_m
  real(8) :: G_0,deltaj0
  real(8) :: time,tmax,time0,t_i
  real(8) :: yy,ran3
  real(8) :: vb1A,vb2A,ystar1A,ystar2A,ystar1A_0,ystar2A_0
  integer :: n_dx,nend,raout,nout_conf,i,j,nimpc_v,nout_v,i_ystar,nt
! for sharp-interface coefficients
  real(8) :: dchi_j,chi_j2

  real(8) :: nA0_s,nA0_l,nB0_s,nB0_l
  complex(8) :: A1_s,A2_s,A3_s,B1_s,B2_s,B3_s,A_l,B_l
  !real(8) :: phi_A1,phi_A2,phi_A3,phi_B1,phi_B2,phi_B3
  real(8) :: A_top,width_per,wid_n,wid_A
  real(8) :: width_nA,width_nB,width_A,width_B,m_gap_nA,m_gap_nB,gamma_nA,gamma_nB,Es
  integer :: i1,i2,i_width,L_mid
  integer :: n_time, itime(15)
  character(10) :: cha
  real(8) :: TOL,err_s ! for iteration of predictor-corrector

  complex(C_DOUBLE_COMPLEX), pointer :: A_tmp(:),Aq_tmp(:)
  real(C_DOUBLE), pointer :: n_tmp(:)

# ifdef open_MP
! use multi-threaded fftw with openMP
  integer iret,nthreads,omp_get_max_threads
# endif
  
  character(:), allocatable :: file ! to be automatically allocated
  character(:), allocatable :: filename ! to be automatically allocated
  !character(5) filename
# ifdef coeffs1D_only ! only calculate coeffs1D based on conf.dat equilibrium profiles
  !character(num_file+2) file_i
  character(:), allocatable :: file_i ! to be automatically allocated
# endif
  character(:), allocatable :: init_cond ! to be automatically allocated

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

! interface orientation theta (with respect to the y axis), q=(n1*qx,n2*qy/2)

! zigzag: theta=0 or +/-60^o (equivalent)
!  i_theta=0 ! direction 1, along y
!  i_theta=60 ! direction 2, along (qx,qy/2)
!  i_theta=-60 ! direction 3, along (-qx,qy/2)

! armchair: theta=90^o or +/-30^o (equivalent)
i_theta=90 ! direction 4, along x
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
  !file='bampl512eps03g05w03aAB05bAB0noise0' ! _4: i_theta=90; _1: i_theta=0; L=512,tmax=10^7, def ini_tanh
                                 ! sharp interface, zeta_D(AB) sensitive on Gibbs surface positions, _4: zeta(1) >0, (2) <0
       ! _1: gamma_nA,B(eV/A)=0.7969663466119443, 0.7969663466969760
       !     sigma(eV/A)=1.347555778200326, 1.347555778194726; sigma_avg(eV/A)=1.347555778197526
       ! _4: gamma_nA,B(eV/A)=0.8026562460117934, 0.8026562460180963
       !     sigma(eV/A)=1.214095473038613, 1.214095473038397; sigma_avg(eV/A)=1.214095473038505
  file='bampl2048eps03g05w03aAB05bAB0noise0' ! L=2048,tmax=10^7; _1,_4: same sign of zeta_DAB for 2 interfaces
       ! _1: gamma_nA,B(eV/A)=0.7969663396406974, 0.7969663541770061
       !     sigma(eV/A)=1.347555773529631, 1.347555773550504; sigma_avg(eV/A)=1.347555773540068
       ! _4: gamma_nA,B(eV/A)=0.8026562457240588, 0.8026562462267081
       !     sigma(eV/A)=1.214095507651008, 1.214095507651016; sigma_avg(eV/A)=1.214095507651012
  ! armchair: theta=90 degree (_4, interface oriented along x)
  ! zigzag: theta=0 degree (_1, interface oriented along y)
  
  if(i_theta.eq.0) then ! _1
     !nA0_s=-0.2682240267782519d0 ! at tmax=10^7, L=512
     !nA0_l=-0.4435164077168946d0
     !A1_s=CMPLX(-8.7833888186763867d-2,-0.2081951907328163d0)
     !A2_s=CMPLX(-0.2197897860826280d0,-5.2463957420903942d-2)
     !A3_s=CMPLX(0.1063354275974040d0,-0.1993810577661679d0)
     !B1_s=CMPLX(-9.4460101349764639d-2,0.2052737650337667d0)
     !B2_s=CMPLX(0.1095166667429461d0,0.1976515029031947d0)
     !B3_s=CMPLX(-0.2189209348939724d0,5.5979046287830070d-2)
     nA0_s=-0.2682240335169304d0 ! at tmax=10^7, L=2048
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
     !wid_A=7.4713937980d0/2.5123056d0 ! use width_A(1) ! at tmax=10^7, L=512
     wid_n=6.4953616398d0/2.5123056d0 ! width/(2*atanh(width_per)), width_per=0.85; use width_nA(1)
     wid_A=7.4711083227d0/2.5123056d0 ! use width_A(1) ! at tmax=10^7, L=2048
# endif
  endif

  if(i_theta.eq.90) then ! _4
     !nA0_s=-0.2682240848947746d0 ! at tmax=10^7, L=512
     !nA0_l=-0.4435164451836281d0
     !A1_s=CMPLX(-8.7833887170658681d-2,-0.2081951788275782d0)
     !A2_s=CMPLX(-0.2197898102873447d0,-5.2463896518067529d-2)
     !A3_s=CMPLX(0.1063353980911257d0,-0.1993810034300522d0)
     !B1_s=CMPLX(-9.4460102249663669d-2,0.2052737521105319d0)
     !B2_s=CMPLX(0.1095166163656826d0,0.1976515415670090d0)
     !B3_s=CMPLX(-0.2189208766383217d0,5.5979030531853054d-2)
     nA0_s=-0.2682240752088968d0 ! at tmax=10^7, L=2048
     nA0_l=-0.4435164435660391d0
     A1_s=CMPLX(-8.7833881982373824d-2,-0.2081951765861624d0)
     A2_s=CMPLX(-0.2197898115052828d0,-5.2463895116867998d-2)
     A3_s=CMPLX(0.1063354123209136d0,-0.1993810340288277d0)
     B1_s=CMPLX(-9.4460100515079734d-2,0.2052737484152882d0)
     B2_s=CMPLX(0.1095166182875313d0,0.1976515414844259d0)
     B3_s=CMPLX(-0.2189209079654175d0,5.5979038035728526d-2)
# ifdef ini_tanh
     !wid_n=6.5233310268d0/2.5123056d0 ! width/(2*atanh(width_per)), width_per=0.85; use width_nA(1)
     !wid_A=1.0817743105d1/2.5123056d0 ! use width_A(1) ! at tmax=10^7, L=512
     wid_n=6.5237979053d0/2.5123056d0 ! width/(2*atanh(width_per)), width_per=0.85; use width_nA(1)
     wid_A=1.0817029688d1/2.5123056d0 ! use width_A(1) ! at tmax=10^7, L=2048
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
  !gamma = 1.214077 eV/A at theta=90 from analytic formula based on 1D ampl

# endif ! ifdef liq_sol
  
  filename=file
  
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
!
! for surface tension and kinetic coefficient only
  if(i_theta.eq.19) then ! direction a1, theta=19.1066 degree
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
!
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
!
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
!
  if(i_theta.eq.5) then ! direction d1, theta=5.2087 degree
     n1=1
     n2=19
     sin_theta=0.5d0*sqrt(3.d0/91)
     cos_theta=sqrt(1-sin_theta*sin_theta)
  endif
!
  if(i_theta.eq.14) then ! direction e1, theta=13.8979 degree
     n1=1
     n2=7
     sin_theta=0.5d0*sqrt(3.d0/13)
     cos_theta=sqrt(1-sin_theta*sin_theta)
  endif
!
  if(i_theta.eq.8) then ! direction f1, theta=7.5891 degree
     n1=1
     n2=13
     sin_theta=0.5d0*sqrt(3.d0/43)
     cos_theta=sqrt(1-sin_theta*sin_theta)
  endif
!
  if(i_theta.eq.9) then ! direction g1, theta=8.9483 degree
     n1=1
     n2=11
     sin_theta=0.5d0*sqrt(3.d0/31)
     cos_theta=sqrt(1-sin_theta*sin_theta)
  endif
!
  if(i_theta.eq.16) then ! direction h1, theta= 16.1021 degree
     n1=2
     n2=12
     sin_theta=1/sqrt(13.d0)
     cos_theta=sqrt(1-sin_theta*sin_theta)
  endif
!
  if(i_theta.eq.17) then ! direction i1, theta= 17.4802 degree
     n1=4
     n2=22
     sin_theta=2.d0*sqrt(3.d0/133)
     cos_theta=sqrt(1-sin_theta*sin_theta)
  endif
!
  if(i_theta.eq.21) then ! direction j1, theta= 20.6330 degree
     n1=5
     n2=23
     sin_theta=2.5d0*sqrt(3.d0/151)
     cos_theta=sqrt(1-sin_theta*sin_theta)
  endif
!
  if(i_theta.eq.22) then ! direction k1, theta= 21.7868 degree
     n1=3
     n2=13
     sin_theta=1.5d0*sqrt(3.d0)/7
     cos_theta=sqrt(1-sin_theta*sin_theta)
  endif
!
  if(i_theta.eq.25) then ! direction l1, theta= 25.2850 degree
     n1=3
     n2=11
     sin_theta=1.5d0*sqrt(3.d0/37)
     cos_theta=sqrt(1-sin_theta*sin_theta)
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

! n_dx: number of grid points per lattice period
  if(i_p.eq.0) n_dx=8 ! for directions 1,2,3
  if(i_p.eq.1) n_dx=8 ! for directions 4,5,6
!  if(i_p.eq.1) n_dx=4
  if(i_p.eq.-1) n_dx=4
  if(i_p.eq.-2) n_dx=2
  if(i_p.eq.-3) n_dx=2
  if(i_p.eq.-4) n_dx=1
  if(i_p.eq.-5) n_dx=2
  if(i_p.eq.-6) n_dx=2
  if(i_p.eq.-7) n_dx=2
  if(i_p.eq.-8) n_dx=2
  if(i_p.eq.-9) n_dx=1
  if(i_p.eq.-10) n_dx=1
  if(i_p.eq.-11) n_dx=2
  if(i_p.eq.-12) n_dx=2

  qx=qx0+delta_x
  qy=qy0+delta_y
  q=sqrt((n1*qx)**2+(n2*qy/2)**2) ! q^2 = q1^2 + q2^2
  du=1 ! if smaller du, need smaller dt to converge

  chi_j2(1)=(-2*qx*cos_theta+qy*sin_theta)**2/2 ! chi_j^2/2 or beta_j^2/2 in the notes
  chi_j2(2)=4*qy*qy*sin_theta*sin_theta/2
  chi_j2(3)=(2*qx*cos_theta+qy*sin_theta)**2/2
  
  dchi_j(1)=2*qx*sin_theta+qy*cos_theta ! dchi_j/dtheta
  dchi_j(2)=-2*qy*cos_theta
  dchi_j(3)=-2*qx*sin_theta+qy*cos_theta

!  noise=0.1d0 ! noise percentage
!  noise0=0.01d0 ! for 0 values
  noise=0
  noise0=0

  idum=-796921

!  dt=0.01d0
!  dt=0.02d0
!  dt=0.05d0
  dt=0.1d0
!  dt=0.2d0 ! diverge for eps=0.3

  t_i=1 ! better for t_i/dt=integer
  dti=0.01d0 ! used for initial transient up to t=t_i

!  tmax=1000
!  nt=10
!  tmax=10000
!  nt=100
!  tmax=1.d5
!  nt=1000
!  tmax=1.d6
!  nt=10000
  tmax=1.d7
  nt=10000

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
  i_width=L/2
!  i_width=L/4

  i1=L/2-i_width/2
  i2=L/2+i_width/2
  L_mid=(i1+i2)/2

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

# ifdef coeffs1D_only ! only calculate coeffs1D based on conf.dat equilibrium profiles
  file_i=filename//'_c'
  open(unit=2,file=file_i//'_para.dat')
# else
! parameters output
  open(unit=2,file=filename//'_para.dat')
# endif

  write(2,*) 'For base state: 1D amplitude equations of Aj^0(u,t),Bj^0(u,t),nA0^0(u,t),nB0^0(u,t)'
  write(2,*) '    n0_reduced: neglect gradient terms of n0'
# ifdef liq_sol
  write(2,*) 'def liq_sol: liquid+solid configuration'
# endif
  write(2,*)
  write(2,*) 'L=',L
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
  write(2,*) 'dt=',dt, '  du=',du
  write(2,*) 'n_dx =',n_dx
  write(2,*) 'interface orientation theta=',dasin(sin_theta)*180/pi
  write(2,*) '                    i_theta=',i_theta
  write(2,*) 'q=(n1*qx,n2*qy/2), n1=',n1,  '  n2=',n2
  write(2,*) 'lambda_x*sin(theta)=n1*n_dx(k)*du=',n1*n_dx*du
  write(2,*) 'lambda_y*cos(theta)=n2*n_dx(k)*du=',n2*n_dx*du
  write(2,*)
  write(2,*) 'qx0=',qx0, '   qy0=', qy0
  write(2,*) 'qx=',qx, '   qy=', qy
  write(2,*) 'q=|(n1*qx,n2*qy/2)|=',q
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
  write(2,*) '   with i_width=',i_width, '  n_layer=', i_width/n_dx
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

  !complex(8) :: A0(3,L),A0q(3,L),B0(3,L),B0q(3,L)
  p_alloc_A=fftw_alloc_complex(int(3*L, C_SIZE_T))
  call c_f_pointer(p_alloc_A, A0, [3,L])
  p_alloc_Aq=fftw_alloc_complex(int(3*L, C_SIZE_T))
  call c_f_pointer(p_alloc_Aq, A0q, [3,L])
  p_alloc_A=fftw_alloc_complex(int(3*L, C_SIZE_T))
  call c_f_pointer(p_alloc_A, B0, [3,L])
  p_alloc_Aq=fftw_alloc_complex(int(3*L, C_SIZE_T))
  call c_f_pointer(p_alloc_Aq, B0q, [3,L])
  !complex(8) :: A_tmp(L),Aq_tmp(L)
  p_alloc_A=fftw_alloc_complex(int(L, C_SIZE_T))
  call c_f_pointer(p_alloc_A, A_tmp, [L])
  p_alloc_Aq=fftw_alloc_complex(int(L, C_SIZE_T))
  call c_f_pointer(p_alloc_Aq, Aq_tmp, [L])
  
  !real(8) :: nA0(L),nA0q(L),nB0(L),nB0q(L)
  p_alloc_n=fftw_alloc_real(int(L, C_SIZE_T))
  call c_f_pointer(p_alloc_n, nA0, [L])
  p_alloc_nq=fftw_alloc_real(int(L, C_SIZE_T))
  call c_f_pointer(p_alloc_nq, nA0q, [L])
  p_alloc_n=fftw_alloc_real(int(L, C_SIZE_T))
  call c_f_pointer(p_alloc_n, nB0, [L])
  p_alloc_nq=fftw_alloc_real(int(L, C_SIZE_T))
  call c_f_pointer(p_alloc_nq, nB0q, [L])
  !real(8) :: n_tmp(L)
  p_alloc_n=fftw_alloc_real(int(L, C_SIZE_T))
  call c_f_pointer(p_alloc_n, n_tmp, [L])
  
! for 1D FFTW
  scale1D_b=1.d0/L ! for backward FFT

  do i=1,L
     if(i.le.L/2+1) then
        qu(i)=2.d0*pi*(i-1)/(L*du)
     else
        qu(i)=-2.d0*pi*(L-i+1)/(L*du)
     endif
     qu2(i)=qu(i)*qu(i)
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
# ifdef coeffs1D_only ! only calculate coeffs1D based on conf.dat equilibrium profiles
  open(unit=2,file=file_i//'_para.dat',position='append')
# else
  open(unit=2,file=filename//'_para.dat',position='append')
# endif
  write(2,*) 'use multi-threaded fftw; nthreads=',nthreads
  close(2)
# endif
  
! initialization for FFTW (out of place; create the plans before initializing the inputs)
  plan1d_c_f=fftw_plan_dft_1d(L,A_tmp,Aq_tmp,FFTW_FORWARD,FFTW_MEASURE)
  plan1d_c_b=fftw_plan_dft_1d(L,Aq_tmp,A_tmp,FFTW_BACKWARD,FFTW_MEASURE)
  plan1d_r_f=fftw_plan_r2r_1d(L,nA0,nA0q,FFTW_R2HC,FFTW_MEASURE)
  plan1d_r_b=fftw_plan_r2r_1d(L,nA0q,nA0,FFTW_HC2R,FFTW_MEASURE)

# ifdef coeffs1D_only ! only calculate coeffs1D based on conf.dat equilibrium profiles
  open(unit=2,file=file_i//'_para.dat',position='append')
  write(2,*)
  write(2,*) 'def coeffs1D_only:  only calculate coeffs1D based on conf.dat equilibrium profiles'
  close(2)

  open(unit=1,file='data/'//filename//'_conf.dat',form='unformatted',status='old')
  read(1) A0,B0,nA0,nB0
  close(1)
  do j=1,3
     A_tmp=A0(j,1:L)
     call fftw_execute_dft(plan1d_c_f,A_tmp,Aq_tmp) ! forward FFT (complex)
     do i=1,L
        A0q(j,i)=Aq_tmp(i)
     enddo
     A_tmp=B0(j,1:L)
     call fftw_execute_dft(plan1d_c_f,A_tmp,Aq_tmp) ! forward FFT (complex)
     do i=1,L
        B0q(j,i)=Aq_tmp(i)
     enddo
  enddo
  n_tmp=nA0
  call fftw_execute_r2r(plan1d_r_f,n_tmp,nA0q) ! forward FFT (real)
  n_tmp=nB0
  call fftw_execute_r2r(plan1d_r_f,n_tmp,nB0q) ! forward FFT (real)

  do i=1,L
     !G_0(1,i)=-qu2(i)+(2*qx*sin_theta+qy*cos_theta)*qu(i)
     G_0(1,i)=-qu2(i)+dchi_j(1)*qu(i)
     !G_0(2,i)=-(qu2(i)+2*qy*cos_theta*qu(i))
     G_0(2,i)=-qu2(i)+dchi_j(2)*qu(i)
     !G_0(3,i)=-qu2(i)-(2*qx*sin_theta-qy*cos_theta)*qu(i)
     G_0(3,i)=-qu2(i)+dchi_j(3)*qu(i)
  enddo
  goto 100
# endif

  yy=ran3(idum)

! set the initial conditions
  
  if(ntype.eq.1) then
    !
    ! put the initial configuration by hand 
# ifdef ini_tanh ! use initial tanh profile
     do i=1,L/2
        A0(1,i)=((A1_s+A_l)/2+(A1_s-A_l)/2*tanh((i-i1)*du/wid_A))*(1+noise*(ran3(idum)-0.5d0))
        A0(2,i)=((A2_s+A_l)/2+(A2_s-A_l)/2*tanh((i-i1)*du/wid_A))*(1+noise*(ran3(idum)-0.5d0))
        A0(3,i)=((A3_s+A_l)/2+(A3_s-A_l)/2*tanh((i-i1)*du/wid_A))*(1+noise*(ran3(idum)-0.5d0))
        B0(1,i)=((B1_s+B_l)/2+(B1_s-B_l)/2*tanh((i-i1)*du/wid_A))*(1+noise*(ran3(idum)-0.5d0))
        B0(2,i)=((B2_s+B_l)/2+(B2_s-B_l)/2*tanh((i-i1)*du/wid_A))*(1+noise*(ran3(idum)-0.5d0))
        B0(3,i)=((B3_s+B_l)/2+(B3_s-B_l)/2*tanh((i-i1)*du/wid_A))*(1+noise*(ran3(idum)-0.5d0))
        nA0(i)=((nA0_s+nA0_l)/2+(nA0_s-nA0_l)/2*tanh((i-i1)*du/wid_n))*(1+noise*(ran3(idum)-0.5d0))
        nB0(i)=((nB0_s+nB0_l)/2+(nB0_s-nB0_l)/2*tanh((i-i1)*du/wid_n))*(1+noise*(ran3(idum)-0.5d0))
     enddo
     do i=L/2+1,L
        A0(1,i)=((A1_s+A_l)/2+(A_l-A1_s)/2*tanh((i-i2)*du/wid_A))*(1+noise*(ran3(idum)-0.5d0))
        A0(2,i)=((A2_s+A_l)/2+(A_l-A2_s)/2*tanh((i-i2)*du/wid_A))*(1+noise*(ran3(idum)-0.5d0))
        A0(3,i)=((A3_s+A_l)/2+(A_l-A3_s)/2*tanh((i-i2)*du/wid_A))*(1+noise*(ran3(idum)-0.5d0))
        B0(1,i)=((B1_s+B_l)/2+(B_l-B1_s)/2*tanh((i-i2)*du/wid_A))*(1+noise*(ran3(idum)-0.5d0))
        B0(2,i)=((B2_s+B_l)/2+(B_l-B2_s)/2*tanh((i-i2)*du/wid_A))*(1+noise*(ran3(idum)-0.5d0))
        B0(3,i)=((B3_s+B_l)/2+(B_l-B3_s)/2*tanh((i-i2)*du/wid_A))*(1+noise*(ran3(idum)-0.5d0))
        nA0(i)=((nA0_s+nA0_l)/2+(nA0_l-nA0_s)/2*tanh((i-i2)*du/wid_n))*(1+noise*(ran3(idum)-0.5d0))
        nB0(i)=((nB0_s+nB0_l)/2+(nB0_l-nB0_s)/2*tanh((i-i2)*du/wid_n))*(1+noise*(ran3(idum)-0.5d0))
     enddo
# else
     do i=1,L
!        u=(i-1)*du
# ifdef liq_sol
        ! for liquid_solid
        if(i.ge.i1.and.i.le.i2) then ! solid region
# endif
           A0(1,i)=A1_s*(1+noise*(ran3(idum)-0.5d0))
           A0(2,i)=A2_s*(1+noise*(ran3(idum)-0.5d0))
           A0(3,i)=A3_s*(1+noise*(ran3(idum)-0.5d0))
           B0(1,i)=B1_s*(1+noise*(ran3(idum)-0.5d0))
           B0(2,i)=B2_s*(1+noise*(ran3(idum)-0.5d0))
           B0(3,i)=B3_s*(1+noise*(ran3(idum)-0.5d0))
           if(abs(nA0_s).lt.1.d-10) then
              nA0(i)=nA0_s+noise0*(ran3(idum)-0.5d0) ! if nA0_s=0
           else
              !nA0(i)=nA0_s+noise*(ran3(idum)-0.5d0)
              nA0(i)=nA0_s*(1+noise*(ran3(idum)-0.5d0))
           endif
           if(abs(nB0_s).lt.1.d-10) then
              nB0(i)=nB0_s+noise0*(ran3(idum)-0.5d0) ! if nB0_s=0
           else
              !nB0(i)=nB0_s+noise*(ran3(idum)-0.5d0)
              nB0(i)=nB0_s*(1+noise*(ran3(idum)-0.5d0))
           endif
# ifdef liq_sol
        ! for liquid_solid
        else ! liquid region
           if(abs(nA0_l).lt.1.d-10) then
              nA0(i)=nA0_l+noise0*(ran3(idum)-0.5d0) ! if nA0_l=0
           else
              !nA0(i)=nA0_l+noise*(ran3(idum)-0.5d0)
              nA0(i)=nA0_l*(1+noise*(ran3(idum)-0.5d0))
           endif
           if(abs(nB0_l).lt.1.d-10) then
              nB0(i)=nB0_l+noise0*(ran3(idum)-0.5d0) ! if nB0_l=0
           else
              !nB0(i)=nB0_l+noise*(ran3(idum)-0.5d0)
              nB0(i)=nB0_l*(1+noise*(ran3(idum)-0.5d0))
           endif
           !if(abs(A_l).lt.1.d-10) then ! liquid film
           do j=1,3
              !A0(j,i)=CMPLX(A_l+noise0*(ran3(idum)-0.5d0),noise0*(ran3(idum)-0.5d0))
              !B0(j,i)=CMPLX(B_l+noise0*(ran3(idum)-0.5d0),noise0*(ran3(idum)-0.5d0))
              A0(j,i)=A_l+noise0*(ran3(idum)-0.5d0)
              B0(j,i)=B_l+noise0*(ran3(idum)-0.5d0)
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
     A_tmp=A0(j,1:L)
     call fftw_execute_dft(plan1d_c_f,A_tmp,Aq_tmp) ! forward FFT (complex)
     do i=1,L
        A0q(j,i)=Aq_tmp(i)
     enddo
     A_tmp=B0(j,1:L)
     call fftw_execute_dft(plan1d_c_f,A_tmp,Aq_tmp) ! forward FFT (complex)
     do i=1,L
        B0q(j,i)=Aq_tmp(i)
     enddo
  enddo
  n_tmp=nA0
  call fftw_execute_r2r(plan1d_r_f,n_tmp,nA0q) ! forward FFT (real)
  n_tmp=nB0
  call fftw_execute_r2r(plan1d_r_f,n_tmp,nB0q) ! forward FFT (real)
  if(ntype.eq.1) then
     if(abs(nA0_s-nB0_s).lt.1.d-10.and.abs(nA0_l-nB0_l).lt.1.d-10) then
        nB0q(1)=nA0q(1)
        call fftw_execute_r2r(plan1d_r_b,nB0q,nB0) ! backward FFT (real)
        nB0=nB0*scale1D_b
        n_tmp=nB0
        call fftw_execute_r2r(plan1d_r_f,n_tmp,nB0q) ! forward FFT (real)
     endif
  endif

  time=time0
  call interface(A_top,ystar1A,ystar2A,L_mid)
  ystar1A_0=ystar1A
  ystar2A_0=ystar2A
  open(unit=7,file=filename//'_ystar.dat',position='append')
  write(7,*) time, ystar1A, ystar2A
  close(7)
  call phase_output(filename//'_phase_t0.dat',filename//'_A0B0n0_t0.dat')

  open(unit=9,file=filename//'_out.dat',position='append')
  write(9,*) 't=',time
  write(9,*) '<nA0>=',sum(nA0)*scale1d_b,nA0q(1)*scale1d_b
  write(9,*) '<nB0>=',sum(nB0)*scale1d_b,nB0q(1)*scale1d_b
  write(9,*) 'A0=',maxval(abs(A0(1:3,1:L))),minval(abs(A0(1:3,1:L)))
  write(9,*) 'B0=',maxval(abs(B0(1:3,1:L))),minval(abs(B0(1:3,1:L)))
  write(9,*) 'nA0=', maxval(nA0(1:L)),minval(nA0(1:L))
  write(9,*) 'nB0=', maxval(nB0(1:L)),minval(nB0(1:L))
  close(9)

  if(iter0.le.n_i) then
     dtime=dti ! for initial transient
  else
     dtime=dt
  endif

  q02_mB=q02*mB
  isigma=0
  isigmaj=0
  do i=1,L
     !G_0(1,i)=-qu2(i)+(2*qx*sin_theta+qy*cos_theta)*qu(i)
     G_0(1,i)=-qu2(i)+dchi_j(1)*qu(i)
     !G_0(2,i)=-(qu2(i)+2*qy*cos_theta*qu(i))
     G_0(2,i)=-qu2(i)+dchi_j(2)*qu(i)
     !G_0(3,i)=-qu2(i)-(2*qx*sin_theta-qy*cos_theta)*qu(i)
     G_0(3,i)=-qu2(i)+dchi_j(3)*qu(i)
     
! for A_j, B_j
     do j=1,3
        alpha_11=-q02*(beta_A*(G_0(j,i)+qA2-q02-deltaj0(j))**2-epsA)
        alpha_22=-q02_mB*(beta_B*(G_0(j,i)+qB2-q02-deltaj0(j))**2-epsB)
        alpha_j12(j,i)=-q02*(alpha_AB+beta_AB*(G_0(j,i)+qAB2-q02-deltaj0(j))**2)
        alpha_j21(j,i)=mB*alpha_j12(j,i)
        
        b=(alpha_11-alpha_22)**2+4*alpha_j12(j,i)*alpha_j21(j,i)
        if(b.lt.0) then
           open(unit=9,file=filename//'_out.dat',position='append')
           write(9,*) 'error: \Delta(Aj,Bj) < 0 at i=',i, '  qu(i)=',qu(i), '  for j=',j
           close(9)
           stop
        endif
        delta_2=sqrt(b) ! note: delta_2 is real here
        if(abs(delta_2).lt.1.d-8) then ! sigma1=sigma2
           isigmaj(j,i)=1
           sigmaj1(j,i)=(alpha_11+alpha_22)/2
           sigmaj2(j,i)=sigmaj1(j,i)
           sigj1_alpha(j,i)=sigmaj1(j,i)-alpha_11

           sig1_dt=sigmaj1(j,i)*dtime
           exp_j11(j,i)=exp(sig1_dt)*(1-sigj1_alpha(j,i)*dtime)
           exp_j12(j,i)=exp(sig1_dt)*alpha_j12(j,i)*dtime
           exp_j21(j,i)=exp(sig1_dt)*alpha_j21(j,i)*dtime
           exp_j22(j,i)=exp(sig1_dt)*(1+sigj1_alpha(j,i)*dtime)
           if(abs(sigmaj1(j,i)).lt.1.d-4) then
              cf_j1(j,i)=dtime*(1.d0+0.5d0*sig1_dt*(1.d0+sig1_dt/3.d0))
              cf2_j1(j,i)=0.5d0*dtime*(1.d0+sig1_dt*(1.d0+0.25d0*sig1_dt)/3.d0)
              cf_j2(j,i)=dtime*dtime*(0.5d0+sig1_dt/3+sig1_dt*sig1_dt/8)
              cf2_j2(j,i)=dtime*dtime*(1.d0/6.d0+sig1_dt/12+sig1_dt*sig1_dt/40)
           else
              cf_j1(j,i)=(exp(sig1_dt)-1)/sigmaj1(j,i)
              cf2_j1(j,i)=(exp(sig1_dt)-(1+sig1_dt))/(sigmaj1(j,i)*sig1_dt)
              cf_j2(j,i)=(1+exp(sig1_dt)*(sig1_dt-1))/(sigmaj1(j,i)*sigmaj1(j,i))
              cf2_j2(j,i)=(2+sig1_dt+(sig1_dt-2)*exp(sig1_dt))/(sigmaj1(j,i)*sigmaj1(j,i)*sig1_dt)
           endif

        else
! use the algorithm of Numerical Recipes: Sec. 5.6
              b=-(alpha_11+alpha_22)
              c=alpha_11*alpha_22-alpha_j12(j,i)*alpha_j21(j,i)
              if(b.ge.0) then ! note: here b is real in this case
                 aq=-(b+delta_2)/2
              else
                 aq=-(b-delta_2)/2
              endif
              sigmaj1(j,i)=aq
              sigmaj2(j,i)=c/aq

           sigmaj12(j,i)=sigmaj1(j,i)-sigmaj2(j,i)
           sigj1_alpha(j,i)=sigmaj1(j,i)-alpha_22
           sigj2_alpha(j,i)=sigmaj2(j,i)-alpha_22

           sig1_dt=sigmaj1(j,i)*dtime
           sig2_dt=sigmaj2(j,i)*dtime
           exp_j11(j,i)=sigj1_alpha(j,i)*exp(sig1_dt)-sigj2_alpha(j,i)*exp(sig2_dt)
           exp_j12(j,i)=alpha_j12(j,i)*(exp(sig1_dt)-exp(sig2_dt))
           exp_j21(j,i)=alpha_j21(j,i)*(exp(sig1_dt)-exp(sig2_dt))
           exp_j22(j,i)=sigj1_alpha(j,i)*exp(sig2_dt)-sigj2_alpha(j,i)*exp(sig1_dt)
           if(abs(sigmaj1(j,i)).lt.1.d-4) then
              cf_j1(j,i)=dtime*(1.d0+0.5d0*sig1_dt*(1.d0+sig1_dt/3.d0))
              cf2_j1(j,i)=0.5d0*dtime*(1.d0+sig1_dt*(1.d0+0.25d0*sig1_dt)/3.d0)
           else
              cf_j1(j,i)=(exp(sig1_dt)-1)/sigmaj1(j,i)
              cf2_j1(j,i)=(exp(sig1_dt)-(1+sig1_dt))/(sigmaj1(j,i)*sig1_dt)
           endif
           if(abs(sigmaj2(j,i)).lt.1.d-4) then
              cf_j2(j,i)=dtime*(1.d0+0.5d0*sig2_dt*(1.d0+sig2_dt/3.d0))
              cf2_j2(j,i)=0.5d0*dtime*(1.d0+sig2_dt*(1.d0+0.25d0*sig2_dt)/3.d0)
           else
              cf_j2(j,i)=(exp(sig2_dt)-1)/sigmaj2(j,i)
              cf2_j2(j,i)=(exp(sig2_dt)-(1+sig2_dt))/(sigmaj2(j,i)*sig2_dt)
           endif
        endif

     enddo ! do j=1,3

! for nA0 and nB0 (note: r2r kind, with qu2(i) already matching that of the 2nd half for imaginary part)
     alpha_11=-qu2(i)*(beta_A*qA2*qA2-epsA) ! if neglecting gradients in n0; mA=1
     alpha_22=-mB*qu2(i)*(beta_B*qB2*qB2-epsB)
     alpha_12(i)=-qu2(i)*(alpha_AB+beta_AB*qAB2*qAB2)
     alpha_21(i)=mB*alpha_12(i)

     b=(alpha_11-alpha_22)**2+4*alpha_12(i)*alpha_21(i)
     if(b.lt.0) then
        open(unit=9,file=filename//'_out.dat',position='append')
        write(9,*) 'error: \Delta(n0) < 0 at i=',i, '  qu(i)=',qu(i)
        close(9)
        stop
     endif
     delta_2=sqrt(b) ! note: delta_2 is real here
     if(abs(delta_2).lt.1.d-8) then ! sigma1=sigma2
        isigma(i)=1
        sigma1(i)=(alpha_11+alpha_22)/2
        sigma2(i)=sigma1(i)
        sig1_alpha(i)=sigma1(i)-alpha_11

        sig1_dt=sigma1(i)*dtime
        exp_11(i)=exp(sig1_dt)*(1-sig1_alpha(i)*dtime)
        exp_12(i)=exp(sig1_dt)*alpha_12(i)*dtime
        exp_21(i)=exp(sig1_dt)*alpha_21(i)*dtime
        exp_22(i)=exp(sig1_dt)*(1+sig1_alpha(i)*dtime)
        if(abs(sigma1(i)).lt.1.d-4) then
           cf_1(i)=dtime*(1.d0+0.5d0*sig1_dt*(1.d0+sig1_dt/3.d0))
           cf2_1(i)=0.5d0*dtime*(1.d0+sig1_dt*(1.d0+0.25d0*sig1_dt)/3.d0)
           cf_2(i)=dtime*dtime*(0.5d0+sig1_dt/3+sig1_dt*sig1_dt/8)
           cf2_2(i)=dtime*dtime*(1.d0/6.d0+sig1_dt/12+sig1_dt*sig1_dt/40)
        else
           cf_1(i)=(exp(sig1_dt)-1)/sigma1(i)
           cf2_1(i)=(exp(sig1_dt)-(1+sig1_dt))/(sigma1(i)*sig1_dt)
           cf_2(i)=(1+exp(sig1_dt)*(sig1_dt-1))/(sigma1(i)*sigma1(i))
           cf2_2(i)=(2+sig1_dt+(sig1_dt-2)*exp(sig1_dt))/(sigma1(i)*sigma1(i)*sig1_dt)
        endif

     else
! use the algorithm of Numerical Recipes: Sec. 5.6
        b=-(alpha_11+alpha_22)
        c=alpha_11*alpha_22-alpha_12(i)*alpha_21(i)
        if(b.ge.0) then ! note: here b is real in this case
           aq=-(b+delta_2)/2
        else
           aq=-(b-delta_2)/2
        endif
        sigma1(i)=aq
        sigma2(i)=c/aq

        sigma12(i)=sigma1(i)-sigma2(i)
        sig1_alpha(i)=sigma1(i)-alpha_22
        sig2_alpha(i)=sigma2(i)-alpha_22

        sig1_dt=sigma1(i)*dtime
        sig2_dt=sigma2(i)*dtime
        exp_11(i)=sig1_alpha(i)*exp(sig1_dt)-sig2_alpha(i)*exp(sig2_dt)
        exp_12(i)=alpha_12(i)*(exp(sig1_dt)-exp(sig2_dt))
        exp_21(i)=alpha_21(i)*(exp(sig1_dt)-exp(sig2_dt))
        exp_22(i)=sig1_alpha(i)*exp(sig2_dt)-sig2_alpha(i)*exp(sig1_dt)
        if(abs(sigma1(i)).lt.1.d-4) then
           cf_1(i)=dtime*(1.d0+0.5d0*sig1_dt*(1.d0+sig1_dt/3.d0))
           cf2_1(i)=0.5d0*dtime*(1.d0+sig1_dt*(1.d0+0.25d0*sig1_dt)/3.d0)
        else
           cf_1(i)=(exp(sig1_dt)-1)/sigma1(i)
           cf2_1(i)=(exp(sig1_dt)-(1+sig1_dt))/(sigma1(i)*sig1_dt)
        endif
        if(abs(sigma2(i)).lt.1.d-4) then
           cf_2(i)=dtime*(1.d0+0.5d0*sig2_dt*(1.d0+sig2_dt/3.d0))
           cf2_2(i)=0.5d0*dtime*(1.d0+sig2_dt*(1.d0+0.25d0*sig2_dt)/3.d0)
        else
           cf_2(i)=(exp(sig2_dt)-1)/sigma2(i)
           cf2_2(i)=(exp(sig2_dt)-(1+sig2_dt))/(sigma2(i)*sig2_dt)
        endif
     endif

  enddo ! do i=1,L

! start running
!
  i_ystar=0 ! for liquid/film interfaces
  do iter=iter0,nend

     if(iter.le.n_i) then ! for initial transient only
        time=iter*dti
     else
        time=(iter-n_i)*dt+t_i
        if(iter.eq.n_i+1) then
           dtime=dt
           do i=1,L
! for A_j and B_j
              do j=1,3
              if(isigmaj(j,i).eq.1) then ! sigma1=sigma2
                 sig1_dt=sigmaj1(j,i)*dtime
                 exp_j11(j,i)=exp(sig1_dt)*(1-sigj1_alpha(j,i)*dtime)
                 exp_j12(j,i)=exp(sig1_dt)*alpha_j12(j,i)*dtime
                 exp_j21(j,i)=exp(sig1_dt)*alpha_j21(j,i)*dtime
                 exp_j22(j,i)=exp(sig1_dt)*(1+sigj1_alpha(j,i)*dtime)
                 if(abs(sigmaj1(j,i)).lt.1.d-4) then
                    cf_j1(j,i)=dtime*(1.d0+0.5d0*sig1_dt*(1.d0+sig1_dt/3.d0))
                    cf2_j1(j,i)=0.5d0*dtime*(1.d0+sig1_dt*(1.d0+0.25d0*sig1_dt)/3.d0)
                    cf_j2(j,i)=dtime*dtime*(0.5d0+sig1_dt/3+sig1_dt*sig1_dt/8)
                    cf2_j2(j,i)=dtime*dtime*(1.d0/6.d0+sig1_dt/12+sig1_dt*sig1_dt/40)
                 else
                    cf_j1(j,i)=(exp(sig1_dt)-1)/sigmaj1(j,i)
                    cf2_j1(j,i)=(exp(sig1_dt)-(1+sig1_dt))/(sigmaj1(j,i)*sig1_dt)
                    cf_j2(j,i)=(1+exp(sig1_dt)*(sig1_dt-1))/(sigmaj1(j,i)*sigmaj1(j,i))
                    cf2_j2(j,i)=(2+sig1_dt+(sig1_dt-2)*exp(sig1_dt))/(sigmaj1(j,i)*sigmaj1(j,i)*sig1_dt)
                 endif
              else
                 sig1_dt=sigmaj1(j,i)*dtime
                 sig2_dt=sigmaj2(j,i)*dtime
                 exp_j11(j,i)=sigj1_alpha(j,i)*exp(sig1_dt)-sigj2_alpha(j,i)*exp(sig2_dt)
                 exp_j12(j,i)=alpha_j12(j,i)*(exp(sig1_dt)-exp(sig2_dt))
                 exp_j21(j,i)=alpha_j21(j,i)*(exp(sig1_dt)-exp(sig2_dt))
                 exp_j22(j,i)=sigj1_alpha(j,i)*exp(sig2_dt)-sigj2_alpha(j,i)*exp(sig1_dt)
                 if(abs(sigmaj1(j,i)).lt.1.d-4) then
                    cf_j1(j,i)=dtime*(1.d0+0.5d0*sig1_dt*(1.d0+sig1_dt/3.d0))
                    cf2_j1(j,i)=0.5d0*dtime*(1.d0+sig1_dt*(1.d0+0.25d0*sig1_dt)/3.d0)
                 else
                    cf_j1(j,i)=(exp(sig1_dt)-1)/sigmaj1(j,i)
                    cf2_j1(j,i)=(exp(sig1_dt)-(1+sig1_dt))/(sigmaj1(j,i)*sig1_dt)
                 endif
                 if(abs(sigmaj2(j,i)).lt.1.d-4) then
                    cf_j2(j,i)=dtime*(1.d0+0.5d0*sig2_dt*(1.d0+sig2_dt/3.d0))
                    cf2_j2(j,i)=0.5d0*dtime*(1.d0+sig2_dt*(1.d0+0.25d0*sig2_dt)/3.d0)
                 else
                    cf_j2(j,i)=(exp(sig2_dt)-1)/sigmaj2(j,i)
                    cf2_j2(j,i)=(exp(sig2_dt)-(1+sig2_dt))/(sigmaj2(j,i)*sig2_dt)
                 endif
              endif
              enddo ! do j=1,3
           
! for nA0 and nB0
              if(isigma(i).eq.1) then ! sigma1=sigma2
                 sig1_dt=sigma1(i)*dtime
                 exp_11(i)=exp(sig1_dt)*(1-sig1_alpha(i)*dtime)
                 exp_12(i)=exp(sig1_dt)*alpha_12(i)*dtime
                 exp_21(i)=exp(sig1_dt)*alpha_21(i)*dtime
                 exp_22(i)=exp(sig1_dt)*(1+sig1_alpha(i)*dtime)
                 if(abs(sigma1(i)).lt.1.d-4) then
                    cf_1(i)=dtime*(1.d0+0.5d0*sig1_dt*(1.d0+sig1_dt/3.d0))
                    cf2_1(i)=0.5d0*dtime*(1.d0+sig1_dt*(1.d0+0.25d0*sig1_dt)/3.d0)
                    cf_2(i)=dtime*dtime*(0.5d0+sig1_dt/3+sig1_dt*sig1_dt/8)
                    cf2_2(i)=dtime*dtime*(1.d0/6.d0+sig1_dt/12+sig1_dt*sig1_dt/40)
                 else
                    cf_1(i)=(exp(sig1_dt)-1)/sigma1(i)
                    cf2_1(i)=(exp(sig1_dt)-(1+sig1_dt))/(sigma1(i)*sig1_dt)
                    cf_2(i)=(1+exp(sig1_dt)*(sig1_dt-1))/(sigma1(i)*sigma1(i))
                    cf2_2(i)=(2+sig1_dt+(sig1_dt-2)*exp(sig1_dt))/(sigma1(i)*sigma1(i)*sig1_dt)
                 endif
              else
                 sig1_dt=sigma1(i)*dtime
                 sig2_dt=sigma2(i)*dtime
                 exp_11(i)=sig1_alpha(i)*exp(sig1_dt)-sig2_alpha(i)*exp(sig2_dt)
                 exp_12(i)=alpha_12(i)*(exp(sig1_dt)-exp(sig2_dt))
                 exp_21(i)=alpha_21(i)*(exp(sig1_dt)-exp(sig2_dt))
                 exp_22(i)=sig1_alpha(i)*exp(sig2_dt)-sig2_alpha(i)*exp(sig1_dt)
                 if(abs(sigma1(i)).lt.1.d-4) then
                    cf_1(i)=dtime*(1.d0+0.5d0*sig1_dt*(1.d0+sig1_dt/3.d0))
                    cf2_1(i)=0.5d0*dtime*(1.d0+sig1_dt*(1.d0+0.25d0*sig1_dt)/3.d0)
                 else
                    cf_1(i)=(exp(sig1_dt)-1)/sigma1(i)
                    cf2_1(i)=(exp(sig1_dt)-(1+sig1_dt))/(sigma1(i)*sig1_dt)
                 endif
                 if(abs(sigma2(i)).lt.1.d-4) then
                    cf_2(i)=dtime*(1.d0+0.5d0*sig2_dt*(1.d0+sig2_dt/3.d0))
                    cf2_2(i)=0.5d0*dtime*(1.d0+sig2_dt*(1.d0+0.25d0*sig2_dt)/3.d0)
                 else
                    cf_2(i)=(exp(sig2_dt)-1)/sigma2(i)
                    cf2_2(i)=(exp(sig2_dt)-(1+sig2_dt))/(sigma2(i)*sig2_dt)
                 endif
              endif

           enddo ! do i=1,L
        endif
     endif

     call An(time,iter,istep,filename)

    if(((iter.eq.n_i).or.(iter.gt.n_i.and.mod(iter+n_0,nout_v).eq.0)).and.(i_ystar.eq.0)) then
      call interface(A_top,ystar1A,ystar2A,L_mid)
      if((ystar1A.le.1.or.ystar1A.ge.L_mid).and.(ystar2A.ge.L.or.ystar2A.le.L_mid)) then
         i_ystar=1
         goto 110
      endif
      vb1A=(ystar1A_0-ystar1A)*du/dtime
      vb2A=(ystar2A-ystar2A_0)*du/dtime
      open(unit=10,file=filename//'_vb.dat',position='append')
      write(10,*) time,vb1A,vb2A
      close(10)
      open(unit=7,file=filename//'_ystar.dat',position='append')
      write(7,*) time, ystar1A, ystar2A
      close(7)
    endif
    if(((iter+1.eq.n_i).or.(iter.gt.n_i.and.mod(iter+n_0+1,nout_v).eq.0)).and.(i_ystar.eq.0)) then 
      call interface(A_top,ystar1A_0,ystar2A_0,L_mid)
    endif
!
110 continue

    if((iter.eq.n_i).or.(iter.gt.n_i.and.mod(iter+n_0,raout).eq.0)) then
       open(unit=9,file=filename//'_out.dat',position='append')
       write(9,*)
       write(9,*) 't=',time,'  istep=',istep
       write(9,*) '<nA0>=',sum(nA0)*scale1d_b,nA0q(1)*scale1d_b
       write(9,*) '<nB0>=',sum(nB0)*scale1d_b,nB0q(1)*scale1d_b
       write(9,*) 'A0=',maxval(abs(A0(1:3,1:L))),minval(abs(A0(1:3,1:L)))
       write(9,*) 'B0=',maxval(abs(B0(1:3,1:L))),minval(abs(B0(1:3,1:L)))
       write(9,*) 'nA0=', maxval(nA0(1:L)),minval(nA0(1:L))
       write(9,*) 'nB0=', maxval(nB0(1:L)),minval(nB0(1:L))
       close(9)

       call surface_tension(time,filename) ! calculate surface tension directly from simulation (not analytic eq)

! output amplitudes and phases
       do j=1,n_time
         if(iter.eq.itime(j)) then
            write(cha,'(i0)') NINT(time)
            call phase_output(filename//'_phase_t'//trim(cha)//'.dat',filename//'_A0B0n0_t'//trim(cha)//'.dat')
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
  call surface_tension(time,filename) ! calculate surface tension directly from simulation (not analytic eq)
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

100 continue
# ifdef coeffs1D_only ! only calculate coeffs1D based on conf.dat equilibrium profiles
  call surface_tension(tmax,file_i) ! calculate surface tension directly from simulation (not analytic eq)
# endif

! calculate coefficients from 1D 0th-order stationary/equilibrium solution
# ifdef coeffs1D_only ! only calculate coeffs1D based on conf.dat equilibrium profiles
  call coeffs1D(file_i)
# else
  call coeffs1D(filename)
# endif

  call fftw_destroy_plan(plan1d_r_f)
  call fftw_destroy_plan(plan1d_r_b)
  call fftw_destroy_plan(plan1d_c_f)
  call fftw_destroy_plan(plan1d_c_b)

  call fftw_free(p_alloc_A)
  call fftw_free(p_alloc_Aq)
  call fftw_free(p_alloc_n)
  call fftw_free(p_alloc_nq)
  
end Program bampl1D_AB


! main routine that solves the 0th order amplitude eqs
!
subroutine An(time,iter,istep,file)
  use global_variables
  implicit none

  common/converge/TOL,err_s,nmbr_eval

  real(8) :: time
  integer :: iter,i,j,nmbr_eval,istep
  real(8) :: TOL,err_s,max_conv_A,max_conv_B,max_conv_nA0,max_conv_nB0,conv,abs_psi
  character file*(*)

  complex(C_DOUBLE_COMPLEX) :: Nonlin_A_q(3,L),sigN_A0q(3,L)
  complex(C_DOUBLE_COMPLEX) :: Nonlin_B_q(3,L),sigN_B0q(3,L)
  real(C_DOUBLE) :: Nonlin1_q(L),Nonlin2_q(L)
  complex(C_DOUBLE_COMPLEX) :: A0_0(3,L),A0q_0(3,L),B0_0(3,L),B0q_0(3,L)!,Aq_tmp(L),A_tmp(L)
  real(C_DOUBLE) :: sigN_nA0q(L),sigN_nB0q(L),sigN_1,sigN_2
  complex(C_DOUBLE_COMPLEX) :: sigN_j1,sigN_j2
  complex(C_DOUBLE_COMPLEX), pointer :: Aq_tmp(:),A_tmp(:)
  real(C_DOUBLE), pointer :: nA0_0(:),nA0q_0(:),nB0_0(:),nB0q_0(:),nq_tmp(:)
  type(C_PTR) :: p_alloc_A_tmp,p_alloc_Aq_tmp
  type(C_PTR) :: p_alloc_nA0_0,p_alloc_nA0q_0,p_alloc_nB0_0,p_alloc_nB0q_0,p_alloc_nq_tmp

  p_alloc_A_tmp=fftw_alloc_complex(int(L, C_SIZE_T))
  call c_f_pointer(p_alloc_A_tmp, A_tmp, [L])
  p_alloc_Aq_tmp=fftw_alloc_complex(int(L, C_SIZE_T))
  call c_f_pointer(p_alloc_Aq_tmp, Aq_tmp, [L])
  
  p_alloc_nA0_0=fftw_alloc_real(int(L, C_SIZE_T))
  call c_f_pointer(p_alloc_nA0_0, nA0_0, [L])
  p_alloc_nA0q_0=fftw_alloc_real(int(L, C_SIZE_T))
  call c_f_pointer(p_alloc_nA0q_0, nA0q_0, [L])
  p_alloc_nB0_0=fftw_alloc_real(int(L, C_SIZE_T))
  call c_f_pointer(p_alloc_nB0_0, nB0_0, [L])
  p_alloc_nB0q_0=fftw_alloc_real(int(L, C_SIZE_T))
  call c_f_pointer(p_alloc_nB0q_0, nB0q_0, [L])
  p_alloc_nq_tmp=fftw_alloc_real(int(L, C_SIZE_T))
  call c_f_pointer(p_alloc_nq_tmp, nq_tmp, [L])
  
!  calculate the nonlinear terms
  call nonlin_q(Nonlin_A_q,Nonlin_B_q,Nonlin1_q,Nonlin2_q)

! First step: predictor

!$OMP PARALLEL PRIVATE(i,j,sigN_1,sigN_2,sigN_j1,sigN_j2)

! for Aj,Bj
  do j=1,3
!$OMP DO
!(Note: DO loop index variable is by default PRIVATE, all other variables are by default SHARED unless declared)
  do i=1,L
     if(isigmaj(j,i).eq.1) then ! sigma1=sigma2
        sigN_j2=alpha_j12(j,i)*Nonlin_B_q(j,i)-sigj1_alpha(j,i)*Nonlin_A_q(j,i)
        A0q_0(j,i)=exp_j11(j,i)*A0q(j,i)+exp_j12(j,i)*B0q(j,i) &
                  +cf_j1(j,i)*Nonlin_A_q(j,i)+cf_j2(j,i)*sigN_j2
        if(nmbr_eval.gt.1) sigN_A0q(j,i)=-(sigN_j2*cf2_j2(j,i)+Nonlin_A_q(j,i)*cf2_j1(j,i)) ! for use in iteration
        sigN_j2=alpha_j21(j,i)*Nonlin_A_q(j,i)+sigj1_alpha(j,i)*Nonlin_B_q(j,i)
        B0q_0(j,i)=exp_j21(j,i)*A0q(j,i)+exp_j22(j,i)*B0q(j,i) &
                  +cf_j1(j,i)*Nonlin_B_q(j,i)+cf_j2(j,i)*sigN_j2
        if(nmbr_eval.gt.1) sigN_B0q(j,i)=-(sigN_j2*cf2_j2(j,i)+Nonlin_B_q(j,i)*cf2_j1(j,i)) ! for use in iteration
     else
        sigN_j1=sigj1_alpha(j,i)*Nonlin_A_q(j,i)+alpha_j12(j,i)*Nonlin_B_q(j,i)
        sigN_j2=sigj2_alpha(j,i)*Nonlin_A_q(j,i)+alpha_j12(j,i)*Nonlin_B_q(j,i)
        A0q_0(j,i)=(exp_j11(j,i)*A0q(j,i)+exp_j12(j,i)*B0q(j,i) &
                  +cf_j1(j,i)*sigN_j1-cf_j2(j,i)*sigN_j2)/sigmaj12(j,i)
        if(nmbr_eval.gt.1) sigN_A0q(j,i)=(sigN_j2*cf2_j2(j,i)-sigN_j1*cf2_j1(j,i))/sigmaj12(j,i) ! for use in iteration
        sigN_j1=alpha_j21(j,i)*Nonlin_A_q(j,i)-sigj2_alpha(j,i)*Nonlin_B_q(j,i)
        sigN_j2=alpha_j21(j,i)*Nonlin_A_q(j,i)-sigj1_alpha(j,i)*Nonlin_B_q(j,i)
        B0q_0(j,i)=(exp_j21(j,i)*A0q(j,i)+exp_j22(j,i)*B0q(j,i) &
                  +cf_j1(j,i)*sigN_j1-cf_j2(j,i)*sigN_j2)/sigmaj12(j,i)
        if(nmbr_eval.gt.1) sigN_B0q(j,i)=(sigN_j2*cf2_j2(j,i)-sigN_j1*cf2_j1(j,i))/sigmaj12(j,i) ! for use in iteration
     endif
  enddo
!$OMP END DO
  enddo

! for nA0,nB0
!$OMP DO
  do i=1,L
     if(isigma(i).eq.1) then ! sigma1=sigma2
        sigN_2=alpha_12(i)*Nonlin2_q(i)-sig1_alpha(i)*Nonlin1_q(i)
        nA0q_0(i)=exp_11(i)*nA0q(i)+exp_12(i)*nB0q(i) &
                  +cf_1(i)*Nonlin1_q(i)+cf_2(i)*sigN_2
        if(nmbr_eval.gt.1) sigN_nA0q(i)=-(sigN_2*cf2_2(i)+Nonlin1_q(i)*cf2_1(i)) ! for use in iteration
        sigN_2=alpha_21(i)*Nonlin1_q(i)+sig1_alpha(i)*Nonlin2_q(i)
        nB0q_0(i)=exp_21(i)*nA0q(i)+exp_22(i)*nB0q(i) &
                  +cf_1(i)*Nonlin2_q(i)+cf_2(i)*sigN_2
        if(nmbr_eval.gt.1) sigN_nB0q(i)=-(sigN_2*cf2_2(i)+Nonlin2_q(i)*cf2_1(i)) ! for use in iteration
     else
        sigN_1=sig1_alpha(i)*Nonlin1_q(i)+alpha_12(i)*Nonlin2_q(i)
        sigN_2=sig2_alpha(i)*Nonlin1_q(i)+alpha_12(i)*Nonlin2_q(i)
        nA0q_0(i)=(exp_11(i)*nA0q(i)+exp_12(i)*nB0q(i) &
                  +cf_1(i)*sigN_1-cf_2(i)*sigN_2)/sigma12(i)
        if(nmbr_eval.gt.1) sigN_nA0q(i)=(sigN_2*cf2_2(i)-sigN_1*cf2_1(i))/sigma12(i) ! for use in iteration
        sigN_1=alpha_21(i)*Nonlin1_q(i)-sig2_alpha(i)*Nonlin2_q(i)
        sigN_2=alpha_21(i)*Nonlin1_q(i)-sig1_alpha(i)*Nonlin2_q(i)
        nB0q_0(i)=(exp_21(i)*nA0q(i)+exp_22(i)*nB0q(i) &
                  +cf_1(i)*sigN_1-cf_2(i)*sigN_2)/sigma12(i)
        if(nmbr_eval.gt.1) sigN_nB0q(i)=(sigN_2*cf2_2(i)-sigN_1*cf2_1(i))/sigma12(i) ! for use in iteration
     endif
  enddo
!$OMP END DO

!$OMP END PARALLEL

  if(iter.eq.1) then
     open(unit=9,file=file//'_out.dat',position='append')
     write(9,*)
     write(9,*) 'iter=',iter, '  time=',time
     write(9,*) 'Nonlin_A_q=',maxval(abs(Nonlin_A_q(1:3,1:L))),minval(abs(Nonlin_A_q(1:3,1:L)))
     write(9,*) 'Nonlin_B_q=',maxval(abs(Nonlin_B_q(1:3,1:L))),minval(abs(Nonlin_B_q(1:3,1:L)))
     write(9,*) 'Nonlin_nA0_q=',maxval(Nonlin1_q(1:L)),minval(Nonlin1_q(1:L))
     write(9,*) 'Nonlin_nB0_q=',maxval(Nonlin2_q(1:L)),minval(Nonlin2_q(1:L))
     write(9,*) 'A0q=',maxval(abs(A0q(1:3,1:L))),minval(abs(A0q(1:3,1:L)))
     write(9,*) 'B0q=',maxval(abs(B0q(1:3,1:L))),minval(abs(B0q(1:3,1:L)))
     write(9,*) 'nA0q=',maxval(nA0q(1:L)),minval(nA0q(1:L))
     write(9,*) 'nB0q=',maxval(nB0q(1:L)),minval(nB0q(1:L))
     write(9,*) 'A0q_0=',maxval(abs(A0q_0(1:3,1:L))),minval(abs(A0q_0(1:3,1:L)))
     write(9,*) 'B0q_0=',maxval(abs(B0q_0(1:3,1:L))),minval(abs(B0q_0(1:3,1:L)))
     write(9,*) 'nA0q_0=',maxval(nA0q_0(1:L)),minval(nA0q_0(1:L))
     write(9,*) 'nB0q_0=',maxval(nB0q_0(1:L)),minval(nB0q_0(1:L))
     write(9,*) 'nA0q_0(1)=',nA0q_0(1), '  nB0q_0(1)=',nB0q_0(1)
     write(9,*) 'sig1_alpha=',maxval(sig1_alpha(1:L)),minval(sig1_alpha(1:L))
     write(9,*) 'sig2_alpha=',maxval(sig2_alpha(1:L)),minval(sig2_alpha(1:L))
     write(9,*) 'alpha_12=',maxval(alpha_12(1:L)),minval(alpha_12(1:L))
     write(9,*) 'alpha_21=',maxval(alpha_21(1:L)),minval(alpha_21(1:L))
     write(9,*) 'sigma12(2:L)=',maxval(sigma12(2:L)),minval(sigma12(2:L))
     do i=2,L
        if(abs(sigma12(i)).le.1.d-10) then
           write(9,*) 'i=',i, '  sigma12(i)=',sigma12(i)
           write(9,*) 'isigma(i)=',isigma(i), '  qu2(i)=',qu2(i)
        endif
     enddo
     write(9,*) 'exp_11=',maxval(exp_11(1:L)),minval(exp_11(1:L))
     write(9,*) 'exp_12=',maxval(exp_12(1:L)),minval(exp_12(1:L))
     write(9,*) 'exp_21=',maxval(exp_21(1:L)),minval(exp_21(1:L))
     write(9,*) 'exp_22=',maxval(exp_22(1:L)),minval(exp_22(1:L))
     write(9,*) 'cf_1=',maxval(cf_1(1:L)),minval(cf_1(1:L))
     write(9,*) 'cf_2=',maxval(cf_2(1:L)),minval(cf_2(1:L))
     write(9,*) 'isigma(2:L)=',maxval(isigma(2:L)),minval(isigma(2:L))
     write(9,*) 'isigmaj(1:3,1:L)=',maxval(isigmaj(1:3,1:L)),minval(isigmaj(1:3,1:L))
     close(9)
  endif
  
  do j=1,3
     Aq_tmp=A0q_0(j,1:L)
     call fftw_execute_dft(plan1d_c_b,Aq_tmp,A_tmp) ! backward FFT (complex)
     do i=1,L
        A0_0(j,i)=A_tmp(i)*scale1D_b
     enddo
     Aq_tmp=B0q_0(j,1:L)
     call fftw_execute_dft(plan1d_c_b,Aq_tmp,A_tmp) ! backward FFT (complex)
     do i=1,L
        B0_0(j,i)=A_tmp(i)*scale1D_b
     enddo
  enddo
  nq_tmp=nA0q_0
  call fftw_execute_r2r(plan1d_r_b,nq_tmp,nA0_0) ! backward FFT (real)
  nA0_0=nA0_0*scale1D_b
  nq_tmp=nB0q_0
  call fftw_execute_r2r(plan1d_r_b,nq_tmp,nB0_0) ! backward FFT (real)
  nB0_0=nB0_0*scale1D_b

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
!(      thus PRIVATE(i,j) is by default and not needed to declare here)
     do i=1,L
        if(isigmaj(j,i).eq.1) then ! sigma1=sigma2
           A0q(j,i)=A0q_0(j,i)+cf2_j1(j,i)*Nonlin_A_q(j,i) &
                   +cf2_j2(j,i)*(alpha_j12(j,i)*Nonlin_B_q(j,i)-sigj1_alpha(j,i)*Nonlin_A_q(j,i))
           B0q(j,i)=B0q_0(j,i)+cf2_j1(j,i)*Nonlin_B_q(j,i) &
                   +cf2_j2(j,i)*(alpha_j21(j,i)*Nonlin_A_q(j,i)+sigj1_alpha(j,i)*Nonlin_B_q(j,i))
        else
           A0q(j,i)=A0q_0(j,i)+(cf2_j1(j,i)*(sigj1_alpha(j,i)*Nonlin_A_q(j,i)+alpha_j12(j,i)*Nonlin_B_q(j,i)) &
                   -cf2_j2(j,i)*(sigj2_alpha(j,i)*Nonlin_A_q(j,i)+alpha_j12(j,i)*Nonlin_B_q(j,i)))/sigmaj12(j,i)
           B0q(j,i)=B0q_0(j,i)+(cf2_j1(j,i)*(alpha_j21(j,i)*Nonlin_A_q(j,i)-sigj2_alpha(j,i)*Nonlin_B_q(j,i)) &
                   -cf2_j2(j,i)*(alpha_j21(j,i)*Nonlin_A_q(j,i)-sigj1_alpha(j,i)*Nonlin_B_q(j,i)))/sigmaj12(j,i)
        endif
     enddo
!$OMP END DO
     enddo

! for nA0,nB0
!$OMP DO
     do i=1,L
        if(isigma(i).eq.1) then ! sigma1=sigma2
           nA0q(i)=nA0q_0(i)+cf2_1(i)*Nonlin1_q(i) &
                   +cf2_2(i)*(alpha_12(i)*Nonlin2_q(i)-sig1_alpha(i)*Nonlin1_q(i))
           nB0q(i)=nB0q_0(i)+cf2_1(i)*Nonlin2_q(i) &
                   +cf2_2(i)*(alpha_21(i)*Nonlin1_q(i)+sig1_alpha(i)*Nonlin2_q(i))
        else
           nA0q(i)=nA0q_0(i)+(cf2_1(i)*(sig1_alpha(i)*Nonlin1_q(i)+alpha_12(i)*Nonlin2_q(i)) &
                   -cf2_2(i)*(sig2_alpha(i)*Nonlin1_q(i)+alpha_12(i)*Nonlin2_q(i)))/sigma12(i)
           nB0q(i)=nB0q_0(i)+(cf2_1(i)*(alpha_21(i)*Nonlin1_q(i)-sig2_alpha(i)*Nonlin2_q(i)) &
                   -cf2_2(i)*(alpha_21(i)*Nonlin1_q(i)-sig1_alpha(i)*Nonlin2_q(i)))/sigma12(i)
        endif
     enddo
!$OMP END DO

!$OMP END PARALLEL
     
     do j=1,3
        Aq_tmp=A0q(j,1:L)
        call fftw_execute_dft(plan1d_c_b,Aq_tmp,A_tmp) ! backward FFT (complex)
        do i=1,L
           A0(j,i)=A_tmp(i)*scale1D_b
        enddo
        Aq_tmp=B0q(j,1:L)
        call fftw_execute_dft(plan1d_c_b,Aq_tmp,A_tmp) ! backward FFT (complex)
        do i=1,L
           B0(j,i)=A_tmp(i)*scale1D_b
        enddo
     enddo
     nq_tmp=nA0q
     call fftw_execute_r2r(plan1d_r_b,nq_tmp,nA0) ! backward FFT (real)
     nA0=nA0*scale1D_b
     nq_tmp=nB0q
     call fftw_execute_r2r(plan1d_r_b,nq_tmp,nB0) ! backward FFT (real)
     nB0=nB0*scale1D_b

! real space (A0,B0,nA0,nB0) convergence

     max_conv_A=0.d0
     do j=1,3
        do i=1,L
           abs_psi=abs(A0(j,i))
           if(abs_psi.gt.1.d5) then
              open(unit=2,file=file//'_out.dat',position='append')
              write(2,*) 'A0(j,i) diverges at t=',time,' istep=',istep, &
                         '  j=',j,' i=',i,' A0=',A0(j,i)
              close(2)
              stop
           endif
           if(abs_psi.le.err_s) then
              conv=abs(A0(j,i)-A0_0(j,i))
           else
              conv=abs(A0(j,i)-A0_0(j,i))/abs(A0(j,i))
           endif
           if(conv.gt.max_conv_A) max_conv_A=conv
        enddo
     enddo

     max_conv_B=0.d0
     do j=1,3
        do i=1,L
           abs_psi=abs(B0(j,i))
           if(abs_psi.gt.1.d5) then
              open(unit=2,file=file//'_out.dat',position='append')
              write(2,*) 'B0(j,i) diverges at t=',time,' istep=',istep, &
                         '  j=',j,' i=',i,' B0=',B0(j,i)
              close(2)
              stop
           endif
           if(abs_psi.le.err_s) then
              conv=abs(B0(j,i)-B0_0(j,i))
           else
              conv=abs(B0(j,i)-B0_0(j,i))/abs(B0(j,i))
           endif
           if(conv.gt.max_conv_B) max_conv_B=conv
        enddo
     enddo
     
     max_conv_nA0=0.d0
     do i=1,L
        abs_psi=abs(nA0(i))
        if(abs_psi.gt.1.d5) then
           open(unit=2,file=file//'_out.dat',position='append')
           write(2,*) 'nA0 diverges at t=',time,' istep=',istep,' i=',i,' nA0(i)=',nA0(i)
           close(2)
           stop
        endif
        if(abs_psi.le.err_s) then
           conv=abs(nA0(i)-nA0_0(i))
        else
           conv=abs((nA0(i)-nA0_0(i))/nA0(i))
        endif
        if(conv.gt.max_conv_nA0) max_conv_nA0=conv
     enddo

     max_conv_nB0=0.d0
     do i=1,L
        abs_psi=abs(nB0(i))
        if(abs_psi.gt.1.d5) then
           open(unit=2,file=file//'_out.dat',position='append')
           write(2,*) 'nB0 diverges at t=',time,' istep=',istep,' i=',i,' nB0(i)=',nB0(i)
           close(2)
           stop
        endif
        if(abs_psi.le.err_s) then
           conv=abs(nB0(i)-nB0_0(i))
        else
           conv=abs((nB0(i)-nB0_0(i))/nB0(i))
        endif
        if(conv.gt.max_conv_nB0) max_conv_nB0=conv
     enddo

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

  enddo ! for istep=2,nmbr_eval

  istep=istep-1
  if(nmbr_eval.gt.2) then
     open(unit=2,file=file//'_out.dat',position='append')
     write(2,*)
     write(2,*) 'Exceeding maximum iterations=',nmbr_eval
     write(2,*) 't=',time,'   TOL=',TOL
     write(2,*) 'max_conv_A=',max_conv_A
     write(2,*) 'A0=',maxval(abs(A0(1:3,1:L))),minval(abs(A0(1:3,1:L)))
     write(2,*) 'max_conv_B=',max_conv_B
     write(2,*) 'B0=',maxval(abs(B0(1:3,1:L))),minval(abs(B0(1:3,1:L)))
     write(2,*) 'max_conv_nA0=',max_conv_nA0
     write(2,*) 'nA0=',maxval(nA0(1:L)),minval(nA0(1:L))
     write(2,*) 'max_conv_nB0=',max_conv_nB0
     write(2,*) 'nB0=',maxval(nB0(1:L)),minval(nB0(1:L))
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

  complex(C_DOUBLE_COMPLEX) :: Nonlin_A_q(3,L),Nonlin_B_q(3,L)
  real(C_DOUBLE) :: Nonlin_nA0_q(L),Nonlin_nB0_q(L)
  complex(C_DOUBLE_COMPLEX) :: A(L),Aq(L),A12,A13,A23,B12,B13,B23,AB12,AB13,AB23
  real(C_DOUBLE) :: n(L),nq(L)
  real(8) :: A_R,A_I,A02(3),B02(3),B_R,ng6A,ng6B,nAB,uw_nAB,A2,B2
  integer :: i,j

!$OMP PARALLEL PRIVATE(A_R,A_I,A02,B02,B_R,A23,A13,A12,B12,B13,B23,AB12,AB13,AB23,ng6A,ng6B,nAB,uw_nAB,A2,B2)
!$OMP DO
  ! (Note: PRIVATE(i,j) is by default and not needed to declare here)
  do i=1,L
     do j=1,3
        A_R=REAL(A0(j,i))
        A_I=AIMAG(A0(j,i))
        A02(j)=A_R*A_R+A_I*A_I
        A_R=REAL(B0(j,i))
        A_I=AIMAG(B0(j,i))
        B02(j)=A_R*A_R+A_I*A_I
     enddo
     A23=conjg(A0(2,i)*A0(3,i))
     A13=conjg(A0(1,i)*A0(3,i))
     A12=conjg(A0(1,i)*A0(2,i))
     B23=conjg(B0(2,i)*B0(3,i))
     B13=conjg(B0(1,i)*B0(3,i))
     B12=conjg(B0(1,i)*B0(2,i))
     AB23=conjg(A0(2,i)*B0(3,i)+A0(3,i)*B0(2,i))
     AB13=conjg(A0(1,i)*B0(3,i)+A0(3,i)*B0(1,i))
     AB12=conjg(A0(1,i)*B0(2,i)+A0(2,i)*B0(1,i))

     ng6A=6*vA*nA0(i)-2*gA
     nAB=nA0(i)*(3*vA*nA0(i)-2*gA)+w*nB0(i)
     uw_nAB=w*nA0(i)+u*nB0(i)
     Nonlin_A_q(1,i)=(nAB+3*vA*(A02(1)+2*(A02(2)+A02(3))))*A0(1,i)+ng6A*A23+uw_nAB*B0(1,i)+u*B23+w*AB23
     Nonlin_A_q(2,i)=(nAB+3*vA*(A02(2)+2*(A02(1)+A02(3))))*A0(2,i)+ng6A*A13+uw_nAB*B0(2,i)+u*B13+w*AB13
     Nonlin_A_q(3,i)=(nAB+3*vA*(A02(3)+2*(A02(1)+A02(2))))*A0(3,i)+ng6A*A12+uw_nAB*B0(3,i)+u*B12+w*AB12

     ng6B=6*vB*nB0(i)-2*gB
     nAB=nB0(i)*(3*vB*nB0(i)-2*gB)+u*nA0(i)
     Nonlin_B_q(1,i)=(nAB+3*vB*(B02(1)+2*(B02(2)+B02(3))))*B0(1,i)+ng6B*B23+uw_nAB*A0(1,i)+w*A23+u*AB23
     Nonlin_B_q(2,i)=(nAB+3*vB*(B02(2)+2*(B02(1)+B02(3))))*B0(2,i)+ng6B*B13+uw_nAB*A0(2,i)+w*A13+u*AB13
     Nonlin_B_q(3,i)=(nAB+3*vB*(B02(3)+2*(B02(1)+B02(2))))*B0(3,i)+ng6B*B12+uw_nAB*A0(3,i)+w*A12+u*AB12

     A_R=real(A0(1,i)*A0(2,i)*A0(3,i))
     A2=A02(1)+A02(2)+A02(3)
     B_R=real(B0(1,i)*B0(2,i)*B0(3,i))
     B2=B02(1)+B02(2)+B02(3)
     A_I=real(A0(1,i)*conjg(B0(1,i))+A0(2,i)*conjg(B0(2,i))+A0(3,i)*conjg(B0(3,i)))

     Nonlin_nA0_q(i)=nA0(i)*nA0(i)*(vA*nA0(i)-gA)+(w*nA0(i)+0.5d0*u*nB0(i))*nB0(i)+ng6A*A2+12*vA*A_R+u*B2+2*w*A_I
     Nonlin_nB0_q(i)=nB0(i)*nB0(i)*(vB*nB0(i)-gB)+(u*nB0(i)+0.5d0*w*nA0(i))*nA0(i)+ng6B*B2+12*vB*B_R+w*A2+2*u*A_I
  enddo
!$OMP END DO
!$OMP END PARALLEL

  do j=1,3
     A=Nonlin_A_q(j,1:L)
     call fftw_execute_dft(plan1d_c_f,A,Aq) ! forward FFT (complex)
     do i=1,L
        Nonlin_A_q(j,i)=Aq(i)
     enddo
     A=Nonlin_B_q(j,1:L)
     call fftw_execute_dft(plan1d_c_f,A,Aq) ! forward FFT (complex)
     do i=1,L
        Nonlin_B_q(j,i)=Aq(i)
     enddo
  enddo

  Nonlin_A_q=-q02*Nonlin_A_q
  Nonlin_B_q=-q02_mB*Nonlin_B_q ! q02_mB = mB*q0^2

  n=Nonlin_nA0_q
  call fftw_execute_r2r(plan1d_r_f,n,nq) ! forward FFT (real)
  Nonlin_nA0_q=-qu2*nq
  n=Nonlin_nB0_q
  call fftw_execute_r2r(plan1d_r_f,n,nq) ! forward FFT (real)
  Nonlin_nB0_q=-mB*qu2*nq

  return
end subroutine nonlin_q


! to determine film/liquid interfaces: ystar1 (bottom one) and ystar2 (top one)
! note: A0_s > A0_l(f)
subroutine interface(A_top,ystar1A,ystar2A,L_mid)
  use global_variables
  implicit none

  real(8) :: psij,psijp
  real(8) :: A_top,ystar1A,ystar2A
  integer :: j,L_mid

  psijp=sum(abs(A0(1:3,1)))/3
! for A_s > A_l
  if(psijp.ge.A_top) then
     ystar1A=1
     goto 11
  endif
  do j=2,L_mid
     psij=sum(abs(A0(1:3,j)))/3
! for A_s > A_l
     if(psij.ge.A_top) then
        ystar1A=j+(A_top-psij)/(psij-psijp) ! psijp=psi(j-1)
        goto 11
     endif
     psijp=psij
  enddo
  ystar1A=L_mid
11 continue

  psijp=sum(abs(A0(1:3,L)))/3
! for A_s > A_l
  if(psijp.ge.A_top) then
     ystar2A=L
     goto 12
  endif
  do j=L-1,L_mid,-1
     psij=sum(abs(A0(1:3,j)))/3
! for A_s > A_l
     if(psij.ge.A_top) then
        ystar2A=j+(A_top-psij)/(psijp-psij) ! psijp=psi(j+1)
        goto 12
     endif
     psijp=psij
  enddo
  ystar2A=L_mid
12 continue

  return
end subroutine interface


! output amplitudes and phases
subroutine phase_output(phase_t,An_t)
  use global_variables
  implicit none

  character phase_t*(*),An_t*(*)

  real(8) :: phi_A(3,L), phiA_old(3)
  real(8) :: phi_B(3,L), phiB_old(3)
  real(8) :: pi2, A_R, A_I
  integer :: i,j,iA(3),ii

!  pi=DACOS(-1.0d0)
  pi2=2.d0*pi
!  pi_2=pi/2.d0

!  phiA_old=100.d0
!  iA=L/2
  iA=1

  open(unit=16,file=An_t,status='unknown')
  open(unit=15,file=phase_t,status='unknown')
! do i=L/2,L
  do i=1,L
     do j=1,3
        
       A_R=REAL(A0(j,i))
       A_I=AIMAG(A0(j,i))
!       if((dabs(A_R).lt.1.e-5).and.(dabs(A_I).lt.1.e-5)) then
       if((dabs(A_R).lt.5.e-5).and.(dabs(A_I).lt.5.e-5)) then
          phi_A(j,i)=0.d0
       else
          phi_A(j,i)=datan(A_I/A_R)
          if(A_R.lt.0.) phi_A(j,i)=phi_A(j,i)+pi
!          if(phi_A(j,i).lt.0.) phi_A(j,i)=phi_A(j,i)+pi2
       endif

       if(i.gt.1) then
       if(phi_A(j,i)-phiA_old(j).gt.pi) then
          do ii=iA(j),i-1
             phi_A(j,ii)=phi_A(j,ii)+pi2
          enddo
!          iA(j)=i
       else
          if(phi_A(j,i)-phiA_old(j).lt.-pi) then
             do ii=iA(j),i-1
                phi_A(j,ii)=phi_A(j,ii)-pi2
             enddo
          endif
       endif
       endif
       phiA_old(j)=phi_A(j,i)
       
       A_R=REAL(B0(j,i))
       A_I=AIMAG(B0(j,i))
!       if((dabs(A_R).lt.1.e-5).and.(dabs(A_I).lt.1.e-5)) then
       if((dabs(A_R).lt.5.e-5).and.(dabs(A_I).lt.5.e-5)) then
          phi_B(j,i)=0.d0
       else
          phi_B(j,i)=datan(A_I/A_R)
          if(A_R.lt.0.) phi_B(j,i)=phi_B(j,i)+pi
!          if(phi_B(j,i).lt.0.) phi_B(j,i)=phi_B(j,i)+pi2
       endif

       if(i.gt.1) then
       if(phi_B(j,i)-phiB_old(j).gt.pi) then
          do ii=iA(j),i-1
             phi_B(j,ii)=phi_B(j,ii)+pi2
          enddo
!          iA(j)=i
       else
          if(phi_B(j,i)-phiB_old(j).lt.-pi) then
             do ii=iA(j),i-1
                phi_B(j,ii)=phi_B(j,ii)-pi2
             enddo
          endif
       endif
       endif
       phiB_old(j)=phi_B(j,i)
       
    enddo
  enddo

  write(16,*) '# i=1,L, abs(A0(1,i)),abs(A0(2,i)),abs(A0(3,i)),abs(B0(1,i)),abs(B0(2,i)),abs(B0(3,i)),nA0(i),nB0(i)'
  write(15,*) '# i=1,L, phi_A(1,i),phi_A(2,i),phi_A(3,i),phi_B(1,i),phi_B(2,i),phi_B(3,i)'
! do i=L/2,L
  do i=1,L
    write(16,51) i,abs(A0(1,i)),abs(A0(2,i)),abs(A0(3,i)),abs(B0(1,i)),abs(B0(2,i)),abs(B0(3,i)),nA0(i),nB0(i)
    write(15,52) i,phi_A(1,i),phi_A(2,i),phi_A(3,i),phi_B(1,i),phi_B(2,i),phi_B(3,i)
  enddo
  close(15)
  close(16)

51  format(1x,i8,8(1x,es17.10))
52  format(1x,i8,6(1x,es17.10))

  return
end subroutine phase_output


! note: here use ave(n0) for determing interface position and calculating surface tension directly from simulation
subroutine surface_tension(time,file)
  use global_variables
  implicit none

  common/q12/G_0(3,L),deltaj0(3),qA2,qB2,qAB2
  common/surf/du,epsA,epsB,beta_A,beta_B,beta_AB,alpha_AB,width_per
  common/interf/m_gap_nA(2),m_gap_nB(2),width_nA(2),width_nB(2),width_A(2),width_B(2),gamma_nA,gamma_nB,Es

  complex(8) :: A(L),Aq(L),B(L)
  real(8) :: duA0(L),nq(L)
  real(8) :: G_0,deltaj0,qA2,qB2,qAB2
  real(8) :: time,du,epsA,epsB,beta_A,beta_B,beta_AB,alpha_AB
  integer :: i,j
  real(8) :: A_R,A_I,A02(3),B02(3),ng6A,ng6B,ng3A,ng3B
  real(8) :: en,enj,enl,enx,nA0_avg,nB0_avg,nA0_s,nA0_l,nB0_s,nB0_l
  real(8) :: Es,A02_s,B02_s,dphi,phi_A,phi_B
  real(8) :: gamma_nA,gamma_nB
  real(8) :: width_nA,width_nB,width_A,width_B,m_gap_nA,m_gap_nB,width_per
  character file*(*)

  nA0_avg=nA0q(1)*scale1d_b ! average nA0
  nB0_avg=nB0q(1)*scale1d_b ! average nB0
  
! calculate the derivatives in F formula
  duA0=0
  do j=1,3
     do i=1,L
        Aq(i)=A0q(j,i)*(G_0(j,i)+qA2-q02-deltaj0(j))
     enddo
     call fftw_execute_dft(plan1d_c_b,Aq,A) ! backward FFT (complex)
     A=A*scale1D_b
     do i=1,L
        Aq(i)=B0q(j,i)*(G_0(j,i)+qB2-q02-deltaj0(j))
     enddo
     call fftw_execute_dft(plan1d_c_b,Aq,B) ! backward FFT (complex)
     B=B*scale1D_b
     do i=1,L
        A_R=REAL(A(i))
        A_I=AIMAG(A(i))
        duA0(i)=duA0(i)+beta_A*(A_R*A_R+A_I*A_I) ! beta_A*\sum_j |(G_0-delta_jA^0) Aj0|^2
        A_R=REAL(B(i))
        A_I=AIMAG(B(i))
        duA0(i)=duA0(i)+beta_B*(A_R*A_R+A_I*A_I) ! beta_B*\sum_j |(G_0-delta_jB^0) Bj0|^2
     enddo
     if(beta_AB.gt.1.e-10) then
     do i=1,L
        Aq(i)=A0q(j,i)*(G_0(j,i)+qAB2-q02-deltaj0(j))
     enddo
     call fftw_execute_dft(plan1d_c_b,Aq,A) ! backward FFT (complex)
     A=A*scale1D_b
     do i=1,L
        Aq(i)=B0q(j,i)*(G_0(j,i)+qAB2-q02-deltaj0(j))
     enddo
     call fftw_execute_dft(plan1d_c_b,Aq,B) ! backward FFT (complex)
     B=B*scale1D_b
     do i=1,L
        duA0(i)=duA0(i)+beta_AB*2*REAL(A(i)*conjg(B(i))) ! beta_AB*\sum_j [(G_0-delta_jAB^0)Aj0][(G_0-delta_jAB^0)Bj0]^*
     enddo
     endif
  enddo

  en=0 ! total free energy
  do i=1,L
     do j=1,3
        A_R=REAL(A0(j,i))
        A_I=AIMAG(A0(j,i))
        A02(j)=A_R*A_R+A_I*A_I
        A_R=REAL(B0(j,i))
        A_I=AIMAG(B0(j,i))
        B02(j)=A_R*A_R+A_I*A_I
     enddo
     
     ng6A=6*vA*nA0(i)-2*gA
     ng3A=nA0(i)*(3*vA*nA0(i)-2*gA)+w*nB0(i)
     ng6B=6*vB*nB0(i)-2*gB
     ng3B=nB0(i)*(3*vB*nB0(i)-2*gB)+u*nA0(i)

     enj=duA0(i)+(-epsA+ng3A)*(A02(1)+A02(2)+A02(3))+1.5d0*vA*(A02(1)*A02(1)+A02(2)*A02(2)+A02(3)*A02(3)) &
        +2*ng6A*real(A0(1,i)*A0(2,i)*A0(3,i))+6*vA*(A02(1)*(A02(2)+A02(3))+A02(2)*A02(3)) &
        +(-epsB+ng3B)*(B02(1)+B02(2)+B02(3))+1.5d0*vB*(B02(1)*B02(1)+B02(2)*B02(2)+B02(3)*B02(3)) &
        +2*ng6B*real(B0(1,i)*B0(2,i)*B0(3,i))+6*vB*(B02(1)*(B02(2)+B02(3))+B02(2)*B02(3)) &
        +(alpha_AB+w*nA0(i)+u*nB0(i))*2*REAL(A0(1,i)*conjg(B0(1,i))+A0(2,i)*conjg(B0(2,i))+A0(3,i)*conjg(B0(3,i))) &
        +2*REAL(u*(A0(1,i)*B0(2,i)*B0(3,i)+A0(2,i)*B0(1,i)*B0(3,i)+A0(3,i)*B0(1,i)*B0(2,i)) &
        +w*(B0(1,i)*A0(2,i)*A0(3,i)+B0(2,i)*A0(1,i)*A0(3,i)+B0(3,i)*A0(1,i)*A0(2,i)))
     enj=enj+nA0(i)*nA0(i)*(0.5d0*(-epsA+beta_A*qA2*qA2)-gA*nA0(i)/3+vA*nA0(i)*nA0(i)/4) &
            +nB0(i)*nB0(i)*(0.5d0*(-epsB+beta_B*qB2*qB2)-gB*nB0(i)/3+vB*nB0(i)*nB0(i)/4) &
            +nA0(i)*nB0(i)*(alpha_AB+beta_AB*qAB2*qAB2+(w*nA0(i)+u*nB0(i))/2)
     en=en+enj
     
     if(i.eq.1) enl=enj ! liquid region
     if(i.eq.L/2) then ! solid region
        enx=enj
        A02_s=A02(1)
        B02_s=B02(1)
     endif
  enddo
  nA0_s=nA0(L/2)
  nA0_l=nA0(1)
  nB0_s=nB0(L/2)
  nB0_l=nB0(1)
  
! surface tension (from lever rule; see the notes and also Dantzig's book p72, WuPRB07)
! in equilibrium of liq-sol coexistence, (nA0_avg-nA0_l)/(nB0_avg-nB0_l) = (nA0_s-nA0_avg)/(nB0_s-nB0_avg)
  gamma_nA=du*(en+L*(enl*(nA0_avg-nA0_s)+enx*(nA0_l-nA0_avg))/(nA0_s-nA0_l))/2
  gamma_nB=du*(en+L*(enl*(nB0_avg-nB0_s)+enx*(nB0_l-nB0_avg))/(nB0_s-nB0_l))/2

  !pi=DACOS(-1.0d0)
  !pi2=2.d0*pi
  A_R=REAL(A0(1,L/2))
  A_I=AIMAG(A0(1,L/2))
!  if((dabs(A_R).lt.1.e-5).and.(dabs(A_I).lt.1.e-5)) then
  if((dabs(A_R).lt.5.e-5).and.(dabs(A_I).lt.5.e-5)) then
     phi_A=0.d0
  else
     phi_A=datan(A_I/A_R)
     if(A_R.lt.0.) phi_A=phi_A+pi
!     if(phi_A.lt.0.) phi_A=phi_A+pi2
  endif
  A_R=REAL(B0(1,L/2))
  A_I=AIMAG(B0(1,L/2))
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
  call width_gap(width_nA,width_nB,width_A,width_B,m_gap_nA,m_gap_nB,width_per,du)
  open(unit=8,file=file//'_width.dat',position='append')
  write(8,*) '# time,epsA,epsB,width_nA(1,2),width_nB(1,2),m_gap_nA(1,2),m_gap_nB(1,2),width_A(1,2),width_B(1,2)'
  write(8,50) time,epsA,epsB,width_nA(1),width_nA(2),width_nB(1),width_nB(2), &
              m_gap_nA(1),m_gap_nA(2),m_gap_nB(1),m_gap_nB(2),width_A(1),width_A(2),width_B(1),width_B(2)
50 format(es11.5,2(1x,f9.3),12(1x,es17.10))
  close(8)

  open(unit=9,file=file//'_surf.dat',position='append')
  if(time.lt.1.1) write(9,*) '# time,epsA,epsB,gamma_nA,gamma_nB,enx,enl,Es,&
                             (nA0_avg-nA0_l)*(nB0_s-nB0_avg)-(nB0_avg-nB0_l)*(nA0_s-nA0_avg)'
  write(9,51) time,epsA,epsB,gamma_nA,gamma_nB,enx,enl,Es,(nA0_avg-nA0_l)*(nB0_s-nB0_avg)-(nB0_avg-nB0_l)*(nA0_s-nA0_avg)
! gamma: surface tension; enx,enl: solid,liquid F density; 
! Es: Young's modulus; in equilibrium (nA0_avg-nA0_l)/(nB0_avg-nB0_l) = (nA0_s-nA0_avg)/(nB0_s-nB0_avg)
51 format(es11.5,2(1x,f9.3),6(1x,es17.10))
  close(9)
  open(unit=10,file=file//'_f.dat',position='append')
  write(10,*) time,en/L ! =en*du/(L*du) total free energy density
  close(10)

  return
end subroutine surface_tension


! for interface width and miscibility gap (determined by n0A or n0B)
subroutine width_gap(width_nA,width_nB,width_A,width_B,m_gap_nA,m_gap_nB,width_per,du)
  use global_variables
  implicit none

  real(8) :: width_nA(2),width_nB(2),m_gap_nA(2),m_gap_nB(2),width_per,du
  real(8) :: nA0_s,nA0_l,nB0_s,nB0_l,width_n1,width_n2,width1,width2
  real(8) :: width_A(2),width_B(2),A0_s,A0_l,m_gap_A,B0_s,B0_l,m_gap_B
  integer :: i

! for first half (1<i<L/2)
  nA0_s=nA0(L/2)
  nA0_l=nA0(1)
  nB0_s=nB0(L/2)
  nB0_l=nB0(1)
  A0_s=abs(A0(1,L/2))
  A0_l=abs(A0(1,1))
  B0_s=abs(B0(1,L/2))
  B0_l=abs(B0(1,1))

  m_gap_nA(1)=nA0_s-nA0_l
  m_gap_nB(1)=nB0_s-nB0_l
  m_gap_A=A0_s-A0_l
  m_gap_B=B0_s-B0_l

  if(abs(m_gap_nA(1)).gt.1.e-6) then
  width1=0
  width2=0
  width_n1=nA0_l+(1-width_per)*m_gap_nA(1)/2
  width_n2=nA0_s-(1-width_per)*m_gap_nA(1)/2
  do i=1,L/2
# ifdef n0s_lt_l ! if n0_s < n0_l
     if(nA0(i).le.width_n1) then
# else  ! if n0_s > n0_l(f)
     if(nA0(i).ge.width_n1) then
# endif
        width1=i+(width_n1-nA0(i))/(nA0(i)-nA0(i-1)) ! interpolation
        goto 1
     endif
  enddo
1 continue
  do i=L/2,1,-1
# ifdef n0s_lt_l ! if n0_s < n0_l
     if(nA0(i).ge.width_n2) then
# else ! if n0_s > n0_l(f)
     if(nA0(i).le.width_n2) then
# endif
        width2=i+(width_n2-nA0(i))/(nA0(i+1)-nA0(i)) ! interpolation
        goto 2
     endif
  enddo
2 continue
  width_nA(1)=(width2-width1)*du

  else
     width_nA(1)=0
  endif

  if(abs(m_gap_nB(1)).gt.1.e-6) then
  width1=0
  width2=0
  width_n1=nB0_l+(1-width_per)*m_gap_nB(1)/2
  width_n2=nB0_s-(1-width_per)*m_gap_nB(1)/2
  do i=1,L/2
# ifdef n0s_lt_l ! if n0_s < n0_l
     if(nB0(i).le.width_n1) then
# else  ! if n0_s > n0_l
     if(nB0(i).ge.width_n1) then
# endif
        width1=i+(width_n1-nB0(i))/(nB0(i)-nB0(i-1)) ! interpolation
        goto 3
     endif
  enddo
3 continue
  do i=L/2,1,-1
# ifdef n0s_lt_l ! if n0_s < n0_l
     if(nB0(i).ge.width_n2) then
# else ! if n0_s > n0_l
     if(nB0(i).le.width_n2) then
# endif
        width2=i+(width_n2-nB0(i))/(nB0(i+1)-nB0(i)) ! interpolation
        goto 4
     endif
  enddo
4 continue
  width_nB(1)=(width2-width1)*du

  else
     width_nB(1)=0
  endif

  if(abs(m_gap_A).gt.1.e-6) then
  width1=0
  width2=0
  width_n1=A0_l+(1-width_per)*m_gap_A/2
  width_n2=A0_s-(1-width_per)*m_gap_A/2
  do i=1,L/2
     if(abs(A0(1,i)).ge.width_n1) then
        width1=i+(width_n1-abs(A0(1,i)))/(abs(A0(1,i))-abs(A0(1,i-1))) ! interpolation
        goto 5
     endif
  enddo
5 continue
  do i=L/2,1,-1
     if(abs(A0(1,i)).le.width_n2) then
        width2=i+(width_n2-abs(A0(1,i)))/(abs(A0(1,i+1))-abs(A0(1,i))) ! interpolation
        goto 6
     endif
  enddo
6 continue
  width_A(1)=(width2-width1)*du

  else
     width_A(1)=0
  endif

  if(abs(m_gap_B).gt.1.e-6) then
  width1=0
  width2=0
  width_n1=B0_l+(1-width_per)*m_gap_B/2
  width_n2=B0_s-(1-width_per)*m_gap_B/2
  do i=1,L/2
     if(abs(B0(1,i)).ge.width_n1) then
        width1=i+(width_n1-abs(B0(1,i)))/(abs(B0(1,i))-abs(B0(1,i-1))) ! interpolation
        goto 7
     endif
  enddo
7 continue
  do i=L/2,1,-1
     if(abs(B0(1,i)).le.width_n2) then
        width2=i+(width_n2-abs(B0(1,i)))/(abs(B0(1,i+1))-abs(B0(1,i))) ! interpolation
        goto 8
     endif
  enddo
8 continue
  width_B(1)=(width2-width1)*du

  else
     width_B(1)=0
  endif

! for second half (L/2<i<L)
  nA0_s=nA0(L/2+1)
  nA0_l=nA0(L)
  nB0_s=nB0(L/2+1)
  nB0_l=nB0(L)
  A0_s=abs(A0(1,L/2+1))
  A0_l=abs(A0(1,L))
  B0_s=abs(B0(1,L/2+1))
  B0_l=abs(B0(1,L))

  m_gap_nA(2)=nA0_s-nA0_l
  m_gap_nB(2)=nB0_s-nB0_l
  m_gap_A=A0_s-A0_l
  m_gap_B=B0_s-B0_l

  if(abs(m_gap_nA(2)).gt.1.e-6) then
  width1=0
  width2=0
  width_n1=nA0_l+(1-width_per)*m_gap_nA(2)/2
  width_n2=nA0_s-(1-width_per)*m_gap_nA(2)/2
  do i=L,L/2+1,-1
# ifdef n0s_lt_l ! if n0_s < n0_l
     if(nA0(i).le.width_n1) then
# else  ! if n0_s > n0_l
     if(nA0(i).ge.width_n1) then
# endif
        width2=i+(width_n1-nA0(i))/(nA0(i+1)-nA0(i)) ! interpolation
        goto 11
     endif
  enddo
11 continue
  do i=L/2+1,L
# ifdef n0s_lt_l ! if n0_s < n0_l
     if(nA0(i).ge.width_n2) then
# else ! if n0_s > n0_l(f)
     if(nA0(i).le.width_n2) then
# endif
        width1=i+(width_n2-nA0(i))/(nA0(i)-nA0(i-1)) ! interpolation
        goto 12
     endif
  enddo
12 continue
  width_nA(2)=(width2-width1)*du

  else
     width_nA(2)=0
  endif

  if(abs(m_gap_nB(2)).gt.1.e-6) then
  width1=0
  width2=0
  width_n1=nB0_l+(1-width_per)*m_gap_nB(2)/2
  width_n2=nB0_s-(1-width_per)*m_gap_nB(2)/2
  do i=L,L/2+1,-1
# ifdef n0s_lt_l ! if n0_s < n0_l
     if(nB0(i).le.width_n1) then
# else  ! if n0_s > n0_l
     if(nB0(i).ge.width_n1) then
# endif
        width2=i+(width_n1-nB0(i))/(nB0(i+1)-nB0(i)) ! interpolation
        goto 13
     endif
  enddo
13 continue
  do i=L/2+1,L
# ifdef n0s_lt_l ! if n0_s < n0_l
     if(nB0(i).ge.width_n2) then
# else ! if n0_s > n0_l(f)
     if(nB0(i).le.width_n2) then
# endif
        width1=i+(width_n2-nB0(i))/(nB0(i)-nB0(i-1)) ! interpolation
        goto 14
     endif
  enddo
14 continue
  width_nB(2)=(width2-width1)*du

  else
     width_nB(2)=0
  endif

  if(abs(m_gap_A).gt.1.e-6) then
  width1=0
  width2=0
  width_n1=A0_l+(1-width_per)*m_gap_A/2
  width_n2=A0_s-(1-width_per)*m_gap_A/2
  do i=L,L/2+1,-1
     if(abs(A0(1,i)).ge.width_n1) then
        width2=i+(width_n1-abs(A0(1,i)))/(abs(A0(1,i+1))-abs(A0(1,i))) ! interpolation
        goto 15
     endif
  enddo
15 continue
  do i=L/2+1,L
     if(abs(A0(1,i)).le.width_n2) then
        width1=i+(width_n2-abs(A0(1,i)))/(abs(A0(1,i))-abs(A0(1,i-1))) ! interpolation
        goto 16
     endif
  enddo
16 continue
  width_A(2)=(width2-width1)*du

  else
     width_A(2)=0
  endif

  if(abs(m_gap_B).gt.1.e-6) then
  width1=0
  width2=0
  width_n1=B0_l+(1-width_per)*m_gap_B/2
  width_n2=B0_s-(1-width_per)*m_gap_B/2
  do i=L,L/2+1,-1
     if(abs(B0(1,i)).ge.width_n1) then
        width2=i+(width_n1-abs(B0(1,i)))/(abs(B0(1,i+1))-abs(B0(1,i))) ! interpolation
        goto 17
     endif
  enddo
17 continue
  do i=L/2+1,L
     if(abs(B0(1,i)).le.width_n2) then
        width1=i+(width_n2-abs(B0(1,i)))/(abs(B0(1,i))-abs(B0(1,i-1))) ! interpolation
        goto 18
     endif
  enddo
18 continue
  width_B(2)=(width2-width1)*du

  else
     width_B(2)=0
  endif

end subroutine width_gap


! calculate coefficients from 1D 0th-order stationary/equilibrium solution
subroutine coeffs1D(file)
  use global_variables
  implicit none
  
  common/q12/G_0(3,L),deltaj0(3),qA2,qB2,qAB2
  common/surf/du,epsA,epsB,beta_A,beta_B,beta_AB,alpha_AB,width_per
  common/interf/m_gap_nA(2),m_gap_nB(2),width_nA(2),width_nB(2),width_A(2),width_B(2),gamma_nA,gamma_nB,Es
  common/interf_coef/dchi_j(3),chi_j2(3),sin_theta,qu(L)

  real(8) :: sin_theta,theta,cos6theta
  real(8) :: G_0,deltaj0,dchi_j,chi_j2,qA2,qB2,qAB2
  real(8) :: du,epsA,epsB,beta_A,beta_B,beta_AB,alpha_AB,width_per,qu
  real(8) :: m_gap_nA,m_gap_nB,width_nA,width_nB,width_A,width_B,gamma_nA,gamma_nB,Es
  integer :: i,j,k
  integer :: i0_nA(2),i0_nB(2),imin(2),imax(2),i_min,i_max
  character file*(*)

  real(8) :: mu_nA0_eq,mu_nA0(L),mu_nB0_eq,mu_nB0(L),mu_min,mu_max
  real(8) :: zeta_D(2),zeta_DAB(2),sigma(2),dpsi0
  complex(8) :: Aa(L),Aaq(L),du2A0(3,L),duA0(3,L),Ba(L),Baq(L),du2B0(3,L),duB0(3,L)
  real(8) :: A_R,A_I,A02(3),B02(3),A2,B2,A03,duA2,duB2
  
  !pi=DACOS(-1.0d0)

  open(unit=2,file=file//'_effs1D.dat',position='append')

! for calculating chemical potentials mu_nA0_eq, mu_nB0_eq

  open(unit=3,file=file//'_mu.dat',position='append')
  do i=1,L
     do j=1,3
        A_R=REAL(A0(j,i))
        A_I=AIMAG(A0(j,i))
        A02(j)=A_R*A_R+A_I*A_I
        A_R=REAL(B0(j,i))
        A_I=AIMAG(B0(j,i))
        B02(j)=A_R*A_R+A_I*A_I
     enddo
     A2=A02(1)+A02(2)+A02(3)
     B2=B02(1)+B02(2)+B02(3)

     A03=real(A0(1,i)*A0(2,i)*A0(3,i))
     A_I=real(A0(1,i)*conjg(B0(1,i))+A0(2,i)*conjg(B0(2,i))+A0(3,i)*conjg(B0(3,i)))
     mu_nA0(i)=(-epsA+beta_A*qA2*qA2)*nA0(i)+(alpha_AB+beta_AB*qAB2*qAB2)*nB0(i) &
               +nA0(i)*nA0(i)*(vA*nA0(i)-gA)+(w*nA0(i)+0.5d0*u*nB0(i))*nB0(i) &
               +(6*vA*nA0(i)-2*gA)*A2+12*vA*A03+u*B2+2*w*A_I

     A03=real(B0(1,i)*B0(2,i)*B0(3,i))
     mu_nB0(i)=(-epsB+beta_B*qB2*qB2)*nB0(i)+(alpha_AB+beta_AB*qAB2*qAB2)*nA0(i) &
               +nB0(i)*nB0(i)*(vB*nB0(i)-gB)+(u*nB0(i)+0.5d0*w*nA0(i))*nA0(i) &
               +(6*vB*nB0(i)-2*gB)*B2+12*vB*A03+w*A2+2*u*A_I
     write(3,*) i, mu_nA0(i), mu_nB0(i)
  enddo
  close(3)

  mu_min=minval(mu_nA0)
  mu_max=maxval(mu_nA0)
  write(2,*) 'mu_nA0_max=',mu_max,'  mu_nA0_min=',mu_min
  write(2,*) 'mu_nA0(1)=',mu_nA0(1), '  mu_nA0(L/2)=',mu_nA0(L/2)
  mu_nA0_eq=sum(mu_nA0)/L
  write(2,*) 'mu_nA0_eq=<mu_nA0>=',mu_nA0_eq
  write(2,*) '|mu_max/mu_min-1|=',abs(mu_max/mu_min-1)
  if((abs(mu_nA0_eq).gt.1.d-10).and.(abs(mu_max/mu_min-1).gt.1.d-2)) then
     write(2,*) 'errors in mu_nA0 calculation (not equilibrium)'
!     close(2)
!     stop
  endif

  mu_min=minval(mu_nB0)
  mu_max=maxval(mu_nB0)
  write(2,*)
  write(2,*) 'mu_nB0_max=',mu_max,'  mu_nB0_min=',mu_min
  write(2,*) 'mu_nB0(1)=',mu_nB0(1), '  mu_nB0(L/2)=',mu_nB0(L/2)
  mu_nB0_eq=sum(mu_nB0)/L
  write(2,*) 'mu_nB0_eq=<mu_nB0>=',mu_nB0_eq
  write(2,*) '|mu_max/mu_min-1|=',abs(mu_max/mu_min-1)
  if((abs(mu_nB0_eq).gt.1.d-10).and.(abs(mu_max/mu_min-1).gt.1.d-2)) then
     write(2,*) 'errors in mu_nB0 calculation (not equilibrium)'
!     close(2)
!     stop
  endif

! Calculate coefficients in sharp-interface equations

! calculate the derivatives
  do j=1,3
     do i=1,L
        Aaq(i)=-qu2(i)*A0q(j,i)
        Baq(i)=-qu2(i)*B0q(j,i)
     enddo
     call fftw_execute_dft(plan1d_c_b,Aaq,Aa) ! backward FFT (complex)
     call fftw_execute_dft(plan1d_c_b,Baq,Ba) ! backward FFT (complex)
     do i=1,L
        du2A0(j,i)=Aa(i)*scale1D_b ! partial_u^2 A0
        du2B0(j,i)=Ba(i)*scale1D_b ! partial_u^2 B0
     enddo
     do i=1,L
        Aaq(i)=cmplx(0,qu(i))*A0q(j,i)
        Baq(i)=cmplx(0,qu(i))*B0q(j,i)
     enddo
     call fftw_execute_dft(plan1d_c_b,Aaq,Aa) ! backward FFT (complex)
     call fftw_execute_dft(plan1d_c_b,Baq,Ba) ! backward FFT (complex)
     do i=1,L
        duA0(j,i)=Aa(i)*scale1D_b ! partial_u A0
        duB0(j,i)=Ba(i)*scale1D_b ! partial_u B0
     enddo
  enddo

! for 1st interface
  imin(1)=1
  imax(1)=L/2
! for 2nd interface
  imin(2)=L/2+1
  imax(2)=L

! interface position j=j0 (u=0)
  do k=1,2
     i_min=imin(k)
     i_max=imax(k)
     dpsi0=nA0(i_max)-nA0(i_min)
     i0_nA(k)=NINT((nA0(i_max)*(i_max+1)-sum(nA0(i_min:i_max))-nA0(i_min)*i_min)/dpsi0)

     dpsi0=nB0(i_max)-nB0(i_min)
     i0_nB(k)=NINT((nB0(i_max)*(i_max+1)-sum(nB0(i_min:i_max))-nB0(i_min)*i_min)/dpsi0)
  enddo
  write(2,*)
  write(2,*) 'interface position 1 (Gibbs surface):'
  write(2,*) 'i0_nA(1)-1=',i0_nA(1)-1, '  i0_nB(1)-1=',i0_nB(1)-1

  write(2,*) 'nA0_l(1)=',nA0(1), '  nA0_s(L/2)=',nA0(L/2)
  write(2,*) 'nB0_l(1)=',nB0(1), '  nB0_s(L/2)=',nB0(L/2)
  write(2,*) 'A0 in solid bulk (L/2):',(A0(j,L/2),j=1,3)
  write(2,*) 'B0 in solid bulk (L/2):',(B0(j,L/2),j=1,3)
  write(2,*)
  write(2,*) 'interface position 2 (Gibbs surface):'
  write(2,*) 'i0_nA(2)=',i0_nA(2), '  i0_nB(2)=',i0_nB(2)

  write(2,*) 'nA0_l(L)=',nA0(L), '  nA0_s(L/2+1)=',nA0(L/2+1)
  write(2,*) 'nB0_l(L)=',nB0(L), '  nB0_s(L/2+1)=',nB0(L/2+1)
  write(2,*) 'A0 in solid bulk (L/2+1):',(A0(j,L/2+1),j=1,3)
  write(2,*) 'B0 in solid bulk (L/2+1):',(B0(j,L/2+1),j=1,3)

  sigma=0
  zeta_D=0
  zeta_DAB=0
  do i=1,L
     k=1 ! for 1st interface
     if(i.gt.L/2) k=2 ! for 2nd interface
     do j=1,3
        A_R=REAL(duA0(j,i))
        A_I=AIMAG(duA0(j,i))
        duA2=A_R*A_R+A_I*A_I
        A_R=REAL(duB0(j,i))
        A_I=AIMAG(duB0(j,i))
        duB2=A_R*A_R+A_I*A_I
        A_R=REAL(du2A0(j,i))
        A_I=AIMAG(du2A0(j,i))
        A2=A_R*A_R+A_I*A_I
        A_R=REAL(du2B0(j,i))
        A_I=AIMAG(du2B0(j,i))
        B2=A_R*A_R+A_I*A_I
        sigma(k)=sigma(k)+beta_A*((deltaj0(j)+q02-qA2+chi_j2(j))*duA2+A2) & ! chi_j2=chi_j^2/2
                 +beta_B*((deltaj0(j)+q02-qB2+chi_j2(j))*duB2+B2) &
                 +beta_AB*2*((deltaj0(j)+q02-qAB2+chi_j2(j))*real(duA0(j,i)*conjg(duB0(j,i))) &
                 +real(du2A0(j,i)*conjg(du2B0(j,i)))) & ! dchi_j=dchi_j/dtheta
                 -dchi_j(j)*(beta_A*aimag(du2A0(j,i)*conjg(duA0(j,i)))+beta_B*aimag(du2B0(j,i)*conjg(duB0(j,i))) &
                 +beta_AB*aimag(du2A0(j,i)*conjg(duB0(j,i))+conjg(duA0(j,i))*du2B0(j,i)))
        zeta_D(k)=zeta_D(k)+2*(duA2+duB2/mB)
        zeta_DAB(k)=zeta_DAB(k)+2*(duA2+duB2/mB)
     enddo
     i_min=imin(k)
     i_max=imax(k)
     if(i.lt.i0_nA(k)) then ! use the nA0 Gibbs surface
        zeta_D(k)=zeta_D(k)+q02*(nA0(i)*nA0(i)-nA0(i_min)*nA0(i_min)+(nB0(i)*nB0(i)-nB0(i_min)*nB0(i_min))/mB)
        zeta_DAB(k)=zeta_DAB(k)+q02*(nA0(i)*nA0(i)-nA0(i_min)*nA0(i_min))
     else
        zeta_D(k)=zeta_D(k)+q02*(nA0(i)*nA0(i)-nA0(i_max)*nA0(i_max)+(nB0(i)*nB0(i)-nB0(i_max)*nB0(i_max))/mB)
        zeta_DAB(k)=zeta_DAB(k)+q02*(nA0(i)*nA0(i)-nA0(i_max)*nA0(i_max))
     endif
     if(i.lt.i0_nB(k)) then ! use the nB0 Gibbs surface
        zeta_DAB(k)=zeta_DAB(k)+q02*(nB0(i)*nB0(i)-nB0(i_min)*nB0(i_min))/mB
     else
        zeta_DAB(k)=zeta_DAB(k)+q02*(nB0(i)*nB0(i)-nB0(i_max)*nB0(i_max))/mB
     endif
  enddo
  sigma=sigma*du*4*q02
  zeta_D=zeta_D*du
  zeta_DAB=zeta_DAB*du

  write(2,*)
  write(2,*) 'sigma=',sigma
  write(2,*) 'sigma_avg=',(sigma(1)+sigma(2))/2
  write(2,*) 'zeta_D=',zeta_D ! use the nA0 Gibbs surface only
  write(2,*) 'zeta_DAB=',zeta_DAB ! use both nA0 and nB0 Gibbs surface
  write(2,*) 'zeta_D(AB)_avg=',(zeta_D(1)+zeta_D(2))/2,(zeta_DAB(1)+zeta_DAB(2))/2

  write(2,*) 'gamma_nA=',gamma_nA, '  gamma_nB=',gamma_nB
  ! unit conversion for surface tension
  theta=2.74d0*(4*pi/sqrt(3.d0))/2.51d0 ! =(2.74 eV)*a_x/(2.51 A), a_x=4pi/sqrt(3); surface tension scale for hBN
  write(2,*) 'surface energy unit (eV/A) =', theta
  write(2,*) 'gamma_nA,B(eV/A)=',gamma_nA*theta,gamma_nB*theta
  write(2,*) 'sigma(eV/A)=',sigma*theta
  write(2,*) 'sigma_avg(eV/A)=',(sigma(1)+sigma(2))*theta/2
  
  theta=dasin(sin_theta)
  cos6theta=cos(6*theta)
  theta=theta*180/pi

  open(unit=10,file=file//'_coeffs1.dat',position='append')
  write(10,*) '# epsA,epsB,theta,cos6theta,sigma,zeta_D,zeta_DAB,mu_nA0_eq,mu_nB0_eq,dnA0=m_gap_nA,dnB0=m_gap_nB,'
  write(10,*) '# gamma_nA,gamma_nB,width_nA,width_nB,width_A,width_B,nA0_l,nA0_s,nB0_l,nB0_s,Es'
  write(10,50) epsA,epsB,theta,cos6theta,sigma(1),zeta_D(1),zeta_DAB(1),mu_nA0_eq,mu_nB0_eq,m_gap_nA(1),m_gap_nB(1),&
               gamma_nA,gamma_nB,width_nA(1),width_nB(1),width_A(1),width_B(1),nA0(1),nA0(L/2),nB0(1),nB0(L/2),Es
! width: interface thickness; gamma_nA,gamma_nB: surface tension from subroutine surface_tension
! m_gap_n=dn0=n0(solid)-n0(liquid); Es: Young's modulus for solid
  close(10)

  open(unit=10,file=file//'_coeffs2.dat',position='append')
  write(10,*) '# epsA,epsB,theta,cos6theta,sigma,zeta_D,zeta_DAB,mu_nA0_eq,mu_nB0_eq,dnA0=m_gap_nA,dnB0=m_gap_nB,'
  write(10,*) '# gamma_nA,gamma_nB,width_nA,width_nB,width_A,width_B,nA0_l,nA0_s,nB0_l,nB0_s,Es'
  write(10,50) epsA,epsB,theta,cos6theta,sigma(2),zeta_D(2),zeta_DAB(2),mu_nA0_eq,mu_nB0_eq,m_gap_nA(2),m_gap_nB(2),&
               gamma_nA,gamma_nB,width_nA(2),width_nB(2),width_A(2),width_B(2),nA0(L),nA0(L/2+1),nB0(L),nB0(L/2+1),Es
! width: interface thickness; gamma_nA,gamma_nB: surface tension from subroutine surface_tension
! m_gap_n=dn0=n0(solid)-n0(liquid); Es: Young's modulus for solid
  close(10)

50 format(2(f9.3,1x),f10.5,19(1x,es17.10))
  
  close(2)

  return
end subroutine coeffs1D
