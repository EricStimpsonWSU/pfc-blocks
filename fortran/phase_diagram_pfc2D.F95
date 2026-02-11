! Calculate one-mode phase diagram for single-component PFC model
! 2D triangular (A0>0) or honeycomb (A0<0) vs liquid
! Equilibrium bulk state; no external strain/stress or misfit

  Program phase_diagram_pfc2D
  implicit none

  common/para1/eps,epsq04,g
  common/para2/dpsi0,xacc,area_acc
  real(8) :: psi0
  real(8) :: psi0_l1,psi0_l2,psi0_s1,psi0_s2
  real(8) :: f_s,f_l,mu_s,mu_l,mu01,mu02,mu0s,mu_s0,mu_l0
  real(8) :: eps,Bx,g,q0,epsq04
  real(8) :: deps,epsMin,epsMax
  real(8) :: dpsi0,psi0_min,psi0_max,psi0_0(2),area_acc,xacc,rtbis
  real(8) :: f_sl,psi0_b1(10),psi0_b2(10)
  integer :: iter1,iter2,nb,nb_old,i,i_mu
  external f_s,f_l,f_sl,mu_s,mu_l,rtbis

  !character(17) file
  character(:), allocatable :: file ! to be automatically allocated

! eps_m=0 ! no external strain/stress or misfit

  !deps=0.01d0
  deps=0.05d0
  epsMin=0.d0
  epsMax=1.d0

  q0=1
 
  !Bx=10
  !g=0.5d0*sqrt(3.d0/Bx)
  !file='pfc2D_diagram_Bx10' ! eps: 0-1, deps=0.05

  !g=-1
  !file='pfc2D_diagram_g_1' ! eps: 0-1, deps=0.05

  g=0.5d0
  file='pfc2D_diagram_g05' ! eps: 0-1, deps=0.05

  Bx=0.75d0/(g*g)
  
  dpsi0=1.d-8
  !dpsi0=1.d-6
  !psi0_min=-1.d0
  psi0_min=-1.5d0
  psi0_max=1.d0
  !psi0_max=1.5d0

  area_acc=1.d-9 ! accuracy for Maxwell equal area construction

!  xacc=dpsi0/100 ! accuracy for bisection method (rtbis)
  xacc=1.d-10

  open(unit=2,file=file//'_para.dat',position='append')
  write(2,*) 'Equilibrium bulk state; no external strain/stress or misfit eps_m=0'
  write(2,*) 'g=',g,'  Bx=',Bx
  write(2,*) 'q0=',q0
  write(2,*) 'deps=',deps, '  dpsi0=',dpsi0
  write(2,*) 'epsMin=',epsMin, '  epsMax=',epsMax
  write(2,*) 'psi0_min=',psi0_min, '  psi0_max=',psi0_max
  write(2,*) 'area_acc=',area_acc, '  xacc=',xacc
  close(2)

  nb_old=0
  do eps=epsMin,epsMax,deps

! for liquid-solid coexistence
     ! get psi0_0(1), psi0_0(2):  cross points of f_s=f_l
     epsq04=q0**4-eps
     nb=10
     call zbrak(f_sl,psi0_min,psi0_max,1000,psi0_b1,psi0_b2,nb) ! bracketing to find root intervals
     nb_old=nb
     if(nb.eq.0) goto 100
     if(nb.gt.2) then
        write(*,*) 'error in finding fs=fl, with number of crossings nb=',nb
        stop
     endif

     do i=1,nb
        psi0_0(i)=rtbis(f_sl,0.d0,psi0_b1(i),psi0_b2(i),xacc) ! solving f_s=f_l via bisection method
     enddo

     mu_s0=mu_s(psi0_0(1),i_mu)
     mu_l0=mu_l(psi0_0(1),i_mu)
     mu01=(mu_s0+mu_l0)/2
     call Maxwell_equal_area(iter1,psi0_s1,psi0_l1,mu01,mu_s0,mu_l0,psi0_min,psi0_max,psi0_0(1),file)
     if(iter1.le.100) then
     if(nb.eq.1) then
        open(unit=20,file=file//'_l-s.dat',position='append')
        if(nb_old.eq.2) then
           write(20,10) -eps,psi0_s1,psi0_l1,psi0_s1,psi0_l1,iter1,iter1
        else
           write(20,10) -eps,psi0_s1,psi0_l1,0,0,iter1,iter1
        endif
        close(20)
        goto 100
     endif
     else
        open(unit=20,file=file//'_error.dat',position='append')
        write(20,*) 'error for liquid-solid coexistence 1: iter1>100 at eps=',eps
        close(20)
     endif

     mu_s0=mu_s(psi0_0(2),i_mu)
     mu_l0=mu_l(psi0_0(2),i_mu)
     mu02=(mu_s0+mu_l0)/2
     call Maxwell_equal_area(iter2,psi0_s2,psi0_l2,mu02,mu_s0,mu_l0,psi0_min,psi0_max,psi0_0(2),file)
     if(iter2.le.100) then
     open(unit=20,file=file//'_l-s.dat',position='append')
     write(20,10) -eps,psi0_s1,psi0_l1,psi0_s2,psi0_l2,iter1,iter2
     close(20)
     else
        open(unit=20,file=file//'_error.dat',position='append')
        write(20,*) 'error for liquid-solid coexistence 2: iter2>100 at eps=',eps
        close(20)
     endif
100  continue

  enddo ! do eps=epsMin,epsMax,deps

10 format(f10.6,4(1x,f20.16),2(i5))

end Program phase_diagram_pfc2D


! Maxwell equal area construction for liquid-solid coexistence
subroutine Maxwell_equal_area(iter,psi0_s,psi0_l,mu0,mu_s0,mu_l0,psi0_min,psi0_max,psi0_0,file)
  implicit none
  common/para2/dpsi0,xacc,area_acc
  real(8) :: psi0_s,psi0_l,mu0,mu_s0,mu_l0,psi0_min,psi0_max,psi0_0
  real(8) :: mu,mu_old,AI_1,AI_2,psi0_1,psi0_2,area_acc,xacc,d,d_per,d_old
  real(8) :: mu_s,mu_l,rtbis,dpsi0,psi0
  external mu_s,mu_l,rtbis
  integer :: iter,i_mu,i_mu_old
  character file*(*)

  iter=0
  d_old=1000
100 continue
  iter=iter+1
  if(iter.gt.100) then
     open(unit=20,file=file//'_error.dat',position='append')
     write(20,*) 'iter exceeds maximum in Maxwell_equal_area: iter=',iter
     write(20,*) 'd_per=',d_per
     write(20,*) 'AI_1=',AI_1, '  AI_2=',AI_2
     write(20,*) 'psi0_s=',psi0_s, '  psi0_l=',psi0_l
     close(20)
     return
!     pause
  endif
  AI_1=0
  AI_2=0
  psi0_1=psi0_min
  psi0_2=psi0_max
  mu_old=mu0+100
  i_mu_old=0
  do psi0=psi0_min,psi0_max,dpsi0
     i_mu=1
     if(psi0.lt.psi0_0) then ! left area of mu--psi0
        if(mu_s0.gt.mu_l0) then
           mu=mu_s(psi0,i_mu)
           if(i_mu.eq.1.and.i_mu_old.eq.1.and.mu.gt.mu0.and.mu_old.le.mu0) &
                psi0_1=rtbis(mu_s,mu0,psi0-dpsi0,psi0,xacc) ! using bisection method
        else
           mu=mu_l(psi0,i_mu)
           if(i_mu.eq.1.and.i_mu_old.eq.1.and.mu.gt.mu0.and.mu_old.le.mu0) &
                psi0_1=rtbis(mu_l,mu0,psi0-dpsi0,psi0,xacc) ! using bisection method
        endif
        if(i_mu.eq.1) then ! valid mu_s result
           if(mu.gt.mu0) AI_1=AI_1+mu-mu0          
        endif
     endif
     if(psi0.gt.psi0_0) then ! right area of mu--psi0
        if(mu_s0.gt.mu_l0) then
           mu=mu_l(psi0,i_mu)
           if(i_mu.eq.1) then ! valid mu_s result
              if(mu.lt.mu0) AI_2=AI_2+mu-mu0
              if(i_mu_old.eq.1.and.mu.gt.mu0.and.mu_old.le.mu0) then
                 psi0_2=rtbis(mu_l,mu0,psi0-dpsi0,psi0,xacc) ! using bisection method
                 goto 200
              endif
           endif
        else
           mu=mu_s(psi0,i_mu)
           if(i_mu.eq.1) then ! valid mu_s result
              if(mu.lt.mu0) AI_2=AI_2+mu-mu0
              if(i_mu_old.eq.1.and.mu.gt.mu0.and.mu_old.le.mu0) then
                 psi0_2=rtbis(mu_s,mu0,psi0-dpsi0,psi0,xacc) ! using bisection method
                 goto 200
              endif
           endif
        endif
     endif
     if(i_mu.eq.1) then
        mu_old=mu
        i_mu_old=i_mu
     endif
  enddo
200 continue
  if(mu_s0.gt.mu_l0) then
     AI_1=abs(AI_1+(mu_s0-mu0)/2)*dpsi0
     AI_2=abs(AI_2+(mu_l0-mu0)/2)*dpsi0
     psi0_s=psi0_1
     psi0_l=psi0_2
  else
     AI_1=abs(AI_1+(mu_l0-mu0)/2)*dpsi0
     AI_2=abs(AI_2+(mu_s0-mu0)/2)*dpsi0
     psi0_l=psi0_1
     psi0_s=psi0_2
  endif

  d=AI_1-AI_2
  d_per=abs(d/AI_1)
!  if(d_per.gt.d_old) then
!     write(*,*) 'error in Maxwell_equal_area: iter=', iter, ' d_per=',d_per, ' d_old=',d_old
!     write(*,*) 'AI_1=',AI_1, '  AI_2=',AI_2
!     write(*,*) 'psi0_s=',psi0_s, '  psi0_l=',psi0_l
!!     pause
!  endif
  if(d_per.lt.area_acc) then
     return
  else
     mu0=mu0+d/2/(psi0_0-psi0_1) ! estimate new mu0 from area difference
     d_old=d_per
     goto 100
  endif

  return
end subroutine Maxwell_equal_area

subroutine A0_sol(A0,psi0,iA0) ! for A0 real
  implicit none
  common/para1/eps,epsq04,g
  real(8) :: eps,epsq04,g
  real(8) :: A0,psi0,A01,A02
  real(8) :: f_s,f1,f2,delta
  integer :: iA0

  iA0=0
  delta=15*eps-12*psi0*(3*psi0-2*g)+g*g
  if(delta.ge.0) then
     iA0=1 ! valid A0 solution
     A01=(g-3*psi0+sqrt(delta))/15
     f1=f_s(A01,psi0)
     A02=(g-3*psi0-sqrt(delta))/15
     f2=f_s(A02,psi0)
     if(f1.le.f2) then
        A0=A01
     else
        A0=A02
     endif
  endif

  return
end subroutine A0_sol

Function f_s(A0,psi0) ! free energy density for solid
  implicit none
  common/para1/eps,epsq04,g
  real(8) :: eps,epsq04,g
  real(8) :: f_s,A0,psi0

  f_s=A0*A0*(3*(psi0*(3*psi0-2*g)-eps)+A0*(4*(3*psi0-g)+45*A0/2)) &
      +psi0*psi0*(epsq04/2+psi0*(psi0/4-g/3)) ! +f_l(psi0)

  return
end Function f_s

Function mu_s(psi0,iA0) ! chemical potential for solid
  implicit none
  common/para1/eps,epsq04,g
  real(8) :: mu_s,psi0,A0
  real(8) :: eps,epsq04,g
  integer :: iA0

  iA0=0
  call A0_sol(A0,psi0,iA0)
  if(iA0.eq.1) & ! iA0=1: valid A0 solution
  mu_s=6*A0*A0*(3*psi0-g+2*A0) &
       +psi0*(epsq04+psi0*(psi0-g)) ! +mu_l(psi0,iA0)

  return
end Function mu_s

Function mu_l(psi0,iA0) ! chemical potential for liquid
  implicit none
  common/para1/eps,epsq04,g
  real(8) :: mu_l,psi0,eps,epsq04,g
  integer iA0

  iA0=1
  mu_l=psi0*(epsq04+psi0*(psi0-g))

  return
end Function mu_l

Function f_l(psi0) ! free energy density for liquid
  implicit none
  common/para1/eps,epsq04,g
  real(8) :: f_l,psi0,eps,epsq04,g

  f_l=psi0*psi0*(epsq04/2+psi0*(psi0/4-g/3))

  return
end Function f_l

Function f_sl(psi0,iA0) ! for f_s-f_l (at same psi0)
  implicit none
  common/para1/eps,epsq04,g
  real(8) :: f_sl,psi0,A0,eps,epsq04,g
  integer :: iA0

  iA0=0
  call A0_sol(A0,psi0,iA0) ! if iA0=1: valid A0 solution
  !if(iA0.eq.1) f_sl=f_s(A0,psi0)-f_l(psi0)
  if(iA0.eq.1) f_sl=A0*(45*A0/2+4*(3*psi0-g))+3*(psi0*(3*psi0-2*g)-eps)

  return
end Function f_sl

! Bisection method to find the root of a function func=f0, known to lie between x1 and x2;
! The root, returned as rtbis, will be refined until its accuracy is +/-xacc
Function rtbis(func,f0,x1,x2,xacc)
  implicit none
  integer jmax,j,i_valid
  real(8) :: rtbis,func,x1,x2,xacc,f0
  external func
  parameter(jmax=100) ! Maximum allowed number of bisections
  real(8) :: dx,f,fmid,xmid
  fmid=func(x2,i_valid)-f0
  if(i_valid.eq.0) return ! no valid func result
  f=func(x1,i_valid)-f0
  if(i_valid.eq.0) return ! no valid func result
  if(f*fmid.ge.0.d0) then
     write(*,*) 'root must be bracketed in rtbis:'
     write(*,*) '  x1=',x1, '  func(x1)-f0=',f
     write(*,*) '  x2=',x2, '  func(x2)-f0=',fmid
     pause
     return
  endif
  if(f.lt.0.d0) then ! Orient the search so that f>0 lies at x+dx
     rtbis=x1
     dx=x2-x1
  else
     rtbis=x2
     dx=x1-x2
  endif
  do j=1,jmax ! Bisection loop
     dx=dx/2
     xmid=rtbis+dx
     fmid=func(xmid,i_valid)-f0
     if(i_valid.eq.0) return ! no valid func result
     if(fmid.le.0) rtbis=xmid
     if(abs(dx).lt.xacc.or.fmid.eq.0.d0) return
  enddo
  write(*,*) 'too many bisections in rtbis; jmax=',jmax
  stop
end Function rtbis

! Bracketing for root finding
! Given a function func defined on x1,x2 interval, subdivide the interval into 
! n equally spaced segments and search for 0 crossings of the function.
! nb: maximum number of roots sought, and reset to the number of
!     bracketing pairs xb1(1:nb), xb2(1:nb) that are found
      SUBROUTINE zbrak(func,x1,x2,n,xb1,xb2,nb)
      INTEGER n,nb
      REAL(8) :: x1,x2,xb1(nb),xb2(nb),func
      EXTERNAL func
      INTEGER i,nbb,ip_valid,ic_valid
      REAL(8) :: dx,fc,fp,x
      nbb=0
      x=x1
      dx=(x2-x1)/n
      fp=func(x,ip_valid)
      do 11 i=1,n
        x=x+dx
        fc=func(x,ic_valid)
        if(ip_valid.eq.1.and.ic_valid.eq.1.and.fc*fp.le.0.d0) then ! no valid func result if i_valid=0
          nbb=nbb+1
          xb1(nbb)=x-dx
          xb2(nbb)=x
          if(nbb.eq.nb)goto 1
        endif
        fp=fc
        ip_valid=ic_valid
11    continue
1     continue
      nb=nbb
      return
      END
