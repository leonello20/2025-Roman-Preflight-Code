#   Copyright 2019 California Institute of Technology
#   Users must agree to abide by the restrictions listed in the 
#   file "LegalStuff.txt" in the CGISim library directory.
#
#  Written by John Krist
#  Jet Propulsion Laboratory, California Institute of Technology
#  28 August 2019


#  thin_film_filter_2.py

import numpy as np
def thin_film_filter_2(n,d,theta,lam,tetm=1,debug1=False, output_opd=False):
  #function [R, T, rr, tt] = thin_film_filter_2(n,d,theta,lam,tetm)
  # [R, T, rr, tt] = thin_film_filter_2(n,d,theta,lam,[tetm])
  # n = index of refraction for each layer. 
  #     n(1) = index of incident medium
  #     n(N) = index of transmission medium
  #     then length(n) must be >= 2
  # d = thickness of each layer, not counting incident medium or transmission
  #     medium. length(d) = length(n)-2
  # theta = angle of incidence [rad], scalar only
  # lam = wavelength. units of lam must be same as d, scalar only
  # tetm: 0 => TE (default is s), 1 => TM (1 is p)
  #
  # outputs:
  # R = normalized reflected intensity coefficient
  # T =        "   transmitted     "
  # rr = complex field reflection coefficient
  # tt =      "        transmission "
  #
  # Ref:
  # H. Angus Macleod, "Thin-Film Optical Filters", Fourth Edition, 2010, p.
  # 89-92, equation 3.16
  #####
  # Dwight added:
  #  d can be a list of arrays of the same shape
  #  output_opd=True makes it OPD before default was False
  ################

  N = len(n);
  if len(d) != N-2:
    errtx = 'n and d mismatch'
    raise BaseException(errtx)
  if type(d[0]) == type(1.0):
    d2 = np.zeros([N], np.float64);
    d2[1:N-1] = d
  else:
    sh1 = d[0].shape
    for ii in range(1,N-2):
      if not np.alltrue(d[ii].shape == sh1):
        errtx = 'thin_film_filter_2() not np alltrue(d[ii].shape == sh1)'
        raise BaseException(errtx)
    d2 = [0 for ii in range(N)]
    d2[0] = d2[-1] = np.zeros(sh1, np.float64)
    d2[1:N-1] = d
  d = d2

  kx = 2*np.pi*n[0]*np.sin(theta)/lam;
  kz = -np.sqrt( (2*np.pi*n/lam)**2 - kx**2 ); # sign agrees with measurement convention
  if debug1:
    print('debug1: kx='+str(kx))
    print('debug1: kz='+str(kz))

  if tetm == 1:
     kzz = kz/(n**2);
  else:
     kzz = kz;

  if type(d[0]) == type(1.0):
    eep = np.exp(-1j*kz*d);
    eem = np.exp(1j*kz*d);
  else:
    eep = [np.exp(-1j*kz[ii]*d[ii]) for ii in range(N-1)]
    eem = [np.exp(1j*kz[ii]*d[ii]) for ii in range(N-1)]

  tin = 0.5*(kzz[0:N-1] + kzz[1:N])/kzz[0:N-1];
  ri  = (kzz[0:N-1] - kzz[1:N])/(kzz[0:N-1] + kzz[1:N]);
  if debug1:
    print('debug1: tin='+str(tin))
    print('debug1: ri='+str(ri))

  Axx = Ayy = 1.0
  Axy = Ayx = 0.0
  for ii in range(0, N-1):
    v1 = tin[ii]
    xx = v1*eep[ii]; 
    xy = v1*ri[ii]*eep[ii];
    yx = v1*ri[ii]*eem[ii]; 
    yy = v1*eem[ii];
    # matrix mulitiply:
    Axx2 = Axx * xx + Axy * yx
    Axy2 = Axx * xy + Axy * yy
    Ayx2 = Ayx * xx + Ayy * yx
    Ayy2 = Ayx * xy + Ayy * yy
    Axx = Axx2
    Axy = Axy2
    Ayx = Ayx2
    Ayy = Ayy2
    if debug1:
      print('A: '+str(np.array([[Axx, Axy], [Ayx, Ayy]])))

  #rr = A[2,1]/A[1,1];
  #tt = 1/A[1,1];
  rr = Ayx/Axx
  tt = 1.0/Axx

  R = abs(rr)**2;
  if tetm == 1:
    Pn = ((kz[N-1]/(n[N-1]**2))/(kz[0]/(n[0]**2))).real;
  else:
    Pn = ((kz[N-1]/kz[0])).real;
  T = Pn*abs(tt)**2;
  tt= np.sqrt(Pn)*tt;
  if output_opd:
    phs = np.arctan2(np.imag(tt),np.real(tt));
    phs = -phs;
    e1 = np.sqrt(np.real(tt*np.conjugate(tt)));
    tt = e1*(np.cos(phs)+1j*np.sin(phs))

  return R, T, rr, tt

