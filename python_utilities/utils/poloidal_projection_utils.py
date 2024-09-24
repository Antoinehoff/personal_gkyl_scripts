import postgkyl as pg
import numpy as np
import matplotlib.pyplot as plt
#.Add pgkyl Utilities to path.
import sys
sys.path.insert(0, '/scratch/gpfs/tbernard/scripts/')
import pgkylUtil as pgu
#.Scipy is used for interpolation and integration.
import scipy.integrate as integrate
from scipy.interpolate import griddata
from scipy.interpolate import pchip_interpolate
#.JAX is used for automatic differentiation (AD).
import jax.numpy as jnp
from jax import grad, jit, vmap
from matplotlib.patches import Rectangle

from mpl_toolkits.axes_grid.inset_locator import (inset_axes, InsetPosition,
                                                  mark_inset)
from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes, mark_inset
from matplotlib import ticker

#................................................................................#
#.
#.Plot the color map of a field on the poloidal plane given the flux-tube data.
#.There are two options:
#.  a) Perform all interpolations in field aligned coordinates and use an FFT
#.     This may only be valid for the potential which is FEM and not DG.
#.  b) Interpolate in the parallel direction onto a finer grid, then transform
#.     to cylindrical and perform another interpolation onto the plotting points.
#.
#.Manaure Francisquez.
#.
#................................................................................#

#.Location of data and directory where to place output figure file.
#dataDir = '/scratch/gpfs/manaurer/gkeyll/d3d/siny_test/'
#outDir  = '/scratch/gpfs/manaurer/gkeyll/d3d/siny_test/'
#simName  = 'd3d-posD04-k135-3x2v-siny'    #.Root name of files to process.
dataDir = '/scratch/gpfs/tbernard/open-closed-fl/negD-ltd/correct-r_x-hiq/' #'/scratch/gpfs/manaurer/gkeyll/d3d/sinphi_test/'
outDir  = '/scratch/gpfs/tbernard/open-closed-fl/negD-ltd/correct-r_x-hiq/clopen-inset/' #'/scratch/gpfs/manaurer/gkeyll/d3d/sinphi_test/'
simName  = 'd3d-posD04-k135-3x2v'    #.Root name of files to process.

dataName = 'elc_gridDiagnostics'          #.Name of data (file) to plot.
fldName = 'M0' # M0, Temp, Upar, etc.
scaleFac = 1 # eV or 1, depending on fldName

outFigureFile    =  False    #.Output a figure file?.
figureFileFormat = '.png'    #.Can be .png, .pdf, .ps, .eps, .svg.
outSuffix = '_poloidalPlaneReal'  #.Append this to the figure file name.

fstart = 600
fend =1000

polyOrder = 1      #.Order of the polynomial basis.

# From TNB input file for +D
# Geometry and magnetic field.
Rdim       = 1.7  # [m]
Zdim       = 3.2  # [m]
Z_axis     = 0.013055028 # [m]
R_axisTrue = 1.6486461 #0.95*R_axisTrue    # Change R_axis to fit geometry better.
R_axis     = 1.6
B_axis     = 2*R_axisTrue/R_axis   # [T]
R_LCFSmid  = 2.17 #2.17885               # Major radius of the LCFS at the outboard midplane [m].
Rmid_min   = R_LCFSmid-2.*0.05          # Minimum midplane major radius of simulation box [m].
Rmid_max   = R_LCFSmid+1.*0.05               # Maximum midplane major radius of simulation box [m].
R0         = 0.5*(Rmid_min+Rmid_max)  # Major radius of the simulation box [m].
a_mid      = R_LCFSmid-R_axis   # Minor radius at outboard midplane [m].
r0         = R0-R_axis          # Minor radius of the simulation box [m].
B0         = B_axis*(R_axis/R0) # Magnetic field magnitude in the simulation box [T].

# MF variables
# B_ref = 2.0            #.T
# R_axis = 1.6            #.m
# a     = 2.17-R_axis     #.minor radius at LCFS.
# r0    = (2.23-2.07)/2.          #.minor radius at center of the flux tube.
# R0    = R_axis          #.Major radius at center of the flux tube.

#.Miller geometry parameters.
kappa = 1.35  #.Elongation.
delta = -0.4  #.Triangularity.

#.Number of parallel (z) points to interpolate at (as a factor of
#.the number of interpolated points returned by pgkyl, typically Nz*(p+1)).
zNumIntFac = 32 #96

#.Minor radius as a function of x:
def r_x(xIn):
  return Rmid_min+xIn-R_axis

#.Magnetic safety factor profile.
def qprofile(rIn):
  #return 2.52*np.power((rIn/aIn),2) - 0.16*(rIn/aIn) + 0.86
  qa = [49.46395467479657, -260.79513158768754, 458.42618139184754, -267.63441353752336]
  return qa[0]*(rIn+R_axis)**3 + qa[1]*(rIn+R_axis)**2 + qa[2]*(rIn+R_axis) + qa[3]

#..................... NO MORE USER INPUTS BELOW (maybe) ....................#

pgu.checkMkdir(outDir)

basisType = 'ms'   #.'ms': modal serendipity, or 'ns': nodal serendipity.

#.Some fontsizes used in plots.
xyLabelFontSize       = 18
titleFontSize         = 18
colorBarLabelFontSize = 18
tickFontSize          = 17
legendFontSize        = 14
textFontSize          = 17

#.Set the font size of the ticks to a given size.
def setTickFontSize(axIn,fontSizeIn):
  for tick in axIn.xaxis.get_major_ticks():
    tick.label.set_fontsize(fontSizeIn)
  for tick in axIn.yaxis.get_major_ticks():
    tick.label.set_fontsize(fontSizeIn)


#.Function that wraps x to [xMin,xMax].
def wrap(x, xMin, xMax):
  return (((x-xMin) % (xMax-xMin)) + (xMax-xMin)) % (xMax-xMin) + xMin

def R_f(r, theta):
  return R_axis + r*np.cos(theta + np.arcsin(delta)*np.sin(theta))
def Z_f(r, theta):
  return kappa*r*np.sin(theta)

#.Analytic derivatives.
def R_f_r(r,theta): 
  return np.cos(theta + np.arcsin(delta)*np.sin(theta))
def R_f_theta(r,theta): 
  return -r*(np.arcsin(delta)*np.cos(theta)+1.)*np.sin(np.arcsin(delta)*np.sin(theta)+theta)
def Z_f_r(r,theta):
  return kappa*np.sin(theta)
def Z_f_theta(r,theta):
  return kappa*r*np.cos(theta)
##.Derivatives with AD. May require replacing 'np.' by 'jnp.' in functions to be differentiated.
#R_f_r     = jit(grad(R_f,argnums=(0)))
#R_f_theta = jit(grad(R_f,argnums=(1)))
#Z_f_r     = jit(grad(Z_f,argnums=(0)))
#Z_f_theta = jit(grad(Z_f,argnums=(1)))

def Jr_f(r, theta):
  return R_f(r,theta)*( R_f_r(r,theta)*Z_f_theta(r,theta)-Z_f_r(r,theta)*R_f_theta(r,theta) )

def integrand(t, r):
  return Jr_f(r,t)/np.power(R_f(r,t),2)

def dPsidr_f(r, theta):
  integral, _ = integrate.quad(integrand, 0., 2.*np.pi, args=(r), epsabs=1.e-8)
  return B0*R_axis/(2.*np.pi*qprofile(r))*integral

def alpha_f(r, theta, phi):
  t = theta
  while (t < -np.pi):
    t = t+2.*np.pi

  while ( np.pi < t):
    t = t-2.*np.pi

  if (0. < t):
    intV, intE = integrate.quad(integrand, 0., t, args=(r), epsabs=1.e-8)
    integral   = intV
  else:
    intV, intE = integrate.quad(integrand, t, 0., args=(r), epsabs=1.e-8)
    integral   = -intV

  return phi - B0*R_axis*integral/dPsidr_f(r,theta)

fldFilePhi = dataDir+simName+'_phi_0.bp'

x, dim, nx, lx, dx, _ = pgu.getRawGrid(fldFilePhi)
#.Get the interpolated grid, nodal coordinates.
xInt, dimInt, nxInt, lxInt, dxInt, _ = pgu.getGrid(fldFilePhi,polyOrder,basisType)
xIntC, _, nxIntC, lxIntC, dxIntC, _ = pgu.getGrid(fldFilePhi,polyOrder,basisType,location='center')

xi_lcfs = int(nxIntC[0]*2/3)
zNumInt = zNumIntFac*nxIntC[2]

#.Magnetic safety factor in the middle of the simulation box.
q0 = qprofile(r0)

toroidal_mode_number = 2.*np.pi*r0/q0/lx[1]  #.n_0 in Goerler et al.

fileRoot = dataDir+simName+'_'+dataName
outFileRoot = outDir+simName+'_'+dataName+outSuffix

#.......................................................................#
#.Precompute grids and arrays needed in transforming/plotting data below.
#.Approach: FFT along y, then follow a procedure similar to that in pseudospectral
#.codes (e.g. GENE, see Xavier Lapillonne's PhD thesis 2010, section 3.2.2, page 55).
esPotInt = np.squeeze(pgu.getInterpData(fileRoot+'_'+str(fstart)+'.bp',polyOrder,basisType,var_name=fldName))

esPotInt_k = np.fft.rfft(esPotInt, axis=1, norm="forward")
kxIntC = esPotInt_k.shape

#.Extend along z by in each direction by applying twist-shift BCs in the 
#.closed-flux region, and just copying the last values (along z) in the SOL.
z_ex = np.concatenate(([xIntC[2][0]-dxIntC[2]],xIntC[2],[xIntC[2][-1]+dxIntC[2]]))
esPotInt_kex = np.zeros(kxIntC+np.array([0,0,2]), dtype=np.cdouble)

bcPhaseShift = 1j*2.0*np.pi*toroidal_mode_number*qprofile(r_x(xIntC[0][:xi_lcfs]))
  
#.Interpolate onto a finer mesh along z.
z_int = np.linspace(z_ex[0],z_ex[-1],zNumInt)
esPotInt_kintPos = np.zeros((kxIntC[0],kxIntC[1],zNumInt), dtype=np.cdouble)

#.Append negative ky values.
esPotInt_kint = np.zeros((kxIntC[0],2*kxIntC[1],zNumInt), dtype=np.cdouble)

#.Compute R(x,z) and Z(x,z).
numInt = [nxIntC[0], zNumInt]
RInt, ZInt = np.zeros((nxIntC[0],zNumInt)), np.zeros((nxIntC[0],zNumInt))
for i in range(nxIntC[0]):
  for k in range(zNumInt):
    x, z = r_x(xIntC[0][i]), z_int[k]
    eps  = x/R_axis

    RInt[i,k] = R_axis + x*np.cos(z + delta*np.sin(z))
    ZInt[i,k] = kappa*x*np.sin(z)

#.Calculate R,Z for LCFS plotting
RInt_lcfs, ZInt_lcfs = np.zeros(zNumInt), np.zeros(zNumInt)
for k in range(zNumInt):
  x, z = r_x(xInt[0][xi_lcfs]), z_int[k]
  eps  = x/R_axis
  
  RInt_lcfs[k] = R_axis + x*np.cos(z + delta*np.sin(z))
  ZInt_lcfs[k] = kappa*x*np.sin(z)
      
#.Compute alpha(r,z,phi=0) which is independent of y:
alpha_rz_phi0 = np.zeros([nxIntC[0],zNumInt])
for i in range(nxIntC[0]):
  for k in range(zNumInt):
    alpha_rz_phi0[i,k]  = alpha_f(r_x(xIntC[0][i]),z_int[k],0.)
  
#.Convert (x,y,z) data to (R,Z):
xyz2RZ = np.zeros([nxIntC[0],2*kxIntC[1],zNumInt], dtype=np.cdouble)
for j in range(kxIntC[1]):
  for k in range(zNumInt):
    #.Positive ky's.
    xyz2RZ[:,j,k]  = np.exp(2.*np.pi*1j*j*(-(r0/q0)*(0. + alpha_rz_phi0[:,k])/lx[1]))  #.phi=0.
    #.Negative ky's.
    xyz2RZ[:,-j,k] = np.conj(xyz2RZ[:,j,k])

esPotInt_RZ = np.zeros([nxIntC[0],zNumInt])

#.Construct nodal coordinates needed for pcolormesh.
RIntN, ZIntN = np.zeros((numInt[0]+1,numInt[1]+1)), np.zeros((numInt[0]+1,numInt[1]+1))
for j in range(numInt[1]):
  for i in range(numInt[0]):
    RIntN[i,j] = RInt[i,j]-0.5*(RInt[1,j]-RInt[0,j])
  RIntN[numInt[0],j] = RInt[-1,j]+0.5*(RInt[-1,j]-RInt[-2,j])
RIntN[:,numInt[1]] = RIntN[:,-2]

for i in range(numInt[0]):
  for j in range(numInt[1]):
    ZIntN[i,j] = ZInt[i,j]-0.5*(ZInt[i,1]-ZInt[i,0])
  ZIntN[i,numInt[1]] = ZInt[i,-1]+0.5*(ZInt[i,-1]-ZInt[i,-2])
ZIntN[numInt[0],:] = ZIntN[-2,:]

del RInt, ZInt

#.Finished precomputing grids and arrays needed in transforming/plotting.
#.......................................................................#

for t in range(fstart,fend+1):
  if t%10 ==0:
    print('t=%d'%t)
    
  fldFile = fileRoot+'_'+str(t)+'.bp'

  # The following code just plots SOL region
  #.Read the donor field, shifted donor field, target field, and target field shifted back.
  #esPotInt = np.squeeze(pgu.getInterpData(fldFile,polyOrder,basisType))
  esPotInt = np.squeeze(pgu.getInterpData(fldFile,polyOrder,basisType,var_name=fldName))

  minSOL = np.amin(esPotInt[xi_lcfs:])
  maxSOL = np.amax(esPotInt[xi_lcfs:])
  
  np.seterr(invalid='ignore')
  #.......................................................................#
  #.Approach: FFT along y, then follow a procedure similar to that in pseudospectral
  #.codes (e.g. GENE, see Xavier Lapillonne's PhD thesis 2010, section 3.2.2, page 55).

  esPotInt_k = np.fft.rfft(esPotInt, axis=1, norm="forward")

  #.Extend along z by in each direction by applying twist-shift BCs in the 
  #.closed-flux region, and just copying the last values (along z) in the SOL.
  esPotInt_kex[:,:,1:-1] = esPotInt_k
  for j in range(kxIntC[1]):
    esPotInt_kex[:xi_lcfs,j,0]  = esPotInt_k[:xi_lcfs,j,-1]*np.exp( bcPhaseShift*j)
    esPotInt_kex[:xi_lcfs,j,-1] = esPotInt_k[:xi_lcfs,j, 0]*np.exp(-bcPhaseShift*j)
    esPotInt_kex[xi_lcfs:,j,0]  = esPotInt_k[xi_lcfs:,j, 1]
    esPotInt_kex[xi_lcfs:,j,-1] = esPotInt_k[xi_lcfs:,j,-2]

  #.Interpolate onto a finer mesh along z.
  for i in range(kxIntC[0]):
    for j in range(kxIntC[1]):
      esPotInt_kintPos[i,j,:] = pchip_interpolate(z_ex, esPotInt_kex[i,j,:], z_int)

  #.Append negative ky values.
  for i in range(kxIntC[0]):
    for j in range(kxIntC[1]):
      esPotInt_kint[i,j,:]  = esPotInt_kintPos[i,j,:]
      esPotInt_kint[i,-j,:] = np.conj(esPotInt_kintPos[i,j,:])

  #.Convert (x,y,z) data to (R,Z):
  for i in range(nxIntC[0]):
    for k in range(zNumInt):
      esPotInt_RZ[i,k] = np.real(np.sum(xyz2RZ[i,:,k]*esPotInt_kint[i,:,k]))

  fldMin = np.amin(esPotInt_RZ)/scaleFac
  fldMax = np.amax(esPotInt_RZ)/scaleFac

  #.Find max for inset
  
  #.Finished transforming data and setting up grids.
  #.......................................................................#

  #.Create the figure.
  figProp1a = (8.5,9.5)
  ax1aPos   = [ [0.10, 0.08, 0.76, 0.88] ]
  cax1aPos  = [0.88, 0.08, 0.02, 0.88]
  fig1a     = plt.figure(figsize=figProp1a)
  ax1a      = list()
  for i in range(len(ax1aPos)):
    ax1a.append(fig1a.add_axes(ax1aPos[i]))
  cbar_ax1a = fig1a.add_axes(cax1aPos)
  
  hpl1a = list()
  hpl1a.append(ax1a[0].pcolormesh(RIntN, ZIntN, np.squeeze(esPotInt_RZ)/scaleFac, shading='auto',cmap='inferno'))

  #fig1a.suptitle
  ax1a[0].set_title('t = %.3f ms' %(t/1000),fontsize=titleFontSize) 
  ax1a[0].set_xlabel(r'$R$ (m)',fontsize=xyLabelFontSize, labelpad=-2)
  setTickFontSize(ax1a[0],tickFontSize)
  ax1a[0].set_ylabel(r'$Z$ (m)',fontsize=xyLabelFontSize, labelpad=-10)
  cbar = plt.colorbar(hpl1a[0],ax=ax1a,cax=cbar_ax1a)
  cbar.ax.tick_params(labelsize=10)#tickFontSize)
  cbar.set_label(r'$n_e(R,\varphi=0,Z)$ (m$^{-3}$)', rotation=270, labelpad=18, fontsize=colorBarLabelFontSize)
  hmag = cbar.ax.yaxis.get_offset_text().set_size(tickFontSize)

  #.Plot lcfs
  ax1a[0].plot(RInt_lcfs,ZInt_lcfs,linewidth=1.5,linestyle='--',color='white',alpha=.8)
  ax1a[0].add_patch(Rectangle((0.95,-0.025),0.08,0.06,color='gray'))

  #.inset data
  axins2 = zoomed_inset_axes(ax1a[0], 6, loc='lower left', bbox_to_anchor=(0.45,-0.4),bbox_transform=ax1a[0].transAxes)  # zoom =
  img_in = axins2.pcolormesh(RIntN, ZIntN, np.squeeze(esPotInt_RZ)/scaleFac, cmap='inferno', shading='auto',vmin=minSOL,vmax=maxSOL)
  #axins2.pcolormesh(RIntN, ZIntN, np.squeeze(esPotInt_RZ)/scaleFac, cmap='inferno', shading='auto',vmin=minSOL,vmax=maxSOL)
  axins2.plot(RInt_lcfs,ZInt_lcfs,linewidth=1.5,linestyle='--',color='white',alpha=.6)
  cax = inset_axes(axins2,
                   width="10%",  # width = 10% of parent_bbox width
                   height="100%",  # height : 50%
                   loc='lower left',
                   bbox_to_anchor=(1.05, 0., 1, 1),
                   bbox_transform=axins2.transAxes,
                   borderpad=0,)
  fig1a.colorbar(img_in,cax=cax)
  
  # sub region of the original image
  x1, x2, y1, y2 = 2.16, 2.22, -.075, .075
  axins2.set_xlim(x1, x2)
  axins2.set_ylim(y1, y2)
  # fix the number of ticks on the inset axes
  axins2.yaxis.get_major_locator().set_params(nbins=7)
  axins2.xaxis.get_major_locator().set_params(nbins=2)
  axins2.xaxis.set_major_formatter(ticker.StrMethodFormatter("{x:.2f}"))
  
  # draw a bbox of the region of the inset axes in the parent axes and
  # connecting lines between the bbox and the inset axes area
  mark_inset(ax1a[0], axins2, loc1=1, loc2=4, fc="none", ec="0.5")

  ax1a[0].set_aspect('equal',adjustable='datalim')
  #plt.tight_layout()
  
  if outFigureFile:
    plt.savefig(outFileRoot+'_%03d'%t+figureFileFormat)
    plt.close()
  else:
    plt.show()

#.......................................................................#