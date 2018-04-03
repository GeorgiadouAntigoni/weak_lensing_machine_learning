"""
Description
"""

import numpy as np 
import matplotlib.pyplot as plt 
from sys import exit
from scipy import integrate
from scipy.interpolate import interp1d, interp2d
#from CosmoPowerLib import CosmoPowerLib as CPL
from glob import glob 

import matplotlib
matplotlib.rc('xtick', labelsize=18) 
matplotlib.rc('ytick', labelsize=18) 

#==============================================================================

class WeakLensingLib(object):
	'''
	Most of the equations here are taken from Takada and Jain 2009 paper.
	Link: http://arxiv.org/pdf/0810.4170v2.pdf
	'''

	def __init__(self, NumberOfBins=1, zm=1.2, \
					zmin=0.2, zmax=5.0, zdim=100, \
					kmin = 0.001, kmax=100, kdim=1000, \
					lmin = 10, lmax=50000, ldim=40, \
					CosmoParams=[0.3,0.8,0.7,0.96,0.046,-1.0,0.0,0.0,0.0], \
					mode='linear', nskip=1, pkfiles_string='*.pk'):
		"""
		Doc string of the constructor
		"""

		self.CosmoParams = CosmoParams
		self.set_cosmology()
		self.zmin = zmin
		self.zmax = zmax
		self.zdim = zdim
		self.kmin = kmin
		self.kmax = kmax
		self.kdim = kdim
		self.lmin = lmin
		self.lmax = lmax
		self.ldim = ldim
		self.nskip = nskip
		self.log_lbin = (self.lmax - self.lmin) / self.ldim
		self.mode = mode
		self.pkfiles_string = pkfiles_string

		# Constants
		#----------
		self.SpeedOfLight = 299792.458 

		# Making Functions for distance redshift relations
		#-------------------------------------------------
		self.zArray = np.linspace(self.zmin, self.zmax, self.zdim)
		self.kiArray = np.zeros((len(self.zArray)))
		for i in range(len(self.zArray)):
			self.kiArray[i] = self.ComovingDistance(self.zArray[i])
		self.Func_z2ki = interp1d(self.zArray, self.kiArray)
		self.Func_ki2z = interp1d(self.kiArray, self.zArray)

		# Setting up Number of Bins and Bin edges
		#----------------------------------------
		self.NumberOfBins = NumberOfBins
		self.z0 = zm/3.0
		self.binedges_z = self.MakeBins(plot=False)
		self.binedges_ki = self.Func_z2ki(self.binedges_z)
		self.nsources_bins = self.n_i_bins()

		# Lensing Weights Matrix
		#-----------------------
		self.qMatrix = np.zeros((len(self.zArray), self.NumberOfBins))
		self.Make_qMatrix(plot=False)

#------------------------------------------------------------------------------

	def set_cosmology(self):
		# Setting up cosmology
		#---------------------
		[self.Omega_m, self.Sigma_8, self.h, self.n_s, self.Omega_b, \
			self.w0, self.wa, self.Omega_r, self.Omega_k] = self.CosmoParams
		self.Omega_l = 1.0 - self.Omega_m - self.Omega_r - self.Omega_k

#------------------------------------------------------------------------------

	def Func_pkmatrix(self, zz, kk):
		return self.Func_pkmatrix_interp(zz, kk)

#------------------------------------------------------------------------------

	def load_pk(self, plot=False):
		# Power Spectrum (Default: Linear)
		#--------------------
		self.kArray = 10**np.linspace(np.log10(self.kmin), \
							np.log10(self.kmax), self.kdim)
		if self.mode=='linear':
			self.PKmatrix = self.PK_linear(self.kArray, self.zArray)
		elif self.mode=='nonlinear':
			self.PKmatrix = self.PK_nonlinear(self.kArray, self.zArray)
		elif self.mode=='custom':
			self.PKmatrix = self.PK_customfolder(self.kArray, self.zArray)			
		else:
			print "Current supported modes are: linear, nonlinear"
			exit()
		self.Func_pkmatrix_interp = interp2d(self.zArray, \
									self.kArray, self.PKmatrix)
		if plot:
			self.plot_pk()

#------------------------------------------------------------------------------

	def plot_pk(self):	
		plt.figure(figsize=(10,6))
		for i in range(len(self.zArray)):
			plt.loglog(self.kArray, self.PKmatrix[:,i], 'k', lw=0.5)
		plt.xlabel('$\mathtt{k\ [h/Mpc]}$', fontsize=22)
		plt.ylabel('$\mathtt{P(k)\ [Mpc/h]^3}$', fontsize=22)
		plt.xlim(min(self.kArray), max(self.kArray))
		plt.tight_layout()
		plt.show()

#------------------------------------------------------------------------------

	def get_pk(self):
		return self.kArray, self.zArray, self.PKmatrix

#------------------------------------------------------------------------------

	def PK_customfolder(self, kk, zz):
		print "=========================================================="
		print "Reading files from: ", self.pkfiles_string
		print "Skipping every %i files"%(self.nskip-1)
		print "=========================================================="
		filenames = glob(self.pkfiles_string)
		pk = []
		k = []
		N = []
		z = []
		self.nfiles = 0
		print len(filenames)
		for i in range((len(filenames)-1)%self.nskip, \
									len(filenames), self.nskip):
			self.nfiles += 1
			name = filenames[i].split('/')[-1]
			name = name.split('a=')[1]
			name = float(name.replace('.dat', ''))
			z.append(1.0/name - 1)
			print i, filenames[i]
			data = np.genfromtxt(filenames[i], skip_header=2)
			pk.append(data[:,1])
			k.append(data[:,0])
			N.append(data[:,2])
		print "=========================================================="
		print "Total number of files used: ", self.nfiles
		print "=========================================================="		
		k = np.mean(np.array(k), axis=0)
		pk = np.array(pk)
		pk = np.transpose(pk)
		z = np.array(z)
		# print z
		# print k
		# print pk
		func = interp2d(z, k, pk)
		return func(zz, kk)

#------------------------------------------------------------------------------

	def PK_linear(self, kk, zz):
		CPLo = CPL(self.CosmoParams, computeGF=True)
		return CPLo.PKL_Camb_MultipleRedshift(kk, zz)

#------------------------------------------------------------------------------

	def PK_nonlinear(self, kk, zz):
		CPLo = CPL(self.CosmoParams, computeGF=True)
		return CPLo.PKNL_CAMB_MultipleRedshift(kk, zz)

#------------------------------------------------------------------------------

	def Ez(self, z=0.0):
		if z==0.0:
			return 1.0
		else:
			a = 1.0 / (1.0 + z)
			return (self.Omega_m/a**3 + \
						self.Omega_k/a**2 + \
						self.Omega_r/a**4 +\
						self.Omega_l/a**(3.0*(1.0+self.w0+self.wa))/\
						np.exp(3.0*self.wa*(1.0-a)))**0.5

#------------------------------------------------------------------------------

	def ComovingDistance(self, z=1.0):
		# Returns Comoving distance in Units Mpc
		func = lambda zz: 1.0/self.Ez(zz)
		ki = integrate.romberg(func, 0.0, z)
		return ki * self.SpeedOfLight / (self.h * 100.0)

#------------------------------------------------------------------------------

	def AngularDiameterDistance(self, z=1.0):
		# Returns Angular Diameter distance in Units Mpc
		return self.ComovingDistance(z)/(1.0+z)

#------------------------------------------------------------------------------

	def LuminosityDistance(self, z=1.0):
		# Returns Luminosity distance in Units Mpc
		return self.ComovingDistance(z) * (1.0+z)

#------------------------------------------------------------------------------

	def p_s(self, z):
		"""
		Refer to paper: https://arxiv.org/pdf/0810.4170.pdf; eqn 20; Sec 4.1
		z_m = 0.7, 1.0, 1.2, 1.5
		z0 = z_m / 3
		n_g = 10, 30 51, 100 arcmin^(-2)
		survey: DES, Subaru, LSST, SNAP
		"""
		return 1.18e9 * 4.0 * z**2 * np.exp(-z/self.z0)

#------------------------------------------------------------------------------

	def n_i(self, z1=0.001, z2=20.0):
		"""
		Refer to paper: https://arxiv.org/pdf/0810.4170.pdf; eqn 20; Sec 4.1
		z_m = 0.7, 1.0, 1.2, 1.5
		z0 = z_m / 3
		n_g = 10, 30 51, 100 arcmin^(-2)
		survey: DES, Subaru, LSST, SNAP
		"""
		result = integrate.quad(self.p_s, z1, z2, limit=500)[0]
		return result

#------------------------------------------------------------------------------

	def n_i_bins(self):
		nsources_bins = np.zeros((self.NumberOfBins))
		for i in range(self.NumberOfBins):
			nsources_bins[i] = self.n_i(self.binedges_z[i], \
								self.binedges_z[i+1])
		return nsources_bins

#------------------------------------------------------------------------------

	def MakeBins(self, plot=False):
		zArray = np.linspace(min(self.zArray)*2, max(self.zArray), 100)
		nArray = np.zeros((len(zArray)))
		for i in range(len(zArray)):
			nArray[i] = self.n_i(min(self.zArray), zArray[i])
		nArray /= max(nArray)
		binedges = [min(self.zArray)]
		for i in range(self.NumberOfBins - 1):
			binedges.append(np.interp(float(i+1)/self.NumberOfBins, \
				nArray, zArray))
		binedges.append(max(self.zArray))
		if plot:
			self.plot_dist(binedges)
		return binedges

#------------------------------------------------------------------------------

	def plot_dist(self, binedges=None):
		if binedges==None:
			binedges = self.binedges_z
		plt.figure(figsize=(10,6))
		plt.plot(self.zArray,self.p_s(self.zArray), 'k', lw=2)
		for i in range(len(binedges)):
			plt.axvline(x=binedges[i], color='b', ls='--', lw=1)
		plt.xlim(0, max(self.zArray))
		plt.xlabel('$\mathtt{Redshift}$', fontsize=22)
		plt.ylabel('$\mathtt{DistributionOfSources}$', fontsize=22)
		plt.tight_layout()
		plt.show()		

#------------------------------------------------------------------------------

	def _q(self, zs, z):
		return self.p_s(zs) * \
				(self.Func_z2ki(zs) - self.Func_z2ki(z)) / \
	    		self.Func_z2ki(zs)

#------------------------------------------------------------------------------

	def q(self, chi, chi1, chi2):
		if chi>chi2:
			return 1e-35
		else:		    
			zz = self.Func_ki2z(chi)
			z1 = self.Func_ki2z(chi1)
			z2 = self.Func_ki2z(chi2)
			zmin = max(zz, z1)
			zbin = 0.01
			zdim = (z2 - zmin)/zbin
			zintarray = np.linspace(zmin, z2, zdim)
			result = 0
			for i in range(len(zintarray)):
				result += self._q(zintarray[i], zz) * zbin
			# result = integrate.quad(self._q, max(zz, z1), z2, \
									# args=tuple([zz]), limit=500)[0]
			return result * 1.5 * (self.h * 100)**2 * self.Omega_m / \
		    		self.SpeedOfLight**2 * chi * (1.0+zz) /self.n_i(z1, z2)

#------------------------------------------------------------------------------

	def Make_qMatrix(self, plot=False):
		for i in range(self.NumberOfBins):
			for j in range(len(self.zArray)):
				self.qMatrix[j,i] = self.q(self.kiArray[j], \
					self.binedges_ki[i], self.binedges_ki[i+1])
		if plot:
			self.plot_q()

#------------------------------------------------------------------------------

	def plot_q(self):
		plt.figure(figsize=(10,6))
		plt.axvline(x=self.binedges_z[0], color='k', ls=':', lw=0.5)
		for i in range(self.NumberOfBins):
			plt.plot(self.zArray, self.qMatrix[:,i], lw=2, \
							label='$\mathtt{Bin:\ %i}$'%(i+1))
			plt.axvline(x=self.binedges_z[i+1], color='k', ls=':', lw=0.5)
		plt.legend(loc=1, fontsize=18)
		plt.xlim(0, max(self.zArray))
		plt.xlabel('$\mathtt{Redshift}$', fontsize=22)
		plt.ylabel('$\mathtt{q(z)\ LensingKernel}$', fontsize=22)
		plt.tight_layout()			
		plt.show()

#------------------------------------------------------------------------------

	def _Cell(self, chi, ell, bin1, bin2):
		zz = self.Func_ki2z(chi)
		integrand = np.interp(zz, self.zArray, \
						(self.qMatrix[:, bin1])) * \
					np.interp(zz, self.zArray, \
						(self.qMatrix[:, bin2])) * \
					self.Func_pkmatrix(zz, ell/chi) / chi**2				
		return integrand

#------------------------------------------------------------------------------

	def Cell(self, ell, bin1, bin2):
		chimin = min(self.kiArray)
		chimax = max(self.kiArray)
		result=0.0
		for i in range(len(self.kiArray)-1):
			result += self._Cell(self.kiArray[i], ell, bin1, bin2) * \
							(self.kiArray[i+1] - self.kiArray[i])
		return result

#------------------------------------------------------------------------------

	def CellVector(self, ell, bin1, bin2):
		cc = np.zeros((len(ell)))
		for i in range(len(ell)):
			cc[i] = self.Cell(ell[i], bin1, bin2)
		return cc

#------------------------------------------------------------------------------

	def CellMatrix(self, ell, plotpk = False, plot=False):
		self.load_pk(plot=plotpk)
		CellArray = np.zeros((self.NumberOfBins, self.NumberOfBins, len(ell)))
		for i in range(self.NumberOfBins):
			for j in range(i, self.NumberOfBins):
					CellArray[i,j,:] = self.CellVector(ell, i, j)
					CellArray[j,i,:] = CellArray[i,j,:]
		if plot:
			self.plot_cell(ell, CellArray)
		return CellArray

#------------------------------------------------------------------------------

	def plot_cell(self, ell=None, CellArray=None):
		if ell==None or CellArray==None:
			ell = self.ellArray
			CellArray = self.CellArray

		plt.figure(figsize=(10,6))
		for i in range(self.NumberOfBins):
			for j in range(i, self.NumberOfBins):
				if i==j:
					plt.loglog(ell, CellArray[i,j,:] * ell * \
						(ell+1)/2.0/np.pi, ls='-', label='%i, %i'%(i,j))
				else:
					plt.loglog(ell, CellArray[i,j,:] * ell * \
						(ell+1)/2.0/np.pi, ls='--')
		plt.legend(loc=2, fontsize=14)
		plt.xlim(min(ell), max(ell))
		plt.xlabel('$\mathtt{\ell}$', fontsize=22)
		plt.ylabel('$\mathtt{C_{\ell}}$', fontsize=22)
		plt.tight_layout()
		plt.show()

#------------------------------------------------------------------------------

	def plot_cell_grid(self):
		f, axarr = plt.subplots(self.NumberOfBins, self.NumberOfBins, \
							sharex=True, sharey=True, figsize=(15,15))
		f.subplots_adjust(wspace=0,hspace=0)

		for j1 in range(self.NumberOfBins):
		    for j2 in range(j1,self.NumberOfBins):
				axarr[j2,j1].loglog(self.ellArray, self.CellArray[j1,j2,:]* \
		        					self.ellArray*(self.ellArray+1)/2.0/np.pi, \
		        					'ok-', ms=8, lw=1)
				if j1==j2:
					maxl = self.kmax * self.binedges_ki[j1+1]
					minl = self.kmin * self.binedges_ki[j1]
					axarr[j2,j1].axvline(x=maxl, color='k', ls=':', lw=0.5)
					axarr[j2,j1].axvline(x=minl, color='k', ls=':', lw=0.5)
		f.text(0.5, 0.07, '$\mathtt{\ell}$', ha='center', fontsize=33)
		f.text(0.04, 0.5, '$\mathtt{C_{\ell}}$', va='center', \
						rotation='vertical', fontsize=33)	
		plt.show()	

#------------------------------------------------------------------------------

	def compute_cell(self, plotpk=False, plot=False):
		self.ellArray = 10**np.linspace(np.log10(self.lmin), \
								np.log10(self.lmax), self.ldim)
		self.CellArray = self.CellMatrix(self.ellArray, \
												plotpk=plotpk, plot=plot)
		return self.ellArray ,self.CellArray

#==============================================================================

if __name__=="__main__":
	from sys import argv
	string = '/Users/mohammed/Dropbox/fermilabwork/with_gnedin/data_ch/ch/10/a/psM*.dat'
	co = WeakLensingLib(NumberOfBins=1, nskip=1, mode='custom', pkfiles_string=string)
	# ell, c_l = co.compute_cell('linear', plot=True)
	# ell, c_nl = co.compute_cell('nonlinear', plot=True)
	ell, c_cu = co.compute_cell(plot=True)
	
