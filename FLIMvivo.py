#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt
from scipy import optimize
from scipy import integrate
from pathlib import Path
import argparse
import time
import os

import warnings
warnings.filterwarnings("ignore")

################################################################################
# Log Likelihood fitting for Poisson distributed data.
################################################################################

def LogLikelihood(p, X, Y):
	A = p[0]
	tau = p[1]
	return (A*np.sum(np.exp(-X/tau)) - np.log(A)*np.sum(Y) + np.sum(Y*X/tau))

def PoissonTauStdError(p, X, Y):
	A = p[0]
	tau = p[1]
	I = np.sum(2*X*Y + (X*X/tau - 2*X)*A*np.exp(-X/tau))/tau**3
	return 1/np.sqrt(I)#/np.sqrt(len(X))

def LogLikelihoodFit(mask, X, Y, p0 = [1.,3.0]):
	return optimize.fmin(LogLikelihood, p0,
							args=(X[mask], Y[mask]),
							disp=False, full_output=True)

################################################################################
# Log Likelihood fitting for Poisson distributed data with biexponential.
################################################################################

def MultiLogLikelihood(p, X, Y):
	A, tau1, B, tau2 = p
	thetaX = np.log(A*np.exp(-X/tau1) + B*np.exp(-X/tau2))
	return np.sum(np.exp(thetaX) - Y*thetaX)

def MultiPoissonTauStdError(p, X, Y):
	A, tau1, B, tau2 = p
	thetaX = np.log(A*np.exp(-X/tau1) + B*np.exp(-X/tau2))
	alpha = np.divide(B*np.exp(-X/tau2), np.exp(thetaX))
	beta = Y - np.exp(thetaX)
	I = np.sum(2/tau2**3*(X + X**2/tau2)*(alpha*beta) + \
				(Y*X**2)/tau2**3*alpha**2)
	# I = np.sum(alpha*beta/tau2**2)**2
	return 1/np.sqrt(I)#/np.sqrt(len(X))

def MultiLogLikelihoodFit(mask, X, Y, p0 = [0.5, 0.5, 0.5, 3.0]):
	return optimize.fmin(MultiLogLikelihood, p0,
							args=(X[mask], Y[mask]),
							disp=False, full_output=True)

################################################################################
# Log Likelihood fitting for Poisson distributed data with biexponential
#  and a predefined autoflourescence timescale.
################################################################################

def MultiLogLikelihoodWithNoise(p, tau1, X, Y):
	A, B, tau2 = p
	thetaX = np.log(A*np.exp(-X/tau1) + B*np.exp(-X/tau2))
	return np.sum(np.exp(thetaX) - Y*thetaX)

def MultiPoissonTauStdErrorWithNoise(p, tau1, X, Y):
	A, B, tau2 = p
	thetaX = np.log(A*np.exp(-X/tau1) + B*np.exp(-X/tau2))
	alpha = np.divide(B*np.exp(-X/tau2), np.exp(thetaX))
	beta = Y - np.exp(thetaX)
	I = np.sum(2/tau2**3*(X + X**2/tau2)*(alpha*beta) + \
				(Y*X**2)/tau2**3*alpha**2)
	# I = np.sum(alpha*beta/tau2**2)**2
	return 1/np.sqrt(I)#/np.sqrt(len(X))

def MultiLogLikelihoodFitWithNoise(mask, tau1, X, Y, p0 = [0.5, 0.5, 3.0]):
	return optimize.fmin(MultiLogLikelihoodWithNoise, p0,
							args=(tau1, X[mask], Y[mask]),
							disp=False, full_output=True)

################################################################################
# Fit a given CSV datafile.
################################################################################

def ProcessFile(filepath, showfits, biexp, autolife):
	# Ignore any broken lines in the file.
	fixedfilepath = Path(str(filepath) + '.fixed')
	with open(fixedfilepath, 'w') as fixedfile, open(filepath, 'r') as infile:
		for linenumber, line in enumerate(infile):
			try:
				t = float(line.split(',')[0])
				x = float(line.split(',')[1])
				fixedfile.write('{0:f},{1:f}'.format(t, x) + os.linesep)
			except ValueError:
				print('Bad line ({0:d}) in {1:s}:'.format(linenumber,
															filepath.name))
				print('\t', line)
	if fixedfilepath.stat().st_size == 0:
		return np.array([-1,-1])
	# 
	data = np.genfromtxt(fixedfilepath, delimiter=',')
	t = data[:,0]/1000. - data[np.argmax(data[:,1]),0]/1000.
	y = data[:,1]/np.amax(data[:,1])
	# before the laser pulse gives a baseline noise estimate
	y -= np.average(y[0:10])
	coarseN = 20
	fineN = 10
	N = coarseN + fineN
	fitmeasure = np.zeros(N)
	tubs = np.zeros(N)
	NoPs = np.zeros(N, dtype=int)
	sols = np.zeros((N,2))
	if biexp:
		if autolife == 0.:
			sols = np.zeros((N,4))
		else:
			sols = np.zeros((N,3))
	#
	tlb = t[np.argmax(y)]
	t0 = t[np.argmax(y) + 36]
	err = 0
	for i in range(coarseN):
		tub = t0 + i*(t[-1] - t0)/coarseN
		tubs[i] = tub
		mask = (t[:] > tlb) & (t[:] < tub)
		NoPs[i] = len(np.where(mask)[0])
		if biexp:
			if autolife == 0.:
				sols[i,:] = MultiLogLikelihoodFit(mask, t, y)[0]
				fitmeasure[i] = np.abs(MultiLogLikelihood(sols[i,:],
										t[mask], y[mask]) / \
											np.sqrt(NoPs[i]))
			else:
				sols[i,:] = MultiLogLikelihoodFitWithNoise(mask,
															autolife, t, y)[0]
				fitmeasure[i] = np.abs(MultiLogLikelihoodWithNoise(sols[i,:],
										autolife, t[mask], y[mask]) / \
											np.sqrt(NoPs[i]))
		else:
			sols[i,:] = LogLikelihoodFit(mask, t, y)[0]
			fitmeasure[i] = np.abs(LogLikelihood(sols[i,:],
									t[mask], y[mask]) / \
										np.sqrt(NoPs[i]))
	iMax = np.argmax(fitmeasure)
	tfineub = t0 + (iMax+1)*(t[-1] - t0)/coarseN
	tfinelb = t0 + (iMax-1)*(t[-1] - t0)/coarseN
	for i in range(fineN):
		tub = tfinelb + i*(tfineub - tfinelb)/fineN
		tubs[i+coarseN] = tub
		mask = (t[:] > tlb) & (t[:] < tub)
		NoPs[i+coarseN] = len(np.where(mask)[0])
		if biexp:
			if autolife == 0.:
				sols[i+coarseN,:] = MultiLogLikelihoodFit(mask, t, y)[0]
				fitmeasure[i+coarseN] = np.abs(
										MultiLogLikelihood(sols[i+coarseN,:],
											t[mask], y[mask]) / \
												np.sqrt(NoPs[i+coarseN]))
			else:
				sols[i+coarseN,:] = MultiLogLikelihoodFitWithNoise(mask,
															autolife, t, y)[0]
				fitmeasure[i+coarseN] = np.abs(
								MultiLogLikelihoodWithNoise(sols[i+coarseN,:],
											autolife, t[mask], y[mask]) / \
												np.sqrt(NoPs[i+coarseN]))
		else:
			sols[i+coarseN,:] = LogLikelihoodFit(mask, t, y)[0]
			fitmeasure[i+coarseN] = np.abs(LogLikelihood(sols[i+coarseN,:],
											t[mask], y[mask]) / \
												np.sqrt(NoPs[i+coarseN]))
	
	iMax = np.argmax(fitmeasure)
	sol = sols[iMax]
	mask = (t[:] > tlb) & (t[:] < tubs[iMax])
	if biexp:
		if autolife == 0.:
			err = MultiPoissonTauStdError(sol, t, y)
		else:
			err = MultiPoissonTauStdErrorWithNoise(sol, autolife, t, y)
	else:
		err = PoissonTauStdError(sol, t, y)
	print(str(filepath))
	print("Life time {0:.3f}ns".format(sol[-1]), end = '\t')
	if biexp:
		if autolife == 0.:
			print("{0:.3f}".format(sol[1]), end='\t')
		else:
			print("{0:.3f}".format(autolife), end='\t')
	print("{0:.3f}".format(sol[-1]))
	
	fig, ax = plt.subplots(1,3, figsize=(16,7))
	ax[0].plot(t[:], y[:], 'o', label='Data')
	ax[0].plot(t[mask], y[mask], 'x', label='Selected for fitting')
	ax[0].plot([t[np.argmax(y)], t[np.argmax(y)]],
				[np.min(y[:]), np.max(y[:])], '--k')
	ax[0].set_xlabel('Time (ns)')
	ax[0].set_ylabel('Intensity (A.U.)')
	#ax[0].set_xlim([0,20])
	ax[0].legend()

	ax[1].plot(t[mask], y[mask], '.', label='Data')
	if biexp:
		if autolife == 0.:
			ax[1].plot(t[mask], sol[0] * np.exp(-t[mask]/sol[1]) + \
								sol[2] * np.exp(-t[mask]/sol[3]), '-',
							label = r'Fit Line ($\tau = ' + \
									'{0:.3f}ns'.format(sol[3]) + r'$)')
		else:
			ax[1].plot(t[mask], sol[0] * np.exp(-t[mask]/autolife) + \
								sol[1] * np.exp(-t[mask]/sol[2]), '-',
							label = r'Fit Line ($\tau = ' + \
									'{0:.3f}ns'.format(sol[2]) + r'$)')
	else:
		ax[1].plot(t[mask], sol[0] * np.exp(-t[mask]/sol[1]), '-',
						label = r'Fit Line ($\tau = ' + \
								'{0:.3f}ns'.format(sol[1]) + r'$)')
	ax[1].text(.04, .03, "Lifetime: {0:.3f} ns +/- {1:.3f} ns".format(
														sol[-1],err),
				bbox=dict(facecolor='orange', alpha=0.5), fontsize=12,
				fontweight='bold', color='blue', transform=ax[1].transAxes)
	if biexp:
		if autolife == 0.:
			ax[1].text(.04, .09, "Autofluorescence: {0:.3f} ns".format(
														sol[1]),
				bbox=dict(facecolor='orange', alpha=0.5), fontsize=12,
				fontweight='bold', color='blue', transform=ax[1].transAxes)
		else:
			ax[1].text(.04, .09, "Autofluorescence: {0:.3f} ns".format(
														autolife),
				bbox=dict(facecolor='orange', alpha=0.5), fontsize=12,
				fontweight='bold', color='blue', transform=ax[1].transAxes)

	ax[1].set_yscale('log')
	ax[1].set_xlabel('Time (ns)')
	ax[1].legend()

	ax[2].plot(tubs, fitmeasure, '.')
	ax[2].plot([tubs[iMax], tubs[iMax]],
				[np.min(fitmeasure)+1, np.max(fitmeasure)],
				'--k')
	if showfits:
		ax[2].plot(tubs, sols[:,-1], '.')
	ax[2].text(tubs[iMax]*0.6, (np.min(fitmeasure)+np.max(fitmeasure))/2 - \
			(np.max(fitmeasure)-np.min(fitmeasure))/2.2,
			"Time upper limit = {0:.2f}ns\nNumber of points = {1:d}".format(
														tubs[iMax], NoPs[iMax]))
	ax[2].set_xlabel('Time Upper Bound (ns)')
	ax[2].set_ylabel('Goodness of Fit')
	fig.suptitle(filepath.name)
	#plt.show()
	fig.savefig(filepath.with_suffix('.pdf'))
	plt.close('all')
	print(sol)
	return np.array([sol[-1],err])

################################################################################
# Convolution fitting option.
################################################################################

def ConvolutionFit(filepath, biexp, autolife, passed_mu, passed_sigma):
	def ME(x,A,tau1,B,tau2):
		return A*np.exp(-x/tau1)+B*np.exp(-x/tau2)
	
	def IRF(x,mu,sigma):
		return np.exp(-(x-mu)**2/2/sigma**2)/sigma/np.sqrt(2*np.pi)
	
	def F(x,B,tau2,A,tau1,mu,sigma):
		return integrate.quad(
			lambda t,x: IRF(t,mu,sigma)*ME(x-t,A,tau1,B,tau2),
				-1.,x,args=x)[0]
	conv_function = np.vectorize(F)
	
	def G(x,B,tau2,A,mu,sigma):
		return integrate.quad(
			lambda t,x: IRF(t,mu,sigma)*ME(x-t,A,autolife,B,tau2),
				-1.,x,args=x)[0]
	conv_function2 = np.vectorize(G)
	
	def H(x,B,tau2,A):
		return integrate.quad(
			lambda t,x: IRF(t,passed_mu,passed_sigma)*ME(x-t,A,autolife,B,tau2),
				-1.,x,args=x)[0]
	conv_function3 = np.vectorize(H)
	
	def K(x,B,tau2, mu,sigma):
		return integrate.quad(
			lambda t,x: IRF(t,mu,sigma)*ME(x-t,0.,1.,B,tau2),
				-1.,x,args=x)[0]
	conv_function4 = np.vectorize(K)
	
	params = np.array([])
	covar = np.array([])
	
	fixedfilepath = Path(str(filepath) + '.fixed')
	with open(fixedfilepath, 'w') as fixedfile, open(filepath, 'r') as infile:
		for linenumber, line in enumerate(infile):
			try:
				t = float(line.split(',')[0])
				x = float(line.split(',')[1])
				fixedfile.write('{0:f},{1:f}'.format(t, x) + os.linesep)
			except ValueError:
				print('Bad line ({0:d}) in {1:s}:'.format(linenumber,
															filepath.name))
				print('\t', line)
	if fixedfilepath.stat().st_size == 0:
		return np.array([-1,-1])
	data = np.genfromtxt(fixedfilepath, delimiter=',')
	
	time_points = (data[:,0] - data[np.argmax(data[:,1]),0])/1000.
	data_points = data[:,1]/np.amax(data[:,1])
	mask = data_points[np.argmax(data_points):] < \
				(np.amin(data_points[10:]) + \
				(np.amax(data_points)-np.amin(data_points[10:])) * 0.03)
	upper_bound = 0
	if np.any(mask):
		upper_bound = np.amin(np.where(mask))
	else:
		upper_bound = len(mask)
	mask = data_points[10:np.argmax(data_points)] < \
				(np.amin(data_points[10:]) + \
				(np.amax(data_points)-np.amin(data_points[10:])) * 0.06)
	lower_bound = 0
	if np.any(mask):
		lower_bound = np.amax(np.where(mask))
	else:
		lower_bound = 0
	# mask = np.arange(np.argmax(data_points)-10, np.argmax(data_points)+340)
	time_points = time_points[lower_bound:upper_bound]
	data_points = data_points[lower_bound:upper_bound]
	
	if biexp:
		if autolife == 0.:
			params, covar = optimize.curve_fit(conv_function,
											time_points, data_points,
					p0=[.8,3.,.2,.3,time_points[np.argmax(data_points)],0.08])
			
		elif passed_mu == 0. or passed_sigma == 0.:
			params, covar = optimize.curve_fit(conv_function2,
											time_points, data_points,
					p0=[.8,3.,.2,time_points[np.argmax(data_points)],0.08])
		else:
			params, covar = optimize.curve_fit(conv_function3,
											time_points, data_points,
					p0=[.8,3.,.2])
	else:
		params, covar = optimize.curve_fit(conv_function4,
										time_points, data_points,
				p0=[1.,3.,time_points[np.argmax(data_points)],0.08])
	print('\t'.join(['{0:2.6f}'.format(param) for param in params[:-2]]))
	fig, ax = plt.subplots(1,1, figsize=(8,7))
	ax.set_xlabel('Time (ns)')
	ax.set_ylabel('Intensity (A.U.)')
	ax.set_yscale('log')
	ax.plot(time_points, data_points,'b.')
	if biexp:
		if autolife == 0.:
			ax.plot(time_points, conv_function(time_points,
													*params),'r-')
			split = 0
			for index, point in enumerate(time_points):
				if point > 0 and \
					ME(point, 0, params[1], params[2], params[3]) / \
						params[0] < 0.2:
					split = index
					break
			a = conv_function(time_points[-1],*params) / \
					np.exp(-time_points[-1]/params[1])
			ax.plot(time_points[split:],
					a*np.exp(-time_points[split:]/params[1]),'c--')
			ax.text(.1, .09, "Autofluorescence: {0:.3f} ns".format(params[3]),
				bbox=dict(facecolor='orange', alpha=0.5), fontsize=12,
				fontweight='bold', color='blue', transform=ax.transAxes)
		
		elif passed_mu == 0. or passed_sigma ==0.:
			ax.plot(time_points, conv_function2(time_points, *params),'r-')
		else:
			ax.plot(time_points, conv_function3(time_points, *params),'r-')
	else:
		ax.plot(time_points, conv_function4(time_points,
												*params),'r-')
	ax.text(.1, .03, "Lifetime: {0:.3f} +/- {1:.3f} ns".format(
										params[1], np.sqrt(covar[1,1])),
				bbox=dict(facecolor='orange', alpha=0.5), fontsize=12,
				fontweight='bold', color='blue', transform=ax.transAxes)
	ax.set_ylim([0.15,1.1])
#	ax.legend()
	fig.suptitle(filepath.name)
	if biexp:
		fig.savefig(filepath.with_suffix('.biexp.convo.pdf'))
	else:
		fig.savefig(filepath.with_suffix('.monoexp.convo.pdf'))
	plt.close('all')
	return np.array([params[1],np.sqrt(covar[1,1])])

################################################################################
# Main function uses argparse to take files and directories and process files.
################################################################################

if __name__ == "__main__":
	# Argparse is a nice way to take command line arguements.
	# We use it to take any number of files and/or directories to process, to
	#  set a flag for whether to use chi-square for Gaussian distributed data
	#  or log-likelyhood for Poisson distributed data, and to set a flag for
	#  whether to show the fit lifetimes with the likelyhood plot.
	parser = argparse.ArgumentParser(description='Fit decay data from FLIMfit.')
	parser.add_argument('-s', '--show', dest='showfits',
						action='store_const',
						const=True, default=False,
						help='Show fit lifetimes for worse fits.')
	parser.add_argument('-b', '--bi', dest='biexp',
						action='store_const',
						const=True, default=False,
						help='Use a biexponential function.')
	parser.add_argument('-a', '--auto', dest='autolife',
						nargs = 1,
						default = [0.],
						type = float,
						required = False,
						help = 'Give an autoflourescent lifetime.')
	parser.add_argument('-r', '--resp', dest='response',
						nargs = 2,
						default = [0., 0.],
						type = float,
						required = False,
						help = 'IRF mean and std. For convolution fitting.')
	parser.add_argument('-c', '--conv', dest='convolution',
						action='store_const',
						const=True, default=False,
						help = 'Do a full convolution fit rather than tail.')
	parser.add_argument('-o', '--output',dest='outputfile',
						nargs = 1,
						default = [Path.cwd() / \
									'FLIMvivo-Output-{0:s}.csv'.format(
									time.strftime("%Y.%m.%d-%H.%M.%S"))],
						type = str,
						required = False,
						help = 'Filename to put output csv data in.')
	parser.add_argument('data', type=str, nargs='*', default='./',
						help='Path to csv data file(s) to process.')

	args = parser.parse_args()
	datapaths = list(map(Path,args.data))
	outfile = open(args.outputfile[0],'w')
	for datapath in datapaths:
		if datapath.exists():
			if datapath.is_dir():
				for datafilepath in datapath.glob('*.csv'): # rglob recurses
					if datafilepath.suffix == '.csv':
						fit = np.array([0,0])
						if args.convolution:
							fit = ConvolutionFit(datafilepath,
													args.autolife[0],
													args.response[0],
													args.response[1])
						else:
							fit = ProcessFile(datafilepath, args.showfits,
														args.biexp,
														args.autolife[0])
						outfile.write('\t'.join([str(datafilepath),
										'Life time: {0:.3f}ns'.format(fit[0]),
										'+/- {0:.3f}ns'.format(fit[1]),
										'{0:.8f}'.format(fit[0])
										]) + os.linesep)
			else:
				if datapath.suffix == '.csv':
					fit = np.array([0,0])
					if args.convolution:
						fit = ConvolutionFit(datapath,
												args.biexp,
												args.autolife[0],
												args.response[0],
												args.response[1])
					else:
						fit = ProcessFile(datapath, args.showfits,
												args.biexp,
												args.autolife[0])
					outfile.write('\t'.join([str(datapath),
									'Life time: {0:.3f}ns'.format(fit[0]),
									'+/- {0:.3f}ns'.format(fit[1]),
									'{0:.8f}'.format(fit[0])
									]) + os.linesep)
				else:
					print('File {0:s} is not a csv file.'.format(str(datapath)))
		else:
			print('Path {0:s} does not seem to exist.'.format(str(datapath)))
	outfile.close()





