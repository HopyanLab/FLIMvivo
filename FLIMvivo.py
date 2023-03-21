#!/usr/bin/env python3
import numpy as np
from scipy import optimize
from matplotlib import pyplot as plt
from matplotlib import gridspec
from pathlib import Path
import argparse
import time
import os

################################################################################
#import warnings
#warnings.filterwarnings("ignore")
################################################################################

#from math import factorial
#from numba import jit

################################################################################
# Functions used for fitting.
################################################################################

# Mono-Exponential model
def ME(x,A,tau):
	return A*np.exp(-x/tau)

# Bi-Exponential model
def BE(x,A,tau1,B,tau2):
	return A*np.exp(-x/tau1)+B*np.exp(-x/tau2)

# Instrument Response Function assuming a gaussian profile
def IRF(x,mu,sigma):
	return np.exp(-(x-mu)**2/2/sigma**2)/sigma/np.sqrt(2*np.pi)

# Mono-Exponential convolution
def MEC(x,A,tau1,mu,sigma):
	return np.fft.ifft(np.fft.fft(IRF(x,mu,sigma)) * \
					np.fft.fft(ME(x,A,tau1))).real * (x[1]-x[0])

# Bi-Exponential convolution
def BEC(x,B,tau2,A,tau1,mu,sigma):
	return np.fft.ifft(np.fft.fft(IRF(x,mu,sigma)) * \
					np.fft.fft(BE(x,A,tau1,B,tau2))).real * (x[1]-x[0])

# Negative Log-Likelihood estimator assuming Poisson statistics
def NLL(p, X, Y, F, startpoint=0, endpoint=-1):
	if endpoint == -1:
		endpoint = len(X)
	FX = F(X, *p)[startpoint:endpoint]
	RY = Y[startpoint:endpoint]
	return np.sum(FX - RY*np.log(FX)) / \
						np.sqrt(endpoint - startpoint)

#TODO: Not sure this is quite right!
def CHI(p, X, Y, F, startpoint=0, endpoint=-1):
	if endpoint == -1:
		endpoint = len(X)
	FX = F(X, *p)[startpoint:endpoint]
	RY = Y[startpoint:endpoint]
	return np.sum(np.divide((FX - RY)**2, FX)) / \
									(endpoint - startpoint)
#	return np.sum((FX - RY)**2) #/ (endpoint - startpoint)

# Minimse the Negative Log-Likelihood for a given function and dataset
def PerformFit(X, Y, F, p, startpoint=0, endpoint=-1, fit_type = 'NLL'):
	if fit_type == 'NLL':
		fit_function = lambda p, X, Y: NLL(p, X, Y, F, startpoint, endpoint)
	else:
		fit_function = lambda p, X, Y: CHI(p, X, Y, F, startpoint, endpoint)
	return optimize.fmin(fit_function, p, args=(X, Y),
						disp=False, full_output=True )

################################################################################
# Weiner Deconvolution on data. Used for Tail Fitting.
################################################################################

def WeinerDeconvolution (time_points, data_points, mu, sigma, alpha = 60.):
	time_zero = time_points(np.argmax(data_points)) - sigma
	H = np.fft.fftshift(np.fft.fft(data_points))
	G = np.fft.fftshift(np.fft.fft(IRF(time_points,time_zero,sigma)))
	M = (np.conj(G)/(np.abs(G)**2 + alpha**2))*H
	m = np.abs(H.shape[0]*np.fft.ifft(M).real)
	return m/np.amax(m)

################################################################################
# Fit for different endpoints. Used for Tail and Fast Convoolution fitting.
################################################################################

def FindEndpoint(time_points, data_points,
					fit_function, guess_params,
					startpoint = 0,
					fit_type = 'NLL', # 'CHI' for Chi Square
					coarse_N = 20, fine_N = 10):
	peak_index = np.argmax(data_points)
	peak_value = data_points[peak_index]
	mask = data_points[peak_index:] < peak_value * 0.2
	initial_index = peak_index + np.amin(np.where(mask))
	index_range = len(time_points) - initial_index
	total_N = coarse_N + fine_N
	endpoints = np.zeros(total_N,dtype=int)
	likelihoods = np.zeros(total_N)
	fit_params = np.zeros((total_N,len(guess_params)))
	for i in range(coarse_N):
		endpoint = int(initial_index + np.floor(i*index_range/coarse_N))
		endpoints[i] = endpoint
		fit = PerformFit(time_points, data_points,
						 fit_function, guess_params,
						 startpoint = startpoint,
						 endpoint = endpoint,
						 fit_type = fit_type)
		fit_params[i] = fit[0]
		likelihoods[i] = fit[1]
	iMax = np.argmax(likelihoods[:coarse_N])
	fine_upper = min(len(time_points),
				initial_index + np.floor((iMax+1)*index_range/coarse_N))
	fine_lower = max(initial_index,
				initial_index + np.floor((iMax-1)*index_range/coarse_N))
	fine_range = fine_upper - fine_lower
	for i in range(coarse_N, coarse_N + fine_N):
		endpoint = int(fine_lower + np.floor((i-coarse_N)*fine_range/fine_N))
		endpoints[i] = endpoint
		fit = PerformFit(time_points, data_points,
						 fit_function, guess_params,
						 startpoint = startpoint,
						 endpoint = endpoint,
						 fit_type = fit_type)
		fit_params[i] = fit[0]
		likelihoods[i] = fit[1]
	return fit_params, endpoints, likelihoods#/np.amax(likelihoods)

################################################################################
# Cut Data bassed on given thresholds as factor of peak
################################################################################

def CutData (time_points, data_points,
				lower_threshold = 0.01,
				upper_threshold = 0.01):
	running_average = (data_points[np.argmax(data_points):-6] + \
					   data_points[np.argmax(data_points)+1:-5] + \
					   data_points[np.argmax(data_points)+2:-4] + \
					   data_points[np.argmax(data_points)+3:-3] + \
					   data_points[np.argmax(data_points)+4:-2] + \
					   data_points[np.argmax(data_points)+5:-1])/6
	peak_index = np.argmax(data_points)
	peak_value = data_points[peak_index]
	mask = running_average < peak_value * upper_threshold
	if np.any(mask):
		upper_bound = np.amin(np.where(mask)) + np.argmax(data_points)
	else:
		upper_bound = len(data_points) - 6
	mask = data_points[10:np.argmax(data_points)] < peak_value * lower_threshold
	lower_bound = 0
	if np.any(mask):
		lower_bound = np.amax(np.where(mask)) + 10
	else:
		lower_bound = np.argmax(data_points)-25
	time_points = time_points[lower_bound:upper_bound]
	data_points = data_points[lower_bound:upper_bound]
	return time_points, data_points

################################################################################
# Process Datafile and Return Data
################################################################################

def ExtractData(filepath):
	# Ignore any broken lines in the file.
	fixedfilepath = Path(str(filepath) + '.fixed')
	with open(fixedfilepath, 'w') as fixedfile, open(filepath, 'r') as infile:
		for linenumber, line in enumerate(infile):
			try:
				t = float(line.split(',')[0])
				x = float(line.split(',')[1])
				fixedfile.write('{0:.12f},{1:.12f}'.format(t, x) + os.linesep)
			except ValueError:
				print('Bad line ({0:d}) in {1:s}:'.format(linenumber,
															filepath.name))
				print('\t', line)
	if fixedfilepath.stat().st_size == 0:
		return None
	data = np.genfromtxt(fixedfilepath, delimiter=',')
	fixedfilepath.unlink()
	# scale time data to nanoseconds (microscope gives picoseconds)
	time_points = data[:,0]/1000.
	data_points = data[:,1]
	###################################################
	## set peak time to zero. (this messes up FLIMseg)
	# peak_index = np.argmax(data_points)
	# time_zero = time_points[peak_index]
	# time_points = time_points - time_zero
	###################################################
	# before the laser pulse gives a baseline noise estimate
	data_points -= np.average(data_points[5:20])
	# scale maximum to unity
	data_points = data_points/np.amax(data_points)
	###################################################
	return time_points, data_points

################################################################################
# Tail Fit
################################################################################

def TailFit(filepath,
			biexp, autolife,
			passed_mu, passed_sigma,
			fit_type = 'NLL'):
	time_points, data_points = ExtractData(datafilepath)
	# if we know about the IRF do a Weiner deconvolution
	if passed_mu != 0. and passed_sigma != 0.:
		data_points = WeinerDeconvolution(time_points, data_points,
											passed_mu, passed_sigma)
	time_points, data_points = CutData(time_points, data_points)
	peak_index = np.argmax(data_points)
	peak_value = data_points[peak_index]
	if biexp:
		fit_function = BE
		initial_guess = [0.8*peak_value, 3.0, 0.2*peak_value, 0.3]
	else:
		fit_function = ME
		initial_guess = [1.0*peak_value, 3.0]
	#
	startpoint = peak_index
	fit_params, endpoints, likelihoods = FindEndpoint(
									time_points[peak_index:],
									data_points[peak_index:],
									fit_function, initial_guess,
									startpoint = startpoint,
									fit_type = fit_type)
	best_fit = np.argmax(likelihoods)
	endpoint = endpoints[best_fit]
	best_params = fit_params[best_fit]
	#
	test_threshold = 0.01
	test_set = np.unique(data_points[endpoint-16:endpoint])
	test = np.abs(test_set[0] - test_set[1]) * peak_value > test_threshold
	#
	fig = plt.figure(figsize=(16,7))
	gs = gridspec.GridSpec(1, 3, width_ratios=[0.8, 1.6, 1])
	ax = list(map(plt.subplot, gs))
	#
	ax[0].plot(time_points, data_points, 'o', label='Data')
	ax[0].plot( time_points[startpoint:endpoint],
				data_points[startpoint:endpoint],
				'x', label='Selected for fitting')
	ax[0].plot([time_points[peak_index], time_points[peak_index]],
				[np.amin(data_points), np.amax(data_points)], '--k')
	ax[0].set_xlabel('Time (ns)')
	ax[0].set_ylabel('Intensity (A.U.)')
	ax[0].legend()
	#
	ax[1].plot( time_points[startpoint:endpoint],
				data_points[startpoint:endpoint],
				marker = '.',
				linestyle = 'none',
				color = 'tab:blue',
				label = 'Data')
	if biexp:
		ax[1].plot( time_points[startpoint:endpoint],
				fit_function(time_points, *best_params)[startpoint:endpoint],
					linestyle = 'solid',
					color = 'tab:red',
					label = 'Full Fit')
		a = fit_function(time_points, *best_params)[endpoint] / \
					np.exp(-time_points[endpoint]/best_params[1])
		ax[1].plot(time_points[startpoint:endpoint],
				a*np.exp(-time_points[startpoint:endpoint]/best_params[1]),
				linestyle = 'dashed',
				color = 'tab:orange',
				label = r'Signal Fit ($\tau = ' + \
						'{0:.3f}ns'.format(best_params[1]) + r'$)')
	else:
		ax[1].plot( time_points[startpoint:endpoint],
				fit_function(time_points, *best_params)[startpoint:endpoint],
					linestyle = 'solid',
					color = 'tab:orange',
					label = r'Fit ($\tau = ' + \
						'{0:.3f}ns'.format(best_params[1]) + r'$)')
	ax[1].set_yscale('log')
	lowest_point = np.argmin(data_points[peak_index:endpoint])
	ax[1].set_ylim([data_points[lowest_point]*0.8,
					data_points[peak_index]*1.1])
	ax[1].set_xlim([time_points[startpoint]-0.2,time_points[endpoint]+0.1])
	ax[1].set_xlabel('Time (ns)')
	ax[1].legend()
#	ax[1].text(.04, .03, "Lifetime: {0:.3f} ns +/- {1:.3f}".format(
#										fit_params[best_fit,1],err),
	ax[1].text(.04, .03, "Lifetime: {0:.3f} ns".format(best_params[1]),
				bbox=dict(facecolor='wheat', alpha=1.0), fontsize=12,
				fontweight='bold', color='blue', transform=ax[1].transAxes)
	if test:
		ax[1].text(.4, .03, "Sparse Data!",
					bbox=dict(facecolor='wheat', alpha=1.0), fontsize=12,
					fontweight='bold', color='red', transform=ax[1].transAxes)
	if biexp:
		if autolife == 0.:
			ax[1].text(.04, .09, "Autofluorescence: {0:.3f} ns".format(
														best_params[3]),
				bbox=dict(facecolor='wheat', alpha=1.0), fontsize=12,
				fontweight='bold', color='blue', transform=ax[1].transAxes)
		else:
			ax[1].text(.04, .09, "Autofluorescence: {0:.3f} ns".format(
														autolife),
				bbox=dict(facecolor='wheat', alpha=1.0), fontsize=12,
				fontweight='bold', color='blue', transform=ax[1].transAxes)
	#
	ax[2].plot(time_points[endpoints], likelihoods, '.')
	ax[2].plot([time_points[endpoint], time_points[endpoint]],
				[np.amin(likelihoods), np.amax(likelihoods)],
				'--k')
#	if showfits:
#		ax[2].plot(time_points[endpoints], fit_params[:,1], '.')
	ax[2].text(.04, .09, "Time upper limit = {0:.2f}ns".format(
										time_points[endpoint]),
				bbox=dict(facecolor='wheat', alpha=1.0), fontsize=12,
				fontweight='bold', color='blue', transform=ax[2].transAxes)
	ax[2].text(.04, .03,"Number of data points = {0:d}".format(
										endpoints[best_fit]-peak_index),
				bbox=dict(facecolor='wheat', alpha=1.0), fontsize=12,
				fontweight='bold', color='blue', transform=ax[2].transAxes)
	ax[2].set_xlabel('Time Upper Bound (ns)')
	ax[2].set_ylabel('Goodness of Fit')
	#
	fig.suptitle(filepath.name)
	fig.savefig(filepath.with_suffix('.tailfit.pdf'))
	plt.close('all')
	print('\t'.join(['{0:2.12f}'.format(param) for param in best_params]))
	return best_params, test
#	return np.array([best_params[1], err]) # TODO: impliment error

################################################################################
# Slow Convolution Fit
################################################################################

def ConvolutionFit(filepath,
					biexp, autolife,
					passed_mu, passed_sigma):
	pass #TODO: this!

################################################################################
# Fast Convolution Fit
################################################################################

def FastConvolutionFit(filepath,
						biexp, autolife,
						passed_mu, passed_sigma,
						fit_type = 'NLL'):
	time_points, data_points = ExtractData(datafilepath)
	time_points, data_points = CutData(time_points, data_points)
	peak_index = np.argmax(data_points)
	peak_value = data_points[peak_index]
	IRF_centre_guess = time_points[peak_index]-0.24 # 0.12
	IRF_width_guess = 0.12 # 0.08
	if biexp:
		if autolife == 0.:
			if passed_mu == 0. or passed_sigma == 0.:
				fit_function = BEC
				initial_guess = [0.85*peak_value, 3.0, 0.15*peak_value, 0.3,
								IRF_centre_guess, IRF_width_guess]
			else:
				fit_function = lambda x, B, tau2, A, tau1: \
								BEC(x, B, tau2, A, tau1,
									passed_mu, passed_sigma)
				initial_guess = [0.85*peak_value, 3.0, 0.15*peak_value,0.3]
		else:
			if passed_mu == 0. or passed_sigma == 0.:
				fit_function = lambda x, B, tau2, A, mu, sigma: \
								BEC(x, B, tau2, A, autolife, mu, sigma)
				initial_guess = [0.85*peak_value, 3.0, 0.15*peak_value,
								IRF_centre_guess, IRF_width_guess]
			else:
				fit_function = lambda x, B, tau2, A: \
								BEC(x, B, tau2, A, autolife,
									passed_mu, passed_sigma)
				initial_guess = [0.85*peak_value, 3.0, 0.15*peak_value]
	else:
		if passed_mu == 0. or passed_sigma == 0.:
			fit_function = MEC
			initial_guess = [1.0*peak_value, 3.0,
								IRF_centre_guess, IRF_width_guess]
		else:
			fit_function = lambda x, A, tau: \
							MEC(x, A, tau, passed_mu, passed_sigma)
			initial_guess = [1.0*peak_value, 3.0]
	#
	startpoint = np.amax([0, np.argmax(data_points)-24])
	fit_params, endpoints, likelihoods = FindEndpoint(
									time_points, data_points,
									fit_function, initial_guess,
									startpoint = startpoint,
									fit_type = fit_type)
	best_fit = np.argmax(likelihoods)
	endpoint = endpoints[best_fit]
	best_params = fit_params[best_fit]
	fit_points = fit_function(time_points, *best_params)
	#
	if endpoint - startpoint < 60:
		test = False
	else:
		test_threshold = 0.03
		test_set = np.unique(data_points[endpoint-60:endpoint])
		test = np.abs(test_set[0] - test_set[1]) * peak_value > test_threshold
	#
	fig = plt.figure(figsize=(16,7))
	gs = gridspec.GridSpec(1, 3, width_ratios=[0.8, 1.6, 1])
	ax = list(map(plt.subplot, gs))
	#
	ax[0].plot(time_points, data_points, 'o', label='Data')
	ax[0].plot( time_points[startpoint:endpoint],
				data_points[startpoint:endpoint],
				'x', label='Selected for fitting')
	#ax[0].plot([time_points[peak_index], time_points[peak_index]],
	#			[np.amin(data_points), np.amax(data_points)], '--k')
	ax[0].set_xlabel('Time (ns)')
	ax[0].set_ylabel('Intensity (A.U.)')
	ax[0].legend()
	#
	ax[1].plot( time_points[startpoint:endpoint],
				data_points[startpoint:endpoint],
				marker = '.',
				linestyle = 'none',
				color = 'tab:blue',
				label = 'Data')
	if biexp:
		ax[1].plot( time_points[startpoint:endpoint],
					fit_points[startpoint:endpoint],
						linestyle = 'solid',
						color = 'tab:red',
						label = 'Full Fit')
		a = fit_points[endpoint] / \
					np.exp(-time_points[endpoint]/best_params[1])
		ax[1].plot(time_points[peak_index:endpoint],
				a*np.exp(-time_points[peak_index:endpoint]/best_params[1]),
				linestyle = 'dashed',
				color = 'tab:orange',
				label = r'Signal Fit ($\tau = ' + \
						'{0:.3f}ns'.format(best_params[1]) + r'$)')
	else:
		ax[1].plot( time_points[peak_index:endpoint],
					fit_points[peak_index:endpoint],
					linestyle = 'solid',
					color = 'tab:red',
					label = r'Fit ($\tau = ' + \
						'{0:.3f}ns'.format(best_params[1]) + r'$)')
	ax[1].set_yscale('log')
	ax[1].set_xlabel('Time (ns)')
#	lowest_point = np.argmin(data_points[peak_index:endpoint])
#	ax[1].set_ylim([data_points[lowest_point] * 0.8,
	ax[1].set_ylim([fit_points[endpoint] * 0.8,
					data_points[peak_index] * 1.1])
	ax[1].set_xlim([time_points[startpoint] - 0.2,
					time_points[endpoint] + 0.1])
	ax[1].legend()
#	ax[1].text(.04, .03, "Lifetime: {0:.3f} ns +/- {1:.3f}".format(
#										fit_params[best_fit,1],err),
	ax[1].text(.04, .03, "Lifetime: {0:.3f} ns".format(best_params[1]),
				bbox=dict(facecolor='wheat', alpha=1.0), fontsize=12,
				fontweight='bold', color='blue', transform=ax[1].transAxes)
	if test:
		ax[1].text(.4, .03, "Sparse Data!",
					bbox=dict(facecolor='wheat', alpha=1.0), fontsize=12,
					fontweight='bold', color='red', transform=ax[1].transAxes)
	if biexp:
		if autolife == 0.:
			ax[1].text(.04, .09, "Autofluorescence: {0:.3f} ns".format(
														best_params[3]),
				bbox=dict(facecolor='wheat', alpha=1.0), fontsize=12,
				fontweight='bold', color='blue', transform=ax[1].transAxes)
		else:
			ax[1].text(.04, .09, "Autofluorescence: {0:.3f} ns".format(
														autolife),
				bbox=dict(facecolor='wheat', alpha=1.0), fontsize=12,
				fontweight='bold', color='blue', transform=ax[1].transAxes)
	#
	ax[2].plot(time_points[endpoints], likelihoods, '.')
	ax[2].plot([time_points[endpoint], time_points[endpoint]],
				[np.amin(likelihoods), np.amax(likelihoods)],
				'--k')
#	if showfits:
#		ax[2].plot(time_points[endpoints], fit_params[:,1], '.')
	ax[2].text(.04, .09, "Time upper limit = {0:.2f}ns".format(
										time_points[endpoint]),
				bbox=dict(facecolor='wheat', alpha=1.0), fontsize=12,
				fontweight='bold', color='blue', transform=ax[2].transAxes)
	ax[2].text(.04, .03,"Number of data points = {0:d}".format(
										endpoints[best_fit]-peak_index),
				bbox=dict(facecolor='wheat', alpha=1.0), fontsize=12,
				fontweight='bold', color='blue', transform=ax[2].transAxes)
	ax[2].set_xlabel('Time Upper Bound (ns)')
	ax[2].set_ylabel('Fit Measure')
	#
	fig.suptitle(filepath.name)
	fig.savefig(filepath.with_suffix('.convofit.pdf'))
	plt.close('all')
	print('\t'.join(['{0:2.12f}'.format(param) for param in best_params]))
	return best_params, test
#	return np.array([best_params[1], err]) # TODO: impliment error

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
	parser.add_argument('-b', '--biex', dest='biexponential',
						action='store_const',
						const=True, default=False,
						help='Use a biexponential function.')
	parser.add_argument('-c', '--conv', dest='convolution',
						action='store_const',
						const=True, default=False,
						help = 'Do a convolution fit rather than tail fit.')
	parser.add_argument('-f', '--fcon', dest='fast_conv',
						action='store_const',
						const=True, default=False,
						help = 'Do a fast convolution fit. ' + \
				'(This requires uniformly temporally spaced data points.)')
	parser.add_argument('-x', '--xsqr', dest='fit_type',
						action='store_const',
						const='CHI', default='NLL',
						help = 'Use Chi Square rather than Poisson.')
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
						help = 'IRF mean and std. For convolution fitting. '+\
								'In tail fit mode use Weiner deconvolution.')
	parser.add_argument('-o', '--output',dest='outputfile',
						nargs = 1,
						default = [Path.cwd() / \
									'FLIMvivo-Output-{0:s}.csv'.format(
									time.strftime("%Y.%m.%d-%H.%M.%S"))],
						type = str,
						required = False,
						help = 'Filename to put output csv data in.')
	parser.add_argument('data', type=str, nargs='*', default=['./'],
						help='Path to csv data file(s) to process.')
	#
	parse_args = parser.parse_args()
	datapaths = list(map(Path, parse_args.data))
	outfilename = parse_args.outputfile[0]
	outfile = open(outfilename,'w')
	args = [parse_args.biexponential,
			parse_args.autolife[0],
			parse_args.response[0],
			parse_args.response[1],
			parse_args.fit_type]
	if not parse_args.biexponential:
		summary_data = np.zeros((0,4), dtype = float)
	else:
		summary_data = np.zeros((0,6), dtype = float)
	autolife = parse_args.autolife[0]
	for datapath in datapaths:
		if datapath.exists():
			if datapath.is_dir():
				datafilepaths = datapath.glob('*.csv') # use rglob to recurse
			else:
				datafilepaths = [datapath]
		else:
			print('Path {0:s} does not seem to exist.'.format(str(datapath)))
			continue
		# if it is a file or directory of files we proceed to extracting data.
		for datafilepath in datafilepaths:
			if datafilepath.suffix != '.csv':
				print('File {0:s} is not a csv file.'.format(
														str(datafilepath)))
				continue
			# if there are parsable csv data files we proceed to fitting.
			fit = np.array([0,0])
			if parse_args.fast_conv:
				fit, sparse_test = FastConvolutionFit(datafilepath, *args)
			elif parse_args.convolution:
			#	fit = ConvolutionFit(datafilepath, *args) #TODO: impliment
				fit, sparse_test = FastConvolutionFit(datafilepath, *args)
			else:
				fit, sparse_test = TailFit(datafilepath, *args)
			outfile.write('\t'.join([str(datafilepath),
			#				'Life time: {0:.3f}ns'.format(fit[0]),
			# TODO				'+/- {0:.3f}ns'.format(fit[1]),
							'{0:.8f}'.format(fit[1]),
							'sparse' if sparse_test else 'good'
							]) + os.linesep)
			try:
				if not parse_args.biexponential:
					summary_data = np.append(summary_data,
						[[int(datafilepath.stem.split('_')[-1]),
						  fit[1],
						  fit[0],
						  int(0 if sparse_test else 1)]],
						axis = 0)
				else:
					if parse_args.autolife[0] == 0.:
						autolife = fit[3]
					summary_data = np.append(summary_data,
						[[int(datafilepath.stem.split('_')[-1]),
						  fit[1],
						  fit[0],
						  autolife,
						  fit[2],
						  int(0 if sparse_test else 1)]],
						axis = 0)
			except:
				pass
	outfile.close()
	summary_data = summary_data[summary_data[:,0].argsort()]
	if not parse_args.biexponential:
		header = 'segment\tsignal_lifetime\t' + \
				 'signal_amplitude\tfit_good'
		fmt = '%d\t%1.9f\t%1.9f\t%d'
	else:
		header = 'segment\tsignal_lifetime\tsignal_amplitude\t' + \
				 'autoflourescent_lifetime\tautoflourescent_amplitude\tfit_good'
		fmt = '%d\t%1.9f\t%1.9f\t%1.9f\t%1.9f\t%d'
	np.savetxt(Path(outfilename).with_suffix('.summary.txt'),
					summary_data,
					delimiter = '\t',
					header = header,
					fmt = fmt)


