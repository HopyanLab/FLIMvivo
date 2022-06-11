#!/usr/bin/env python3
from readPTU_FLIM import PTUreader
import numpy as np
from matplotlib import pyplot as plt
from scipy import optimize
#from scipy import ndimage
from astropy.convolution import Gaussian2DKernel
from astropy.convolution import convolve, convolve_fft
from PIL import Image
from pathlib import Path
import argparse
import os
import re

'''
####################
####################
## MakeHeatmap.py ##
####################
####################
use after FLIMseg.py
'''

rootpath = Path(os.path.realpath(__file__)).parent.resolve()

################################################################################
# 
################################################################################

def process_dir (directory, channel = 0, smoothing = 0,
				 opacity = 0.5, remove_bad = False,
				 correlation = False, flip_axis = False):
	print('Processing: ', directory)
	output_dir = directory/'FLIMseg-output'
	summary_file = output_dir/'FLIMvivo.output.summary.txt'
	results = np.genfromtxt(summary_file, delimiter = '\t', dtype = float)
	has_fit_check = (results.shape[1] == 3)
	if has_fit_check:
		good_segments = np.array([results[:,0],results[:,2]], dtype = int).T
		results[:,2] = 0
	else:
		results = np.pad(results, ((0,0),(0,1)))
	ptu_files = np.empty(shape = [0,2])
	tif_files = np.empty(shape = [0,2])
	for ptu_file in directory.glob('*.ptu'):
		print('Found ptu file: \n', ptu_file, '\n')
		number = 0
		if re.split('[\s_]+',ptu_file.stem)[-1].lstrip('t').isdigit():
			number = int(re.split('[\s_]+',ptu_file.stem)[-1].lstrip('t'))
		ptu_files = np.append(ptu_files, np.array([[number,
													ptu_file]]), axis = 0)
	for tif_file in directory.glob('*.tif'):
		print('Found tif file: \n', tif_file, '\n')
		number = 0
		if re.split('[\s_]+', tif_file.stem)[-2].lstrip('t').isdigit():
			number = int(re.split('[\s_]+', tif_file.stem)[-2].lstrip('t'))
		tif_files = np.append(tif_files, np.array([[number,
													tif_file]]), axis = 0)
	# sort them by number
	ptu_files = ptu_files[ptu_files[:, 0].argsort()]
	tif_files = tif_files[tif_files[:, 0].argsort()]
	try:
		print('Attempting to read: \n', ptu_files[0,1].name)
		ptu_stream = PTUreader(ptu_files[0,1], print_header_data = False)
		flim_data_stack = ptu_stream.get_flim_data_stack()
		if flip_axis:
			flim_data_stack = flim_data_stack[::-1,:,:,:]
		intensity_image = np.sum(flim_data_stack[:,:,channel,:], axis=2)
		print('Successfully read.')
	except:
		print('There was a problem reading. \nTerminating.\n')
		print('-------------------------------------\n')
		return
	(Image.fromarray(intensity_image.astype('uint8'))).save(output_dir / \
			(ptu_files[0,1].with_suffix('.intensity.tif').name))
	seg_mask = np.array(Image.open(
			tif_files[np.abs(tif_files[:,0] - ptu_files[0,0]).argmin(), 1]))
	scaled_intensity = intensity_image[
						::int(len(intensity_image)/len(seg_mask)),
						::int(len(intensity_image[0])/len(seg_mask[0]))]
	seg_image = (seg_mask > 0)*1.
	seg_image[seg_mask == 0] = np.nan
	results[results[:,0] == 0, 2] = np.mean(scaled_intensity)
	for segment in np.unique(seg_mask[seg_mask > 0]):
		segment_points = np.where(seg_mask == segment)
		seg_image[segment_points] = results[results[:,0] == segment,1]
		results[results[:,0] == segment,2] = \
				np.mean(scaled_intensity[segment_points])
		if has_fit_check and remove_bad:
			if not good_segments[good_segments[:,0] == segment, 1]:
				seg_image[segment_points] = np.nan
	seg_alpha = (seg_mask > 0)*opacity
	if smoothing > 0:
		kernel = Gaussian2DKernel(x_stddev = smoothing,
								  y_stddev = smoothing)
		seg_image = convolve(seg_image, kernel, boundary = 'extend')
	if correlation:
		def fit_function(x, a, b):
			return -a*x + b
		fit_params, covar = optimize.curve_fit(fit_function,
									results[:,1], results[:,2])
		fig, (ax,ax2) = plt.subplots(1,2, figsize=(10,6))
		ax2.plot(results[:,1],results[:,2],
					marker = '.', linestyle = '',
					color = 'black')
		ax2.plot(results[:,1],fit_function(results[:,1],*fit_params),
					marker = '', linestyle = '-',
					color = 'red')
		aspect = np.diff(ax2.get_xlim())[0] / np.diff(ax2.get_ylim())[0]
		ax2.set_aspect(aspect)
		print(fit_params)
		print(covar)
	else:
		fig, ax = plt.subplots(1,1, figsize = (6,6))
	heatmap = ax.imshow(seg_image,
				cmap = plt.get_cmap('jet'),
				interpolation = 'none',
				origin = 'lower')
	ax.imshow(np.sqrt(scaled_intensity),
				cmap = plt.get_cmap('binary_r'),
				interpolation = 'none',
				origin = 'lower')
	ax.imshow(seg_image, alpha = seg_alpha,
				cmap = plt.get_cmap('jet'),
				interpolation = 'none',
				origin = 'lower')
	plt.colorbar(heatmap, ax=ax, orientation = 'vertical',
					fraction=0.046, pad=0.04)
	fig.tight_layout()
	plt.savefig(output_dir / (ptu_files[0,1].with_suffix(
								'.heatmap.png').name))
#	plt.subplots_adjust(
#		left  = 0.125, # the left side of the subplots of the figure
#		right = 0.9,   # the right side of the subplots of the figure
#		bottom = 0.1,  # the bottom of the subplots of the figure
#		top = 0.9,     # the top of the subplots of the figure
#		wspace = 0.2,  # the width reserved for blank space between subplots
#		hspace = 0.2)  # the height reserved for white space between subplots
	plt.clf()
	plt.close()


################################################################################
# Main function uses argparse to take directories and process files.
################################################################################

if __name__ == '__main__':
	parser = argparse.ArgumentParser(description='')
	parser.add_argument('-c', '--chan', dest='channel',
						nargs = 1,
						default = [0],
						type = int,
						required = False,
						help = 'use a different channel (default 0)')
	parser.add_argument('-s', '--smooth', dest='smoothing',
						nargs = 1,
						default = [0],
						type = int,
						required = False,
						help = 'guassian smooth result (default 0)')
	parser.add_argument('-o', '--opac', dest='opacity',
						nargs = 1,
						default = [0.5],
						type = float,
						required = False,
						help = 'opacity for heatmap overlay (default 0.5)')
	parser.add_argument('-r', '--remove', dest='remove_bad',
						action='store_const',
						const=True, default=False,
						help = 'remove segments where fit was not great')
	parser.add_argument('-x', '--corr', dest='correlation',
						action='store_const',
						const=True, default=False,
						help = 'check correlation between fits and intensity')
	parser.add_argument('-f', '--flip', dest='flip_axis',
						action='store_const',
						const=True, default=False,
						help = 'flip Y-axis (old FLIMfit standard)')
	parser.add_argument('data', type=str, nargs='*', default=['./'],
						help='data file directory path')
	args = parser.parse_args()
	datapaths = list(map(Path,args.data))
	dirs_to_process = np.array([])
	for datapath in datapaths:
		if datapath.exists():
			if datapath.is_dir():
				for datafilepath in datapath.rglob('*.ptu'):
					parentdirpath = datafilepath.resolve().parent
					if parentdirpath not in dirs_to_process:
						dirs_to_process = np.append(dirs_to_process,
													parentdirpath)
			else:
				print('Path {0:s} is not a directory.'.format(str(datapath)))
		else:
			print('Path {0:s} does not seem to exist.'.format(str(datapath)))
		for directory in dirs_to_process:
		#	try:
			process_dir(directory,
							args.channel[0],
							args.smoothing[0],
							args.opacity,
							args.remove_bad,
							args.correlation,
							args.flip_axis)
		#	except:
		#		print('-------------------------------------\n')
		#		print('There was a problem with: ' + str(directory) +'\n')
		#		print('Have you run FLIMseg.py yet?')
		#		print('-------------------------------------\n')
