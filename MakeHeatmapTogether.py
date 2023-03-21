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
#############################
#############################
## MakeHeatmap_Together.py ##
#############################
#############################
use after FLIMseg.py
'''

rootpath = Path(os.path.realpath(__file__)).parent.resolve()

################################################################################
# 
################################################################################

def process_dir (directory, channel = 0, smoothing = 0,
				 cbar_min = -1, cbar_max = -1,
				 opacity = 0.5, remove_bad = False,
				 flip_axis = False,
				 backward_order = False,
				 correlation = False):
	print(backward_order)
	# expecting nd_z1.ptu, nd_z1.tif, "FLIMseg-output z1",
	#           nd_z2.ptu, nd_z2.tif, "FLIMseg-output z2",
	#           nd_z3.ptu, nd_z3.tif, "FLIMseg-output z3", etc.
	print('Processing: ', directory)
	# assemble lists of ptu, tif, and output directories
	ptu_files = np.empty(shape = [0,2])
	tif_files = np.empty(shape = [0,2])
	output_dirs = np.empty(shape = [0,2])
	for ptu_file in directory.glob('*.ptu'):
		print('Found ptu file: \n', ptu_file, '\n')
		number = 0
		if re.split('[\s_]+',ptu_file.stem)[-1].lstrip('z').isdigit():
			number = int(re.split('[\s_]+',ptu_file.stem)[-1].lstrip('z'))
			ptu_files = np.append(ptu_files, np.array([[number,
													ptu_file]]), axis = 0)
	ptu_files = ptu_files[np.argsort(ptu_files[:,0])]
	for tif_file in directory.glob('*.tif'):
		print('Found tif file: \n', tif_file, '\n')
		number = 0
		if re.split('[\s_]+', tif_file.stem)[-1].lstrip('z').isdigit():
			number = int(re.split('[\s_]+', tif_file.stem)[-1].lstrip('z'))
			tif_files = np.append(tif_files, np.array([[number,
													tif_file]]), axis = 0)
	tif_files = tif_files[np.argsort(tif_files[:,0])]
	for output_dir in directory.glob('FLIMseg-output*'):
		print('Found FLIMseg output directory: \n', output_dir, '\n')
		number = 0
		if re.split('[\s_]+', output_dir.stem)[-1].lstrip('z').isdigit():
			number = int(re.split('[\s_]+', output_dir.stem)[-1].lstrip('z'))
			output_dirs = np.append(output_dirs, np.array([[number,
													output_dir]]), axis = 0)
	output_dirs = output_dirs[np.argsort(output_dirs[:,0])]
	# check there is a tif and output for each ptu
	for index, number in enumerate(ptu_files[:,0]):
		if number not in tif_files[:,0]:
			print('Error: No tif file for ptu file: ',
						ptu_files[index,1].name,
						'\n Exiting. \n')
			return
		if number not in output_dirs[:,0]:
			print('Error: No output directory for ptu file: ',
						ptu_files[index,1].name,
						'\n Exiting. \n')
			return
	# setup figure for output
	number_plots = ptu_files.shape[0]
	columns = min(4,number_plots)
	rows = int(np.ceil(number_plots/4))
	fig, ax = plt.subplots(rows, columns, figsize = (6*columns, 6*rows))
	fig.tight_layout()
	int_images = np.empty(ptu_files.shape[0],dtype = object)
	seg_images = np.empty(ptu_files.shape[0],dtype = object)
	seg_alphas = np.empty(ptu_files.shape[0],dtype = object)
	max_values = np.zeros(ptu_files.shape[0],dtype = float)
	min_values = np.zeros(ptu_files.shape[0],dtype = float)
	for ptu_index, number in enumerate(ptu_files[:,0]):
		tif_index = np.where(tif_files[:,0] == number)[0]
		dir_index = np.where(output_dirs[:,0] == number)[0]
		summary_file = output_dirs[dir_index,1]/'FLIMvivo.output.summary.txt'
		results = np.genfromtxt(summary_file[0], delimiter = '\t', dtype = float)
		if remove_bad:
			has_fit_check = (results.shape[1] == 6)
			if has_fit_check:
				good_segments = np.array([results[:,0],results[:,5]],
														dtype = int).T
				results = results[:,-1]
			else:
				print('Cannot remove bad points. ',
					  'No check in FLIMseg output.')
		results = results[:,:3]
		try:
			print('Attempting to read: \n', ptu_files[ptu_index,1].name)
			ptu_stream = PTUreader(ptu_files[ptu_index,1],
									print_header_data = False)
			flim_data_stack = ptu_stream.get_flim_data_stack()
			if flip_axis:
				flim_data_stack = flim_data_stack[::-1,:,:,:]
			intensity_image = np.sum(flim_data_stack[:,:,channel,:], axis=2)
			print('Successfully read.')
		except:
			print('There was a problem reading. \nTerminating.\n')
			print('-------------------------------------\n')
			return
		seg_mask = np.array(Image.open(tif_files[tif_index, 1][0]))
		if seg_mask.shape[0] != intensity_image.shape[0] or \
		   seg_mask.shape[1] != intensity_image.shape[1]:
			print('Size mismatch! Scaling tif.')
			y_factor = int(intensity_image.shape[0]/seg_mask.shape[0])
			x_factor = int(intensity_image.shape[1]/seg_mask.shape[1])
			new_seg_mask = np.zeros_like(intensity_image)
			for i in range(intensity_image.shape[0]):
				for j in range(intensity_image.shape[1]):
					new_seg_mask[i,j] = seg_mask[int(np.floor(i/y_scale)),
												 int(np.floor(j/x_scale))]
			seg_mask = new_seg_mask
		seg_image = (seg_mask > 0)*1.
		seg_image[seg_mask == 0] = np.nan
		for segment in np.unique(seg_mask[seg_mask > 0]):
			segment_points = np.where(seg_mask == segment)
			seg_image[segment_points] = results[results[:,0] == segment,1]
			results[results[:,0] == segment,2] = \
					np.mean(intensity_image[segment_points])
			if remove_bad:
				if not good_segments[good_segments[:,0] == segment, 1]:
					seg_image[segment_points] = np.nan
		seg_alpha = (seg_mask > 0)*opacity
		if smoothing > 0:
			kernel = Gaussian2DKernel(x_stddev = smoothing,
									  y_stddev = smoothing)
			seg_image = convolve(seg_image, kernel, boundary = 'extend')
		int_images[ptu_index] = intensity_image
		seg_images[ptu_index] = seg_image
		seg_alphas[ptu_index] = seg_alpha
		max_values[ptu_index] = np.nanmax(seg_image)
		min_values[ptu_index] = np.nanmin(seg_image)
	max_value = np.amax(max_values)
	min_value = np.amin(min_values)
	if cbar_max != -1:
		max_value = cbar_max
	if cbar_min != -1:
		min_value = cbar_min
	print(max_value)
	print(min_value)
	for original_index, number in enumerate(ptu_files[:,0]):
		if backward_order:
			index = len(ptu_files[:,0]) - 1 - original_index
		else:
			index = original_index
		plot_row = int(np.floor(original_index/4))
		plot_col = original_index - plot_row*4
		ax[plot_row,plot_col].set_title(f'Z Slice {number:d}')
		heatmap = ax[plot_row, plot_col].imshow(seg_images[index],
					cmap = plt.get_cmap('jet'),
					vmax = max_value,
					vmin = min_value,
					interpolation = 'none',
					origin = 'lower',
					zorder = 5)
		ax[plot_row, plot_col].imshow(np.sqrt(int_images[index]),
					cmap = plt.get_cmap('binary_r'),
					interpolation = 'none',
					origin = 'lower',
					zorder = 6)
		ax[plot_row, plot_col].imshow(seg_images[index],
					alpha = seg_alphas[index],
					cmap = plt.get_cmap('jet'),
					vmax = max_value,
					vmin = min_value,
					interpolation = 'none',
					origin = 'lower',
					zorder = 7)
	cbar = fig.colorbar(heatmap, ax = ax.ravel().tolist(),
						orientation = 'vertical',  shrink=0.95)
	cbar.mappable.set_clim(vmin = min_value, vmax = max_value)
	plt.savefig(directory / 'heatmap.svg')
	plt.clf()
	plt.close()

#	plt.colorbar(heatmap, ax=ax, orientation = 'vertical',
#					fraction=0.046, pad=0.04)
#	
#	plt.savefig(output_dir / (ptu_files[0,1].with_suffix(
#								'.heatmap.svg').name))
##	plt.subplots_adjust(
##		left  = 0.125, # the left side of the subplots of the figure
##		right = 0.9,   # the right side of the subplots of the figure
##		bottom = 0.1,  # the bottom of the subplots of the figure
##		top = 0.9,     # the top of the subplots of the figure
##		wspace = 0.2,  # the width reserved for blank space between subplots
##		hspace = 0.2)  # the height reserved for white space between subplots
#	plt.clf()
#	plt.close()


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
	parser.add_argument('-n', '--min', dest='min',
						nargs = 1,
						default = [-1],
						type = float,
						required = False,
						help = 'min value for colorbar (-1 = dynamic)')
	parser.add_argument('-x', '--max', dest='max',
						nargs = 1,
						default = [-1],
						type = float,
						required = False,
						help = 'max value for colorbar (-1 = dynamic)')
	parser.add_argument('-r', '--remove', dest='remove_bad',
						action='store_const',
						const=True, default=False,
						help = 'remove segments where fit was not great')
	parser.add_argument('-f', '--flip', dest='flip_axis',
						action='store_const',
						const=True, default=False,
						help = 'flip Y-axis (old FLIMfit standard)')
	parser.add_argument('-b', '--backward', dest='backward_order',
						action='store_const',
						const=True, default=False,
						help = 'switch order of plots')
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
							args.min[0],
							args.max[0],
							args.opacity,
							args.remove_bad,
							args.flip_axis,
							args.backward_order)
		#	except:
		#		print('-------------------------------------\n')
		#		print('There was a problem with: ' + str(directory) +'\n')
		#		print('Have you run FLIMseg.py yet?')
		#		print('-------------------------------------\n')
