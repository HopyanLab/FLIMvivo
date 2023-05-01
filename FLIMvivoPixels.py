#!/usr/bin/env python3
from readPTU_FLIM import PTUreader
import numpy as np
from matplotlib import pyplot as plt
from scipy.spatial import ConvexHull
from scipy.ndimage import binary_dilation
from PIL import Image
from pathlib import Path
import argparse
import os
import subprocess
import re

'''
#######################
#######################
## FLIMvivoPixels.py ##
#######################
#######################
'''

rootpath = Path(os.path.realpath(__file__)).parent.resolve()

################################################################################
# 
################################################################################

def process_file (file_path, channel = 0, flip_axis = False,
					colormap_name = 'afmhot'):
	print('Processing: ', file_path.name)
	print('-------------------------------------\n')
	output_dir = file_path.parent/'FLIMvivo-output'
	Path.mkdir(output_dir, exist_ok = True)
	Path.mkdir(output_dir /'pixels', exist_ok = True)
	try:
		print('Attempting to read: \n', file_path.name)
		ptu_stream = PTUreader(file_path, print_header_data = False)
		flim_data_stack = ptu_stream.get_flim_data_stack()
		space_resolution = ptu_stream.head['ImgHdr_PixResol']
		time_resolution = ptu_stream.head['MeasDesc_Resolution'] * 1.e12
		flim_data_stack = flim_data_stack[:,:,channel,:].astype(int)
		if flip_axis:
			flim_data_stack = flim_data_stack[::-1,:,:]
	#	intensity_image = np.sum(flim_data_stack[:,:,:], axis=2)
		print('Successfully read.')
	except:
		print('There was a problem reading file. \nAborting.\n')
		print('-------------------------------------\n')
		return
	full_field = np.sum(flim_data_stack, axis=(0,1))
	print('Successfully processed.\n')
	print('-------------------------------------\n')
	time_points = np.arange(0,flim_data_stack.shape[2]).astype(int)
	peak_index = np.argmax(full_field)
	time_points = time_points - time_points[peak_index]
	np.savetxt(output_dir / 'pixels' / 'full_field.csv',
							np.vstack([time_points, full_field]).T,
								fmt = '%d',
								delimiter = ',',
								header = '*{0:1.12f}'.format(time_resolution))
	for index_y in range(int(flim_data_stack.shape[0]/32)):
		for index_x in range(int(flim_data_stack.shape[1]/32)):
			np.savetxt(output_dir / 'pixels' / \
						'pixel_x{0:d}_y{1:d}.csv'.format(index_x, index_y),
								np.vstack([
									time_points,
									flim_data_stack[index_y,index_x]]).T,
								fmt = '%d',
								delimiter = ',',
								header = '*{0:1.12f}'.format(time_resolution))
	proc = subprocess.Popen(['python3', str(rootpath / 'FLIMvivo.py'),
						'-o', str(output_dir / 'FLIMvivo.output.txt'),
						'-f',
						'-b',
						str(output_dir / 'pixels' / 'full_field.csv')],
					stdout=subprocess.PIPE)
	procout = str(proc.stdout.read().decode('utf-8'))
	print('Biexponetial convolution fit parameters :')
	print(procout)
	params = np.fromstring(procout, dtype = float, sep = '\t')
	fit_mu = params[-2]
	fit_sigma = params[-1]
	autolifetime = params[-3]
	subprocess.call(['python3', str(rootpath / 'FLIMvivo.py'),
					'-o', str(output_dir / 'FLIMvivo.output.txt'),
					'-f',
					'-b',
					'-n',
					'-r', str(fit_mu), str(fit_sigma),
					str(output_dir / 'pixels')])
	print('\n-------------------------------------\n')

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
	parser.add_argument('-f', '--flip', dest='flip_axis',
						action='store_const',
						const=True, default=False,
						help = 'flip Y-axis (old FLIMfit standard)')
	parser.add_argument('data', type=str,
						help='data file path')
	args = parser.parse_args()
	datapath = Path(args.data)
#	try:
	process_file(datapath, args.channel[0], args.flip_axis)
#	except:
#		print('-------------------------------------\n')
#		print('There was a problem with: ' + str(directory) +'\n')
#		print('-------------------------------------\n')
