#!/usr/bin/env python3
from readPTU_FLIM import PTUreader
import numpy as np
from matplotlib import pyplot as plt
from scipy import ndimage
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
				 opacity = 0.3, flip_axis = False):
	print('Processing: ', directory)
	output_dir = directory/'FLIMseg-output'
	summary_file = output_dir/'FLIMvivo.output.summary.txt'
	results = np.genfromtxt(summary_file, delimiter = '\t', dtype = float)
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
	seg_mask = np.array(Image.open(
			tif_files[np.abs(tif_files[:,0] - ptu_files[0,0]).argmin(), 1]))
	fig, ax = plt.subplots(1,1)
	scaled_intensity = intensity_image[
						::int(len(intensity_image)/len(seg_mask)),
						::int(len(intensity_image[0])/len(seg_mask[0]))]
	seg_image = (seg_mask > 0)*1.
	seg_image[seg_mask == 0] = np.amin(results[:,1])
	for segment in np.unique(seg_mask[seg_mask > 0]):
		segment_points = np.where(seg_mask == segment)
		seg_image[segment_points] = results[results[:,0] == segment,1]
	seg_alpha = (seg_mask > 0)*opacity
	if smoothing > 0:
		seg_image = ndimage.gaussian_filter(seg_image,
						sigma=(smoothing, smoothing), order=0)
	heatmap = ax.imshow(seg_image,
				cmap = plt.get_cmap('jet'),
				origin = 'lower')
	ax.imshow(np.sqrt(scaled_intensity),
				cmap = plt.get_cmap('Greys'),
				origin = 'lower')
	ax.imshow(seg_image, alpha = seg_alpha,
				cmap = plt.get_cmap('jet'),
				origin = 'lower')
	plt.colorbar(heatmap)
	plt.savefig(output_dir / (ptu_files[0,1].with_suffix(
								'.heatmap.png').name))
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
						default = [0.3],
						type = float,
						required = False,
						help = 'opacity for heatmap overlay (default 0.3)')
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
			try:
				process_dir(directory, args.channel[0], 
							args.smoothing[0], args.opacity,
							args.flip_axis)
			except:
				print('-------------------------------------\n')
				print('There was a problem with: ' + str(directory) +'\n')
				print('Have you run FLIMseg.py yet?')
				print('-------------------------------------\n')
