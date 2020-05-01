#!/usr/bin/env python3
from readPTU_FLIM import PTUreader
import numpy as np
from matplotlib import pyplot as plt
from scipy.spatial import ConvexHull
from PIL import Image
from pathlib import Path
import argparse
import os
import subprocess
import re

'''
################
################
## FLIMseg.py ##
################
################
'''

rootpath = Path(os.path.realpath(__file__)).parent.resolve()

################################################################################
# 
################################################################################

def process_dir (directory, time_bin_factor = 1, channel = 0):
	print('Processing: ', directory)
	output_dir = directory/'FLIMseg-output'
	Path.mkdir(output_dir, exist_ok = True)
	Path.mkdir(output_dir / 'segments', exist_ok = True)
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
		shape_from_tif(tif_file, output_dir = output_dir)
	print('-------------------------------------\n')
	if len(tif_files) == 0:
		print('No segmentation files in:\n', directory.name, '\nAborting')
		print('-------------------------------------\n')
		return
	if len(ptu_files) == 0:
		print('Somehow there are no PTU files in:\n', directory.name,
			'\nThis should never happen. There is likely a bug in the code.')
		print('-------------------------------------\n')
		return
	# indices: time_point, segment, decay_time_bin
	data_array = np.empty(shape = [0,0,0])
	time_resolution = 0. # 1.6e-11 s
	number_segments = 0
	number_time_bins = 0
	lower_bound = 0
	upper_bound = 0
	for ptu_file in ptu_files:
		try:
			print('Attempting to read: \n', ptu_file[1].name)
			ptu_stream = PTUreader(ptu_file[1], print_header_data = False)
			flim_data_stack, intensity_image = ptu_stream.get_flim_data_stack()
		except:
			print('There was a problem reading. \nDisregarding file.\n')
			print('-------------------------------------\n')
			continue
		if lower_bound == 0:
			all_data = np.sum(np.sum(flim_data_stack,
								axis=0), axis=0)[channel,:]
			lower_bound = 1 # first point is usually weird
			mask = all_data[np.argmax(all_data):] < (np.amin(all_data[10:]) + \
							(np.amax(all_data)-np.amin(all_data[10:])) * 0.03)
			upper_bound = np.amin(np.where(mask)) # first point below threshold
		flim_data_stack = flim_data_stack[:,:,channel,lower_bound:upper_bound]
		seg_mask = np.array(Image.open(
				tif_files[np.abs(tif_files[:,0] - ptu_file[0]).argmin(), 1]))
		if data_array.shape[2] == 0:
			number_time_bins = int(np.ceil(flim_data_stack.shape[-1] / \
											 time_bin_factor))
			data_array.resize(0,0, number_time_bins)
			time_resolution = ptu_stream.head['MeasDesc_Resolution'] * 1.e12 * \
																time_bin_factor
		binning_factor_x = int(flim_data_stack.shape[0]/seg_mask.shape[0])
		binning_factor_y = int(flim_data_stack.shape[1]/seg_mask.shape[1])
		for roll_number_x in range(1,binning_factor_x):
			for roll_number_y in range(1,binning_factor_y):
				flim_data_stack = flim_data_stack + \
						np.roll(flim_data_stack, -roll_number_x, axis=0) + \
						np.roll(flim_data_stack, -roll_number_y, axis=1) + \
						np.roll(np.roll(flim_data_stack,
											-roll_number_x, axis=0),
											-roll_number_y, axis=1)
		#TODO: It might be better to use np.pad and slicing rather than roll?
		flim_data_stack = flim_data_stack[::binning_factor_x,
										  ::binning_factor_y,:]
		binned_data_stack = flim_data_stack[:,:,::time_bin_factor]
		for start_id in range(1,time_bin_factor):
			binned_data_stack += np.pad(flim_data_stack[:,:,
											start_id::time_bin_factor],
										((0,0),(0,0),(0,
					len(flim_data_stack[0,0,::time_bin_factor]) - \
					len(flim_data_stack[0,0,start_id::time_bin_factor]))),
										mode = 'constant')
		if data_array.shape[1] == 0:
			number_segments = len(np.unique(seg_mask))
			data_array.resize(0, number_segments, number_time_bins)
		current_file_data = np.zeros([number_segments, number_time_bins])
		current_file_data[0,:] = np.sum(binned_data_stack, axis = (0,1))
		data_array = np.append(data_array, [current_file_data], axis = 0)
		for segment in range(1, number_segments):
			current_file_data[segment,:] = np.sum(
					(seg_mask == segment)[:,:,np.newaxis] * \
											binned_data_stack, axis = (0,1))
		data_array = np.append(data_array, [current_file_data], axis = 0)
		print('Successfully processed.\n')
		print('-------------------------------------\n')
	# This was to align peaks across time points. Not sure it is a good idea.
	#peak_time = np.argmax(np.sum(data_array, axis = 1), axis = 1)
	#time_shifts = peak_time - np.min(peak_time)
	#for time_point in range(len(data_array[:,0,0])):
	#	data_array[time_point,:,:] = np.roll(data_array[time_point,:,:],
	#										 -time_shifts[time_point],
	#										 axis = -1)
	
	time_points = np.linspace(0,data_array.shape[2]*time_resolution,
													data_array.shape[2])
	peak_index = 0
	for segment in range(data_array.shape[1]):
		data = np.vstack([time_points,
				np.sum(data_array[:,segment,:],axis=0)]).T
		data[:,1] = data[:,1] / np.max(data[:,1])
		if segment == 0:
			peak_index = np.argmax(data[:,1])
		np.savetxt(output_dir / 'segments' / \
						'segment_{0:d}.csv'.format(segment),
											data,
											delimiter = ',')
	proc = subprocess.Popen(['python', str(rootpath / 'FLIMvivo.py'),
							'-o', str(output_dir / 'FLIMvivo.output.txt'),
							'-c',
							'-b',
							str(output_dir / 'segments' / 'segment_0.csv')],
						stdout=subprocess.PIPE)
	procout = str(proc.stdout.read().decode('utf-8'))
	print('Biexponetial convolution fit parameters :')
	print(procout)
	params = np.fromstring(procout, dtype = np.float, sep = '\t')
	autolifetime = params[-1]
	# Want to check if the multiexponential went well
	#  or we should have used a single one.
	# We check if at a time point 1/2 the way from the peak to the end,
	#  the autoflourescence has damped away so that the biexponential is
	#  within a couple percent of a monoexponetial that matches at the end.
	# If not, it probably wasn't autoflourescence, but just fitting two badly
	#  correlated exponents within the signal distribution.
	test_location = 0.5   # half way
	threshold     = 0.02  # 2%
	test_point = int((len(time_points)-peak_index)*test_location)
	a = (params[0]*np.exp(-time_points[-1]/1000./params[1]) + \
		 params[2]*np.exp(-time_points[-1]/1000./params[3])) / \
			np.exp(-time_points[-1]/1000./params[1])
	deviation = (params[0]*np.exp(-time_points[test_point]/1000./params[1]) + \
			params[2]*np.exp(-time_points[test_point]/1000./params[3]) - \
				a*np.exp(-time_points[test_point]/1000./params[1])) / \
					a*np.exp(-time_points[test_point]/1000./params[1])
	print('At test point fit deviation: {0:2.2f}%\n'.format(deviation*100.))
	print('-------------------------------------\n')
	if deviation > threshold:
		print('Biexponetial fit questionable. Reverting to monoexponential.\n')
		print('-------------------------------------\n')
		proc = subprocess.Popen(['python', str(rootpath / 'FLIMvivo.py'),
							'-o', str(output_dir / 'FLIMvivo.output.txt'),
							'-c',
							str(output_dir / 'segments' / 'segment_0.csv')],
						stdout=subprocess.PIPE)
		procout = str(proc.stdout.read().decode('utf-8'))
		params = np.fromstring(procout, dtype = np.float, sep = '\t')
		autolifetime = params[-1]
		subprocess.call(['python', str(rootpath / 'FLIMvivo.py'),
						'-o', str(output_dir / 'FLIMvivo.output.txt'),
						'-a', str(autolifetime),
						str(output_dir / 'segments')])
	else:
		subprocess.call(['python', str(rootpath / 'FLIMvivo.py'),
						'-o', str(output_dir / 'FLIMvivo.output.txt'),
						'-b',
						'-a', str(autolifetime),
						str(output_dir / 'segments')])
	print('\n-------------------------------------\n')

################################################################################
# Do a simple shape analysis on a segmentation 'tif' file
################################################################################

def shape_from_tif (tif_file, output_dir = Path.cwd()):
	im = Image.open(tif_file)
	imarray = np.array(im)
	output_data = np.array([['#channel','long/short',
							 'perimeter/sqrt(area)','area']])
	for channel in np.unique(imarray):
		if channel == 0:
			continue
		shape = np.where(imarray == channel)
		centroid = np.mean(shape,axis=1)
		x = shape[0]-centroid[0]
		y = shape[1]-centroid[1]
		points = np.vstack([x,y]).T
		hull = ConvexHull(points)
		boundary_x = points[hull.vertices,0]
		boundary_y = points[hull.vertices,1]
		boundary_points = np.vstack([boundary_x, boundary_y]).T
		plt.plot(np.append(boundary_x, boundary_x[0]) + centroid[0],
				 np.append(boundary_y, boundary_y[0]) + centroid[1],
				 'b--', lw=2)
		plt.plot(boundary_x + centroid[0],
				 boundary_y + centroid[1], 'bo')
		plt.plot(boundary_x + centroid[0],
				 boundary_y + centroid[1], 'w.')
		plt.fill(np.append(boundary_x, boundary_x[0]) + centroid[0],
				 np.append(boundary_y, boundary_y[0]) + centroid[1],
				 'lightgray')
		average_length = np.mean(np.linalg.norm(boundary_points, axis = 1))
		U, s, V = np.linalg.svd(boundary_points / average_length)
		s_scaled = s / np.mean(s)
		plt.plot([-V[0,0]*s_scaled[0]*average_length + centroid[0],
					V[0,0]*s_scaled[0]*average_length + centroid[0]],
				 [-V[0,1]*s_scaled[0]*average_length + centroid[1],
					V[0,1]*s_scaled[0]*average_length + centroid[1]],'ro-')
		plt.plot([-V[0,0]*s_scaled[0]*average_length + centroid[0],
					V[0,0]*s_scaled[0]*average_length + centroid[0]],
				 [-V[0,1]*s_scaled[0]*average_length + centroid[1],
					V[0,1]*s_scaled[0]*average_length + centroid[1]],'w.')
		plt.plot([-V[1,0]*s_scaled[1]*average_length + centroid[0],
					V[1,0]*s_scaled[1]*average_length + centroid[0]],
				 [-V[1,1]*s_scaled[1]*average_length + centroid[1],
					V[1,1]*s_scaled[1]*average_length + centroid[1]],'go-')
		plt.plot([-V[1,0]*s_scaled[1]*average_length + centroid[0],
					V[1,0]*s_scaled[1]*average_length + centroid[0]],
				 [-V[1,1]*s_scaled[1]*average_length + centroid[1],
					V[1,1]*s_scaled[1]*average_length + centroid[1]],'w.')
		plt.text(centroid[0], centroid[1], '{0:d}'.format(channel),
					color = 'k', fontsize = 'large', fontweight = 'bold')
		axes = plt.gca()
		figure = plt.gcf()
		figure.set_size_inches(16,16)
		axes.set_xlim([0,imarray.shape[0]])
		axes.set_ylim([0,imarray.shape[1]])
		axes.set_aspect('equal')
		output_data = np.append(output_data,
					np.array([['{0:d}'.format(channel),
						'{0:.6f}'.format(s_scaled[0] / s_scaled[1]),
						'{0:.6f}'.format(hull.area / np.sqrt(hull.volume)),
						'{0:.6f}'.format(hull.area)]]),
					axis = 0)
	np.savetxt(output_dir / (tif_file.with_suffix('.shape.output.csv').name),
					output_data, delimiter = ',', fmt='%s')
	plt.savefig(output_dir / (tif_file.with_suffix('.shape.output.png').name))
	plt.clf()

################################################################################
# Main function uses argparse to take directories and process files.
################################################################################

if __name__ == '__main__':
	parser = argparse.ArgumentParser(description='')
	parser.add_argument('-b', '--bin', dest='bin_factor',
						nargs = 1,
						default = [1],
						type = int,
						required = False,
						help = 'how much should we bin time (default 1)')
	parser.add_argument('-c', '--chan', dest='channel',
						nargs = 1,
						default = [0],
						type = int,
						required = False,
						help = 'use a different channel (default 0)')
	parser.add_argument('data', type=str, nargs='*', default='./',
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
				process_dir(directory, args.bin_factor[0], args.channel[0])
			except:
				print('-------------------------------------\n')
				print('There was a problem with: ' + str(directory) +'\n')
				print('-------------------------------------\n')
