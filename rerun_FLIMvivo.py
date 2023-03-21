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

def process_dir (directory):
	print('Processing: ', directory)
	output_dir = directory
	proc = subprocess.Popen(['python3', str(rootpath / 'FLIMvivo.py'),
							'-o', str(output_dir / 'FLIMvivo.output.txt'),
							'-f',
							'-b',
							str(output_dir / 'segments' / 'segment_0.csv')],
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
					'-a', str(autolifetime),
					'-r', str(fit_mu), str(fit_sigma),
					str(output_dir / 'segments')])
	print('\n-------------------------------------\n')

################################################################################
# Main function uses argparse to take directories and process files.
################################################################################

if __name__ == '__main__':
	parser = argparse.ArgumentParser(description='')
	parser.add_argument('data', type=str, nargs='*', default=['./'],
						help='data file directory path')
	args = parser.parse_args()
	datapaths = list(map(Path,args.data))
	dirs_to_process = np.array([])
	for datapath in datapaths:
		if datapath.exists():
			if datapath.is_dir():
				for datafilepath in datapath.rglob('FLIMseg-output*'):
					if datafilepath not in dirs_to_process:
						dirs_to_process = np.append(dirs_to_process,
												datafilepath)
			else:
				print('Path {0:s} is not a directory.'.format(str(datapath)))
		else:
			print('Path {0:s} does not seem to exist.'.format(str(datapath)))
		for directory in dirs_to_process:
		#	try:
			process_dir(directory)
		#	except:
		#		print('-------------------------------------\n')
		#		print('There was a problem with: ' + str(directory) +'\n')
		#		print('-------------------------------------\n')
