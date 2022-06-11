#!/usr/bin/env python3
import numpy as np
from PIL import Image
import argparse
from matplotlib import pyplot as plt
from pathlib import Path

def generate_grid (n = 4, maskfile = None):
	N = 2**n
	M = int(np.floor(512/N))
	grid = np.zeros((512,512), dtype=np.uint16)
	counter = 1
	if maskfile is not None:
		mask = np.array(Image.open(maskfile))
	else:
		mask = None
	for i in range(M):
		for j in range(M):
			if mask is not None:
				if mask[int(np.floor((i+1/2)*N)),
						int(np.floor((j+1/2)*N)),0]==0:
					continue
			for a in range(N):
				for b in range(N):
					grid[i*N+a,j*N+b] = counter
			counter += 1
#	print(grid)
#	plt.imshow(grid)
#	plt.show()
	im = Image.fromarray(grid)
	im.save(Path.cwd() / './grid.tif')

if __name__ == "__main__":
	parser = argparse.ArgumentParser(description='Generate grid tiff.')
	parser.add_argument('-m', '--mask', dest = 'mask',
						default = [Path.cwd() / 'mask.tif'],
						type = str,
						required = False,
						help = 'file to mask off grid boxes')
	parser.add_argument('n', type=int, nargs='?', default=5,
						help='level of courseness (min 1 / default 4 / max 8)')
	#
	parse_args = parser.parse_args()
	maskfile = parse_args.mask[0]
	if parse_args.n < 1 or parse_args.n > 9:
		print('illegal user choice.')
	elif maskfile.exists():
		generate_grid(parse_args.n, maskfile)
	else:
		generate_grid(parse_args.n)
