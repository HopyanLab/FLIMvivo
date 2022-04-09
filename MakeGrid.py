#!/usr/bin/env python3
import numpy as np
from PIL import Image
import argparse
from matplotlib import pyplot as plt

def generate_grid (n = 4):
	N = 2**n
	M = int(np.floor(256/N))
	grid = np.zeros((256,256), dtype=np.uint16)
	counter = 1
	for i in range(M):
		for j in range(M):
			for a in range(N):
				for b in range(N):
					grid[i*N+a,j*N+b] = counter
			counter += 1
	print(grid)
#	plt.imshow(grid)
#	plt.show()
	im = Image.fromarray(grid)
	im.save('./grid.tif')

if __name__ == "__main__":
	parser = argparse.ArgumentParser(description='Generate grid tiff.')
	parser.add_argument('n', type=int, nargs='?', default=4,
						help='level of courseness (min 1 / default 4 / max 8)')
	#
	parse_args = parser.parse_args()
	if parse_args.n < 1 or parse_args.n > 9:
		print('illegal user choice.')
	else:
		generate_grid(parse_args.n)
