#!/usr/bin/env python3
import numpy as np
from matplotlib import pyplot as plt
from pathlib import Path
import argparse

if __name__ == "__main__":
	parser = argparse.ArgumentParser(description='Plot CSV file.')
	parser.add_argument('data', type=str,
						help='Path to csv data file.')
	parse_args = parser.parse_args()
	data = np.loadtxt(Path(parse_args.data), delimiter = ',')
	plt.plot(data[:,0],data[:,1],'.')
	plt.show()
