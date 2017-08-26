import pandas as pd
import numpy as np
import argparse
from lmfit import Model
import matplotlib.pyplot as plt

parser=argparse.ArgumentParser()
#parser.parse_args()
parser.add_argument("file",help="input file")
parser.add_argument("-f", help="inputfile")
args=parser.parse_args()
print(args.file)


def model_circ(ZT,amp,phase,mesor):
	return amp*np.cos(ZT/24.0*(2.0*np.pi)+phase)+mesor
def get_data():
	data=pd.read_csv(args.file,sep='\t')
	gmodel=Model(model_circ)
	#params = gmodel.make_params(amp=1, phase=0, mesor=0)
	result = gmodel.fit(data['Control'], ZT=data['ZT'], amp=1, phase=0, mesor=0)
	#gmodel.eval(params, ZT=data['ZT'])
	print(gmodel.param_names)
	print(gmodel.independent_vars)
	print(result.fit_report())
	#axes = plt.gca()
	plt.plot(data['ZT'], data['Control'],'bo')
	plt.plot(data['ZT'], result.init_fit, 'k--')
	plt.plot(data['ZT'], result.best_fit, 'r-')
	plt.ylim([-0.03,0.03])
	plt.show()
	print(result.params['amp'].value)
	
	from scipy.optimize import curve_fit
	y =data['ZT']
	x=data['Control']
	init_vals=[1,0,0]
	bst_vals,covar=curve_fit(model_circ,y,x,p0=init_vals)
	print(bst_vals)
	print(covar)
	
	
def main():
	get_data()
	
if __name__=='__main__':
	main()