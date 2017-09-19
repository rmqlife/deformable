import os, cv2
import matplotlib.pyplot as plt
import numpy as np

def gabor_feat(gray, num_theta = 4, grid = 80):
    ks = 101 # kernel size
    avg = np.zeros(gray.shape)
    count = 0
    rows, cols = gray.shape
    hist = []
    for i in range(num_theta):
        g_kernel1 = cv2.getGaborKernel((ks, ks), sigma = 6.0, theta = i*np.pi/num_theta, lambd = 10.0, gamma=0.5, psi=0, ktype=cv2.CV_32F)
        filtered = cv2.filter2D(gray, cv2.CV_8UC3, g_kernel1)
        count = count+1
        avg = avg + filtered
        count = 0
        for col in np.arange(0, cols, grid):
            for row in np.arange(0, rows, grid):
                block = filtered[row:row+grid,col:col+grid]
                val =  float(np.sum(block))
                hist.append(val)
    avg = avg/float(num_theta)
    avg = np.uint8(avg)
    return avg, hist


if __name__ == '__main__':
	from util import *
	import sys,os
	home = sys.argv[1]
	rgblist = get_filelist(os.path.join(home,'rgb'))
	print home, rgblist
	# build up the dataset
	feat = np.array([])
	for i in range(len(rgblist)):
		gabor, hist = gabor_feat(cv2.imread(rgblist[i]), num_theta = 8, show_step = 0)
		#gabor = cv2.resize(gabor,(50,50))
	#     plt.imshow(gabor)
	#     plt.show()
		hist = np.array(hist)
		feat = np.vstack((feat,hist)) if feat.size else hist
		print feat.shape

	# find the edges
	data = np.load(os.path.join(home+'data.npz'))
	pos = data['pos']
	np.savez(home+'/data2',feat=feat,pos=pos)

