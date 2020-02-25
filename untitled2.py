import matplotlib.pyplot as plt
import matplotlib.animation as animation

from matplotlib import style
import numpy as np
import time
import pickle

for i in range(100):
	time.sleep(0.1)
	with open('example', 'wb') as f:
		pickle.dump(np.random.rand(32,32), f)