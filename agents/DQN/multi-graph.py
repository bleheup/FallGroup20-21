from matplotlib import pyplot as plt
import sys
import getopt
from datetime import datetime
import os
import numpy as np
import scipy as sp
from scipy.signal import argrelextrema

time = datetime.now().strftime('%Y%m%d_%H%M%S')
parent_dir = "./"
save_dir = "results-{}".format(time)
path = os.path.join(parent_dir, save_dir)
os.mkdir(path)
validate=True
title = "Rainbow DQN testing winrates agaisnt random actions"
labels = ["Base DQN", "Double DQN", "Dueling DQN", "PER DQN"]
validate = sys.argv[len(sys.argv) - 1]
if validate == "-v":
    validate = True
else:
    title = "Rainbow DQN training winrates against random actions"
    validate = False

fig = plt.figure()
ax = fig.add_subplot(111)
for i in range(1, len(sys.argv) - 1):
    with open(sys.argv[i]) as f:
        if not validate: 
            list = np.array([float(line.split(' ')[3][:-1]) for line in f])
        else:
            list = np.array([float(line.split(' ')[3].split('\n')[0][:-1]) for line in f])
        
        if not validate:
            for k in range(len(list[:100])):
                if list[k] == 100.0:
                    list[k] = 0.0

        if not validate:
            max = np.argmax(list)
        else:
            max = len(list) - 1

        plt.plot(list, label = labels[i-1])

        ax.text(max, list[max], "%.2f" %list[max], ha="center")
        
plt.xlabel("Episodes")
plt.ylabel("Winrates")
plt.title(title)
plt.ylim(ymin=0)
plt.legend()
plt.savefig("{}/winrate.png".format(path))
plt.clf()