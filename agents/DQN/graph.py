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

with open(sys.argv[1]) as f:
    title = sys.argv[2]
    validate = sys.argv[3]
    if validate == "-v":
        validate = True
    else:
        validate = False
    if not validate: 
        list = np.array([float(line.split(' ')[3][:-1]) for line in f])
    else:
        list = np.array([float(line.split(' ')[3].split('\n')[0][:-1]) for line in f])

if not validate:
    for i in range(len(list[:100])):
        if list[i] == 100.0:
            list[i] = 0.0
    
fig = plt.figure()
ax = fig.add_subplot(111)

if not validate:
    max = np.argmax(list)
else:
    max = len(list) - 1
plt.plot(list)

ax.text(max, list[max], "%.2f" %list[max], ha="center")
plt.xlabel("Episodes")
plt.ylabel("Winrates")
plt.title(title)
plt.ylim(ymin=0)
plt.savefig("{}/winrate.png".format(path))
plt.clf()

# if len(sys.argv) > 2:
#     with open(sys.argv[2]) as f:
#         list = [float(line.rstrip()) for line in f]
#     plt.plot(list)
#     plt.xlabel("Steps")
#     plt.ylabel("Losses")
#     plt.savefig("{}/loss.png".format(path))
