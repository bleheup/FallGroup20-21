from matplotlib import pyplot as plt
import sys
import getopt
from datetime import datetime
import os

time = datetime.now().strftime('%Y%m%d_%H%M%S')
parent_dir = "/mnt/d/everglades-ai-wargame/agents/bd3qn"
save_dir = "results-{}".format(time)
path = os.path.join(parent_dir, save_dir)
os.mkdir(path)


with open(sys.argv[1]) as f:
    list = [float(line.rstrip()) for line in f]
plt.plot(list)
plt.xlabel("Episodes")
plt.ylabel("Winrates")
plt.savefig("{}/winrate.png".format(path))
plt.clf()

if len(sys.argv) > 2:
    with open(sys.argv[2]) as f:
        list = [float(line.rstrip()) for line in f]
    plt.plot(list)
    plt.xlabel("Steps")
    plt.ylabel("Losses")
    plt.savefig("{}/loss.png".format(path))
