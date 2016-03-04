import re
fh = open("output.txt")

idx = 0
sums = []
for line in fh:
	line = line.strip()
	if re.search("reward =", line):
		sums.append( float( line[line.index("reward")+len("reward ="):].strip()) )
print sum(sums)


import matplotlib.pyplot as plt
import numpy as np
fh = open("output.txt")	
rewards = []
for line in fh:
	line = line.strip()
	if re.search("Averate overall reward:", line):
		rewards.append( float( line[line.index(": ")+2:].strip()) )


mv_avg = []
n_points = 25
for i in range(len(rewards)):
	if i> n_points:
		mv_avg.append( sum(rewards[i-n_points:i])/n_points )
mv_avg = [None for i in range(n_points)] + mv_avg
plt.plot( rewards )
plt.plot(mv_avg)
plt.show()