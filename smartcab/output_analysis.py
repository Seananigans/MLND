import matplotlib.pyplot as plt
import numpy as np
import re


fh = open("output.txt")
idx = 0
total_rewards = 0
rewards = []
completions = []
for line in fh:
	line = line.strip()
	if re.search("reward =", line):
		total_rewards += float( line[line.index("reward")+len("reward ="):].strip() )
	if re.search("Cummulative reward for trial", line):
		rewards.append( float( line[line.index(": ")+2:].strip()) )
	if re.search("Primary agent has reached destination!", line):
		completions.append( 1 )
	elif re.search("Primary agent could not reach destination within deadline!", line):
		completions.append( 0 )
print total_rewards
fh.close()

mv_avg = []
n_points = 20
for i in range(len(rewards)):
	if i> n_points:
		mv_avg.append( sum(rewards[i-n_points:i])/n_points )
mv_avg = [None for i in range(n_points)] + mv_avg

plt.plot( rewards )
plt.plot( mv_avg )
plt.show()

# Report how well the learning agent does as it approaches the end of its trials.
print "The agent completed {0:.2f}% of its runs".format(
float(sum(completions))/len(completions)*100
)

def percent_last_number_of_trials_completed(count):
	print "Of the last {0:d} trials, the agent completed {1:.2f}%".format(
	count,
	float(sum(completions[-count:]))/len(completions[-count:])*100
	)

percent_last_number_of_trials_completed(50)
percent_last_number_of_trials_completed(25)
percent_last_number_of_trials_completed(10)
