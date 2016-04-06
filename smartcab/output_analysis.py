import matplotlib.pyplot as plt
import numpy as np
import re


fh = open("output2.txt")
idx = 0
total_rewards = 0
rewards = []
completions = []
neg_rewards = []
for line in fh:
	line = line.strip()
	if re.search("reward =", line):
		total_rewards += float( line[line.index("reward")+len("reward ="):].strip() )
	if re.search("Cummulative reward for trial", line):
		rewards.append( float( line[line.index(": ")+2:].strip()) )
	if re.search("Negative rewards for trial", line):
		neg_rewards.append( float( line[line.index(": ")+2:].strip()) )
	if re.search("Primary agent has reached destination!", line):
		completions.append( 1 )
	elif re.search("Primary agent could not reach destination within deadline!", line):
		completions.append( 0 )
	if re.search("\[", line):
		changes = line[1:-1].split(", ")
print total_rewards
fh.close()

changes = [float(i) for i in changes]


def plot_metric(item, moving_average=False, window=20,
				title="Total Rewards per Trial",
				xlabel="Trials", 
				ylabel="Total Rewards"):
	mv_avg = []
	mv_std = []
	for i in range(len(item)):
		if i> window:
			in_window = item[i-window:i]
			mean = sum(in_window)/window
			stdev = (sum([(i-mean)**2 for i in in_window])/window)**0.5
			mv_avg.append( mean )
			mv_std.append( stdev )
	upper_band = [x+y for x,y in zip(mv_avg, mv_std)]
	lower_band = [x-y for x,y in zip(mv_avg, mv_std)]
	mv_avg = [mv_avg[0] for i in range(window)] + mv_avg
	upper_band = [upper_band[0] for i in range(window)] + upper_band
	lower_band = [lower_band[0] for i in range(window)] + lower_band

	plt.plot( item, color="k" )
	if moving_average:
		plt.plot( upper_band, color="m")
		plt.plot( lower_band, color="m")
		plt.plot( mv_avg, color="r")
	plt.title(title)
	plt.xlabel(xlabel)
	plt.ylabel(ylabel)
	plt.show()

#Plot changes to Q table values over moves
plot_metric(changes, moving_average=True, window=200, 
			title="Changes to Q-values over time", 
			xlabel="Moves", 
			ylabel="Change to Q-value")
#Plot rewards per trial
plot_metric(rewards, moving_average=True)
#Plot the number of negative rewards per trial
plot_metric(neg_rewards, title="Total Negative Rewards per Trial", ylabel="Total Negative Rewards")


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
