import re
fh = open("output.txt")

idx = 0
sums = []
for line in fh:
	line = line.strip()
	if re.search("reward", line):
		sums.append( float( line[line.index("reward")+len("reward ="):].strip()) )
print sum(sums)
	
