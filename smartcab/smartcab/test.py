
dct = {
'light': "green",
'oncoming': 'oncoming',
'right': 'right',
'left': 'left',
'next_waypoint':"way",
'deadline':2.0
}

for i in dct.items():
	print i
	
from environment import Agent, Environment
import random
action = random.choice(Environment.valid_actions)
print action

print (tuple(dct.items()), action)