import random
from environment import Agent, Environment
from planner import RoutePlanner
from simulator import Simulator

class LearningAgent(Agent):
    """An agent that learns to drive in the smartcab world."""

    def __init__(self, env):
        super(LearningAgent, self).__init__(env)  # sets self.env = env, state = None, next_waypoint = None, and a default color
        self.color = 'red'  # override color
        self.planner = RoutePlanner(self.env, self)  # simple route planner to get next_waypoint
        # TODO: Initialize any additional variables here
        self.gamma = 0.95
        self.alpha = 0.3
        self.q_table = {}
        self.cummulative_reward = 0.0
        self.ddlines = []
        self.neg_reward_total=0
        self.abs_changes=[]

    def reset(self, destination=None):
    	"""Reset variables used to record information about each trial.
    	
    	No parameters.
    	
    	No return value."""
        self.planner.route_to(destination)
        # TODO: Prepare for a new trip; reset any variables here, if required
        print "Cummulative reward for trial: {}".format(self.cummulative_reward)
        print "Negative rewards for trial: {}".format(self.neg_reward_total)
        self.cummulative_reward = 0
        self.ddlines = []
        self.neg_reward_total=0
    
    def update_state(self):
    	"""Gather inputs to use as the state, then return the state, and 
    	inputs and deadline for further use.
    	
    	No parameters.
    	
    	return values:
    	state - variables that describe current condition of learning agent.
    	inputs - information gathered from observing current condition.
    	deadline - The amount of time left until trial is over.
    	"""
        # Gather inputs
        self.next_waypoint = self.planner.next_waypoint()  # from route planner, also displayed by simulator
        inputs = self.env.sense(self)
        deadline = self.env.get_deadline(self)
        self.ddlines.append(deadline)
        time_left = self.discretize_deadline(deadline)
        
        state = tuple({
        'next_waypoint':self.next_waypoint,
        'light': inputs['light'],
        'oncoming': inputs['oncoming'],
        'left': inputs['left']
        }.items())
        return state, inputs, deadline
        
    def discretize_deadline(self, deadline):
    	"""Decide on random action rate based on moves left until deadline
    	
    	parameters: 
    	deadline - The amount of time left until trial is over.
    	
    	return value:
    	random action rate - probability of taking a random action"""
        time_left = float(deadline)/max(self.ddlines)
        if time_left>0.5:
            return 0.40
        else:
            return 0.05
        
    def choose_action(self, state):
    	"""Choose an action that maximizes potential reward"""
        actions = Environment.valid_actions
        max_q = None
        for act in actions:
            self.q_table[(tuple(state), act)] = self.q_table.get((tuple(state), act), 0.0)
            q_value = self.q_table[(tuple(state), act)]
            if q_value > max_q:
                max_q = q_value
                max_action = act
        
        deadline = self.env.get_deadline(self)
        time_left = self.discretize_deadline(deadline)
        if random.random()<time_left:
            max_action = random.choice(actions[1:])
        return max_action, max_q
        
    def update_q_table(self, table, action, reward, max_q_prime):
    	"""Update the q table based on information gathered from (state, action) pair.
    	
    	parameters:
    	table - a q-table to update.
    	action - the action taken for the current state to get to the new state.
    	reward - the reward for moving into the new state.
    	max_q_prime - the maximum value observed for actions that 
    	              can be taken in the new state.
    	              
    	No return value."""
        value = reward + self.gamma * max_q_prime
        value -= table[(self.state, action)]
        temp = table[(self.state, action)]
        table[(self.state, action)] += self.alpha * value
        self.abs_changes.append(abs(temp - table[(self.state, action)]))
        
    def update(self, t):
    	"""Update the current state of the learning agent.
    	Update the q-table based on the action chosen.
    	
    	No Parameters.
    	
    	No Return value."""
        # TODO: Update state
        self.state = self.update_state()[0]

        # TODO: Select action according to your policy
        actions = Environment.valid_actions
        action, max_q = self.choose_action(self.state)

        # Execute action and get reward
        reward = self.env.act(self, action)
        #Update the number of negative rewards if reward<0
        if reward<0:
        	self.neg_reward_total += 1
        state_prime, inputs, deadline = self.update_state()
        max_q_prime = self.choose_action(self.state)[1]
        
        # TODO: Learn policy based on state, action, reward
        self.update_q_table(self.q_table,action, reward, max_q_prime)
        
        self.cummulative_reward += reward
        self.state = state_prime
        
        print "LearningAgent.update(): deadline = {}, inputs = {}, action = {}, reward = {}".format(deadline, inputs, action, reward)  # [debug]
        

def run():
    """Run the agent for a finite number of trials."""

    # Set up environment and agent
    e = Environment()  # create environment (also adds some dummy traffic)
    a = e.create_agent(LearningAgent)  # create agent
    e.set_primary_agent(a, enforce_deadline=True)  # set agent to track

    # Now simulate it
    sim = Simulator(e, update_delay=0.0001)  # reduce update_delay to speed up simulation
    sim.run(n_trials=100)  # press Esc or close pygame window to quit
    print a.abs_changes


if __name__ == '__main__':
    run()
