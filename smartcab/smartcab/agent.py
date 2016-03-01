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
        self.state = None
        self.q_table = {}

    def reset(self, destination=None):
        self.planner.route_to(destination)
        # TODO: Prepare for a new trip; reset any variables here, if required

    def update(self, t):
        # Gather inputs
        self.next_waypoint = self.planner.next_waypoint()  # from route planner, also displayed by simulator
        inputs = self.env.sense(self)
        deadline = self.env.get_deadline(self)

        # TODO: Update state
        self.state = {
        'next_waypoint':self.next_waypoint,
        'light': inputs['light'],
        'oncoming': inputs['oncoming'],
        'right': inputs['right'],
        'left':inputs['left'],
        'deadline':deadline,
        }.items()
        
        {
        'location':None,
        'heading':None,
        'distance':None
        }
        
        actions = Environment.valid_actions[1:]
        
        largest = 0
        max_act = random.choice(Environment.valid_actions[1:])
        for act in actions:
        	current_q_val = self.q_table.get((tuple(self.state), act),0)
        	if current_q_val > largest:
        		largest = current_q_val
        		max_act = act
        # TODO: Select action according to your policy
        action = max_act

        # Execute action and get reward
        reward = self.env.act(self, action)

        # TODO: Learn policy based on state, action, reward
        self.q_table[(tuple(self.state), action)] = self.q_table.get((tuple(self.state), action),0)
        print "LearningAgent.update(): deadline = {}, inputs = {}, action = {}, reward = {}".format(deadline, inputs, action, reward)  # [debug]


def run():
    """Run the agent for a finite number of trials."""

    # Set up environment and agent
    e = Environment()  # create environment (also adds some dummy traffic)
    a = e.create_agent(LearningAgent)  # create agent
    e.set_primary_agent(a, enforce_deadline=False)  # set agent to track

    # Now simulate it
    sim = Simulator(e, update_delay=1.0)  # reduce update_delay to speed up simulation
    sim.run(n_trials=10)  # press Esc or close pygame window to quit


if __name__ == '__main__':
    run()
