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
        self.gamma = 0.9
        self.alpha = 0.5
        self.q_table = {}
        self.prev_state = ()
        self.prev_action = None
        self.prev_reward = 0.0

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
        
        actions = Environment.valid_actions[1:]
#         qs = [self.q_table[(tuple(self.state), a)] for a in actions]
#         max_q = max(qs)
        
        # TODO: Select action according to your policy
        max_q = 0.
        max_act = random.choice(Environment.valid_actions[1:])
        for act in actions:
        	self.q_table[(tuple(self.state), act)] = self.q_table.get((tuple(self.state), act), 0.0)
        	if self.q_table[(tuple(self.state), act)] > max_q:
        		max_q = self.q_table[(tuple(self.state), act)]
        		max_act = act
        action = max_act

        # Execute action and get reward
        reward = self.env.act(self, action)

        # TODO: Learn policy based on state, action, reward
        value = self.prev_reward + self.gamma * max_q
        if (self.q_table.has_key( (tuple(self.prev_state), self.prev_action) ) ):
        	update = self.alpha * (value - self.q_table[(tuple(self.prev_state), self.prev_action)])
        	self.q_table[(tuple(self.prev_state), self.prev_action)] += update
        else:
        	update = self.prev_reward
        	self.q_table[(tuple(self.state), action)] += update
        #self.q_table[(tuple(self.state), action)] = reward + self.q_table.get((tuple(self.state), action),0)
        
        self.prev_state = self.state
        self.prev_action = action
        self.prev_reward = reward
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
