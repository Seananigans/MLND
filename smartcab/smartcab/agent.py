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
        self.alpha = 0.7
        self.q_table = {}
        self.ego_q_table = {}
        self.allo_q_table = {}
        self.cummulative_reward = 0.0
        self.ddlines = []

    def reset(self, destination=None):
        self.planner.route_to(destination)
        # TODO: Prepare for a new trip; reset any variables here, if required
        print "Cummulative reward for trial: {}".format(self.cummulative_reward)
        self.cummulative_reward = 0
        self.ddlines = []
    
    def update_state(self):
        # Gather inputs
        self.next_waypoint = self.planner.next_waypoint()  # from route planner, also displayed by simulator
        inputs = self.env.sense(self)
        deadline = self.env.get_deadline(self)
        self.ddlines.append(deadline)
        time_left = self.discretize_deadline(deadline)
        
        state = tuple({
        'next_waypoint':self.next_waypoint,
        'light': inputs['light'],
#         'oncoming': inputs['oncoming'],
        'time_left': time_left,
#         'left':inputs['left'],
        }.items())
        return state, inputs, deadline
        
    def discretize_deadline(self, deadline):
        time_left = float(deadline)/max(self.ddlines)
        if time_left>0.5:
            return 2
#         elif time_left>0.33:
#             return 1
        else:
            return 0
        
    def choose_action(self, state, epsilon=0.1):
        actions = Environment.valid_actions
        max_q = None
        for act in actions:
            self.q_table[(tuple(state), act)] = self.q_table.get((tuple(state), act), 0.0)
            q_value = self.q_table[(tuple(state), act)]
            if q_value > max_q:
                max_q = q_value
                max_action = act
        if random.random()<epsilon:
            max_action = random.choice(actions[1:])
        return max_action, max_q
    
    def choose_action_2(self, state, epsilon=0.1):
        actions = Environment.valid_actions
        max_q = None
        for act in actions:
            self.ego_q_table[(tuple(state), act)] = self.ego_q_table.get((tuple(state), act), 0.0)
            self.allo_q_table[(tuple(state), act)] = self.allo_q_table.get((tuple(state), act), 0.0)
            q_value = 0.5*(self.ego_q_table[(tuple(state), act)] + self.allo_q_table[(tuple(state), act)])
            if q_value > max_q:
                max_q = q_value
                max_action = act
        if random.random()<epsilon:
            max_action = random.choice(actions[1:])
        return max_action, max_q
        
    def update_q_table(self, table, action, reward, max_q_prime, gamma):
        value = reward + gamma * max_q_prime
        value -= table[(self.state, action)]
        table[(self.state, action)] += self.alpha * value
        
    def update(self, t):
        # TODO: Update state
        self.state = self.update_state()[0]

        # TODO: Select action according to your policy
        actions = Environment.valid_actions
        action, max_q = self.choose_action(self.state)
#         action, max_q = self.choose_action_2(self.state)

        # Execute action and get reward
        reward = self.env.act(self, action)
        state_prime, inputs, deadline = self.update_state()
        action_prime, max_q_prime = self.choose_action(self.state)
        
        # TODO: Learn policy based on state, action, reward
        self.update_q_table(self.q_table,action, reward, max_q_prime, gamma=self.gamma)
#         self.update_q_table(self.ego_q_table, action, reward, max_q_prime, gamma=0.0)
#         self.update_q_table(self.allo_q_table, action, reward, max_q_prime, gamma=self.gamma)
        
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


if __name__ == '__main__':
    run()
