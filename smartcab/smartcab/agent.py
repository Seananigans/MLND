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
        self.alpha = 0.7
        self.q_table = {}
        self.prev_state = ()
        self.prev_action = None
        self.prev_reward = 0.0

    def reset(self, destination=None):
        self.planner.route_to(destination)
        # TODO: Prepare for a new trip; reset any variables here, if required
    
    def update_state(self):
        # Gather inputs
        self.next_waypoint = self.planner.next_waypoint()  # from route planner, also displayed by simulator
        inputs = self.env.sense(self)
        deadline = self.env.get_deadline(self)
        
        state = tuple({
        'next_waypoint':self.next_waypoint,
        'light': inputs['light'],
        'oncoming': inputs['oncoming'],
    #         'right': inputs['right'], 
    #         remove oncoming traffic from right because it will learn this rule 
    #         through the color of the light
        'left':inputs['left'],
        'deadline':deadline,
        }.items())
        return state, inputs, deadline

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
            max_action = random.choice(actions)
        return max_action, max_q
        
    def update(self, t):
        # TODO: Update state
        self.state = self.update_state()[0]

        # TODO: Select action according to your policy
        actions = Environment.valid_actions
        action, max_q = self.choose_action(self.state)

        # Execute action and get reward
        reward = self.env.act(self, action)
        state_prime, inputs, deadline = self.update_state()
        action_prime, max_q_prime = self.choose_action(self.state)
        
        # TODO: Learn policy based on state, action, reward
        value = reward + self.gamma * max_q_prime
        value -= self.q_table[(self.state, action)]
        self.q_table[(self.state, action)] += self.alpha * value
        
        if (self.q_table.has_key( (self.prev_state, self.prev_action) ) ):
            value = self.prev_reward + self.gamma * max_q
            update = self.alpha * (value - self.q_table[(self.prev_state, self.prev_action)])
            self.q_table[(self.prev_state, self.prev_action)] += update
            #self.hallucinate(100)
        else:
            update = self.prev_reward
            self.q_table[(tuple(self.state), action)] += update
        
        self.prev_state = self.state
        self.prev_action = action
        self.prev_reward = reward
        print "LearningAgent.update(): deadline = {}, inputs = {}, action = {}, reward = {}".format(deadline, inputs, action, reward)  # [debug]
        
    
    def hallucinate(self, iterations):
        for i in range(iterations):
            #Choose random state from q_table and act randomly
            s, a = choice(self.q_table.keys())
            "---------------------------------------"
            s_prime = self.execute_action(s, a)
            a_prime = self.choose_action(s_prime)
            #Update Q table with state, action, reward, state_prime, and action_prime
            q_prime = self.q_table[(s_prime, a_prime)]
            value = self.reward(s) + self.gamma*q_prime - self.q_table[(s, a)]
            self.q_table[(s, a)] += self.alpha * value

def run():
    """Run the agent for a finite number of trials."""

    # Set up environment and agent
    e = Environment()  # create environment (also adds some dummy traffic)
    a = e.create_agent(LearningAgent)  # create agent
    e.set_primary_agent(a, enforce_deadline=True)  # set agent to track

    # Now simulate it
    sim = Simulator(e, update_delay=0.05)  # reduce update_delay to speed up simulation
    sim.run(n_trials=100)  # press Esc or close pygame window to quit


if __name__ == '__main__':
    run()
