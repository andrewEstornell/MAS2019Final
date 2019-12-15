import random as rand
import numpy as np
import math as math

UPPER = 0
LOWER = 1
HIGHW = 2


FICTITIOUS_PLAY = 0
EPSILON_GREEDY = 1
UCB = 2
THOMPSON = 3


class Agent:

    def __init__(self, i, num_routes, learning_algo, epsilon):
        self.learning_algo = learning_algo # String: name of the algorithm used to determine the agents decsiion
        self.i = i # Int : id for the agents
        self.num_routes = num_routes # Boolean dtereming if the road network has a highway
        self.route = None
        self.network = None
        self.highway = False
        if self.num_routes > 2:
            self.highway = True
        
        #self.algorithms = {FICTITIOUS_PLAY: self.fictitious_play_selection,
        #                   EPSILON_GREEDY:  self.epsilon_greedy_selection,
        #                   UCB:             self.UCB1,
        #                   THOMPSON:        self.thompson_sampling}
        self.epsilon = epsilon
        self.a = [1, 1]
        self.b = [1, 1]
        if self.highway:
            self.a.append(1)
            self.b.append(1)

        if self.highway:
            self.N = [0, 0, 0] # number of times the agent has used each route
            self.avg_route_costs = [0.0, 0.0, 0.0] # emperical average cost of the route
            self.raw_route_costs = [0.0, 0.0, 0.0]
        else:
            self.N = [0, 0]
            self.avg_route_costs = [0.0, 0.0]
            self.raw_route_costs = [0.0, 0.0]

    def decision_function(self):
        if self.learning_algo == FICTITIOUS_PLAY:
            self.fictitious_play_selection()
        elif self.learning_algo == EPSILON_GREEDY:
            self.epsilon_greedy_selection()
        elif self.learning_algo == UCB:
            self.UCB1()
        elif self.learning_algo == THOMPSON:
            self.thompson_sampling()

    def fictitious_play_selection(self):
        min_val = min(self.avg_route_costs)
        self.route = rand.sample([i for i in range(self.num_routes) if self.avg_route_costs[i] == min_val], 1)[0]

    def epsilon_greedy_selection(self):
        if rand.uniform(0, 1) < self.epsilon:
            min_val = min(self.avg_route_costs)
            self.route = rand.sample([i for i in range(self.num_routes) if self.avg_route_costs[i] == min_val], 1)[0]
        else:
            self.route = rand.randint(0, self.num_routes - 1)

    def UCB1(self):
        total = float(sum(self.avg_route_costs))
        route_value_bounds = [-self.avg_route_costs[i]/total + math.sqrt((2*math.log(sum(self.N)))/float(self.N[i])) for i in range(self.num_routes)]
        max_val = max(route_value_bounds)
        self.route = rand.sample([i for i in range(self.num_routes) if route_value_bounds[i] == max_val], 1)[0]

    def thompson_sampling(self):
        # theta is our belief over all actions, saying how optimistic we are about each action
        self.theta = np.random.beta(self.a, self.b)
        # pick the one with highest posterior p of success
        self.thompson_action = np.argmax(self.theta)
        self.route = self.thompson_action

    def update(self, full_observation):
        if self.learning_algo in [FICTITIOUS_PLAY, EPSILON_GREEDY, UCB]:
            if full_observation:
                for r in range(self.num_routes):
                    self.raw_route_costs[r] += self.network.route_costs[r]
                    self.N[r] += 1
                    self.avg_route_costs[r] = self.raw_route_costs[r] / float(self.N[r])
            else:
                self.raw_route_costs[self.route] += self.network.route_costs[self.route]
                self.N[self.route] += 1
                self.avg_route_costs[self.route] = self.raw_route_costs[self.route] / float(self.N[self.route])

        elif self.learning_algo == THOMPSON:
            if self.network.route_costs[self.route] == self.network.min_cost:    
                self.a[self.route] += 1
            else:
                self.b[self.route] += 1

        

class Network:

    def __init__(self, demographics, congestion_function, num_routes):
        self.demographics = demographics
        self.agents = [] # List: agent objects
        for demo in self.demographics:
            self.agents += demo
        self.congestion_function = congestion_function # Function f: Z -> R, such that f(x) = conjestion on road segment where x is the number of agents 
        self.num_routes = num_routes
        self.highway = True # Boolean determining if the road network has a highway
        if self.num_routes < 3:
            self.highway = False
        
        self.min_cost = None

        if self.highway:
            self.route_costs = [0.0, 0.0, 0.0]
        else:
            self.route_costs = [0.0, 0.0]

    def calculate_route_costs(self):
        s_up = sum(1 for agent in self.agents if agent.route == UPPER)
        s_lr = sum(1 for agent in self.agents if agent.route == LOWER)
        s_hw = sum(1 for agent in self.agents if agent.route == HIGHW)

        self.route_costs[UPPER] = 1 + self.congestion_function(s_up + s_hw)
        self.route_costs[LOWER] = 1 + self.congestion_function(s_hw + s_lr)
        if self.highway:
            self.route_costs[HIGHW] = self.congestion_function(s_up + s_hw) + self.congestion_function(s_hw + s_lr)
        self.min_cost = min(self.route_costs)
        return self.route_costs

    def calculate_average_demographic_costs(self, agents):
        s_up = sum(1 for agent in agents if agent.route == UPPER)
        s_lr = sum(1 for agent in agents if agent.route == LOWER)
        s_hw = sum(1 for agent in agents if agent.route == HIGHW)

        cost = s_up*self.route_costs[UPPER] + s_lr*self.route_costs[LOWER]
        if self.highway:
            cost += s_hw*self.route_costs[HIGHW]
        return cost/float(s_up + s_lr + s_hw)

    def starting_rounds(self, full_observation):
        agents_starting_order = []
        for agent in self.agents:
            arms = [i for i in range(len(self.route_costs))]
            rand.shuffle(arms)
            agents_starting_order.append(arms)
        for i in range(len(self.route_costs)):
            for j in range(len(self.agents)):
                self.agents[j].route = agents_starting_order[j][i]
            self.calculate_route_costs()
            for j in range(len(self.agents)):
                agent = self.agents[j]
                agent.update(full_observation)
            #print("###################")
            #print(network.s)
            #print(network.costs)



def comparison_of_learning_across_all_learning_algorithms(n, num_routes, congestion_function, full_observation, rounds, display_rate):
    agents = [Agent(i, num_routes, FICTITIOUS_PLAY, epsilon) for  i in range(n)]
    network = Network([agents], congestion_function, num_routes)
    for agent in agents:
        agent.network = network
    network.starting_rounds(full_observation)
    average_cost_per_agent = [network.calculate_average_demographic_costs(agents)]
    for r in range(3, rounds):
        for agent in network.agents:
            agent.decision_function()
        network.calculate_route_costs()
        if r%display_rate == 0:
            average_cost_per_agent.append(network.calculate_average_demographic_costs(agents))
        for agent in network.agents:
            agent.update(full_observation)
    print("fictitious_play =", end=' ')
    print(average_cost_per_agent)

    agents = [Agent(i, num_routes, EPSILON_GREEDY, epsilon) for  i in range(n)]
    network = Network([agents], congestion_function, num_routes)
    for agent in agents:
        agent.network = network
    network.starting_rounds(full_observation)
    average_cost_per_agent = [network.calculate_average_demographic_costs(agents)]
    for r in range(3, rounds):
        for agent in network.agents:
            agent.decision_function()
        network.calculate_route_costs()
        if r%display_rate == 0:
            average_cost_per_agent.append(network.calculate_average_demographic_costs(agents))
        for agent in network.agents:
            agent.update(full_observation)
    print("epsilon_greedy =", end=' ')
    print(average_cost_per_agent)
    
    agents = [Agent(i, num_routes, UCB, epsilon) for  i in range(n)]
    network = Network([agents], congestion_function, num_routes)
    for agent in agents:
        agent.network = network
    network.starting_rounds(full_observation)
    average_cost_per_agent = [network.calculate_average_demographic_costs(agents)]
    for r in range(3, rounds):
        for agent in network.agents:
            agent.decision_function()
        network.calculate_route_costs()
        if r%display_rate == 0:
            average_cost_per_agent.append(network.calculate_average_demographic_costs(agents))
        for agent in network.agents:
            agent.update(full_observation)
    print("UCB1 =", end=' ')
    print(average_cost_per_agent)
    
    agents = [Agent(i, num_routes, THOMPSON, epsilon) for  i in range(n)]
    network = Network([agents], congestion_function, num_routes)
    for agent in agents:
        agent.network = network
    average_cost_per_agent = []
    for r in range(rounds):
        for agent in network.agents:
            agent.decision_function()
        network.calculate_route_costs()
        if r%display_rate == 0:
            average_cost_per_agent.append(network.calculate_average_demographic_costs(agents))
        for agent in network.agents:
            agent.update(full_observation)
    print("thompson =", end=' ')
    print(average_cost_per_agent)

    print("x = ", end=' ')
    print([i for i in range(rounds) if i%display_rate == 0])
    



def f(x):
    return x/n

###########################
#### HYPTER PARAMETERS ####
###########################
epsilon = 0.1       
n = 1000
rounds = 100
display_rate = 1

num_routes = 3
full_observation = False
###########################
comparison_of_learning_across_all_learning_algorithms(n, num_routes, f, full_observation, rounds, display_rate)








































































