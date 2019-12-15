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
        self.mu_up = 0
        self.mu_lr = 0
        self.mu_hw = 0

        # the history of costs agent incurred by taking each route
        self.cost_history_up = np.array([])
        self.cost_history_lr = np.array([])
        self.cost_history_hw = np.array([])
        self.cost_history = [self.cost_history_up, self.cost_history_lr, self.cost_history_hw]

        self.sigma_up = 1
        self.sigma_lr = 1
        self.sigma_hw = 1

        self.mu = [self.mu_up, self.mu_lr]
        self.sigma = [self.sigma_up, self.sigma_lr]
        if self.highway:
            self.a.append(1)
            self.b.append(1)
            self.mu.append(self.mu_hw)
            self.sigma.append(self.sigma_hw)

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
        # theta is our belief of cost for each action in this case
        self.theta = np.random.normal(self.mu, self.sigma)
        # pick the action with lower cost
        self.route = np.argmin(self.theta)

       

    def _update_gaussian_posterior_belief(self, action, cost):
        """
        update according to equation 4.3
        https://djrusso.github.io/RLCourse/papers/TS_Tutorial.pdf

        """

        # keep track of cost history for each route
        self.cost_history[action] = np.append(self.cost_history[action], cost)

        if len(self.cost_history[action]) > 1:
            # update the posterior only if there are more than 1 observations
            if np.var(self.cost_history[action]) > 0:
                # variance must be positive (in case it is 0, don't update?
                self.mu[action] = ((1/self.sigma[action])*(self.mu[action]) + (1/np.var(self.cost_history[action]))*(np.log(cost) + np.var(self.cost_history[action])/2)) / ((1 / self.sigma[action]) + (1 / np.var(self.cost_history[action])))
                self.sigma[action] = 1 / ((1 / self.sigma[action]) + (1 / np.var(self.cost_history[action])))


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
            if full_observation:
                for route in range(self.num_routes):
                    self._update_gaussian_posterior_belief(route, self.network.route_costs[route])

            else:
                reward = self.network.route_costs[self.route]
                self._update_gaussian_posterior_belief(self.route, reward)


        

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
        self.s_up = sum(1 for agent in self.agents if agent.route == UPPER)
        self.s_lr = sum(1 for agent in self.agents if agent.route == LOWER)
        self.s_hw = sum(1 for agent in self.agents if agent.route == HIGHW)

        self.route_costs[UPPER] = 1 + self.congestion_function(self.s_up + self.s_hw)
        self.route_costs[LOWER] = 1 + self.congestion_function(self.s_hw + self.s_lr)
        if self.highway:
            self.route_costs[HIGHW] = self.congestion_function(self.s_up + self.s_hw) + self.congestion_function(self.s_hw + self.s_lr)
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
            #print(network.s_up, network.s_lr, network.s_hw)
            average_cost_per_agent.append(network.calculate_average_demographic_costs(agents))
        for agent in network.agents:
            agent.update(full_observation)
    print("thompson =", end=' ')
    print(average_cost_per_agent)

    print("x = ", end=' ')
    print([i for i in range(rounds) if i%display_rate == 0])

def small_number_of_agents_final_policy_comparison(n, num_routes, congestion_function, full_observation, learning_rounds):
    # returns the final policy of an agent when it is the only agent on the road
    agents = [Agent(i, num_routes, FICTITIOUS_PLAY, epsilon) for  i in range(n)]
    network = Network([agents], congestion_function, num_routes)
    for agent in agents:
        agent.network = network
    
    ####### LEARNING PROCESS ##########
    network.starting_rounds(full_observation)
    
    for r in range(3, rounds):
        for agent in network.agents:
            agent.decision_function()
        network.calculate_route_costs()
       
        for agent in network.agents:
            agent.update(full_observation)
    ######## PLAYING WITH FINAL POLICY ##################
    for agent in agents:
        agent.route = np.argmin(agent.avg_route_costs)
    network.calculate_route_costs()
    print("Fictitious play ", n, network.calculate_average_demographic_costs(agents))
    
    agents = [Agent(i, num_routes, EPSILON_GREEDY, epsilon) for  i in range(n)]
    network = Network([agents], congestion_function, num_routes)
    for agent in agents:
        agent.network = network
    
    ####### LEARNING PROCESS ##########
    network.starting_rounds(full_observation)
    
    for r in range(3, rounds):
        for agent in network.agents:
            agent.decision_function()
        network.calculate_route_costs()
       
        for agent in network.agents:
            agent.update(full_observation)
    ######## PLAYING WITH FINAL POLICY ##################
    for agent in agents:
        agent.route = np.argmin(agent.avg_route_costs)
    network.calculate_route_costs()
    print("Epsilon Greedy ", n, network.calculate_average_demographic_costs(agents))


    agents = [Agent(i, num_routes, UCB, epsilon) for  i in range(n)]
    network = Network([agents], congestion_function, num_routes)
    for agent in agents:
        agent.network = network
    
    ####### LEARNING PROCESS ##########
    network.starting_rounds(full_observation)
    
    for r in range(3, rounds):
        for agent in network.agents:
            agent.decision_function()
        network.calculate_route_costs()
       
        for agent in network.agents:
            agent.update(full_observation)
    ######## PLAYING WITH FINAL POLICY ##################
    for agent in agents:
        agent.route = np.argmin(agent.avg_route_costs)
    network.calculate_route_costs()
    print("UCB1 ", n, network.calculate_average_demographic_costs(agents))

    agents = [Agent(i, num_routes, THOMPSON, epsilon) for  i in range(n)]
    network = Network([agents], congestion_function, num_routes)
    for agent in agents:
        agent.network = network
    
    ####### LEARNING PROCESS ##########
    network.starting_rounds(full_observation)
    
    for r in range(3, rounds):
        for agent in network.agents:
            agent.decision_function()
        network.calculate_route_costs()
       
        for agent in network.agents:
            agent.update(full_observation)
    ######## PLAYING WITH FINAL POLICY ##################
    for agent in agents:
        agent.route = np.argmin(agent.mu)
    network.calculate_route_costs()
    print("Thompson Sampling ", n, network.calculate_average_demographic_costs(agents))




def average_agent_reward_as_a_function_of_n(max_n, num_routes, congestion_function, full_observation, rounds):
    n = 1
    avg_cost = []
    while n < max_n:
        agents = [Agent(i, num_routes, FICTITIOUS_PLAY, epsilon) for  i in range(n)]
        network = Network([agents], congestion_function, num_routes)
        for agent in agents:
            agent.network = network
        network.starting_rounds(full_observation)
        average_agent_costs = 0
        for r in range(3, rounds):
            for agent in network.agents:
                agent.decision_function()
            network.calculate_route_costs()
            average_agent_costs += network.calculate_average_demographic_costs(agents)
            for agent in network.agents:
                agent.update(full_observation)
        average_agent_cost /= float(rounds)
        avg_cost.append(average_agent_costs)
    print("fictitious_play =", end=' ')
    print(avg_cost)
        

    n = 1
    avg_cost = []
    while n < max_n:
        agents = [Agent(i, num_routes, EPSILON_GREEDY, epsilon) for  i in range(n)]
        network = Network([agents], congestion_function, num_routes)
        for agent in agents:
            agent.network = network
        network.starting_rounds(full_observation)
        average_agent_costs = 0
        for r in range(3, rounds):
            for agent in network.agents:
                agent.decision_function()
            network.calculate_route_costs()
            average_agent_costs += network.calculate_average_demographic_costs(agents)
            for agent in network.agents:
                agent.update(full_observation)
        average_agent_cost /= float(rounds)
        avg_cost.append(average_agent_costs)
    print("epsilon_greedy =", end=' ')
    print(avg_cost)

    n = 1
    avg_cost = []
    while n < max_n:
        agents = [Agent(i, num_routes, UCB, epsilon) for  i in range(n)]
        network = Network([agents], congestion_function, num_routes)
        for agent in agents:
            agent.network = network
        network.starting_rounds(full_observation)
        average_agent_costs = 0
        for r in range(3, rounds):
            for agent in network.agents:
                agent.decision_function()
            network.calculate_route_costs()
            average_agent_costs += network.calculate_average_demographic_costs(agents)
            for agent in network.agents:
                agent.update(full_observation)
        average_agent_cost /= float(rounds)
        avg_cost.append(average_agent_costs)
    print("UCB1 =", end=' ')
    print(avg_cost)

    n = 1
    avg_cost = []
    while n < max_n:
        agents = [Agent(i, num_routes, FICTITIOUS_PLAY, epsilon) for  i in range(n)]
        network = Network([agents], congestion_function, num_routes)
        for agent in agents:
            agent.network = network
        network.starting_rounds(full_observation)
        average_agent_costs = 0
        for r in range(3, rounds):
            for agent in network.agents:
                agent.decision_function()
            network.calculate_route_costs()
            average_agent_costs += network.calculate_average_demographic_costs(agents)
            for agent in network.agents:
                agent.update(full_observation)
        average_agent_cost /= float(rounds)
        avg_cost.append(average_agent_costs)
    print("thompson =", end=' ')
    print(avg_cost)

    n = 1
    x = []
    while n < max_n:
        x.append(n)
    print("x =", x)






def f(x):
    return 1.5*x/n

###########################
#### HYPTER PARAMETERS ####
###########################
epsilon = 0.5      
n = 1000
rounds = 100
display_rate = 1

num_routes = 3
full_observation = True
###########################
print("exp 1, close up")
comparison_of_learning_across_all_learning_algorithms(n, num_routes, f, full_observation, rounds, display_rate)


###########################
#### HYPTER PARAMETERS ####
###########################
epsilon = 0.5      
n = 1000
rounds = 5001
display_rate = 250

num_routes = 3
full_observation = True
###########################
print("exp 2, stable")
comparison_of_learning_across_all_learning_algorithms(n, num_routes, f, full_observation, rounds, display_rate)



print("exp 3, average agent reward over all rounds, n changes")








































































