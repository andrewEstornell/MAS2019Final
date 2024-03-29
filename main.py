import numpy as np
import random as rand
import matplotlib.pyplot as plt
import math as math


UPPER = 0
LOWER = 1
HIGHW = 2



class Network:

    def __init__(self, highway, f, agents):
        # True if there is a highway
        self.highway = highway
        # Cost function
        self.f = f
        self.agents = agents

        # num agents on each path
        self.s_up = 0
        self.s_hw = 0
        self.s_lr = 0
        self.s = [self.s_up, self.s_lr, self.s_hw]
        
        # Cost of each path under the current paths agents have selected
        # The paths each agent is currently taking can be found with 
        # [agent.last_route for agent in self.agents]
        self.upper_cost = 0
        self.highw_cost = 0
        self.lower_cost = 0
        self.costs = [0, 0]
        if self.highway:
            self.costs.append(0)

    # Finds the current cost of every path under the current action choice of agents
    # should be run at the end of every round once ALL agents have selected their path
    def calculate_route_costs(self):
        # reset the number of agetns on each path
        self.s_up = 0
        self.s_hw = 0
        self.s_lr = 0
        # Count each agent taking each path based on the route that agent took
        for agent in self.agents:
            if agent.last_route == UPPER:
                self.s_up += 1
            elif agent.last_route == HIGHW:
                self.s_hw += 1
            elif agent.last_route == LOWER:
                self.s_lr += 1
        self.s = [self.s_up, self.s_lr, self.s_hw]

        # calcualte new cost of each route
        self.upper_cost = 1 + self.f(self.s_up + self.s_hw)
        self.highw_cost = self.f(self.s_up + self.s_hw) + self.f(self.s_hw + self.s_lr)
        self.lower_cost = 1 + self.f(self.s_hw + self.s_lr)
        costs = [self.upper_cost, self.lower_cost]

        if self.highway:
            costs.append(self.highw_cost)
        self.costs = costs

    def total_network_cost(self):
        return sum(self.s[i]*self.costs[i] for i in range(len(self.costs)))

<<<<<<< HEAD
    
=======
>>>>>>> 8c40235353180600bc0f4b3d17368eb58e227cd2


class Agent:

    def __init__(self, i):
        self.i = i
        # path the agent is taking
        self.last_route = None
        self.avg_route_costs = [0, 0, 0]
        self.raw_route_costs = [0, 0, 0]

        # the history of costs agent incurred by taking each route
        self.cost_history_up = np.array([])
        self.cost_history_lr = np.array([])
        self.cost_history_hw = np.array([])
        self.cost_history = [self.cost_history_up, self.cost_history_lr, self.cost_history_hw]

        # number of times the agent has taken each path
        self.num_up = 0
        self.num_lr = 0
        self.num_hw = 0
        self.num = [self.num_up, self.num_lr, self.num_hw]

        # hyper-parameter (alpha) of a beta distribution; 1 indicates uniform prior
        self.a_up = 1
        self.a_lr = 1
        self.a_hw = 1

        # hyper-parameter (beta) of a beta distribution; 1 indicates uniform prior
        self.b_up = 1
        self.b_lr = 1
        self.b_hw = 1

        self.a = [self.a_up, self.a_lr]
        self.b = [self.b_up, self.b_lr]

        # parameters for gaussian distribution
        self.mu_up = 0
        self.mu_lr = 0
        self.mu_hw = 0

        self.sigma_up = 1
        self.sigma_lr = 1
        self.sigma_hw = 1

        self.mu = [self.mu_up, self.mu_lr]
        self.sigma = [self.sigma_up, self.sigma_lr]


        if highway:
            self.a.append(self.a_hw)
            self.b.append(self.b_hw)

            self.mu.append(self.mu_hw)
            self.sigma.append(self.sigma_hw)




    def _update_beta_posterior_belief(self, action, reward):
        # posterior is (a,b) = (a,b)+(r,1-r)
        self.a[action] += reward
        self.b[action] += 1 - reward

    def _pick_beta_thompson_action(self):
        # theta is our belief over all actions, saying how optimistic we are about each action
        self.theta = np.random.beta(self.a, self.b)
        # pick the one with highest posterior p of success
        self.thompson_action = np.argmax(self.theta)

        self.last_route = self.thompson_action

        return self.thompson_action

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

    def _pick_gaussian_thompson_action(self):
        # theta is our belief of cost for each action in this case
        self.theta = np.random.normal(self.mu, self.sigma)
        # pick the action with lower cost
        self.thompson_action = np.argmin(self.theta)

        self.last_route = self.thompson_action

        return self.thompson_action

    def update_averages(self, network, route, full_obs=False):
        if full_obs:
            for r in range(len(network.costs)):
                self.raw_route_costs[r] += network.costs[r]
                self.num[r] += 1
                self.avg_route_costs[r] = self.raw_route_costs[r] / float(self.num[r])
        else:
            self.raw_route_costs[route] += network.costs[route]
            self.num[route] += 1
            self.avg_route_costs[route] = self.raw_route_costs[route] / float(self.num[route])


def starting_rounds(network, full_obs=False):
    agents_starting_order = []
    for agent in network.agents:
        arms = [i for i in range(len(network.costs))]
        rand.shuffle(arms)
        agents_starting_order.append(arms)
    for i in range(len(network.costs)):
        for j in range(len(network.agents)):
            network.agents[j].last_route = agents_starting_order[j][i]
        network.calculate_route_costs()
        for j in range(len(network.agents)):
            agent = network.agents[j]
            agent.update_averages(network, agent.last_route, full_obs)
        #print("###################")
        #print(network.s)
        #print(network.costs)



def ficticious_play(network, rounds, full_obs=False):
    # runs fictiious play for all agents
    starting_rounds(network, full_obs)

    average_agent_cost_per_round = []
    for r in range(3, rounds):
        for agent in network.agents:
            min_avg_cost = min(agent.avg_route_costs)
            agent.last_route = rand.sample([i for i in range(len(network.costs)) if agent.avg_route_costs[i] == min_avg_cost], 1)[0]
        network.calculate_route_costs()
        for agent in network.agents:
            agent.update_averages(network, agent.last_route, full_obs)
        average_agent_cost_per_round.append(network.total_network_cost()/float(len(network.agents)))
        #print("#################")
        #print(r)
        #print(network.costs)
        #print(network.s)
    return average_agent_cost_per_round



def epsilon_greedy(network, agents, rounds, epsilon, full_obs=False):
    starting_rounds(network, full_obs)

    average_agent_cost_per_round = []
    for r in range(3, rounds):
        for agent in agents:
            # Chooses to explore or exploite with probability epsilon
            if rand.uniform(0, 1) < epsilon:
                min_avg_cost = min(agent.avg_route_costs)
                agent.last_route = rand.sample([i for i in range(len(network.costs)) if agent.avg_route_costs[i] == min_avg_cost], 1)[0]
            else:
                agent.last_route = rand.randint(0, len(network.costs) - 1)
        # After all agents have commited to s stratagy, the network costs are updated
        network.calculate_route_costs()
        for agent in agents:
            # Each agent then gets to see the cost of the route they took, and their understanding of the game is updated accordingly
            agent.update_averages(network, agent.last_route)
        average_agent_cost_per_round.append(network.total_network_cost()/float(len(network.agents)))
    return average_agent_cost_per_round
        


def UCB1(network, agents, rounds, display_rate, full_obs=False):

    starting_rounds(network, full_obs)

    average_agent_cost_per_round = []
    for r in range(3, rounds):
        log_r = 2*math.log(r)
        for agent in agents:
            # Computes the maximum upper bound on the reward for each agent
            
            total = float(sum(agent.avg_route_costs))
            action_values = [-agent.avg_route_costs[i]/total + np.sqrt(log_r/float(agent.num[i])) for i in range(len(network.costs))]
        
            #print(action_values)

            # Plays the action with the greatest upperbound at this round
            #print(action_values)
            max_val = max(action_values)
            action = rand.sample([i for i in range(len(action_values)) if action_values[i] == max_val], 1)[0]
            agent.last_route = action

        # Once all agents have commited to an action, recalculates the costs
        network.calculate_route_costs()

        # After the network costs are updated, each agent then gets to see the value of the route they took
        for agent in agents:
            # Updates the the agent's ideas of the route cost
            agent.update_averages(network, agent.last_route, full_obs)
        if r % display_rate == 0:
            print([np.sqrt(log_r/float(agents[0].num[i])) for i in range(len(network.costs))])
            print(agents[0].avg_route_costs)
            average_agent_cost_per_round.append(network.total_network_cost()/float(len(network.agents)))

        #print("#################")
        #print(r)
        #print(network.costs)
        #print(network.s)
    return average_agent_cost_per_round



def bernoulli_thompson_sampling(function_network, agents, rounds, noise):
    total_system_cost = np.zeros(rounds)
    #average_agent_cost_per_round = []
    for r in range(rounds):
        print("### Round %d ###" % r)

        # pick action based on current belief
        for agent in agents:
            agent._pick_beta_thompson_action()

        # compute societal outcome once agents commit to actions
        function_network.calculate_route_costs()
        #print(function_network.costs, "Route Costs")
        #print(function_network.s, "Route Congestion")

        total_system_cost[r] = sum([function_network.costs[agent.last_route] for agent in agents])
        

        # compute reward update agent beliefs
        for agent in agents:
            reward = 0
            if function_network.costs[agent.last_route] == min(function_network.costs):
                reward = 1

            # add noise: with noise probability, the reward changes to the wrong value
            random_value = rand.random()
            if random_value < noise:
                # alternate the reward from zero
                reward = abs(reward - 1)

            agent._update_beta_posterior_belief(agent.last_route, reward)

    average_agent_costs_per_round = total_system_cost / len(agents)

    plt.plot(list(range(rounds)), average_agent_costs_per_round, c='black')
    plt.title("Total Societal Cost vs. Iteration")
    plt.xlabel("Iteration")
    plt.ylabel("Total Societal Cost (Time on the road)")
    plt.show()

    return average_agent_costs_per_round



def gaussian_thompson_sampling(function_network, agents, rounds, full_obs=False):
    """
    based on equation 4.3 in the following tutorial:
    https://djrusso.github.io/RLCourse/papers/TS_Tutorial.pdf
    """

    total_system_cost = np.zeros(rounds)
    # average_agent_cost_per_round = []
    for r in range(rounds):
        print("### Round %d ###" % r)

        # pick action based on current belief
        for agent in agents:
            agent._pick_gaussian_thompson_action()

        # compute societal outcome once agents commit to actions
        function_network.calculate_route_costs()
        #print(function_network.costs, "Route Costs")
        #print(function_network.s, "Route Congestion")

        total_system_cost[r] = sum([function_network.costs[agent.last_route] for agent in agents])

        # compute reward update agent beliefs
        for agent in agents:
            if full_obs:
                for route in range(len(function_network.costs)):
                    agent._update_gaussian_posterior_belief(route, function_network.costs[route])

            else:
                reward = function_network.costs[agent.last_route]
                agent._update_gaussian_posterior_belief(agent.last_route, reward)

    average_agent_costs_per_round = total_system_cost / len(agents)

    plt.plot(list(range(rounds)), average_agent_costs_per_round, c='black')
    plt.title("Total Societal Cost vs. Iteration")
    plt.xlabel("Iteration")
    plt.ylabel("Total Societal Cost (Time on the road)")
    plt.show()

    return average_agent_costs_per_round



def f(x):
    # cost function
    # FEEL FREE TO MAKE NEW COST FUNCTIONS
    return 1.5*x/float(n)



def f_2(x):
    # cost function
    return x / float(n)




if __name__ == "__main__":
    ##########################
    ### HYPER PARAMETERS #####
    highway = False
    n = 4000
    rounds = 500

    epsilon = 0.9
    thompson_noise = 0.5
    ############################

    agents = [Agent(i) for i in range(n)]
    network = Network(highway, f, agents)

    network_2 = Network(highway, f_2, agents)

    # Plays each learning stratagy
    #average_agent_costs = ficticious_play(network, rounds)
    #average_agent_costs = epsilon_greedy(network, agents, rounds, epsilon)
    #average_agent_costs = UCB1(network, agents, rounds, display_rate)
    #average_agent_costs = bernoulli_thompson_sampling(network_2, agents, rounds, thompson_noise)
    average_agent_costs = gaussian_thompson_sampling(network_2, agents, rounds, full_obs=False)
    print(average_agent_costs)


    
    
    
    
    
    
