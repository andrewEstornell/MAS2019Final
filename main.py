import numpy as np
import random as rand

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


class Agent:

    def __init__(self, i):
        self.i = i
        # path the agent is taking
        self.last_route = None
        self.avg_route_costs = [0, 0, 0]
        self.raw_route_costs = [0, 0, 0]
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

        if highway:
            self.a.append(self.a_hw)
            self.b.append(self.b_hw)

    def _ficticious_play(self, network):
        # simulates each agent best responding simoltaniously to the last set of route costs
        best_route = np.argmin(network.costs)
        self.last_route = best_route

    def _update_posterior_belief(self, action, reward):
        # posterior is (a,b) = (a,b)+(r,1-r)
        self.a[action] += reward
        self.b[action] += 1 - reward

    def _pick_thompson_action(self):
        # theta is our belief over all actions, saying how optimistic we are about each action
        self.theta = np.random.beta(self.a, self.b)
        # pick the one with highest posterior p of success
        self.thompson_action = np.argmax(self.theta)

        return self.thompson_action

    




def ficticious_play(network, rounds):
    # runs fictiious play for all agents
    for r in range(rounds):
        network.calculate_route_costs()
        for agent in network.agents:
            agent._ficticious_play(network)

        print("#################")
        print(r)
        print(network.costs)
        print(network.s)


def epsilon_greedy(network, agents, rounds, epsilon):
    for r in range(rounds):
        for agent in agents:
            # Chooses to explore or exploite with probability epsilon
            if rand.uniform(0, 1) < epsilon:
                agent.last_route = np.argmin(agent.avg_route_costs)
            else:
                agent.last_route = rand.randint(0, len(network.costs) - 1)
        # After all agents have commited to s stratagy, the network costs are updated
        network.calculate_route_costs()
        for agent in agents:
            # Each agent then gets to see the cost of the route they took, and their understanding of the game is updated accordingly
            agent.raw_route_costs[agent.last_route] += network.costs[agent.last_route]
            agent.num[agent.last_route] += 1
            agent.avg_route_costs[agent.last_route] = agent.raw_route_costs[agent.last_route] / float(agent.num[agent.last_route])
        
        print("#################")
        print(r)
        print(network.costs)
        print(network.s)

    # Afrter all learning rounds are done, the agents commit to a a mixed stratagey and play one more round
    print("###############")
    for agent in agents:
        t = sum(agent.avg_route_costs)
        agent_route_likelyhood = [t - c for c in agent.avg_route_costs]
        t = sum(agent_route_likelyhood)
        selection = rand.uniform(0, t)
        if 0 <= selection <= agent_route_likelyhood[0]:
            agent.last_route = UPPER
        elif agent_route_likelyhood[0] <= selection <= sum(agent_route_likelyhood[:2]):
            agent.last_route = LOWER
        elif sum(agent_route_likelyhood[:2]) <= selection <= t:
            agent.last_route = HIGHW
    network.calculate_route_costs()
    print(network.costs)
    print(network.s)

    #for agent in agents:
    #    print(agent.avg_route_costs)
    print(agents[0].avg_route_costs)


def UCB1(network, agents, rounds):

    for r in range(rounds):
        for agent in agents:
            # Computes the maximum upper bound on the reward for each agent
            action_values = [-agent.avg_route_costs[i] + np.sqrt((2*np.log(r))/ (agent.num[i])) for i in range(len(network.costs))]
            # Plays the action with the greatest upperbound at this round
            action = np.argmax(action_values)
            agent.last_route = action
        # Once all agents have commited to an action, recalculates the costs
        network.calculate_route_costs()

        # After the network costs are updated, each agent then gets to see the value of the route they took
        for agent in agents:
            # Updates the the agent's ideas of the route cost
            agent.raw_route_costs[agent.last_route] += network.costs[agent.last_route]
            agent.num[agent.last_route] += 1
            agent.avg_route_costs[agent.last_route] = agent.raw_route_costs[agent.last_route] / float(agent.num[agent.last_route])
                    
        print("#################")
        print(r)
        print(network.costs)
        print(network.s)

    # After all the rounds are over, each agent plays a mixed strategy where the proability of playing a route is porporitonal to -cost
    print("###############")
    for agent in agents:
        t = sum(agent.avg_route_costs)
        agent_route_likelyhood = [t - c for c in agent.avg_route_costs]
        t = sum(agent_route_likelyhood)
        selection = rand.uniform(0, t)
        if 0 <= selection <= agent_route_likelyhood[0]:
            agent.last_route = UPPER
        elif agent_route_likelyhood[0] <= selection <= sum(agent_route_likelyhood[:2]):
            agent.last_route = LOWER
        elif sum(agent_route_likelyhood[:2]) <= selection <= t:
            agent.last_route = HIGHW
    network.calculate_route_costs()
    print(network.costs)
    print(network.s)

    #for agent in agents:
    #    print(agent.avg_route_costs)
    print(agents[0].avg_route_costs)

def f(x):
    # cost function
    # FEEL FREE TO MAKE NEW COST FUNCTIONS
    return 2*x/float(n)







if __name__ == "__main__":
    ##########################
    ### HYPER PARAMETERS #####
    highway = True
    n = 1000
    rounds = 1000
    epsilon = 0.9
    ############################

    agents = [Agent(i) for i in range(n)]
    network = Network(highway, f, agents)

    # Plays each learning stratagy
    #ficticious_play(network, rounds)
    #epsilon_greedy(network, agents, rounds, epsilon)
    UCB1(network, agents, rounds)
