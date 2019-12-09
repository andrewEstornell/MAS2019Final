import numpy as np
import random as rand

UPPER = 0
LOWER = 1
HIGHW = 2



class Network:

    def __init__(self, highway, f, agents):
        self.highway = highway
        self.f = f
        self.agents = agents

        self.s_up = 0
        self.s_hw = 0
        self.s_lr = 0
        self.s = [self.s_up, self.s_lr, self.s_hw]
        
        self.upper_cost = 0
        self.highw_cost = 0
        self.lower_cost = 0
        self.costs = []

    def calculate_route_costs(self):
        self.s_up = 0
        self.s_hw = 0
        self.s_lr = 0
        for agent in self.agents:
            if agent.last_route == UPPER:
                self.s_up += 1
            elif agent.last_route == HIGHW:
                self.s_hw += 1
            elif agent.last_route == LOWER:
                self.s_lr += 1
        self.s = [self.s_up, self.s_lr, self.s_hw]

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
        self.last_route = None
        self.avg_route_costs = [0, 0, 0]

    def _ficticious_play(self, network):
        best_route = np.argmin(network.costs)
        self.last_route = best_route

    def epsilon_greedy(self, network, e, rounds):
        if rand.uniform(0, 1) < e:
            self.last_route = rand.randint(0, len(network.costs) - 1)
    







def ficticious_play(network, rounds):

    for r in range(rounds):
        network.calculate_route_costs()
        for agent in network.agents:
            agent._ficticious_play(network)

        print("#################")
        print(r)
        print(network.costs)
        print(network.s)



        

        
        





def f(x):
    return 2*x/float(n)







if __name__ == "__main__":
    highway = True
    n = 100
    rounds = 100
    agents = [Agent(i) for i in range(n)]
    network = Network(highway, f, agents)

    ficticious_play(network, rounds)
