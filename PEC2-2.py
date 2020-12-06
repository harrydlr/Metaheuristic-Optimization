import pandas as pd
import numpy as np
import operator
import math
#data = pd.read_csv('/home/harry/PycharmProjects/MASTERINGCOMP/Metaheuristic Optimization/data/A-n38-k5_input_nodes.txt',
#                   header=None, delimiter= '\s+', index_col=False)


class Node:
    def __init__(self, ID, x, y, demand):
        self.ID = ID    # node identifier (depot ID = 0)
        self.x = x  # Euclidean x-coordinate
        self.y = y  # Euclidean y-coordinate
        self.demand = demand    # demand (is 0 for depot and positive for the rest)
        self.inRoute = None     # route to which node belongs
        self.isInterior = False     # an interior node is not connected to depot
        self.dnEdge = None  # edge (arc) from depot to this node
        self.ndEdge = None  # edge (arc) from this node to depot

class Edge:
    def __init__(self, origin, end):
        self.origin = origin    # origin node of the edge (arc)
        self.end = end  # end node of the edge (arc)
        self.cost = 0.0     # edge cost
        self.savings = 0.0  # edge savings (Clarke & Wright)
        self.invEdge = None     # inverse edge (arc)

class Route:
    def __init__(self):
        self.cost = 0.0     # cost of this route
        self.edges = []     # sorted edges in this route
        self.demand = 0.0   # total demand covered by this route

    def reverse(self):
        size = len(self.edges)
        for k in range(size):
            edge = self.edges[k]
            invEdge = edge.invEdge
            self.edges.remove(edge)
            self.edges.insert(0, invEdge)

class Solution:
    last_ID = -1    # counts the number of solutions, starts with 0

    def __init__(self):
        Solution.last_ID += 1
        self.ID = Solution.last_ID
        self.routes = []
        self.cost = 0.0
        self.demand = 0.0   # total demand covered by this solution


vehCap = 100.0 # update vehicle capacity for each instance
epoch = 10000
best_solutions = []
while epoch > 0:
    #alpha = np.random.uniform(0, 1)
    with open('/home/harry/PycharmProjects/MASTERINGCOMP/Metaheuristic Optimization/data/A-n80-k10_input_nodes.txt') as instance:
        i = 0
        nodes = []
        for line in instance:
            # array data with node data: x, y, demand
            data = [float(x) for x in line.split()]
            aNode = Node(i, data[0], data[1], data[2])
            nodes.append(aNode)
            i += 1

    depot = nodes[0]    # node 0 is the depot

    for node in nodes[1:]:  # excludes the depot
        dnEdge = Edge(depot, node)  # creates the (depot, node) edge (arc)
        ndEdge = Edge(node, depot)
        dnEdge.invEdge = ndEdge     # sets the inverse edge (arc)
        ndEdge.invEdge = dnEdge
        # compute the Euclidean distance as cost
        dnEdge.cost = math.sqrt((node.x - depot.x)**2 + (node.y-depot.y)**2)
        ndEdge.cost = dnEdge.cost   # assume symmetric costs
        # save in node a reference to the (depot, node) edge (arc)
        node.dnEdge = dnEdge
        node.ndEdge = ndEdge

    savingsList = []
    for i in range(1, len(nodes)-1):    # excludes the depot
        iNode = nodes[i]
        for j in range(i+1, len(nodes)):
            jNode = nodes[j]
            ijEdge = Edge(iNode, jNode)     # creates the (i,j) edge
            jiEdge = Edge(jNode, iNode)
            ijEdge.invEdge = jiEdge     # sets the inverse edge (arc)
            jiEdge.invEdge = ijEdge
            # compute the Euclidean distance as cost
            ijEdge.cost = math.sqrt((jNode.x - iNode.x)**2 + (jNode.y - iNode.y)**2)
            jiEdge.cost = ijEdge.cost   # assume symmetric costs
            # compute savings as proposed by Clark & Wright
            ijEdge.savings = iNode.ndEdge.cost + jNode.dnEdge.cost - ijEdge.cost
            jiEdge.savings = ijEdge.savings
            # save one edge in the savings list
            savingsList.append(ijEdge)
    # sort the list of edges from higher to lower savings
    savingsList.sort(key=operator.attrgetter("savings"), reverse=True)

    ## Construct the dummy solution ##
    sol = Solution()
    for node in nodes[1:]:  # excludes the depot
        dnEdge = node.dnEdge    # get the (depot, node) edge
        ndEdge = node.ndEdge
        dndRoute = Route()  # construct the route (depot, node, depot)
        dndRoute.edges.append(dnEdge)
        dndRoute.demand += node.demand
        dndRoute.cost += dnEdge.cost
        dndRoute.edges.append(ndEdge)
        dndRoute.cost += ndEdge.cost
        node.inRoute = dndRoute     # save in node a reference to its current route
        node.isInterior = False     # this node is currently exterior (connected to depot)
        sol.routes.append(dndRoute)     # add this route to the solution
        sol.cost += dndRoute.cost
        sol.demand += dndRoute.demand

    ## Perform the edge-selection & routing-merging iterative process ##
    def checkMergingConditions(iNode, jNode, iRoute, jRoute):
        # Condition 1: iRoute and jRoute are not the same route object
        if iRoute == jRoute:
            return False
        # Condition 2: both nodes are exterior nodes in their respective routes
        if iNode.isInterior == True or jNode.isInterior == True:
            return False
        # Condition3: demand after merging can be covered by a single vehicle
        if vehCap < iRoute.demand + jRoute.demand:
            return False
        # else, merging is feasible
        return True
    def getDepotEdge(aRoute, aNode):
        ## returns the edge in aRoute that contains aNode and the depot (it will be the first or the last one) ##
        # check if first edge in aRoute contains aNode and depot
        origin = aRoute.edges[0].origin
        end = aRoute.edges[0].end
        if ((origin == aNode and end == depot) or
                (origin == depot and end == aNode)):
            return aRoute.edges[0]
        else:   # return last edge in aRoute
            return aRoute.edges[-1]

    while len(savingsList) > 0:     #list is not empty
        ##
        alpha = np.random.uniform(0.05, 0.35)
        q = 99999
        while q > len(savingsList) - 1:
            q = np.random.geometric(alpha, size=1) - 1
        q = int(q)
        ijEdge = savingsList.pop(q)
        ##
        #ijEdge = savingsList.pop(0)     # select the next edge from the list
        # determine the nodes i < j that define the edge
        iNode = ijEdge.origin
        jNode = ijEdge.end
        # determine the routes associated with each node
        iRoute = iNode.inRoute
        jRoute = jNode.inRoute
        # check if merge is possible
        isMergeFeasible = checkMergingConditions(iNode, jNode, iRoute, jRoute)
        # if all necessary conditions are satisfied, merge
        if isMergeFeasible == True:
            # iRoute will contain either edge (depot, i) or edge (i, depot)
            iEdge = getDepotEdge(iRoute, iNode)     # iEdge is either (0, i) or (i, o)
            # remove iEdge from iRoute and update iRoute cost
            iRoute.edges.remove(iEdge)
            iRoute.cost -= iEdge.cost
            # if there are multiple edges in iRoute, then i will be interior
            if len(iRoute.edges) > 1:
                iNode.isInterior = True
            # if new iRoute does not start at 0 it must be reversed
            if iRoute.edges[0].origin != depot:
                iRoute.reverse()
            # jRoute will contain either edge (depot, j) or edge (j, depot)
            jEdge = getDepotEdge(jRoute, jNode)     # jEdge is either (0,j) or (j,0)
            # remove jEdge from jRoute and update jRoute cost
            jRoute.edges.remove(jEdge)
            jRoute.cost -= jEdge.cost
            # if there are multiple edges in jRoute, then j will be interior
            if len(jRoute.edges) > 1:
                jNode.isInterior = True
            # if new jRoute starts at 0 it must be reversed
            if jRoute.edges[0].origin == depot:
                jRoute.reverse()
            # add ijEdge to iRoute
            iRoute.edges.append(ijEdge)
            iRoute.cost += ijEdge.cost
            iRoute.demand += jNode.demand
            jNode.inRoute = iRoute
            # add jRoute to new iRoute
            for edge in jRoute.edges:
                iRoute.edges.append(edge)
                iRoute.cost += edge.cost
                iRoute.demand += edge.end.demand
                edge.end.inRoute = iRoute
            # delete jRoute from emerging solution
            sol.cost -= ijEdge.savings
            sol.routes.remove(jRoute)
    #print("Cost of C&W savings sol=", "{:.{}f}".format(sol.cost, 2))
    #for route in sol.routes:
        #s = str(0)
        #for edge in route.edges:
            #s = s + "-" + str(edge.end.ID)
        #print("Route: " + s + "|| cost = " + "{:.{}f}".format(route.cost, 2))
    best_solutions.append(sol)
    epoch -= 1

best_sol_score = []
best_sol_route_edges = []
for best_sol in best_solutions:
    best_sol_score.append(best_sol.cost)
print(min(best_sol_score))
best_index = best_sol_score.index(min(best_sol_score))
for route in best_solutions[best_index].routes:
    s = str(0)
    list_of_edges = []
    list_of_edges.append(0)
    for edge in route.edges:
        s = s + "-" + str(edge.end.ID)
        list_of_edges.append(edge.end.ID)
    case = {'route': list_of_edges, 'cost': route.cost}
    best_sol_route_edges.append(case)
    print("Route: " + s + "|| cost = " + "{:.{}f}".format(route.cost, 2))
print("Improving the best solution")
final_routes_list = []
for sol in best_solutions:
    for route in sol.routes:
        list_of_edges= []
        list_of_edges.append(0)
        for edge in route.edges:
            list_of_edges.append(edge.end.ID)
        case = {'route': list_of_edges, 'cost': route.cost}
        final_routes_list.append(case)

for i in range(len(best_sol_route_edges)):
    for j in range(len(final_routes_list)):
        if sorted(best_sol_route_edges[i]["route"]) == sorted(final_routes_list[j]["route"]):
            if best_sol_route_edges[i]["cost"] > final_routes_list[j]["cost"]:
                best_sol_route_edges[i] = final_routes_list[j]

for i in range(len(best_sol_route_edges)):
    print("Route: " + '-'.join(map(str, best_sol_route_edges[i]["route"])) + "|| cost: " + str(best_sol_route_edges[i]["cost"]))
cost_sum = 0
for i in range(len(best_sol_route_edges)):
    cost_sum += best_sol_route_edges[i]["cost"]
print(cost_sum)



