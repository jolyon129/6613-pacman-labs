# pacmanAgents.py
# ---------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
#
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


from pacman import Directions
from game import Agent
from heuristics import *
import random


class RandomAgent(Agent):
    # Initialization Function: Called one time when the game starts
    def registerInitialState(self, state):
        return

    # GetAction Function: Called with every frame
    def getAction(self, state):
        # get all legal actions for pacman
        actions = state.getLegalPacmanActions()
        # returns random action from all the valide actions
        return actions[random.randint(0, len(actions)-1)]


class OneStepLookAheadAgent(Agent):
    # Initialization Function: Called one time when the game starts
    def registerInitialState(self, state):
        return

    # GetAction Function: Called with every frame
    def getAction(self, state):
        # get all legal actions for pacman
        legal = state.getLegalPacmanActions()
        # get all the successor state for these actions
        successors = [(state.generatePacmanSuccessor(action), action)
                      for action in legal]
        # evaluate the successor states using scoreEvaluation heuristic
        scored = [(admissibleHeuristic(state), action)
                  for state, action in successors]
        # get best choice
        bestScore = min(scored)[0]
        # get all actions that lead to the highest score
        bestActions = [pair[1] for pair in scored if pair[0] == bestScore]
        # return random action from the list of the best actions
        return random.choice(bestActions)


class BFSAgent(Agent):
    # Initialization Function: Called one time when the game starts
    def registerInitialState(self, state):
        return

    # GetAction Function: Called with every frame
    def getAction(self, state):
        if state.isWin() or state.isLose():
            return Directions.STOP
        # Use fringe_paths_queue to track all paths and states
        visited = set()
        queue = []
        root_node = Node(state, None, admissibleHeuristic(state), 0)
        visited.add(root_node.state)
        queue.append(root_node)
        while queue:
            node = queue.pop(0)
            visited.add(node.state)
            legal = node.state.getLegalPacmanActions()
            successors = [(node.state.generatePacmanSuccessor(action), action) for action in legal]
            for successor in successors:
                new_state, new_action = successor[0], successor[1]
                # If exceed the limit
                if new_state is None:
                    min_node = node
                    # Iterate all nodes in queue, find the one with lowest total cost
                    for n in queue:
                        if n.tot_cost < min_node.tot_cost:
                            min_node = n
                    # return the action lead to this node.
                    action = min_node.action_finder(root_node)
                    return action
                if new_state not in visited:
                    h = 0 if new_state is None else admissibleHeuristic(new_state)
                    new_node = Node(new_state, new_action, h, node.g_cost+1)
                    new_node.prev = node
                    if new_node.state.isWin():
                        action = new_node.action_finder(root_node)
                        return action
                    # If in a deadlock, skip this one, continue the loop.
                    if new_node.state.isLose():
                        continue
                    queue.append(new_node)


class DFSAgent(Agent):
    # Initialization Function: Called one time when the game starts
    def registerInitialState(self, state):
        return

    # GetAction Function: Called with every frame
    def getAction(self, state):
        if state.isWin() or state.isLose():
            return Directions.STOP
        visited = set()
        stack = []
        root_node = Node(state, None, admissibleHeuristic(state), 0)
        visited.add(state)
        stack.append(root_node)
        while stack:
            node = stack.pop(-1)
            visited.add(node.state)          
            legal = node.state.getLegalPacmanActions()
            successors = [(node.state.generatePacmanSuccessor(action), action) for action in legal]
            for successor in successors:
                # If current state has been visted, skip
                new_state, new_action = successor[0], successor[1]
                # If exceed the limit
                if new_state is None:
                    min_node = node
                    # Iterate all nodes in stack, find the one with lowest total cost
                    for n in stack:
                        if n.tot_cost < min_node.tot_cost:
                            min_node = n
                    # return the action lead to this node.
                    action = min_node.action_finder(root_node)
                    return action
                if new_state not in visited:
                    h = admissibleHeuristic(new_state)
                    new_node = Node(new_state, new_action, h, node.g_cost+1)
                    new_node.prev = node
                    if new_node.state.isWin():
                        action = new_node.action_finder(root_node)
                        return action
                    # If in a deadlock, skip this one, continue the loop.
                    if new_node.state.isLose():
                        continue
                    stack.append(new_node)


class AStarAgent(Agent):
    # Initialization Function: Called one time when the game starts
    def registerInitialState(self, state):
        return

    # GetAction Function: Called with every frame
    def getAction(self, state):
        if state.isWin() or state.isLose():
            return Directions.STOP
        # Create new node
        root_node = Node(state, None, admissibleHeuristic(state), 0)
        closed = set()
        open_pq = []
        graph = dict()
        open_pq.append(root_node)
        closed.add(root_node.state)
        # This is a dictionary of {state: Node}
        graph[state] = root_node
        while open_pq:
            node = open_pq.pop(0)
            closed.add(node.state)
            graph[node.state]= node
            if node.state.isWin():
                return node.action_finder(root_node)
            if node.state.isLose():
                continue
            legal = node.state.getLegalPacmanActions()
            successors = [(node.state.generatePacmanSuccessor(action), action)
                          for action in legal]
            for successor in successors:
                parent_node = node
                new_state, new_action = successor[0], successor[1]
                if new_state is None:
                    # If exceed the limit, the node with lowest total cost is the current node
                    # return the action lead to this node.
                    action = parent_node.action_finder(root_node)
                    return action
                h = admissibleHeuristic(new_state)
                new_node = Node(new_state, new_action,
                                h, parent_node.g_cost+1)
                new_node.prev = parent_node
                if new_state not in graph:
                    open_pq.append(new_node)
                elif new_node.tot_cost < graph[state].tot_cost:
                    # If this node already in the graph, and has a better total cost,
                    # we need to redirect the node
                    graph[new_state].prev = parent_node
                    # Update the total cost of the node which already exist
                    graph[new_state].tot_cost= new_node.tot_cost
            # sort the pq first by the total cost, then by the negative g_cost(the depth of the node)
            open_pq.sort(key=lambda node: [node.tot_cost, -node.g_cost])


class Node:
    def __init__(self, state, action, heuristic, g_cost):
        self.state = state
        self.heuristic = heuristic
        self.g_cost = g_cost
        self.prev = None
        self.tot_cost = self.g_cost + self.heuristic
        self.action = action

    def action_finder(self, root):
        n = self
        while n.prev != root:
            n = n.prev
        return n.action
