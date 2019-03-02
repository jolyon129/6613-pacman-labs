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
        fringe_paths_queue = []
        init_node = [state, None]
        fringe_paths_queue.append([init_node])
        while fringe_paths_queue:
            # Get the current path
            path = fringe_paths_queue.pop(0)
            # Get the current state, which is the last one in the path
            state = path[-1][0]
            if state is None or state.isWin():
                # Choose the action on the corresponding path,
                # wich lead to the win state.
                action = path[1][1]
                return action
            # If in a deadlok, skip this one, continue to the next one in the loop.
            if state.isLose():
                continue
            legal = state.getLegalPacmanActions()
            successors = [(state.generatePacmanSuccessor(action), action)
                          for action in legal]
            for successor in successors:
                # In each iteration the state is different,
                # so I don't need to check whether the state is in explored or fringe_path.
                # shallow copy the original path
                new_path = list(path)
                # create a new path
                new_path.append(successor)
                fringe_paths_queue.append(new_path)


class DFSAgent(Agent):
    # Initialization Function: Called one time when the game starts
    def registerInitialState(self, state):
        return

    # GetAction Function: Called with every frame
    def getAction(self, state):
        # TODO: write DFS Algorithm instead of returning Directions.STOP
        # return Directions.STOP
        if state.isWin() or state.isLose():
            return Directions.STOP
        init_node = [state, None]
        fringe_paths_stack = [[init_node]]
        while fringe_paths_stack:
            path = fringe_paths_stack.pop()
            state = path[-1][0]
            if state is None or state.isWin():
                action = path[1][1]
                return action
            if state.isLose():
                continue
            legal = state.getLegalPacmanActions()
            successors = [(state.generatePacmanSuccessor(action), action)
                          for action in legal]
            for successor in successors:
                new_path = list(path)
                new_path.append(successor)
                fringe_paths_stack.append(new_path)


class AStarAgent(Agent):
    # Initialization Function: Called one time when the game starts
    def registerInitialState(self, state):
        return

    # GetAction Function: Called with every frame
    def getAction(self, state):
        if state.isWin() or state.isLose():
            return Directions.STOP
        node = Node(state, admissibleHeuristic(state), cost=0)
        node.tot_cost = node.heuristic + 0
        sudo_priority_queue = []
        sudo_priority_queue.append(node)
        while sudo_priority_queue:
            sudo_priority_queue.sort(key = lambda node: node.tot_cost)
            


class Node:
    def __init__(self, state, heuristic, cost):
        self.state = state
        self.heuristic = heuristic
        self.g_cost = cost
        self.prev = None
        self.next = None
        self.tot_cost = None 
