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
import math


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


class RandomSequenceAgent(Agent):
    # Initialization Function: Called one time when the game starts
    def registerInitialState(self, state):
        self.actionList = []
        for i in range(0, 10):
            self.actionList.append(Directions.STOP)
        return

    # GetAction Function: Called with every frame
    def getAction(self, state):
        # get all legal actions for pacman
        possible = state.getAllPossibleActions()
        for i in range(0, len(self.actionList)):
            self.actionList[i] = possible[random.randint(0, len(possible)-1)]
        tempState = state
        for i in range(0, len(self.actionList)):
            if tempState.isWin() + tempState.isLose() == 0:
                tempState = tempState.generatePacmanSuccessor(
                    self.actionList[i])
            else:
                break
        # returns random action from all the valide actions
        return self.actionList[0]


class HillClimberAgent(Agent):
    # Initialization Function: Called one time when the game starts
    def registerInitialState(self, state):
        possible = state.getAllPossibleActions()
        self.action_list =possible
        while len(self.action_list) < 5:
            self.action_list.append(possible[random.randint(0, len(possible)-1)])
        return

    # GetAction Function: Called with every frame
    def getAction(self, state):
        # TODO: write Hill Climber Algorithm instead of returning Directions.STOP
        # return Directions.STOP
        possible = state.getAllPossibleActions()
        init_state = state
        best_action_list = self.action_list
        best_value = -float('inf')
        exceed_limit = False
        while True:  # Loop until reach the limit of
            new_action_list = []
            for i in range(5):
                # 50% chance to flip the cureent action
                flip = random.randint(0, 1)
                # If we flip
                if flip:  
                    # Randomly choose a possible action to replace the current action
                    new_action_list.append(
                        possible[random.randint(0, len(possible)-1)])
                else:
                    # else leave it
                    new_action_list.append(self.action_list[i])
            # Keep track of the successive states
            states = [init_state] 
            # Loop the action list and find the successive states
            for action in new_action_list:
                # If already achieve a terminal state, then break.
                if states[-1].isWin()+ states[-1].isLose() != 0:
                    break
                temp_state = states[-1].generatePacmanSuccessor(action)
                if temp_state:
                    # If temp_state not None add to states
                    states.append(temp_state)
                # If temp_state is None, we exceed the limit. 
                else:
                    # Make a flag here.
                    exceed_limit = True
                    break
            # Evaluation
            val = gameEvaluation(init_state, states[-1])
            if val > best_value:
                best_action_list = new_action_list
                best_value = val
            # if already exceed the limit
            if exceed_limit:
                return best_action_list[0]

class GeneticAgent(Agent):
    # Initialization Function: Called one time when the game starts
    def registerInitialState(self, state):
        return

    # GetAction Function: Called with every frame
    def getAction(self, state):
        # TODO: write Genetic Algorithm instead of returning Directions.STOP
        return Directions.STOP


class MCTSAgent(Agent):
    # Initialization Function: Called one time when the game starts
    def registerInitialState(self, state):
        return

    # GetAction Function: Called with every frame
    def getAction(self, state):
        # TODO: write MCTS Algorithm instead of returning Directions.STOP
        return Directions.STOP
