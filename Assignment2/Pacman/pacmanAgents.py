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


##
# Name: Zhuolun Li
# NetId: zl2501
#
##

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
         # the size of population
        self.P_SIZE = 8
        # The length of chromosome
        self.C_SIZE = 5
        possible = state.getAllPossibleActions()
        # Randomly initialize the action list
        self.action_list = [possible[random.randint(
            0, len(possible)-1)] for i in range(self.C_SIZE)]
        return

    # GetAction Function: Called with every frame
    def getAction(self, state):
        # TODO: write Hill Climber Algorithm instead of returning Directions.STOP
        # return Directions.STOP
        init_state = state
        possible = init_state.getAllPossibleActions()
        best_action_list = self.action_list
        best_value = -float('inf')
        exceed_limit = False
        while True:  # Loop until reach the limit
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
                    # else leave it the same as in the current action list
                    new_action_list.append(self.action_list[i])
            # Keep track of the successive states
            states = [init_state]
            # Loop the action list and find the successive states
            for action in new_action_list:
                # If already achieve a terminal state, then break.
                if states[-1].isWin() + states[-1].isLose() != 0:
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
                # Let the current action list be the new action list
                self.action_list = best_action_list
                best_value = val
            # if already exceed the limit
            if exceed_limit:
                self.action_list = best_action_list
                return best_action_list[0]


class GeneticAgent(Agent):
    # Initialization Function: Called one time when the game starts
    def registerInitialState(self, state):
        # the size of population
        self.P_SIZE = 8
        # The length of chromosome
        self.C_SIZE = 5
        # Store the last valid generation in case of exceeding the limit
        self.previous_population = None
        self.exceed_limit = False
        # Store the initial population
        self.init_population = []
        possible = state.getAllPossibleActions()
        for i in range(self.P_SIZE):
            # Generate 8 random chromosome as initial population
            action_list = [possible[random.randint(
                0, len(possible)-1)] for i in range(self.C_SIZE)]
            self.init_population.append(action_list)
        return

    # GetAction Function: Called with every frame
    def getAction(self, state):
        # Set the initial flag
        self.exceed_limit = False
        self.init_population = []
        possible = state.getAllPossibleActions()
        for i in range(self.P_SIZE):
            # Generate 8 random chromosome as initial population
            action_list = [possible[random.randint(
                0, len(possible)-1)] for i in range(self.C_SIZE)]
            self.init_population.append(action_list)

        population = self.init_population
        # Calculate the evaluation on the inital population list
        pop_with_eval = self.getAllEvaluation(state, population)
        # If doesn't exceed the limit
        while not self.exceed_limit:
            # Assign the current population to the prev_pop_w_eval
            prev_pop_w_eval = pop_with_eval
            # Run rank selection on the new population to select parents that are used to reproduce.
            selected_parents = self.rank_selection(state, pop_with_eval)
            # Use the selected pairs to reproduce new generation
            new_generation = self.reproduce(state, selected_parents)
            # Mute some chromosomes in the new generation with possibility of 10%
            for chrom in new_generation:
                if random.random() <= 0.1:
                    # Randomly mute one action in each chromosome
                    possible = state.getAllPossibleActions()
                    chrom[random.randint(
                        0, len(chrom)-1)] = possible[random.randint(0, len(possible)-1)]
            # Assign to the new population and repeat
            pop_with_eval = self.getAllEvaluation(state, new_generation)

        # When exceeds the limit, use the previous one as our new population
        new_population_with_eval = prev_pop_w_eval
        new_population_with_eval.sort(key=lambda p: p[1])
        # find the chromosome with highest evaluation value, and return its first action
        return new_population_with_eval[-1][0][0]

    def rank_selection(self, state, population_with_eval):
        """Run rank selection on a population to select pairs that are used to reproduce

        Arguments:
            population_with_eval: A list of population with evaluation
            state: the current state

        Returns:
            A list of selected chromosomes
        """
        # Sort population based on its evaluation value.
        temp = sorted(population_with_eval, key=lambda p: p[1])
        # Rank population
        ranked_population = [(j[0], i) for i, j in enumerate(temp, 1)]
        selected_parents = []
        for i in range(self.P_SIZE):
            # Rank selection
            choice, _ = self.rank_selection_choice(ranked_population)
            selected_parents.append(choice)
        return selected_parents

    def getAllEvaluation(self, state, population):
        """
        Calculate the evluation of each chromosome in the population list.
        Return a list which has evluation on each chromosome and the total sum of evaluation value
        Arguments:
            population: the population list to be processed
            state: the current state

        Returns:
            [(chromosome1, evaluation1),(chromosome2, evaluation2),...]
            If exceeds the limit, return None
        """
        selected_parents = []
        for chromosome in population:
            evaluaton = self.getEvaluationOfActionList(state, chromosome)
            selected_parents.append((chromosome, evaluaton))
            if self.exceed_limit:
                return None
        return selected_parents

    def rank_selection_choice(self, ranked_population):
        """ Fast weighted random selection.
        Choose a chromosome based on its rank.
        Arguments:
            ranked_population: [(chrom1, 1),(chrom2,2)...(chrom8,8)]
        Returns:
            chrom_k
        """
        rnd = random.random()*36
        for p in ranked_population:
            rnd -= p[1]
            if rnd < 0:
                return p[0], p[1]

    def getEvaluationOfActionList(self, init_state, chromosome):
        ''' Get evaluation for a action list. 
        If generatePacmanSuccessor runs out of calls in the middle, flat self.exceed_limit = True

        Arguments:
            init_state -- current state
            chromosome -- the action list

        Returns:
            float -- the value of evaluation
        '''
        # keep track of all successive states
        states = [init_state]
        self.exceed_limit = False
        temp_state = init_state
        for action in chromosome:
                # If already achieve a terminal state, then break.
            if states[-1].isWin() + states[-1].isLose() != 0:
                break
            temp_state = temp_state.generatePacmanSuccessor(action)
            if temp_state:
                states.append(temp_state)
            else:
                # If exceed the limit, FLAG!
                self.exceed_limit = True
                break
        # Only evaluate on the initial state and the last valid state in state array
        # Return evluateion and the flag of whether exceeds the limit
        return gameEvaluation(init_state, states[-1])

    def reproduce(self, state, selected_parents):
        """
        Reproduce new generation. 
        Returns:
            the new generation list -- List
        """
        new_generation = []
        pairs = [(selected_parents[i], selected_parents[i+1])
                 for i in range(0, self.P_SIZE, 2)]
        for pair in pairs:
            rdn = random.randint(1, 10)
            new_chrom_1 = [None]*self.C_SIZE
            new_chrom_2 = [None]*self.C_SIZE
            # 70% chance that the pair will generate new chromosome by cross-over
            if rdn <= 7:
                # start mutation
                for i in range(self.C_SIZE):
                    # 50%-50% chance
                    if random.random() < 0.5:
                        new_chrom_1[i] = pair[0][i]
                        new_chrom_2[i] = pair[1][i]
                    else:
                        new_chrom_1[i] = pair[1][i]
                        new_chrom_2[i] = pair[0][i]
            else:
                # keep the orignal pair in the next generation
                new_chrom_1, new_chrom_2 = pair[0], pair[1]
            new_generation.append(new_chrom_1)
            new_generation.append(new_chrom_2)
        return new_generation


class MCTSAgent(Agent):
    # Initialization Function: Called one time when the game starts
    def registerInitialState(self, state):
        self.ROLLOUT_LIMIT = 5
        return

    # GetAction Function: Called with every frame
    def getAction(self, state):
        root = Node(None, state)
        self.rootstate = state
        self.exceed_limit = False
        while not self.exceed_limit:
            new_node = self.treePolicy(root, self.rootstate)
            # Check if exceed the limit
            if self.exceed_limit:
                break
            node, reward = self.defaultPolicy(new_node)
            self.backup(node, reward)

        best_child = None
        largest_count = -1
        # Find the child node with largest visited count
        for child in root.children:
            if child.visited_count > largest_count:
                best_child = child
                largest_count = child.visited_count
            # If it's tie, randomly choose one
            elif child.visited_count == largest_count and random.randint(0,1):
                best_child = child
        return best_child.action

    def treePolicy(self, node, rootstate):
        """ 
        Return a new slected node and the corresponding state
        """
        c_node = node
        while len(c_node.legal_actions) != 0 and (rootstate.isWin()+rootstate.isLose() == 0):
            if len(c_node.untried_actions) > 0:
                return self.expand(c_node)
            else:
                new_child = self.bestChild(c_node)
                c_node = new_child
        return c_node

    def expand(self, node):
        # Choose an untried action
        action = node.untried_actions.pop()
        temp_node = node
        predecessors = []
        while temp_node.parent is not None:
            # Find all predecessors from the current node
            predecessors.insert(0, temp_node)
            temp_node = temp_node.parent
        temp_state = self.rootstate
        # Apply all the actions from predecessors to generate the current state
        for p in predecessors:
            temp_state = temp_state.generatePacmanSuccessor(p.action)
            # if exceed the limit
            if temp_state is None:
                self.exceed_limit = True
                return None
            elif temp_state.isWin() or temp_state.isLose():
                # If win, use this action and state
                action = p.action
                break
        new_node = Node(action, temp_state)
        # Add the new node to the children of the parent
        node.children.append(new_node)
        # Link to the parent
        new_node.parent = node
        # Return the child node and the corresponding state
        return new_node

    def bestChild(self, node):
        """Find the best child
        """
        # initialize
        largest_ucb = -float('inf')
        best_child = None
        for child in node.children:
            ucb = child.reward/child.visited_count + \
                math.sqrt(2*math.log(node.visited_count)/child.visited_count)
            if ucb > largest_ucb:
                best_child = child
                largest_ucb = ucb
                # If equal, randomly replace
            elif ucb == largest_ucb and random.randint(0, 1):
                best_child = child
        return best_child

    def defaultPolicy(self, node):
        temp_node = node
        predecessors = []
        # Find all the predecessors
        while temp_node.parent is not None:
            predecessors.insert(0, temp_node)
            temp_node = temp_node.parent
        temp_state = self.rootstate
        # Apply all the actions from predecessors to generate the current state
        for p in predecessors:
            prev_state = temp_state
            temp_state = temp_state.generatePacmanSuccessor(p.action)
            if temp_state is None:
                self.exceed_limit = True
                return p, gameEvaluation(self.rootstate, prev_state)
            elif temp_state.isWin() or temp_state.isLose():
                return p, gameEvaluation(self.rootstate, temp_state)

        n_state = temp_state
        count = 0
        while (count < self.ROLLOUT_LIMIT):
            count += 1
            actions = n_state.getLegalPacmanActions()
            # Randomly choose an action and generate the successor
            prev_state = n_state
            n_state = n_state.generatePacmanSuccessor(
                actions[random.randint(0, len(actions)-1)])
            # If exceed the limit
            if n_state is None:
                self.exceed_limit = True
                n_state = prev_state
                break
            if n_state.isWin() or n_state.isLose():
                n_state = prev_state
                break

        # Calculate the reward
        reward = gameEvaluation(self.rootstate, n_state)
        return node, reward

    def backup(self, node, reward):
        while node is not None:
            node.visited_count += 1
            node.reward += reward
            node = node.parent
        return


class Node:
    def __init__(self, action, state):
        # Store the cumulative reward
        self.reward = 0
        self.visited_count = 1
        self.parent = None
        # Store the children
        self.children = []
        # The action leading to the current state
        self.action = action
        # store the legal actions
        self.legal_actions = state.getLegalPacmanActions()
        # Keep track of the unexpanded actions
        self.untried_actions = state.getLegalPacmanActions()
