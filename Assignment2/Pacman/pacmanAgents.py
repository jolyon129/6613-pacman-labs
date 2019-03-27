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
                # Let the current action list be the new  action list
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
        new_population_with_eval.sort(key= lambda p: p[1])
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
        return

    # GetAction Function: Called with every frame
    def getAction(self, state):
        # TODO: write MCTS Algorithm instead of returning Directions.STOP
        return Directions.STOP
