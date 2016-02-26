# multiAgents.py
# --------------
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

import sys
import math
from util import manhattanDistance
from game import Directions
import random, util
from decimal import Decimal
from game import Agent, Actions


class ReflexAgent(Agent):
    """
      A reflex agent chooses an action at each choice point by examining
      its alternatives via a state evaluation function.

      The code below is provided as a guide.  You are welcome to change
      it in any way you see fit, so long as you don't touch our method
      headers.
    """


    def getAction(self, gameState):
        """
        You do not need to change this method, but you're welcome to.

        getAction chooses among the best options according to the evaluation function.

        Just like in the previous project, getAction takes a GameState and returns
        some Directions.X for some X in the set {North, South, West, East, Stop}
        """
        # Collect legal moves and successor states
        legalMoves = gameState.getLegalActions()

        # Choose one of the best actions
        scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices) # Pick randomly among the best

        "Add more of your code here if you want to"

        return legalMoves[chosenIndex]

    def evaluationFunction(self, currentGameState, action):
        """
        Design a better evaluation function here.

        The evaluation function takes in the current and proposed successor
        GameStates (pacman.py) and returns a number, where higher numbers are better.

        The code below extracts some useful information from the state, like the
        remaining food (newFood) and Pacman position after moving (newPos).
        newScaredTimes holds the number of moves that each ghost will remain
        scared because of Pacman having eaten a power pellet.

        Print out these variables to see what you're getting, then combine them
        to create a masterful evaluation function.
        """
        # Useful information you can extract from a GameState (pacman.py)
        successorGameState = currentGameState.generatePacmanSuccessor(action)
        newPos = successorGameState.getPacmanPosition()
        newFood = successorGameState.getFood()
        currentFood = currentGameState.getFood()
        currentPos = currentGameState.getPacmanPosition()
        newGhostStates = successorGameState.getGhostStates()
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

        "*** YOUR CODE HERE ***"

	#handle food
        foodLocations = currentGameState.getFood().asList()
        best = sys.maxint
        index = 0
        while index < len(foodLocations):
            dist = manhattanDistance(newPos,foodLocations[index])
            if dist < best:
                best = dist
            index = index + 1;
        #ghost
        ghostPositions = successorGameState.getGhostPositions()
        newValue = 0
        for ghostposition in ghostPositions:
            if newValue == 0:
                newValue = manhattanDistance(newPos,ghostposition)
            elif manhattanDistance(newPos, ghostposition) < newValue:
                newValue = manhattanDistance(newPos, ghostposition)
        if newValue < 2:
            return -sys.maxint - 1
        returnValue = -2 * best
        return returnValue



def scoreEvaluationFunction(currentGameState):
    """
      This default evaluation function just returns the score of the state.
      The score is the same one displayed in the Pacman GUI.

      This evaluation function is meant for use with adversarial search agents
      (not reflex agents).
    """
    return currentGameState.getScore()

class MultiAgentSearchAgent(Agent):
    """
      This class provides some common elements to all of your
      multi-agent searchers.  Any methods defined here will be available
      to the MinimaxPacmanAgent, AlphaBetaPacmanAgent & ExpectimaxPacmanAgent.

      You *do not* need to make any changes here, but you can if you want to
      add functionality to all your adversarial search agents.  Please do not
      remove anything, however.

      Note: this is an abstract class: one that should not be instantiated.  It's
      only partially specified, and designed to be extended.  Agent (game.py)
      is another abstract class.
    """

    def __init__(self, evalFn = 'scoreEvaluationFunction', depth = '2'):
        self.index = 0 # Pacman is always agent index 0
        self.evaluationFunction = util.lookup(evalFn, globals())
        self.depth = int(depth)

class MinimaxAgent(MultiAgentSearchAgent):
    """
      Your minimax agent (question 2)
    """

    def getAction(self, gameState):
        """
          Returns the minimax action from the current gameState using self.depth
          and self.evaluationFunction.

          Here are some method calls that might be useful when implementing minimax.

          gameState.getLegalActions(agentIndex):
            Returns a list of legal actions for an agent
            agentIndex=0 means Pacman, ghosts are >= 1

          gameState.generateSuccessor(agentIndex, action):
            Returns the successor game state after an agent takes an action

          gameState.getNumAgents():
            Returns the total number of agents in the game
        """
        "*** YOUR CODE HERE ***"
	getAllAgents = gameState.getNumAgents()

        def calculateMax(gamestate, current_depth):
            pacmanMoves = gamestate.getLegalActions(0)

            if current_depth > self.depth or gamestate.isWin() or not pacmanMoves:
                return self.evaluationFunction(gamestate), None

            nextMoveCost = []
            for action in pacmanMoves:
                nextMove = gamestate.generateSuccessor(0, action)
                nextMoveCost.append((calculateMin(nextMove, 1, current_depth), action))

            return max(nextMoveCost)

        def calculateMin(gamestate, agent_index, current_depth):
            ghostMoves = gamestate.getLegalActions(agent_index)
            if not ghostMoves or gamestate.isLose():
                return self.evaluationFunction(gamestate), None

            allMoves = [gamestate.generateSuccessor(agent_index, action) for action in ghostMoves]

            if agent_index == getAllAgents - 1:
                nextMoveCost = []
                for nextMove in allMoves:
                    nextMoveCost.append(calculateMax(nextMove, current_depth + 1))
            else:
                nextMoveCost = []
                for nextMove in allMoves:
                    nextMoveCost.append(calculateMin(nextMove, agent_index + 1, current_depth))

            return min(nextMoveCost)


        return calculateMax(gameState, 1)[1]

class AlphaBetaAgent(MultiAgentSearchAgent):
    """
      Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState):
        """
          Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"
        getAllAgents = gameState.getNumAgents()
        def calculateMax(gamestate, current_depth, a, b):
            pacmanMoves = gamestate.getLegalActions(0)
            if current_depth > self.depth or gamestate.isWin() or not pacmanMoves:
                return self.evaluationFunction(gamestate), Directions.STOP
            bestCaseMove = Directions.STOP
            neg_inf = Decimal('-Infinity')
            for action in pacmanMoves:
                successor = gamestate.generateSuccessor(0, action)
                cost = calculateMin(successor, 1, current_depth, a, b)[0]
                if cost > neg_inf:
                    neg_inf = cost
                    bestCaseMove = action
                if neg_inf > b:
                    return neg_inf, bestCaseMove
                a = max(a, neg_inf)
            return neg_inf, bestCaseMove
    
        def calculateMin(gamestate, agent_index, current_depth, a, b):
            ghostMoves = gamestate.getLegalActions(agent_index)
            if not ghostMoves or gamestate.isLose():
                return self.evaluationFunction(gamestate), Directions.STOP
            bestCaseMove = Directions.STOP
            pacmanAgent = agent_index == getAllAgents - 1
            pos_inf = Decimal('Infinity')
            for action in ghostMoves:
                successor = gamestate.generateSuccessor(agent_index, action)
                if pacmanAgent:
                    cost = calculateMax(successor, current_depth + 1, a, b)[0]
                else:
                    cost = calculateMin(successor, agent_index + 1, current_depth, a, b)[0]
                if cost < pos_inf:
                    pos_inf = cost
                    bestCaseMove = action
                if pos_inf < a:
                    return pos_inf, bestCaseMove
                b = min(b, pos_inf)
            return pos_inf, bestCaseMove
        negativeAlpha = Decimal('-Infinity')
        positiveBeta = Decimal('Infinity')
        return calculateMax(gameState, 1, negativeAlpha, positiveBeta)[1]


class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """

    def getAction(self, gameState):
        """
          Returns the expectimax action using self.depth and self.evaluationFunction

          All ghosts should be modeled as choosing uniformly at random from their
          legal moves.
        """
        "*** YOUR CODE HERE ***"
        getAllAgents = gameState.getNumAgents()
        
        def probability(nms):
            retList = []
            for z in nms:
                retList.append(z[0]/len(nms))
            return retList
    
        def calculateMax(gamestate, current_depth):
            pacmanMoves = gamestate.getLegalActions(0)
            
            if current_depth > self.depth or gamestate.isWin() or not pacmanMoves:
                return self.evaluationFunction(gamestate), None
            
            nextMoveScore = []
            for action in pacmanMoves:
                nextMove = gamestate.generateSuccessor(0, action)
                nextMoveScore.append((calculateMin(nextMove, 1, current_depth)[0], action))
            return max(nextMoveScore)
        
        def calculateMin(gamestate, agent_index, current_depth):
            ghostMoves = gamestate.getLegalActions(agent_index)
            if not ghostMoves or gamestate.isLose():
                return self.evaluationFunction(gamestate), None
        
            nextMoves = [gamestate.generateSuccessor(agent_index, action) for action in ghostMoves]
            
            nextMoveScore = []
            for nextMove in nextMoves:

                if agent_index == getAllAgents - 1:
                    nextMoveScore.append(calculateMax(nextMove, current_depth + 1))
                else:
                    nextMoveScore.append(calculateMin(nextMove, agent_index + 1, current_depth))
    
            percent = probability(nextMoveScore)
            retAvg = 0
            for val in percent:
                retAvg += val
            
            return retAvg, None
    
    
        return calculateMax(gameState, 1)[1]
        


def betterEvaluationFunction(currentGameState):
    """
      Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
      evaluation function (question 5).

      DESCRIPTION: I took into account the closest distance to food, the avg distance to food
      (so as to aim for clusters of food), the distance to the closest capsule, the number of
      food left on the board, and the current score. I weighted the last two in order to get the
      highest score, acheived through experimentation. I also got the ghost distance which I check
      in order to maintain a minimum distance from the ghost.
    """
    "*** YOUR CODE HERE ***"
    if currentGameState.isWin() == True:
        return sys.maxint
    currPos = currentGameState.getPacmanPosition()
    currFood = currentGameState.getFood().asList()
    ghostStates = currentGameState.getGhostStates()
    scaredTimes = [ghostState.scaredTimer for ghostState in ghostStates]
    #ghost
    ghostPositions = currentGameState.getGhostPositions()
    newValue = 0
    for ghostposition in ghostPositions:
        newValue = manhattanDistance(currPos,ghostposition)
        if newValue < 1:
            return -sys.maxint - 1
    #avg
    avgList = []
    avgDist = 0
    index = 0
    while index < len(currFood):
        avgDist += manhattanDistance(currPos,currFood[index])
        index = index + 1
    avgDist = avgDist/ (len(currFood))

	#handle food
    best = sys.maxint
    index = 0
    while index < len(currFood):
        dist = manhattanDistance(currPos,currFood[index])
        if dist < best:
            best = dist
        index = index + 1;
    cap = sys.maxint
    index = 0
    dist = 0
    capsuleList = currentGameState.getCapsules()
    while index < len(capsuleList):
        dist = manhattanDistance(currPos,capsuleList[index])
        if dist < cap:
            cap = dist
        index = index + 1;
    #returnvalue
    bestF = best + avgDist
    numFoodWeighted = 20*currentGameState.getNumFood()
    scoreWeighted = -15* currentGameState.getScore()
    returnValue =  bestF+cap+numFoodWeighted+scoreWeighted
    return returnValue * -1

# Abbreviation
better = betterEvaluationFunction

