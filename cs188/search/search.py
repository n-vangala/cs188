# search.py
# ---------
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


"""
In search.py, you will implement generic search algorithms which are called by
Pacman agents (in searchAgents.py).
"""

import util
from game import Directions
from typing import List

class SearchProblem:
    """
    This class outlines the structure of a search problem, but doesn't implement
    any of the methods (in object-oriented terminology: an abstract class).

    You do not need to change anything in this class, ever.
    """

    def getStartState(self):
        """
        Returns the start state for the search problem.
        """
        util.raiseNotDefined()

    def isGoalState(self, state):
        """
          state: Search state

        Returns True if and only if the state is a valid goal state.
        """
        util.raiseNotDefined()

    def getSuccessors(self, state):
        """
          state: Search state

        For a given state, this should return a list of triples, (successor,
        action, stepCost), where 'successor' is a successor to the current
        state, 'action' is the action required to get there, and 'stepCost' is
        the incremental cost of expanding to that successor.
        """
        util.raiseNotDefined()

    def getCostOfActions(self, actions):
        """
         actions: A list of actions to take

        This method returns the total cost of a particular sequence of actions.
        The sequence must be composed of legal moves.
        """
        util.raiseNotDefined()




def tinyMazeSearch(problem: SearchProblem) -> List[Directions]:
    """
    Returns a sequence of moves that solves tinyMaze.  For any other maze, the
    sequence of moves will be incorrect, so only use this for tinyMaze.
    """
    s = Directions.SOUTH
    w = Directions.WEST
    return  [s, s, w, s, w, w, s, w]

def depthFirstSearch(problem: SearchProblem) -> List[Directions]:
    """
    Search the deepest nodes in the search tree first.

    Your search algorithm needs to return a list of actions that reaches the
    goal. Make sure to implement a graph search algorithm.

    To get started, you might want to try some of these simple commands to
    understand the search problem that is being passed in:

    print("Start:", problem.getStartState())
    print("Is the start a goal?", problem.isGoalState(problem.getStartState()))
    print("Start's successors:", problem.getSuccessors(problem.getStartState()))
    """
    dfs_stack = util.Stack()
    startState = problem.getStartState() # ONE LETTER
    visitedSet = set() # set of LETTERS
    # currState is only ONE LETTER

    dfs_stack.push((startState, []))
    while not dfs_stack.isEmpty():
        currState, path = dfs_stack.pop()
        if currState in visitedSet:
            continue
        visitedSet.add(currState)
        if problem.isGoalState(currState):
            return path
        for successor, action, stepCost in problem.getSuccessors(currState):
            if successor not in visitedSet:
                dfs_stack.push((successor, path + [action]))
    return []

def breadthFirstSearch(problem: SearchProblem) -> List[Directions]:
    bfs_queue = util.Queue()
    startState = problem.getStartState() # ONE LETTER
    visitedSet = set() # set of LETTERS
    # currState is only ONE LETTER

    bfs_queue.push((startState, []))
    while not bfs_queue.isEmpty():
        currState, path = bfs_queue.pop()
        if currState in visitedSet:
            continue
        visitedSet.add(currState)
        if problem.isGoalState(currState):
            return path
        for successor, action, stepCost in problem.getSuccessors(currState):
            if successor not in visitedSet:
                bfs_queue.push((successor, path + [action]))
    return []

def uniformCostSearch(problem: SearchProblem) -> List[Directions]:
    """Search the node of least total cost first."""
    ucs_queue = util.PriorityQueue()
    startState = problem.getStartState() # ONE LETTER
    visitedSet = set() # set of LETTERS
    # currState is only ONE LETTER

    ucs_queue.push((startState, []), 0)
    while not ucs_queue.isEmpty():
        currState, path = ucs_queue.pop()
        if currState in visitedSet:
            continue
        visitedSet.add(currState)
        if problem.isGoalState(currState):
            return path
        for successor, action, stepCost in problem.getSuccessors(currState):
            if successor not in visitedSet:
                ucs_queue.push((successor, path + [action]), problem.getCostOfActions(path + [action]))
    return []

def nullHeuristic(state, problem=None) -> float:
    """
    A heuristic function estimates the cost from the current state to the nearest
    goal in the provided SearchProblem.  This heuristic is trivial.
    """
    return 0

def aStarSearch(problem: SearchProblem, heuristic= nullHeuristic) -> List[Directions]:
    """Search the node that has the lowest combined cost and heuristic first."""
    def priority(tuple):
        state, path = tuple
        pathCost = problem.getCostOfActions(path)
        heuristicCost = heuristic(state, problem)
        return pathCost + heuristicCost
    
    startState = problem.getStartState() # ONE LETTER
    astar_queue = util.PriorityQueueWithFunction(priority)
    visitedDictionary = dict() # set of LETTERS with min_cost so far to get to that letter
    # currState is only ONE LETTER

    astar_queue.push((startState, []))
    while not astar_queue.isEmpty():
        currState, path = astar_queue.pop()
        if currState not in visitedDictionary or problem.getCostOfActions(path) < visitedDictionary.get(currState):
            visitedDictionary[currState] = problem.getCostOfActions(path)
            if problem.isGoalState(currState):
                return path
            for successor, action, stepCost in problem.getSuccessors(currState):
                astar_queue.push((successor, path + [action]))
    return []

# Abbreviations
bfs = breadthFirstSearch
dfs = depthFirstSearch
astar = aStarSearch
ucs = uniformCostSearch
