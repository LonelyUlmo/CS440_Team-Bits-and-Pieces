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


def tinyMazeSearch(problem):
    """
    Returns a sequence of moves that solves tinyMaze.  For any other maze, the
    sequence of moves will be incorrect, so only use this for tinyMaze.
    """
    from game import Directions
    s = Directions.SOUTH
    w = Directions.WEST
    return  [s, s, w, s, w, w, s, w]

def depthFirstSearch(problem: SearchProblem):
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
    "*** YOUR CODE HERE ***"
    #Stack DFS implementation
    dfsStack = util.Stack()

    #Array to keep track of visited nodes
    dfsVisited = []

    #List of actions
    actions = []

    #Push starting node
    dfsStack.push((problem.getStartState(), '' , 0))

    while(dfsStack.isEmpty() != True):
        #Pop last node from stack
        current = dfsStack.pop()
        
        if current[0] not in dfsVisited: 
            #add node to visited list
            dfsVisited.append(current[0])
            #Check if we have achieved goal
            if problem.isGoalState(current[0]):
                return actions;
            #Explore child nodes
            else:
                for node in problem.getSuccessors(current[0]):
                     dfsStack.push((node[0], node[1], node[2]))

    print(actions)
    return actions

def breadthFirstSearch(problem: SearchProblem):
    """Search the shallowest nodes in the search tree first."""
    "*** YOUR CODE HERE ***"
    util.raiseNotDefined()

def uniformCostSearch(problem: SearchProblem):
    """Search the node of least total cost first."""
    "*** YOUR CODE HERE ***"
    util.raiseNotDefined()

def nullHeuristic(state, problem=None):
    """
    A heuristic function estimates the cost from the current state to the nearest
    goal in the provided SearchProblem.  This heuristic is trivial.
    """
    return 0

def aStarSearch(problem: SearchProblem, heuristic=nullHeuristic):
    """Search the node that has the lowest combined cost and heuristic first."""
    from util import PriorityQueue

    # Initialize the frontier (open list) with the start state
    start_state = problem.getStartState()
    frontier = PriorityQueue()
    frontier.push((start_state, [], 0), heuristic(start_state, problem))

    # Initialize the explored set (closed list)
    explored = set()

    while not frontier.isEmpty():
        # Choose the node with the lowest f(n) = g(n) + h(n)
        current_state, path, cost = frontier.pop()

        # Check if we've reached the goal
        if problem.isGoalState(current_state):
            return path

        # Skip if we've already explored this state
        if current_state in explored:
            continue

        # Mark the current state as explored
        explored.add(current_state)

        # Expand the current state
        for next_state, action, step_cost in problem.getSuccessors(current_state):
            if next_state not in explored:
                new_path = path + [action]
                new_cost = cost + step_cost
                new_priority = new_cost + heuristic(next_state, problem)
                frontier.push((next_state, new_path, new_cost), new_priority)

    # If we've exhausted all possible paths and haven't found the goal, return None
    return None


# Abbreviations
bfs = breadthFirstSearch
dfs = depthFirstSearch
astar = aStarSearch
ucs = uniformCostSearch
