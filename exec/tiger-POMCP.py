import sys
sys.path.append('../src/')

import numpy as np
import random
from POMCP import  Search, Simulate, RollOut, OutputAction, Expand, upperConfidenceBound, SelectAction, update

class SimulatePOMDP(object):
    
    def __init__(self, stateSpace, actionSpace, observationSpace, transitionFunction, rewardFunction, observationFunction):
        self.stateSpace=stateSpace
        self.actionSpace=actionSpace
        self.observationSpace=observationSpace
        self.transitionFunction=transitionFunction
        self.rewardFunction=rewardFunction
        self.observationFunction=observationFunction
    
    def __call__(self, s, a):
        sPrimeDistribution={sPrime: self.transitionFunction(s, a, sPrime) for sPrime in self.stateSpace}
        sPrime=list(sPrimeDistribution.keys())[np.random.choice(len(sPrimeDistribution.keys()), p=list(sPrimeDistribution.values()))]
        r=self.rewardFunction(s, a, sPrime)
        oDistribution={o: self.observationFunction(sPrime, a, o) for o in self.observationSpace}
        o=list(oDistribution.keys())[np.random.choice(len(oDistribution.keys()), p=list(oDistribution.values()))]
        return (sPrime, r, o)

class TigerTransition():
    def __init__(self):
        self.transitionMatrix = {
            ('listen', 'tiger-left', 'tiger-left'): 1.0,
            ('listen', 'tiger-left', 'tiger-right'): 0.0,
            ('listen', 'tiger-right', 'tiger-left'): 0.0,
            ('listen', 'tiger-right', 'tiger-right'): 1.0,

            ('open-left', 'tiger-left', 'tiger-left'): 0.5,
            ('open-left', 'tiger-left', 'tiger-right'): 0.5,
            ('open-left', 'tiger-right', 'tiger-left'): 0.5,
            ('open-left', 'tiger-right', 'tiger-right'): 0.5,

            ('open-right', 'tiger-left', 'tiger-left'): 0.5,
            ('open-right', 'tiger-left', 'tiger-right'): 0.5,
            ('open-right', 'tiger-right', 'tiger-left'): 0.5,
            ('open-right', 'tiger-right', 'tiger-right'): 0.5
        }

    def __call__(self, state, action, nextState):
        nextStateProb = self.transitionMatrix.get((action, state, nextState), 0.0)
        return nextStateProb


class TigerReward():
    def __init__(self, rewardParam):
        self.rewardMatrix = {
            ('listen', 'tiger-left'): rewardParam['listen_cost'],
            ('listen', 'tiger-right'): rewardParam['listen_cost'],

            ('open-left', 'tiger-left'): rewardParam['open_incorrect_cost'],
            ('open-left', 'tiger-right'): rewardParam['open_correct_reward'],

            ('open-right', 'tiger-left'): rewardParam['open_correct_reward'],
            ('open-right', 'tiger-right'): rewardParam['open_incorrect_cost']
        }

    def __call__(self, state, action, sPrime):
        rewardFixed = self.rewardMatrix.get((action, state), 0.0)
        return rewardFixed


class TigerObservation():
    def __init__(self, observationParam):
        self.observationMatrix = {
            ('listen', 'tiger-left', 'tiger-left'): observationParam['obs_correct_prob'],
            ('listen', 'tiger-left', 'tiger-right'): observationParam['obs_incorrect_prob'],
            ('listen', 'tiger-right', 'tiger-left'): observationParam['obs_incorrect_prob'],
            ('listen', 'tiger-right', 'tiger-right'): observationParam['obs_correct_prob'],

            ('open-left', 'tiger-left', 'Nothing'): 1,
            ('open-left', 'tiger-right', 'Nothing'): 1,
            ('open-right', 'tiger-left', 'Nothing'): 1,
            ('open-right', 'tiger-right', 'Nothing'): 1,
        }

    def __call__(self, state, action, observation):
        observationProb = self.observationMatrix.get((action, state, observation), 0.0)
        return observationProb



def main():
    
    rewardParam={'listen_cost':-1, 'open_incorrect_cost':-100, 'open_correct_reward':10}
    rewardFunction=TigerReward(rewardParam)
    
    observationParam={'obs_correct_prob':0.85, 'obs_incorrect_prob':0.15}
    observationFunction=TigerObservation(observationParam)
    
    transitionFunction=TigerTransition()
    
    stateSpace=['tiger-left', 'tiger-right']
    observationSpace=['tiger-left', 'tiger-right', 'Nothing']
    actionSpace=['open-left', 'open-right', 'listen']
    
    
    b={'tiger-left':0.85, 'tiger-right':0.15}
    bPrime={'tiger-left':0.5, 'tiger-right':0.5}
    a='open-left'
    
    T={}
    h=()
    
    
    simulatePOMDP=SimulatePOMDP(stateSpace, actionSpace, observationSpace, transitionFunction, rewardFunction, observationFunction)
    
    epsilon=0.1
    gamma=0.8
    rollOutPolicy=lambda h: actionSpace[np.random.choice(len(actionSpace))] 
    rollOut=RollOut(rollOutPolicy, simulatePOMDP, epsilon, gamma)
    
    expand=Expand(T, actionSpace)
    
    
    c=2
    uCB=lambda Q, N: upperConfidenceBound(Q, N, c)
    selectAction=SelectAction(T, actionSpace, uCB)
    
    simulate=Simulate(rollOut, expand, selectAction, simulatePOMDP, update, epsilon, gamma, T)
    
    getB=lambda h: b
    outputAction=OutputAction(T, actionSpace)
    sampleStateFromBelief=lambda b: random.choices(list(b.keys()), weights=b.values())[0]
    numberOfSimulation=100
    search=Search(getB, sampleStateFromBelief, simulate, outputAction, numberOfSimulation)
    a=search(h)
    print(T)
    print(a)

if __name__=='__main__': 
    main()
