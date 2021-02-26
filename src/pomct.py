
import numpy as np

class Search(object):
    
    def __init__(self, getB, sampleStateFromBelief, simulate, outputAction, numberOfSimulation):
        self.getB=getB
        self.sampleStateFromBelief=sampleStateFromBelief
        self.simulate=simulate
        self.outputAction=outputAction
        self.numberOfSimulation=numberOfSimulation

    def __call__(self, h):
        b=self.getB(h)
        for simulation in range(self.numberOfSimulation):
            s=self.sampleStateFromBelief(b)
            R=self.simulate(s, h, 0)
        a=self.outputAction(h)
        return a
    
class Simulate(object):
    
    def __init__(self, rollOut, expand, selectAction, simulatePOMDP, update, epsilon, gamma, T):
        self.rollOut=rollOut
        self.expand=expand
        self.selectAction=selectAction
        self.simulatePOMDP=simulatePOMDP
        self.update=update
        self.epsilon=epsilon
        self.gamma=gamma
        self.T=T
        
    def __call__(self, s, h, depth):
        if self.gamma ** depth < self.epsilon:
            return 0
        if h not in self.T:
            self.T=self.expand(h)
            return self.rollOut(s, h, depth)
        a=self.selectAction(h)
        sPrime, r, o=self.simulatePOMDP(s, a)
        hao=h + (a, o)
        R=r+self.gamma*self.__call__(s, hao, depth+1)
        self.T=self.update(self.T, h, a, R)
        return R
 

class RollOut(object):
    
    def __init__(self, rollOutPolicy, simulatePOMDP, epsilon, gamma):
        self.rollOutPolicy=rollOutPolicy
        self.simulatePOMDP=simulatePOMDP
        self.epsilon=epsilon
        self.gamma=gamma
        
    def __call__(self, s, h, depth):
        if self.gamma ** depth < self.epsilon:
            return 0
        a=self.rollOutPolicy(h)
        sPrime, r, o=self.simulatePOMDP(s, a)
        hao=h+(a,o)
        return r+self.gamma*self.__call__(sPrime, hao, depth+1)


class OutputAction(object):
    
    def __init__(self, T, actionSpace):
        self.T=T
        self.actionSpace=actionSpace
        
    def __call__(self, h):
        vDict={a:self.T.get(h+(a,), {}).get('V', -np.inf) for a in self.actionSpace}
        a=max(vDict, key=vDict.get)
        return a
       

class Expand(object):
    
    def __init__(self, T, actionSpace):
        self.T=T
        self.actionSpace=actionSpace
        
    def __call__(self, h):
        self.T[h]={'N':0}
        for a in self.actionSpace:
            ha=h+(a,)
            self.T[ha]={'N':0, 'V':0}
        return self.T
    

def upperConfidenceBound(Q, N, c):
    t=sum(N.values())+1
    candidates=[action for action in N.keys() if N[action]==0]
    if len(candidates)==0:
        QNBalance={action: Q[action]+c*np.sqrt(np.log(t)/N[action]) for action in Q.keys()}
        maxQNBalance=max(QNBalance.values())
        candidates=[action for action in Q.keys() if QNBalance[action]==maxQNBalance]
    action=np.random.choice(candidates)
    return action



class SelectAction(object):
    
    def __init__(self, T, actionSpace, uCB):
        self.T=T
        self.actionSpace=actionSpace
        self.uCB=uCB
        
    def __call__(self, h):
        Q={a:self.T.get(h+(a,), {}).get('V', -np.inf) for a in self.actionSpace}
        N={a:self.T.get(h+(a,), {}).get('N', np.finfo(float).eps) for a in self.actionSpace}
        action=self.uCB(Q, N)
        return action
       


def update(T, h, a, reward):
    ha=h+(a,)   
    T[h]['N']=T[h]['N']+1
    T[ha]['N']=T[ha]['N']+1
    T[ha]['V']=T[ha]['V']+(reward-T[ha]['V'])/T[ha]['N']
    return T
        
            
  
