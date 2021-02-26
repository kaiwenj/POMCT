import sys
sys.path.append('../src/')

import unittest
from ddt import ddt, data, unpack
import POMCP as targetCode


@ddt
class TestOutputAction(unittest.TestCase):
    
    def setUp(self):
        self.T={(0, 1, 0): {'N': 25, 'V': 12},
                (0, 1, 1): {'N': 2, 'V': 17},
                (0, 1, 2): {'N': 23, 'V': 15},
                (0, 2, 0): {'N': 31, 'V': 16},
                (0, 1): {'N': 31, 'V': -1},
                (0, 2): {'N': 3, 'V': 1},
                (0, 3): {'N': 32, 'V': 19}}
        self.actionSpace=[0, 1, 2]
        
    @data(((0, 1), 1))
    @unpack
    def testOutputActionOrdinary(self, h, expectedResult):
        outputAction=targetCode.OutputAction(self.T, self.actionSpace)
        calculatedResult=outputAction(h)
        self.assertAlmostEqual(calculatedResult, expectedResult)
        
    @data(((0,), 2))
    @unpack
    def testOutputActionOutOfActionSpace(self, h, expectedResult):
        outputAction=targetCode.OutputAction(self.T, self.actionSpace)
        calculatedResult=outputAction(h)
        self.assertAlmostEqual(calculatedResult, expectedResult)
        
    def tearDown(self):
        pass

@ddt
class TestExpand(unittest.TestCase):

    def setUp(self):
        self.T={(0, 1, 0): {'N': 25, 'V': 12},
                (0, 2, 0): {'N': 31, 'V': 16},
                (0, 1): {'N': 31, 'V': -1},
                (0, 3): {'N': 3, 'V': 1}}
        self.actionSpace=[0, 1, 2]
        
    @data(((0, 3), {(0, 1, 0): {'N': 25, 'V': 12},
                (0, 2, 0): {'N': 31, 'V': 16},
                (0, 1): {'N': 31, 'V': -1},
                (0, 3): {'N':0},
                (0, 3, 0): {'N': 0, 'V': 0},
                (0, 3, 1): {'N': 0, 'V': 0},
                (0, 3, 2): {'N': 0, 'V': 0}}))
    @unpack
    def testExpandBranchAlreadyIn(self, h, expectedResult):
        expand=targetCode.Expand(self.T, self.actionSpace)
        calculatedResult=expand(h)
        self.assertEqual(calculatedResult, expectedResult)
        
    @data(((0, 1), {(0, 2, 0): {'N': 31, 'V': 16},
                (0, 1): {'N':0},
                (0, 3): {'N': 3, 'V': 1},
                (0, 1, 0): {'N': 0, 'V': 0},
                (0, 1, 1): {'N': 0, 'V': 0},
                (0, 1, 2): {'N': 0, 'V': 0}}))
    @unpack
    def testExpandLeafAlreadyIn(self, h, expectedResult):
        expand=targetCode.Expand(self.T, self.actionSpace)
        calculatedResult=expand(h)
        self.assertEqual(calculatedResult, expectedResult)
    
    @data(((0, 1, 0), {(0, 1, 0): {'N':0},
                (0, 2, 0): {'N': 31, 'V': 16},
                (0, 1): {'N': 31, 'V': -1},
                (0, 3): {'N': 3, 'V': 1},
                (0, 1, 0, 0): {'N': 0, 'V': 0},
                (0, 1, 0, 1): {'N': 0, 'V': 0},
                (0, 1, 0, 2): {'N': 0, 'V': 0}}))
    @unpack
    def testExpandOrdinaryCase(self, h, expectedResult):
        expand=targetCode.Expand(self.T, self.actionSpace)
        calculatedResult=expand(h)
        self.assertEqual(calculatedResult, expectedResult)
        
        
    def tearDown(self):
        pass

@ddt
class TestSelectAction(unittest.TestCase):
    
    def setUp(self):
        self.T={(0, 1, 0): {'N': 25, 'V': 12},
                (0, 1, 1): {'N': 2, 'V': 17},
                (0, 1, 2): {'N': 23, 'V': 15},
                (0, 2, 0): {'N': 31, 'V': 16},
                (0, 1): {'N': 31, 'V': -1},
                (0, 2): {'N': 3, 'V': 1},
                (0, 3): {'N': 32, 'V': 19}}
        self.actionSpace=[0, 1, 2]
        
    @data(((0, 1), lambda Q, N: max(Q, key=Q.get), 1))
    @unpack
    def testSelectActionLargestQ(self, h, uCB, expectedResult):
        selectAction=targetCode.SelectAction(self.T, self.actionSpace, uCB)
        calculatedResult=selectAction(h)
        self.assertAlmostEqual(calculatedResult, expectedResult)
        
    @data(((0,), lambda Q, N: min(Q, key=Q.get), 0))
    @unpack
    def testSelectActionLowestQ(self, h, uCB, expectedResult):
        selectAction=targetCode.SelectAction(self.T, self.actionSpace, uCB)
        calculatedResult=selectAction(h)
        self.assertAlmostEqual(calculatedResult, expectedResult)
        
    @data(((0, 1), lambda Q, N: max(N, key=N.get), 0))
    @unpack
    def testSelectActionLargestN(self, h, uCB, expectedResult):
        selectAction=targetCode.SelectAction(self.T, self.actionSpace, uCB)
        calculatedResult=selectAction(h)
        self.assertAlmostEqual(calculatedResult, expectedResult)
        
    def tearDown(self):
        pass


@ddt
class TestUpdate(unittest.TestCase):

    def setUp(self):
        self.T={(0, 1, 0): {'N': 2, 'V': 12},
                (0, 1, 1): {'N': 0, 'V': 0},
                (0, 1): {'N': 31, 'V': -1},
                (0, 3): {'N': 3, 'V': 1}}
        
    @data(((0, 1), 0, 15, {(0, 1, 0): {'N': 3, 'V': 13},
                (0, 1, 1): {'N': 0, 'V': 0},
                (0, 1): {'N': 32, 'V': -1},
                (0, 3): {'N': 3, 'V': 1}}))
    @unpack
    def testUpdateOrdinary(self, h, a, reward, expectedResult):
        calculatedResult=targetCode.update(self.T, h, a, reward)
        self.assertEqual(calculatedResult, expectedResult)
    
    @data(((0, 1), 1, 15, {(0, 1, 0): {'N': 2, 'V': 12},
                (0, 1, 1): {'N': 1, 'V': 15},
                (0, 1): {'N': 32, 'V': -1},
                (0, 3): {'N': 3, 'V': 1}}))
    @unpack
    def testUpdateZeroVisitTime(self, h, a, reward, expectedResult):
        calculatedResult=targetCode.update(self.T, h, a, reward)
        self.assertEqual(calculatedResult, expectedResult)   
        
    def tearDown(self):
        pass


@ddt
class TestRollOut(unittest.TestCase):
    
    def setUp(self):
        self.rollOutPolicy=lambda h: h[0]
        
    @data((2, (0, 1), 0, lambda s, a: (s, a+10, a+1), 0.5, 0.3, 15))
    @unpack
    def testRollOutOrdinary(self, s, h, depth, simulatePOMDP, gamma, epsilon, expectedResult):
        rollOut=targetCode.RollOut(self.rollOutPolicy, simulatePOMDP, epsilon, gamma)
        calculatedResult=rollOut(s, h, depth)
        self.assertAlmostEqual(calculatedResult, expectedResult)
    
    @data((2, (0, 1), 1, lambda s, a: (s, a+10, a+1), 0.5, 0.3, 10))
    @unpack
    def testRollOutFromNonZeroDepth(self, s, h, depth, simulatePOMDP, gamma, epsilon, expectedResult):
        rollOut=targetCode.RollOut(self.rollOutPolicy, simulatePOMDP, epsilon, gamma)
        calculatedResult=rollOut(s, h, depth)
        self.assertAlmostEqual(calculatedResult, expectedResult)
        
    @data((2, (0, 1), 0, lambda s, a: (s, 10, 1), 0.4, 0.1, 15.6))
    @unpack
    def testRollOutChangeGamma(self, s, h, depth, simulatePOMDP, gamma, epsilon, expectedResult):
        rollOut=targetCode.RollOut(self.rollOutPolicy, simulatePOMDP, epsilon, gamma)
        calculatedResult=rollOut(s, h, depth)
        self.assertAlmostEqual(calculatedResult, expectedResult)
        
    @data((2, (0, 1), 0, lambda s, a: (s, 10, 1), 0.5, 0.1, 18.75))
    @unpack
    def testRollOutChangeEpsilon(self, s, h, depth, simulatePOMDP, gamma, epsilon, expectedResult):
        rollOut=targetCode.RollOut(self.rollOutPolicy, simulatePOMDP, epsilon, gamma)
        calculatedResult=rollOut(s, h, depth)
        self.assertAlmostEqual(calculatedResult, expectedResult)
        
    @data((2, (0, 1), 0, lambda h: h[-1], lambda s, a: (s, a+10, a+1), 0.5, 0.3, 17))
    @unpack
    def testRollOutChangeActionReward(self, s, h, depth, rollOutPolicy, simulatePOMDP, gamma, epsilon, expectedResult):
        rollOut=targetCode.RollOut(rollOutPolicy, simulatePOMDP, epsilon, gamma)
        calculatedResult=rollOut(s, h, depth)
        self.assertAlmostEqual(calculatedResult, expectedResult)
        
    def tearDown(self):
        pass

@ddt
class TestUCB(unittest.TestCase):
    
    @data(({0:0, 1:0, 2:0, 3:0, 4:0, 5:0}, {0:0, 1:0, 2:0, 3:0, 4:0, 5:0}, 1, 1, 1/6), #Q from 0
          ({0:124, 2:1000, 3:200, 4:0}, {0:1, 2:1, 3:1, 4:0}, 0, 4, 1, 1), #zero first, always 4
          ({0:124, 2:1000, 3:200, 4:0}, {0:1, 2:1, 3:1, 4:0}, 0, 2, 0, 1),
          ({2:125, 3:125, 5:20, 4:50}, {2:125, 3:125, 5:125, 4:125}, 0.5, 2, 0.5), #random tie
          ({2:125, 3:125, 5:20, 4:50}, {2:125, 3:125, 5:125, 4:125}, 0.5, 5, 0, 1),
          ({0:124, 2:1000, 3:200, 4:400}, {0:2, 2:300, 3:400, 4:500}, 0, 2, 1, 1), #no penalty
          ({0:1, 5:2, 10:3}, {0:1000, 5:2000, 10:3000}, 400, 0, 1, 1),
          ({0:1, 5:2, 10:3}, {0:1000, 5:2000, 10:3000}, 400, 5, 0, 1)) #large penalty
    @unpack
    def test_UCB(self, Q, N, c, prime, probability, threshold=0.95, maxiter=10000):
        randomResult=[targetCode.upperConfidenceBound(Q,N,c) for i in range(maxiter)]
        countResult=[1 for i in randomResult if i==prime]
        calculatedResult=sum(countResult)
        expectedResult=maxiter*probability
        self.assertTrue(expectedResult-(1-threshold)/2*maxiter <= calculatedResult <= expectedResult+(1-threshold)/2*maxiter)
        
    def tearDown(self):
        pass
 
if __name__ == '__main__':
	unittest.main(verbosity=2)
