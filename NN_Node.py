import math
import random


def signum(x):
    if x > 0:
        return 1
    elif x < 0:
        return -1
    return 0


class NN_Node:
    """ generated source for class NN_Node """
    backConnections = []
    forwardConnections = []
    biasRef = None
    output = 0.0
    error = 0.0
    slope = 1.0
    intercept = 0.0
    avg = 0.0
    setAvg = 0.0
    var = 1.0
    zOutput = 0.0
    avgWt = 0.01
    shadowAvgTotal = 0.0
    shadowVarTotal = 0.0

    def __init__(self):
        """ generated source for method __init__ """
        self.backConnections = []
        self.forwardConnections = []
        # preload bias connection
        self.biasRef = NN_NodeConnect(self, None)
        self.backConnections.append(self.biasRef)

    def getBiasRef(self):
        """ generated source for method getBiasRef """
        return self.biasRef

    def addInputNodes(self, nodes: list):
        """ generated source for method addInputNodes """
        # preload bias connection
        i = 0
        while i < len(nodes):
            newConnection = NN_NodeConnect(self, nodes[i])
            self.backConnections.append(newConnection)
            nodes[i].forwardConnections.append(newConnection)
            i += 1

    def addOutputNodes(self, nodes):
        """ generated source for method addOutputNodes """
        i = 0
        while i < len(nodes):
            newConnection = NN_NodeConnect(nodes[i], self)
            self.forwardConnections.append(newConnection)
            nodes[i].backConnections.append(newConnection)
            i += 1

    def removeConnections(self):
        i = 0
        while i < len(self.backConnections):
            connect = self.backConnections[i]
            if connect is not None:
                connect.backNode.forwardConnections.remove(connect)
            i += 1
        self.backConnections.clear()

        i = 0
        while i < len(self.forwardConnections):
            connect = self.forwardConnections[i]
            connect.forwardNode.backConnections.remove(connect)
            i += 1
        self.forwardConnections.clear()

    def sumInputs(self):
        in_sum = 0.0
        i = 0
        while i < len(self.backConnections):
            in_sum += self.backConnections[i].getInput()
            i += 1
        return in_sum

    def sumError(self):
        error = 0.0
        i = 0
        while i < len(self.forwardConnections):
            error += self.forwardConnections[i].getError()
            i += 1
        return error

    def setOutput(self, out: float = None):
        if out is None:
            out = NN_Node.tanh(self.sumInputs())
        self.output = out
        self.avg = NN_Node.avgWt * self.output + (1 - NN_Node.avgWt) * self.avg
        self.var = NN_Node.avgWt * math.pow(self.output - self.avg, 2) + (1 - NN_Node.avgWt) * self.var
        self.zOutput = self.standardizeOut(out)

    def getOutput(self) -> float:
        return self.output - self.setAvg;

    def setNodeError(self, error: float = None):
        if error is None:
            self.error = self.sumError() * NN_Node.invTanh(self.output)
        else:
            self.error = error * NN_Node.invTanh(self.output)

    def getNodeError(self):
        return self.error

    def updateWeights(self):
        updated = False
        self.shadowAvgTotal = 0.0
        self.shadowVarTotal = 0.0
        maxi = -1
        maxwt = -math.inf

        mini = -1;
        minwt = math.inf
        for i in range(len(self.backConnections)):
            connect = self.backConnections[i]
            self.shadowAvgTotal += abs(connect.shadowAvg)
            self.shadowVarTotal += math.sqrt(connect.shadowVar)
            # always keep bias active
            if connect.backNode is not None:
                connect.freeze = False  # Math.random() > 0.5;
                # if (Math.abs(connect.shadowVar) > maxwt) {
                if abs(connect.shadowAvg) > maxwt:
                    # maxwt = abs(connect.shadowVar)
                    maxwt = abs(connect.shadowAvg)
                    maxi = i
                elif abs(connect.shadowAvg) < minwt:
                    minwt = abs(connect.shadowAvg)
                    mini = i
        if maxi > -1:
            self.backConnections[maxi].freeze = False  # True  # unfreeze for var*avg 62, unfreeze for var 71, avg 74
        if mini > -1:
            self.backConnections[mini].freeze = False
        for i in range(len(self.backConnections)):
            updated = self.backConnections[i].updateWt()
        return updated

    def standardizeWts(self):
        dev = math.sqrt(self.var)
        for i in range(len(self.forwardConnections)):
            forward = self.forwardConnections[i]
            f_bias = forward.forwardNode.getBiasRef()
            forward.shadowWt = forward.wt * dev
            forward.shadowAvg = NN_Node.avgWt * forward.shadowWt + (1 - NN_Node.avgWt) * forward.shadowAvg
            forward.shadowVar = NN_Node.avgWt * math.pow(forward.shadowWt - forward.shadowAvg, 2) + (
                        1 - NN_Node.avgWt) * forward.shadowVar
            f_bias.shadowWt += forward.wt * (self.avg - self.setAvg)  # initialized to wt in layer

    def standardizeOut(self, x):
        dev = math.sqrt(self.var)
        if dev > 0.0000001:
            self.slope = 1.0 / dev
            self.intercept = -self.avg / dev
        else:
            self.slope = 1.0 / 0.0000001
            self.intercept = -self.avg / 0.0000001
        return self.slope * x + self.intercept

    def getAvgDev(self):
        devSum = 0.0
        for i in range(len(self.backConnections)):
            devSum += math.sqrt(self.backConnections[i].shadowVar)
        return devSum / float(len(self.backConnections))

    @staticmethod
    def tanh(val):
        if val > 6:
            return 0.99999
        elif val < -6:
            return -0.99999
        return math.tanh(val)

    @staticmethod
    def invTanh(val):
        if val > 6 or val < -6:
            return 0.00002

        return 1 - math.pow(math.tanh(val), 2)


class NN_NodeConnect:
    """ generated source for class NN_NodeConnect """
    updateCount = 0

    # controlled by network training update
    batchSize = 10
    lrUp = 1.2
    lrDn = 0.5
    maxLr = 0.1
    minLr = 0.00001
    maxWt = 10.0
    minWt = -10.0
    forwardNode = None
    backNode = None
    wt = 0.0
    shadowWt = 0.0
    shadowAvg = 0.0
    shadowVar = 0.0
    lr = 0.1
    prevWtChange = 0.0
    prevErr = 0.01
    errSum = 0.0
    freeze = False
    debug = 0.1

    def __init__(self, forward: NN_Node, back: NN_Node = None):
        """ generated source for method __init__ """
        self.forwardNode = forward
        if back is not None:
            self.backNode = back
        NN_NodeConnect.debug += 0.05
        self.wt = random.random() - 0.5
        #  -0.5 to 0.5
        self.shadowWt = self.wt
        self.shadowAvg = self.shadowWt
        self.shadowVar = 1.0

    #
    #      * Pull input from a previous node, modify by weight and return
    #      * @return
    #
    def getInput(self):
        """ generated source for method getInput """
        if self.backNode is None:
            return self.wt
        return self.backNode.getOutput() * self.wt

    #
    #      * Pull error from the current node, modify by weight and return
    #      * @return
    #
    def getError(self):
        """ generated source for method getError """
        return self.forwardNode.getNodeError() * self.wt

    def updateWt(self):
        """ generated source for method updateWt """
        inputx = 1.0
        if self.backNode is not None:
            inputx = self.backNode.getOutput() + self.backNode.setAvg
        self.errSum += self.forwardNode.getNodeError() * signum(inputx) * math.sqrt(abs(inputx))

        if NN_NodeConnect.updateCount % NN_NodeConnect.batchSize == 0:
            if self.freeze and self.backNode is not None:
                self.prevErr = self.errSum
                self.errSum = 0.0
                return False
            # if NN_NodeConnect.updateCount % (NN_NodeConnect.batchSize * 4) == 0:
            #     if self.backNode is None:
            #         self.wt = self.shadowWt
            #         self.forwardNode.setAvg = self.forwardNode.avg
            #     self.prevErr = 0.0
            #     self.errSum = 0.0
            #     return True

            wtAdjustment2 = math.pow(1.5 - abs(self.shadowAvg) / self.forwardNode.shadowAvgTotal, 1)
            wtAdjustment = wtAdjustment2  # math.sqrt(wtAdjustment1 * wtAdjustment2)
            errSign = signum(self.errSum)
            prevErrSign = signum(self.prevErr)
            errDelta = errSign * prevErrSign

            if errDelta > 0:
                self.lr = min(NN_NodeConnect.maxLr, self.lr * NN_NodeConnect.lrUp * wtAdjustment)
            elif errDelta < 0:
                self.lr = max(NN_NodeConnect.minLr, self.lr * NN_NodeConnect.lrDn)
                if abs(self.errSum) > abs(self.prevErr):
                    self.wt -= self.prevWtChange * 0.5
                    #  (1 - Math.abs(this.prevErr)/Math.abs(this.errSum));
                self.prevErr = 0.0
                self.errSum = 0.0
                return True
            self.prevWtChange = errSign * self.lr
            # constrain weights to a defined magnitude
            if self.wt > NN_NodeConnect.maxWt and self.prevWtChange > 0 \
                    or self.wt < NN_NodeConnect.minWt and self.prevWtChange < 0:
                self.prevWtChange = 0.0
            else:
                self.wt += self.prevWtChange
            self.prevErr = self.errSum
            self.errSum = 0.0
            return True
        return False
