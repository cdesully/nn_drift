import math
from NN_Node import NN_Node


class NN_Layer:
    """ generated source for class NN_Layer """
    prevLayer = None

    # previous layer for reference when adding a new node
    nextLayer = None

    # next layer for reference for node connections
    nodes = []

    def __init__(self, count, prevLayer=None):
        """ generated source for method __init__ """
        self.nodes = []
        if prevLayer is not None:
            self.prevLayer = prevLayer
            prevLayer.nextLayer = self

        for i in range(count):
            self.addNode(NN_Node())

    def addNode(self, node: NN_Node):
        self.nodes.append(node)

        if self.prevLayer is not None:
            node.addInputNodes(self.prevLayer.nodes)
            if self.prevLayer.prevLayer is not None:
                # node.addInputNodes(this.prevLayer.prevLayer.nodes);
                pass
        if self.nextLayer is not None:
            node.addOutputNodes(self.nextLayer.nodes)

    def removeNode(self, node):
        self.nodes.remove(node)
        node.removeConnections()

    def setOutput(self, inputs: list = None):
        if inputs is None:
            for i in range(len(self.nodes)):
                self.nodes[i].setOutput()
        elif len(inputs) != len(self.nodes):
            raise Exception("input length mismatch")
        else:
            for i in range(len(self.nodes)):
                self.nodes[i].setOutput(inputs[i])

    def setError(self, errors: list = None):
        if errors is not None:
            if len(errors) != len(self.nodes):
                raise Exception("error length mismatch")
            for i in range(len(self.nodes)):
                self.nodes[i].setNodeError(errors[i])
        else:
            for i in range(len(self.nodes)):
                self.nodes[i].setNodeError()

    def updateWeights(self):
        for i in range(len(self.nodes)):
            self.nodes[i].updateWeights()

    def standardizeWeights(self):
        for i in range(len(self.nodes)):
            self.nodes[i].standardizeWts()

    def getOutput(self):
        outputs = [];
        for i in range(len(self.nodes)):
            outputs.append(self.nodes[i].getOutput())
        return outputs

    def getLayerDev(self):
        layerDevSum = 0.0;
        for i in range(len(self.nodes)):
            layerDevSum += self.nodes[i].getAvgDev()
        return layerDevSum / float(len(self.nodes))

    def getNodeShadowStats(self):
        collect = []
        for i in range(len(self.nodes)):
            collect.append(self.nodes[i].getShadowStats())
        return collect

    def resetShadowBias(self):
        for i in range(len(self.nodes)):
            bias = self.nodes[i].getBiasRef()
            bias.shadowWt = bias.wt

    def finalizeShadowBias(self):
        for i in range(len(self.nodes)):
            bias = self.nodes[i].getBiasRef()
            bias.shadowAvg = NN_Node.avgWt * bias.shadowWt + (1 - NN_Node.avgWt) * bias.shadowAvg
            bias.shadowVar = NN_Node.avgWt * math.pow(bias.shadowWt - bias.shadowAvg, 2) + (
                        1 - NN_Node.avgWt) * bias.shadowVar
