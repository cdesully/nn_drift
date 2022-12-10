import numpy
import pandas as pd
import matplotlib.pyplot as plt

import math
import random
from NN_Layer import NN_Layer
from NN_Node import NN_NodeConnect


class NN_Network:
    """ generated source for class NN_Network """
    layers = []
    trainingError = 0.0
    errReportWt = 0.005

    # scan = Scanner()

    #
    #      * Produce a new neural net with predetermined layer structures
    #      * @param layerCount
    #
    def __init__(self, layerCount: list):
        """ generated source for method __init__ """
        self.layers = []
        prevLayer = None
        i = 0
        while i < len(layerCount):
            if layerCount[i] > 0:
                newLayer = NN_Layer(layerCount[i], prevLayer)
                prevLayer = newLayer
                self.layers.append(newLayer)
            i += 1
        # NN_Network.scan = Scanner(System.in_)

    def predict(self, inputFeed: list):
        """ generated source for method predict """
        return self.feedForward(inputFeed)

    def trainEpoch(self, inputFeed: list, target: list):
        result = self.feedForward(inputFeed)
        errors = []
        entropyError = 0.0
        rmsError = 0.0
        for i in range(len(result)):
            errors.append(target[i] - result[i])
            entropyError += NN_Network.crossEntropy((target[i] + 1) / 2, (result[i] + 1) / 2)
            rmsError += math.pow(target[i] - result[i], 2)

        entropyError /= float(len(result))
        rmsError = math.sqrt(rmsError / float(len(result)))

        # self.trainingError = self.errReportWt * entropyError + (1.0 - self.errReportWt) * self.trainingError
        self.trainingError = self.errReportWt * rmsError + (1.0 - self.errReportWt) * self.trainingError
        self.backPropagate(errors)
        self.updateWeights()

    def feedForward(self, inputFeed: list):
        inputLayer = self.layers[0]  # first layer
        inputLayer.setOutput(inputFeed)
        for i in range(1, len(self.layers)):
            self.layers[i].setOutput()

        i = len(self.layers) - 1
        while i > -1:
            self.layers[i].resetShadowBias()
            i -= 1

        # avoid standardizing last layer because there is no forward layer weights to adjust
        i = len(self.layers) - 2
        while i > -1:
            self.layers[i].standardizeWeights()
            i -= 1

        i = len(self.layers) - 1
        while i > -1:
            self.layers[i].finalizeShadowBias()
            i -= 1

        # get nodes in last layer, should be the only output node
        return self.layers[len(self.layers) - 1].getOutput()

    def backPropagate(self, errors: list):
        self.layers[len(self.layers) - 1].setError(errors)
        i = len(self.layers) - 2
        while i > 0:
            self.layers[i].setError()  # hidden layers
            i -= 1

    def updateWeights(self):
        NN_NodeConnect.updateCount += 1
        self.layers[len(self.layers) - 1].updateWeights()

        i = len(self.layers) - 2
        while i > 0:
            self.layers[i].updateWeights()
            i -= 1

    def getLayerDeviations(self):
        collect = []
        for i in range(len(self.layers)):
            collect.append(self.layers[i].getLayerDev())
        return collect

    def getNodeShadowStats(self):
        collect = [None for x in range(len(self.layers))]
        i = 6
        while i < len(self.layers):
            collect[i] = self.layers[i].getNodeShadowStats()
            i += 1
        # collect[5] = this.layers.get(5).getNodeShadowStats();
        return collect

    @staticmethod
    def crossEntropy(target, result):
        return -(target * math.log(result) + (1 - target) * math.log(1 - result))

    @staticmethod
    def validateRMS(nn, inputs, target):
        hit = 0.0
        total = len(inputs)
        for i in range(len(inputs)):
            predict = nn.predict(inputs[i])
            error = 0.0
            for j in range(len(predict)):
                error += math.pow(target[i][j] - predict[j], 2)
            hit += math.sqrt(error / float(len(predict)))
        accuracy = hit / float(total)
        return accuracy

    @staticmethod
    def main():
        shadowStats = []
        trainingCount = 9000
        dInput = [[1, 1], [1, -1], [-1, 1], [-1, -1]]
        dTarget = [[-1], [1], [1], [-1]]
        nnStructure = [len(dInput[0]), 3, 3, len(dTarget[0])]
        NN_NodeConnect.batchSize = 8
        nn = NN_Network(nnStructure)

        df = pd.read_csv('C:/Users/chuck.desully/Downloads/AI/test/dataverse_files/mixed_0101_abrupto.csv')
        df.plot()
        plt.show()

        for epoch in range(400):
            pattern = int(random.random() * len(dTarget))
            # pattern = epoch % len(dTarget)
            nn.trainEpoch(dInput[pattern], dTarget[pattern])
            if epoch % 10 == 0:
                accuracy = NN_Network.validateRMS(nn, dInput, dTarget)
                print("{} RMS: {}".format(epoch, accuracy))

        for i in range(len(dTarget)):
            output = nn.predict(dInput[i])
            print(output[0])


if __name__ == '__main__':
    NN_Network.main()
