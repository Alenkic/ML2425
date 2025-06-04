import random
import math
from data import trainingSet, testSet, outputDict


class Node:
    nodeValue = int
    layer = int

    def __init__(self, layer, value):
        self.layer = layer
        self.nodeValue = value

    def getValue(self):
        return self.nodeValue

    def getLayer(self):
        return self.layer

    def addValue(self, value):
        self.nodeValue += value

    def setValue(self, value):
        self.nodeValue = value


class Link:
    weight = float
    originNode = Node
    destinationNode = Node

    def __init__(self, originNode, destinationNode, weight):
        self.originNode = originNode
        self.destinationNode = destinationNode
        self.weight = weight

    def getOriginNode(self):
        return self.originNode

    def getDestinationNode(self):
        return self.destinationNode

    def getWeight(self):
        return self.weight


class Network:
    nodes = []
    links = []

    def __init__(self):
        self.nodes = []
        self.links = []

    def addNode(self, node):
        self.nodes.append(node)

    def addLink(self, link):
        self.links.append(link)

    def getNodes(self):
        return self.nodes

    def getLinks(self):
        return self.links

    def forwardPass(self):
        for i in range(9, len(self.nodes)):
            self.nodes[i].setValue(0)

        for link in self.links:
            origin = link.getOriginNode()
            destination = link.getDestinationNode()
            weight = link.getWeight()

            weighted_value = origin.getValue() * weight
            destination.addValue(weighted_value)

        rawOutputs = [self.nodes[9].getValue(), self.nodes[10].getValue()]
        expValues = [math.exp(val) for val in rawOutputs]
        sumExpValues = sum(expValues)
        finalOutputs = [exp / sumExpValues for exp in expValues]

        self.nodes[9].setValue(finalOutputs[0])
        self.nodes[10].setValue(finalOutputs[1])

        return finalOutputs


def calculateMSE(predicted, expected):
    n = len(predicted)
    squaredErrors = [(predicted[i] - expected[i]) ** 2 for i in range(n)]
    mse = sum(squaredErrors) / n
    return mse


def train_network(network, training_set, output_dict):
    def calculate_average_error():
        total_error = 0

        for example, label in training_set:
            flat_input = [pixel for row in example for pixel in row]

            for i, value in enumerate(flat_input):
                network.nodes[i].setValue(value)

            predicted_output = network.forwardPass()

            expected_output = output_dict[label]

            error = calculateMSE(predicted_output, expected_output)
            total_error += error

        average_error = total_error / len(training_set)
        return average_error

    learning_rate = 0.1
    max_iterations = 10000
    target_error = 0.01

    iteration = 0
    current_error = float("inf")

    while iteration < max_iterations and current_error > target_error:
        current_error = calculate_average_error()

        if iteration % 100 == 0:
            print(f"Iteratie {iteration}, Fout: {current_error}")

        best_link = None
        best_change = 0
        best_error_improvement = 0

        for link_index, link in enumerate(network.links):
            original_weight = link.weight

            link.weight = original_weight + learning_rate
            error_after_increase = calculate_average_error()

            link.weight = original_weight - learning_rate
            error_after_decrease = calculate_average_error()

            link.weight = original_weight

            if (
                error_after_increase < error_after_decrease
                and error_after_increase < current_error
            ):
                improvement = current_error - error_after_increase
                if improvement > best_error_improvement:
                    best_link = link
                    best_change = learning_rate
                    best_error_improvement = improvement
            elif (
                error_after_decrease < error_after_increase
                and error_after_decrease < current_error
            ):
                improvement = current_error - error_after_decrease
                if improvement > best_error_improvement:
                    best_link = link
                    best_change = -learning_rate
                    best_error_improvement = improvement

        if best_link is not None:
            best_link.weight += best_change
            current_error -= best_error_improvement
        else:
            learning_rate *= 0.9
            if learning_rate < 1e-6:
                print("Learning rate te klein, training stopt.")
                break

        iteration += 1

    print(f"Training voltooid na {iteration} iteraties. Eindfout: {current_error}")
    return current_error