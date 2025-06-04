from data import trainingSet, testSet, outputDict
import random
import math

class MatrixNetwork:
    def __init__(self, input_size, hidden_size, output_size):
        self.inputWeightMatrix = [
            [random.uniform(-1, 1) for _ in range(hidden_size)]
            for _ in range(input_size)
        ]

        self.outputWeightMatrix = [
            [random.uniform(-1, 1) for _ in range(output_size)]
            for _ in range(hidden_size)
        ]
    
    def forward_pass(self, input_vector):        
        hidden_raw = matrix_multiply(input_vector, self.inputWeightMatrix)
        
        hidden_activated = sigmoid(hidden_raw)
        
        output_raw = matrix_multiply(hidden_activated, self.outputWeightMatrix)
        
        output_probabilities = softmax(output_raw)
        return output_probabilities

def matrix_multiply(vector, matrix):   
    num_outputs = len(matrix[0])    
    result = []

    for col_index in range(num_outputs):
        column_sum = 0
        for row_index in range(len(vector)):
            input_value = vector[row_index]
            weight = matrix[row_index][col_index]
            column_sum += input_value * weight
        result.append(column_sum)
    
    return result

def sigmoid(values):
    result = []
    
    for value in values:
        sigmoid_value = 1 / (1 + math.exp(-value))
        result.append(sigmoid_value)
    
    return result

def softmax(values):
    exp_values = []
    
    for value in values:
        exp_values.append(math.exp(value))
    
    sum_exp_values = sum(exp_values)
    result = []

    for exp_value in exp_values:
        probability = exp_value / sum_exp_values
        result.append(probability)
    
    return result

def calculate_mse(predicted, expected):
    n = len(predicted)
    squared_errors = [(predicted[i] - expected[i]) ** 2 for i in range(n)]
    mse = sum(squared_errors) / n
    return mse

def convert_example_to_vector(example):
    flat_vector = []
    for row in example:
        for pixel in row:
            flat_vector.append(pixel)
    return flat_vector

def train_with_gradient_descent(network, training_set, learning_rate=0.1, max_iterations=1000):
    
    def calculate_average_error():
        total_error = 0
        for example, label in training_set:
            input_vector = convert_example_to_vector(example)
            predicted_output = network.forward_pass(input_vector)
            expected_output = list(outputDict[label])
            error = calculate_mse(predicted_output, expected_output)
            total_error += error
        return total_error / len(training_set)
    
    for iteration in range(max_iterations):
        current_error = calculate_average_error()
        
        if current_error < 0.01:
            break

        for example, label in training_set:
            input_vector = convert_example_to_vector(example)
            expected_output = list(outputDict[label])
            
            hidden_raw = matrix_multiply(input_vector, network.inputWeightMatrix)
            hidden_activated = sigmoid(hidden_raw)
            output_raw = matrix_multiply(hidden_activated, network.outputWeightMatrix)
            output_probabilities = softmax(output_raw)

            output_error_gradients = []
            for i in range(len(output_probabilities)):
                error_gradient = 2 * (output_probabilities[i] - expected_output[i])
                output_error_gradients.append(error_gradient)

            for i in range(len(hidden_activated)):
                for j in range(len(output_error_gradients)):
                    gradient = hidden_activated[i] * output_error_gradients[j]
                    
                    network.outputWeightMatrix[i][j] -= learning_rate * gradient

            hidden_error_gradients = []
            for h in range(len(hidden_activated)):
                hidden_error = 0
                for j in range(len(output_error_gradients)):
                    contribution = output_error_gradients[j] * network.outputWeightMatrix[h][j]
                    hidden_error += contribution
                
                sigmoid_derivative = hidden_activated[h] * (1 - hidden_activated[h])
                hidden_error_gradients.append(hidden_error * sigmoid_derivative)

            for i in range(len(input_vector)):
                for h in range(len(hidden_error_gradients)):
                    gradient = input_vector[i] * hidden_error_gradients[h]
                    
                    network.inputWeightMatrix[i][h] -= learning_rate * gradient

        if iteration % 10 == 0:
            print(f"Iteratie {iteration}, Fout: {current_error:.4f}")
    
    print(f"Training voltooid na {iteration} iteraties.")