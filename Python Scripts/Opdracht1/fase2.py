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

def train_matrix_network(network, training_set, output_dict, learning_rate=0.1, max_iterations=1000):
    def calculate_average_error():
        total_error = 0
        for example, label in training_set:
            input_vector = convert_example_to_vector(example)
            
            predicted_output = network.forward_pass(input_vector)
            
            expected_output = list(output_dict[label])
            
            error = calculate_mse(predicted_output, expected_output)
            total_error += error
        
        return total_error / len(training_set)
    
    print("Starting training...")
    current_error = calculate_average_error()
    print(f"Initial error: {current_error:.4f}")

    iteration = 0
    
    while iteration < max_iterations and current_error > 0.01:  # Stop bij 1% fout
        best_improvement = 0
        best_change = None  # (matrix_type, row, col, delta)
        
        # Probeer alle gewichten in inputWeightMatrix
        for row in range(len(network.inputWeightMatrix)):
            for col in range(len(network.inputWeightMatrix[row])):
                original_weight = network.inputWeightMatrix[row][col]
                
                # Probeer +learning_rate
                network.inputWeightMatrix[row][col] = original_weight + learning_rate
                error_after_increase = calculate_average_error()
                
                # Probeer -learning_rate  
                network.inputWeightMatrix[row][col] = original_weight - learning_rate
                error_after_decrease = calculate_average_error()
                
                # Herstel originele waarde
                network.inputWeightMatrix[row][col] = original_weight
                
                # Check welke verandering het beste is
                if error_after_increase < current_error:
                    improvement = current_error - error_after_increase
                    if improvement > best_improvement:
                        best_improvement = improvement
                        best_change = ("input", row, col, learning_rate)
                
                if error_after_decrease < current_error:
                    improvement = current_error - error_after_decrease
                    if improvement > best_improvement:
                        best_improvement = improvement
                        best_change = ("input", row, col, -learning_rate)
        
        # Hetzelfde voor outputWeightMatrix
        for row in range(len(network.outputWeightMatrix)):
            for col in range(len(network.outputWeightMatrix[row])):
                original_weight = network.outputWeightMatrix[row][col]
                
                network.outputWeightMatrix[row][col] = original_weight + learning_rate
                error_after_increase = calculate_average_error()
                
                network.outputWeightMatrix[row][col] = original_weight - learning_rate
                error_after_decrease = calculate_average_error()
                
                network.outputWeightMatrix[row][col] = original_weight
                
                if error_after_increase < current_error:
                    improvement = current_error - error_after_increase
                    if improvement > best_improvement:
                        best_improvement = improvement
                        best_change = ("output", row, col, learning_rate)
                
                if error_after_decrease < current_error:
                    improvement = current_error - error_after_decrease
                    if improvement > best_improvement:
                        best_improvement = improvement
                        best_change = ("output", row, col, -learning_rate)
        
        if best_change is not None:
            matrix_type, row, col, delta = best_change
            if matrix_type == "input":
                network.inputWeightMatrix[row][col] += delta
            else:
                network.outputWeightMatrix[row][col] += delta
            current_error -= best_improvement
        else:
            learning_rate *= 0.9
            if learning_rate < 1e-6:
                print("Learning rate te klein, training stopt.")
                break
        
        if iteration % 10 == 0:
            print(f"Iteratie {iteration}, Fout: {current_error:.4f}")
        
        iteration += 1
    
    print(f"Training voltooid na {iteration} iteraties. Eindfout: {current_error:.4f}")
    
    return current_error

def convert_example_to_vector(example):
    flat_vector = []
    for row in example:
        for pixel in row:
            flat_vector.append(pixel)
    return flat_vector