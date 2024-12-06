import numpy as np
import os
from random import sample

anomaly_vid =["Abuse","Arrest","Arson","Assault","Burglary","Explosion","Fighting","RoadAccidents","Robbery","Shooting","Shoplifting","Stealing","Vandalism"]
normal_vid = ["Normal_Videos_event","Testing_Normal_Videos_Anomaly","Training_Normal_Videos_Anaomaly"]
ano_data = []
normal_data = []
train_ano = []
train_norm = []
test_ano = []
test_norm = []
ano_width = []
norm_width = []


#parameters
n_input = 4096
n_nodes_hl1 = 512
n_nodes_hl2 = 32
n_classes = 1
batch_size = 32
dropout_rate = 0.6

# Initialize weights and biases
def initialize_weights(shape):
	return np.random.normal(size=shape)

weights = {
	"hidden_1": initialize_weights((n_input, n_nodes_hl1)),
	"hidden_2": initialize_weights((n_nodes_hl1, n_nodes_hl2)),
	"output": initialize_weights((n_nodes_hl2, n_classes))
}
biases = {
	"hidden_1": initialize_weights((n_nodes_hl1,)),
	"hidden_2": initialize_weights((n_nodes_hl2,)),
	"output": initialize_weights((n_classes,))
}

# Activation functions
def relu(x):
	return np.maximum(0, x)

def sigmoid(x):
	return 1 / (1 + np.exp(-x))

# Dropout function
def dropout(layer, rate):
	mask = np.random.binomial(1, 1 - rate, size=layer.shape)
	return layer * mask

# Forward propagation
def forward_propagation(data):
	# Layer 1
	l1 = relu(np.dot(data, weights["hidden_1"]) + biases["hidden_1"])
	l1 = dropout(l1, dropout_rate)

	# Layer 2
	l2 = np.dot(l1, weights["hidden_2"]) + biases["hidden_2"]
	l2 = dropout(l2, dropout_rate)

	# Output
	output = sigmoid(np.dot(l2, weights["output"]) + biases["output"])
	return output

# Cost function and backpropagation
def compute_cost(predictions, labels):
	l2_reg = 0.5 * (
		np.sum(weights["hidden_1"]**2)
		+ np.sum(weights["hidden_2"]**2)
		+ np.sum(weights["output"]**2)
	)
	loss = np.mean((predictions - labels)**2) + l2_reg
	return loss

#back prop
def backward_propagation(data, labels, learning_rate=0.001):

    l1_input = np.dot(data, weights["hidden_1"]) + biases["hidden_1"]
    l1 = relu(l1_input)
    l1 = dropout(l1, dropout_rate)

    l2_input = np.dot(l1, weights["hidden_2"]) + biases["hidden_2"]
    l2 = relu(l2_input)
    l2 = dropout(l2, dropout_rate)

    predictions = sigmoid(np.dot(l2, weights["output"]) + biases["output"])  


    output_error = predictions - labels  
    grad_output_weights = np.dot(l2.T, output_error) / data.shape[0] 
    grad_output_biases = np.mean(output_error, axis=0)

    l2_error = np.dot(output_error, (weights["output"].T)) 
    l2_error *= (l2 > 0).astype(float)
    grad_hidden_2_weights = np.dot(l1.T, l2_error) / data.shape[0] 
    grad_hidden_2_biases = np.mean(l2_error, axis=0) 

    l1_error = np.dot(l2_error, weights["hidden_2"].T) 
    l1_error *= (l1 > 0).astype(float) 
    grad_hidden_1_weights = np.dot(data.T, l1_error) / data.shape[0] 
    grad_hidden_1_biases = np.mean(l1_error, axis=0)

    # Update weights and biases
    weights["output"] -= learning_rate * grad_output_weights
    biases["output"] -= learning_rate * grad_output_biases

    weights["hidden_2"] -= learning_rate * grad_hidden_2_weights
    biases["hidden_2"] -= learning_rate * grad_hidden_2_biases

    weights["hidden_1"] -= learning_rate * grad_hidden_1_weights
    biases["hidden_1"] -= learning_rate * grad_hidden_1_biases




# Accuracy calculation
def calculate_accuracy(predictions, labels):
	pred_labels = (predictions >= 0.5).astype(int)
	return np.mean(pred_labels == labels) * 100

# Training function
def train_neural_network(train_x, train_y, test_x, test_y, epochs=10, learning_rate=0.001):
	for epoch in range(epochs):
		epoch_loss = 0
		for i in range(0, len(train_x), batch_size):
			batch_x = np.array(train_x[i:i+batch_size])
			batch_y = np.array(train_y[i:i+batch_size])
			
			# Forward and backward pass
			predictions = forward_propagation(batch_x)
			loss = compute_cost(predictions, batch_y)
			#backward_propagation(batch_x, batch_y, learning_rate)
			epoch_loss += loss
		
		print(f"Epoch {epoch + 1}/{epochs}, Loss: {epoch_loss:.4f}")

	# Evaluate accuracy on test set
	test_predictions = forward_propagation(test_x)
	accuracy = calculate_accuracy(test_predictions, test_y)
	print(f"Test Accuracy: {accuracy:.2f}%")

for folder in anomaly_vid:
	path = "./FOutput/" + folder
	files = os.listdir(path)
	if len(files) == 0:
		continue
	for fl in files:
		with open(path +"/"+fl,'r') as f:
			ano_width.append(float(f.readline().split("\n")[0]))
			for cnt,line in enumerate(f):
				if cnt == 32:
					break
				line = line.split("\n")[0]
				ano_data.append(list(map(float,line.split(","))))
for folder in normal_vid:
	path = "./FOutput/" + folder
	files = os.listdir(path)
	if len(files) == 0:
		continue
	for fl in files:
		with open(path +"/"+fl,'r') as f:
			norm_width.append(float(f.readline().split("\n")[0]))
			for cnt,line in enumerate(f):
				if cnt == 32:
					break
				line = line.split("\n")[0]
				normal_data.append(list(map(float,line.split(","))))

train_x = ano_data[0:4000]
train_y = normal_data[0:4000]
test_x = ano_data[4000:]
test_y = normal_data[4000:]
# Train the model
train_neural_network(train_x, train_y, test_x, test_y)
