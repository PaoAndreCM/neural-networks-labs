import numpy as np
import matplotlib.pyplot as plt

def sigmoid(X):
    return 1/(1+np.exp(-X))

class neuralNetwork:
    def __init__(self, iNodes, hNodes, oNodes, lRate):
        # Number of input, hidden and output nodes
        self.iNodes = iNodes
        self.hNodes = hNodes
        self.oNodes = oNodes  
        # Weight matrices, wih and who, initialized with random numbers that follow a normal distribution
        self.wih = np.random.normal(0.0, 0.5, (self.hNodes,self.iNodes))
        self.who = np.random.normal(0.0, 0.5, (self.oNodes,self.hNodes))   
        # Learning rate (for session 7)
        self.lRate = lRate
        # Activation function is the sigmoid function
        self.actFunc = sigmoid
    
    def __str__(self):
        return f"Input nodes: {self.iNodes}, Hidden nodes: {self.hNodes}, Output nodes: {self.oNodes}, Learning rate: {self.lRate}, wih matrix shape: {self.wih.shape}, who matrix shape: {self.who.shape}"
        
    def query(self, imgArr):
        # Transform the image into a vector    
        inputs = imgArr.flatten()
        # Move signal into hidden layer
        hiddenInputs = np.dot(self.wih, inputs)
        # Apply the activation function
        hiddenOutputs = self.actFunc(hiddenInputs)
        # Move signal into output layer
        outputs = np.dot(self.who, hiddenOutputs)
        # Apply the activation function
        prediction = self.actFunc(outputs)
        return prediction 