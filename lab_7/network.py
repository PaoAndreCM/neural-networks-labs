import numpy as np

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
        return f"Input nodes: {self.iNodes} \nHidden nodes: {self.hNodes} \nOutput nodes: {self.oNodes} \nLearning rate: {self.lRate} \nwih matrix shape: {self.wih.shape} \nwho matrix shape: {self.who.shape}"
        
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
    
    def train(self, imgArr, target):
        # Transform the image to a vector
        inputs = imgArr.flatten()
        # Move signal into hidden layer
        hiddenInputs = np.dot(self.wih, inputs)
        # Apply the activation function
        hiddenOutputs = self.actFunc(hiddenInputs)
        # Move signal into output layer
        outputs = np.dot(self.who, hiddenOutputs)
        # Apply the activation function
        prediction = self.actFunc(outputs)
        # output layer error is (target-actual)
        outputErrors = target - prediction
        # for the hidden layer error, we need to invert who
        whoT = self.who.T
        whoSq = np.dot(self.who,whoT)
        whoInv = np.linalg.inv(whoSq)
        whoInv = np.dot(whoT,whoInv)
        hiddenErrors = np.dot(whoInv, outputErrors)
        # update the weights between the hidden and output layer
        err = outputErrors*prediction*(1.-prediction)
        self.who += self.lRate * np.dot(err[:,np.newaxis], hiddenOutputs[np.newaxis,:])
        # update the weights between the input and the hidden layer
        err = hiddenErrors * hiddenOutputs * (1.-hiddenOutputs)
        self.wih += self.lRate * np.dot(err[:,np.newaxis], inputs[np.newaxis,:])

    def evaluate(self, test_data, test_labels):
            num_correct_predictions = 0

            for i, test_label in enumerate(test_labels):
                prediction = self.query(test_data[i])  # query the network

                # Compare index of the largest element in `prediction` with label
                if test_label == np.argmax(prediction):
                    num_correct_predictions += 1

            performance = num_correct_predictions / len(test_labels)
            return performance
    
    def saveWeights(self, base_path):
        wih_filename = 'wih.npy'
        who_filename = 'who.npy'
        np.save(base_path + wih_filename,self.wih)
        np.save(base_path + who_filename,self.who)

    def restoreWeights(self, weights_path):
        wih_filename = 'wih.npy'
        who_filename = 'who.npy'
        self.wih = np.load(weights_path+wih_filename)
        self.who = np.load(weights_path+who_filename)
