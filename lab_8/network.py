import numpy as np

def sigmoid(X):
    return 1/(1+np.exp(-X))

def softmax(arr):
    numerator = np.exp(arr)
    denominator = np.sum(np.exp(arr))
    return (numerator / denominator)

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
            '''
            Evaluate the performance of the neural network on a test dataset.

            Parameters:
            - test_data (list or array): Input data for testing.
            - test_labels (list or array): Corresponding labels for the test data.

            Returns:
            - performance (float): The accuracy of the neural network on the provided test dataset,
            calculated as the ratio of correct predictions to the total number of test samples.

            Usage:
            1. Instantiate an object of the neural network class.
            2. Train the neural network using the `train` method.
            3. Call this method with the test data and labels to evaluate the network's performance.
            '''
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

    def batchTrain(self, train_data, train_labels):
        for i, label in enumerate(train_labels):
            # create target vector
            target = np.zeros(10, dtype='float') + 0.01 # Set the target vector
            target[label] = 0.99

            # feed image with target vector into method `train`
            self.train(train_data[i], target)


