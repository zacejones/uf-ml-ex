import numpy as np

X = np.array(([2,9], [1,5],[3,6]),dtype=float)
y = np.array(([92], [86],[89]), dtype=float)

X = X/np.amax(X, axis=0) # maximum of X array
y = y/100 # Max test score is 100

class Neural_Network(object):
    def __init__(self):
        #parameters
        self.inputSize = 2
        self.outputSize = 1
        self.hiddenSize = 3
        self.W1 = np.random.randn(self.inputSize, self.hiddenSize)
        self.W2 = np.random.randn(self.hiddenSize, self.outputSize)
        
    def forward(self, X):
        #forward propagation throug hour network
        self.z = np.dot(X, self.W1)
        self.z2 = self.sigmoid(self.z)
        self.z3 = np.dot(self.z2, self.W2)
        o = self.sigmoid(self.z3)
        return o
    
    def sigmoid(self, s):
        #activation function
        return 1/(1+np.exp(-s))
    
    def sigmoidPrime(self, s):
        #derivative of sigmoid function
        return s*(1-s)
    
    def backward(self, X, y, o):
        # Backward propagate through the network
        self.o_error = y - o #output error
        self.o_delta = self.o_error*self.sigmoidPrime(o) #derivative of sigmoid on error
        
        self.z2_error = self.o_delta.dot(self.W2.T) # z2 error: how much our hidden layer weights contributed to output error
        self.z2_delta = self.z2_error*self.sigmoidPrime(self.z2) # apply derivative of sigmoid to z2 error
        
        self.W1 += X.T.dot(self.z2_delta) #adjusting first input weights
        self.W2 += self.z2.T.dot(self.o_delta) #adjusting hidden layer weights
        
    def train(self, X, y):
        o= self.forward(X)
        self.backward(X, y, o)
    
NN = Neural_Network()

#defining our output
o = NN.forward(X)



NN = Neural_Network()
for i in range(10000):
  print( "Input: \n" + str(X) )
  print ("Actual Output: \n" + str(y) )
  print ("Predicted Output: \n" + str(NN.forward(X)) )
  print ("Loss: \n" + str(np.mean(np.square(y - NN.forward(X))))) # mean sum squared loss
  print ("\n")
  NN.train(X, y)
  
  
  
  