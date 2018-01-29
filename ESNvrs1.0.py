
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
import scipy.linalg as sl

data = np.loadtxt('MackeyGlass_t17.txt')

plt.figure(10).clear()
plt.plot(data[0:1000])
plt.title('A sample of data')
plt.show()

trainLen =2000    
testLen = 2000    
wshLen = 100     # washout length


#Parameters of the Reservoir
inSize = 1          #input units- K
outSize = 1         #output untis-L
resSize = 400     #size of the reservoir/internal units- N
a = 0.3             #leanking rate
alpha = 0.9         #scaling factor for W 
initType= 'u';      #type of initializatoin (u-uniform, g-gaussian)
initScale = 0.5     #scaling factor for weight matrix initializations
reg = 1e-4          #regularization coefficient
errorLen = 500      #sample size for mse calculation 
#*****************************************************************
np.random.seed(45)
#    W_in = np.random.uniform(-initScale,initScale,resSize*(1+inSize)).reshape(resSize,1+inSize)
W_in = (np.random.rand(resSize,1+inSize)-initScale)*1 #rand nos[0,1) from uniform dist
W = np.random.rand(resSize,resSize)-initScale
print ('Computing spectral radius...')
rhoW = np.max(np.abs(sl.eigvals(W))) #spectral radius of mat W
#rhoW = np.max(np.abs(sl.eig(W)[0]))
W *= alpha / rhoW

X = np.zeros((1+inSize+resSize,trainLen-wshLen))    #allocate memory for state collecting matrix
Yt = data[None,wshLen+1:trainLen+1]    #allocate memeory for target matrix 
x = np.zeros((resSize,1))   #initial state
for t in range(trainLen):
    u = data[t]
    x = (1-a)*x + a*np.tanh( np.dot(W_in,np.vstack((1,u)) ) + np.dot( W, x ))    
    if t >= wshLen:
        #replace colums in X by stack
        X[:,t-wshLen] = np.vstack((1,u,x))[:,0]
# train the output
X_T = X.T
Wout = np.dot( np.dot(Yt,X_T), sl.inv( np.dot(X,X_T) + \
reg*np.eye(1+inSize+resSize) ) )
    
Y = np.zeros((outSize,testLen))
u = data[trainLen]
for t in range(testLen):
    x = (1-a)*x + a*np.tanh( np.dot( W_in, np.vstack((1,u)) ) + np.dot( W, x ) )
    y = np.dot( Wout, np.vstack((1,u,x)) )
    Y[:,t] = y
    # generative mode:
    u = y
#MSE computation

mse = sum( np.square( data[trainLen+1:trainLen+errorLen+1] - Y[0,0:errorLen] ) ) / errorLen
print ('MSE = ' + str( mse ))

# plot some signals
plt.figure(1).clear()
plt.plot(data[trainLen+1:trainLen+testLen+1], 'g' )
plt.plot( Y.T, 'b' )
plt.title('Target and generated signals $y(n)$ starting at $n=0$')
plt.legend(['Target signal', 'Free-running predicted signal'])

plt.figure(2).clear()
plt.plot( X[0:20,0:200].T )
plt.title('Some reservoir activations $\mathbf{x}(n)$')

plt.figure(3).clear()
plt.bar( range(1+inSize+resSize), Wout.T )
plt.title('Output weights $\mathbf{W}^{out}$')
plt.show()

def printline():
    print("-------------------------------------------------------------------")