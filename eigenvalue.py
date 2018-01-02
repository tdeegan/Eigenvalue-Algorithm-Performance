import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

def power_iteration(A,tolerance):
    '''The purpose of the power_iteration function is to solve for the largest eigenvalue of a matrix.
    The function receives as arguments a square matrix A and a tolerance level.
    It returns a tuple containing the final estimate of the largest eigenvalue, the final estimate
    of the corresponding eigenvector and the number of iterations it took to reach convergence'''

    count = 0 #variable to track iterations
    
    A_shape = A.shape #returns tuple (# rows, # cols) from our square matrix
    rows = A_shape[0] # number of rows
    cols = A_shape[1] # number of cols
    
    x = np.ones((cols,1)) #x is the initial guess for the eigenvector (column vector of ones)
    
    while(True):
        count = count + 1 #increment count - new iteration underway
        y = np.matmul(A,x) #multiply A by current eigenvector
        k = np.linalg.norm(y) #estimate of largest eigenvalue
        x = y / k #estimate of eigenvector
        res = np.linalg.norm(np.matmul(A,x) - k*x) #infinity norm of Ax - kx
    
        if(res < tolerance): #determine is infinity norm of residual less than specified tolerance
            return (k,x,count) #if it is, return tuple containing (eigenvalue, eigenvector, # iterations)

def main():
    
    np.random.seed(1) #random number seed set to 1
    A = np.random.rand(5,5) #'random' 5x5 matrix
    
    tols = list() #will be a list of the tolerance levels used
    its = list() #will be a list of iterations needed at each tolerance level
    
    for i in range(6):
        e = -1*(i+3) #determine exponent
        tol = 10**e #calculate tolerance level
        tols.append(tol) #append tolerance level to tols list
        result = power_iteration(A,tol) #perform power_iteration on A at tolerance level tol
        its.append(result[2]) #append number of iterations to its list
    
    plt.semilogx(tols,its) #tols varies over multiple orders of magnitude, so we use plt.semilogx
    plt.title('Performance of power_iteration') #title of plot
    plt.xlabel('Tolerance') #x axis label for plot
    plt.ylabel('Iterations') #y axis label for plot
    plt.savefig('iterations.png') #save file as 'iterations.png'

if __name__ == "__main__":
    main()
