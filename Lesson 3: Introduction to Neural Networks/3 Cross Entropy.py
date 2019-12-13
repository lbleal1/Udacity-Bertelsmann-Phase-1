import numpy as np

'''

Cross-Entropy is an error function that we want to minimize.
To calculate Cross-Entropy(CE):

CE = negative summation of the logarithms of the probabilities whenever there is or there's none
simply
 = - summ [(y_i)ln(p_i) + (1-y_i)ln(1-p_i)]
where:
y is either 1 if there is or 0 if there's none
p is the probability that there is

'''

def cross_entropy(Y, P):
    result = []
    for i in range(0,len(Y)):
        result.append(-1*(Y[i]*np.log(P[i]) + (1-Y[i])*np.log(1-P[i])))
    CE = np.sum(result)
    return CE