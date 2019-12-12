import numpy as np

'''
the formula for softmax is:
P(class_i) = e^(Z_i)/(e^Z_1 + e^Z_2 + ... + e^Z_n)
'''

# Write a function that takes as input a list of numbers, and returns
# the list of values given by the softmax function.
def softmax(L):
    # numerator - convert the list into exponents
    expL = np.exp(L)
    
    # denominator - summation of the exponential values from the list
    sumExpL = sum(expL)
    
    # we need answers for each of the classes or value in the list
    result = []
    for i in expL:
        result.append(i*1.0/sumExpL)
    return result
