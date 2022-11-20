import numpy as np
import matplotlib.pyplot as plt

#Use a for loop to compute the 10th triangular number. The nth triangular number is defined as 1+2+3+...+n. 
def P1A():
    n = 10
    sum = 0
    for i in range(1,n+1):
        sum = sum + i
    print(sum)
P1A()

#Use a for loop to compute 10!, the factorial of 10. Recall that the factorial of n is 1*2*3*...*n.
def P1B():
    n = 10
    sum = 1
    for i in range(1,n+1):
        sum = sum * i
    print(sum)

P1B()
#Write code to print 10 factorials in reverse order. 
def P1C():
    numlines = 10
    for i in range(numlines,0,-1):
        sum = 1
        for j in range(1,i+1):
            sum = sum * j
        print(sum)
P1C()