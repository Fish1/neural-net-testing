import numpy, random, os

lr = 1 # learning rate
bias = 1 # value of bias

weights = numpy.random.rand((64 * 2) + 1)

def Perception(input1, input2, output, train) :
    binary1 = "{0:b}".format(int(input1)).rjust(64, '0') # binary representation of the input1
    binary2 = "{0:b}".format(int(input2)).rjust(64, '0') # binary representation of the input2

    outputP = 0

    for index, element in enumerate(binary1) :
        outputP += weights[index] * int(element)

    for index, element in enumerate(binary2) :
        outputP += weights[index + 64] * int(element)

    outputP += bias * weights[128]


    if outputP > 0 : # activation function (here Heaviside)
        outputP = 1 / ( 1 + numpy.exp(-outputP)) # sigmoid function
    else :
        outputP = 0

    if train == True :
        error = output - outputP
        for index, element in enumerate(binary1) :
            weights[index] += lr * error * int(element)
        for index, element in enumerate(binary2) :
            weights[index + 64] += lr * error * int(element)
        weights[128] += lr * error * bias
    else :
        print(outputP)

for i1 in range(10000) :
    a = numpy.random.randint(10000000)
    b = numpy.random.randint(10000000)
    even =  1 if (a % 2) == 0 and (b % 2) == 0 else 0 # if both are even, c = 1
    Perception(a, b, even, True) # if both true then 1 else 0
    Perception(b, a, even, True) # if both true then 1 else 0


while True :
    x = int(input("Input 1: "))
    y = int(input("Input 2: "))
    Perception(x, y, 1, False) # if both true then 1 else 0:26