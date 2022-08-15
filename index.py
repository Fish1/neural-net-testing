import numpy

lr = 1 # learning rate
bias = 1 # value of bias

weights = numpy.random.rand((64 * 2) + 1)

def Perception(input1, input2, output, train) :
    input1Copy = input1
    input2Copy = input2
    outputP = 0
    index = 1

    while index != 64 :
        c = input1Copy & 1
        outputP += weights[index] * c
        input1Copy >>= 1
        index += 1

    while index != 64 * 2 :
        c = input2Copy & 1
        outputP += weights[index] * c
        input2Copy >>= 1
        index += 1

    outputP += bias * weights[128]

    if outputP > 0 : # activation function (here Heaviside)
        outputP = 1 / ( 1 + numpy.exp(-outputP)) # sigmoid function
    else :
        outputP = 0

    if train == True :
        error = output - outputP

        index = 0
        input1Copy = input1
        input2Copy = input2

        while index != 64 :
            c = input1Copy & 1
            weights[index] += lr * error * c
            input1Copy >>= 1
            index += 1

        while index != 64 * 2 :
            c = input2Copy & 1
            weights[index] += lr * error * c
            input2Copy >>= 1
            index += 1

        weights[128] += lr * error * bias
    else :
        print(int(outputP * 100), '%')

# for i in range(10000) :
#    a = numpy.random.randint(10000000)
#    b = numpy.random.randint(10000000)
#    even =  1 if (a % 2) == 0 and (b % 2) == 0 else 0 # if both are even, c = 1
#    Perception(a, b, even, True) # if both true then 1 else 0
#    Perception(b, a, even, True) # if both true then 1 else 0

for i in range(100000) :
    question = numpy.random.randint(100) + 1
    answer = 0 if (question % 3) == 0 and (question % 5) == 0 else 1 if (question % 3) == 0 else 2 if (question % 5) == 0 else 3
    # print(question, guess, answer, truth)
    for g in range(4) :
        guess = g
        truth = 1 if answer == guess else 0
        Perception(question, guess, truth, True) # if both true then 1 else 0

# 0 = fizbuzz
# 1 = fizz
# 2 = buzz
# 3 = number
while True :
    x = int(input("Input 1: "))
    y = int(input("Input 2: "))
    Perception(x, y, 1, False)
