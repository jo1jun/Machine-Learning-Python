import matplotlib.pyplot as plt

def plotData(X,y):
    plt.plot(X, y, 'rx', markersize=10, label='Training data')
    plt.xlabel('Population of City in 10,000s')
    plt.ylabel('Profit in $10,000s')
    plt.show()