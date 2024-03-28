
import numpy as np
import matplotlib.pyplot as plt



def loss(bias,weights,points):
    total=0.00
    for i in range(0,len(points)):
        x=points[i][0]
        y=points[i][1]
        total+=(y-(weights*x+bias))**2
    total=total/float(len(points))
    return total
def step(points,bias,weights,learning_rate):
    biasg=0
    weightg=0
    for i in range(0,len(points)):
        x=points[i][0]
        y=points[i][1]
        biasg+=-(y-(weights*x+bias))*(2/len(points))
        weightg+=-x*(y-(weights*x+bias))*(2/len(points))
    biasn=bias-learning_rate*biasg
    weightn=weights-learning_rate*weightg
    return biasn,weightn

def gradientd(points,bias,weights,lr,iterations):
    for i in range(iterations):
        bias,weights=step(points,bias,weights,lr)
    return weights,bias

def result(bias,weight,input):
    return input*weight+bias

def run():
    points=np.genfromtxt('data.csv',delimiter=',')
    #defhyperparameters
    lr=0.0001
    initial_bias=0
    initial_weight = 0
    epochs=1000
    print("initial weights:{0},initial bias:{1},loss:{2}".format(initial_bias,initial_weight,loss(initial_bias,initial_weight,points)))
    initial_weight,initial_bias=gradientd(points,initial_bias,initial_weight,lr,epochs)
    print("end weights:{0},initial bias:{1},loss:{2}".format(initial_bias, initial_weight, loss(initial_bias, initial_weight,points)))
    xs=[i[0]for i in points]
    ys=[i[1]for i in points]
    plt.scatter(xs,ys)
    a=np.linspace(0,100,1000)
    b=result(initial_bias,initial_weight,a)
    plt.plot(a,b)
    plt.show()
if __name__==('__main__'):
    run()
