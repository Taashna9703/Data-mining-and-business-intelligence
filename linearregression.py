import pandas as pd
import numpy as np
from matplotlib import pyplot as plt 
import seaborn as sns   
df= pd.read_csv('student.csv')
def coefficients(x,y):
    n = np.size(x)
    m_x = np.mean(x)
    m_y = np.mean(y)
    SS_xy = np.sum(y*x) - n*m_y*m_x
    SS_xx = np.sum(x*x) - n*m_x*m_x
    b_1 = SS_xy / SS_xx
    b_0 = m_y - b_1*m_x
    return (b_0, b_1)
def plot_regression(x,y,b):
    plt.scatter(x,y,)
    y_pred = b[0] + b[1]*x
    plt.plot(x, y_pred, color = "g")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.show()
x=df["years"]
y=df["salary"]
b=coefficients(x,y)
plot_regression(x,y,b)
