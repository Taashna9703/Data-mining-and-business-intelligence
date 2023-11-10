dataset=[10, 12, 3, 6, 5, 25, 17, 100, 1000, 98, 11, 27, 78, 33, 9, 18, 23, 44, 690, 200]
maxi=max(dataset)
digit=len(str(maxi))-1
div=10**digit
import pandas as pd
import numpy as np
dataset=[10, 12, 3, 6, 5, 25, 17, 100, 1000, 98, 11, 27, 78, 33, 9, 18, 23, 44, 690, 200]
mean=np.mean(dataset)
sd=np.std()
new_dataset=[]
for i in dataset:
    data=(i-mean)/sd
    new_dataset.append(data)   
print(new_dataset)
new_dataset=[]
for i in dataset:
    data=(i/div)
    new_dataset.append(data)    
print(new_dataset)
