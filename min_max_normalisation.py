dataset=[10, 12, 3, 6, 5, 25, 17, 100, 1000, 98, 11, 27, 78, 33, 9, 18, 23, 44, 690, 200]
old_max=max(dataset)
old_min =  min(dataset)
new_dataset=[]
new_dataset1=[]
new_min=0
new_max=1
print("Min max normalisation from range -1 to 1")
for i in dataset:  
     data=(((i-old_min)/(old_max-old_min))*2) -1
     new_dataset.append(data) 
print(new_dataset)
print("Min max normalisation From range 0 to 1:")  
for i in dataset:    
     data=(((i-old_min)/(old_max-old_min))) 
     new_dataset1.append(data)     
print(new_dataset1)
