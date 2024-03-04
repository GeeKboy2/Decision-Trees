import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier

classes_df = pd.read_csv("./Test.py/classes.csv")
data_df = pd.read_csv("./Test.py/zoo.csv")


#y has the number of each line class
y = data_df['class_type']
#print('y=',y[0:5])

#Hold item names
names = data_df['animal_name']
#print('names=',names[0:5])

#x has all the info about the items except their classes
x = data_df.drop(columns=['class_type', 'animal_name']).to_numpy()
#print('x=',x[0:5])

#create a DecisionTree
classifier = DecisionTreeClassifier()

#ask the program to follow these examples
#each ligne has tha caracteristics from x and the class from y
classifier.fit(x, y)


gorilla = [1,0,0,1,0,0,0,1,1,1,0,0,2,0,0,1] # expect 1
pony = [1,0,0,1,0,0,0,1,1,1,0,0,4,1,1,1] # expect 1
sole = [0,0,1,0,0,1,0,1,1,0,0,1,0,1,0,0] # expect 4
tortoise = [0,0,1,0,0,0,0,0,1,1,0,0,4,1,0,1] # expect 3

test_set = [gorilla, pony, sole, tortoise]

#make predictions acording to the choices we made by hand
predictions = classifier.predict(test_set)

print(predictions)
