import os
import pickle

actors = os.listdir('Data')

filename = []

for actor in actors:
    for file in os.listdir(os.path.join("Data",actor)):
        filename.append(os.path.join("Data",actor,file))

pickle.dump(filename, open("filename.pkl", "wb"))