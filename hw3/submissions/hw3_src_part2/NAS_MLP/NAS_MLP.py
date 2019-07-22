import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from keras.models import Sequential
from keras.layers import Dense
from keras.models import load_model
import matplotlib.pyplot as plt
from scipy import stats

import random


def loaddata():
    data=pd.read_csv('./data/data.csv')
    data
    x=data[['0','1']]
    y=data['label']



def get_train_test_5_fold(k):
    data=pd.read_csv('./data/data.csv')
    data
    x=data[['0','1']]
    y=data['label']
    kf = KFold(n_splits=5,shuffle=True)
    x_train=[]
    y_train=[]
    x_test=[]
    y_test=[]
    
    q=0
    for train_index, test_index in kf.split(x):
        if q==k:
            for i in train_index:
                x_train.append(x.loc[i].values)
#                 print(y.loc[i])
                y_train.append(y.loc[i])
            for i in test_index:
                x_test.append(x.loc[i].values)
                y_test.append(y.loc[i])
            return np.array(x_train),np.array(y_train),np.array(x_test),np.array(y_test)
        q+=1


x_train, y_train, x_test, y_test=get_train_test_5_fold(1)




# Settings

def func(x):
    return np.sin(x * np.pi) # Modify this line to add your function

layer_max = 10
neuron_max = 200 
num_epochs = 100
activation_function = 'relu' 


def basic_model(num_layers = 3, num_neurons = 256):

    # Create Model

    model = Sequential()

    # Add Hidden Layers

    for _ in range(num_layers):
        model.add(Dense(num_neurons, input_dim = 2, kernel_initializer='normal', activation = activation_function))
    model.add(Dense(1, kernel_initializer='normal'))
    
    # Compile
    
    model.compile('adam', loss = 'mean_squared_error', metrics = ['accuracy'])
    return model

counter = 0

s


a=[]

for neuron_indx in range(neuron_max // 10):
    for layer_indx in range(layer_max):

        # Convert indx to number of layers/neurons 

        layers = layer_indx + 1
        neurons = (neuron_indx + 1) * 10

        x_train, y_train, x_test, y_test=get_train_test_5_fold(random.randint(0,4))
        # Calculate predictions of model
        
        model = basic_model(layers, neurons)
        model.fit(x_train, y_train, validation_data = (x_test, y_test), epochs = num_epochs, verbose = 0)
        y_model = model.predict_classes(x_test)
        res=model.evaluate(x=x_test, y=y_test)
        a.append(res)
        print(res)
        
        # Compute rsq value and put in array
        
        # rsq_matrix[counter] = [layers, neurons, rsq(y_model, y_test)]

        # Printing Progress
        
        counter += 1
        print('step %d of %d'%(counter,(neuron_max // 10 * layer_max))) 
    b=np.array(a)
    np.save('b'+str(counter),b)



b=np.array(a)
np.save('b',)
# layers_opt, neurons_opt,  _ = max(rsq_matrix, key = lambda x: x[2]) # Find optimal layers and neurons from array

#Generate and Save Model

# opt_model = basic_model(layers_opt, neurons_opt)
# opt_model.fit(x_train, y_train, validation_data = (x_test, y_test), epochs = 10, verbose = 0)
# opt_model.save('my_model.h5')

# #Plot Information

# plt.plot(x_test, model.predict(x_test), 'rx', x_test, y_test, 'kx')
# plt.legend(['Target', 'Model Prediction'])
# plt.title('Optimal Model Configuration (rsq = %.3f): %d Layers, %d Neurons'%(round(rsq(model.predict(x_test),y_test),3),layers,neurons))
# plt.show()

