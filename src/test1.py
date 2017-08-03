from utilities import image_reader as ir
import numpy as np
import pandas as pd
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D
from keras.layers.normalization import BatchNormalization
from keras.models import load_model
import math
from utilities.support_functions import *
from utilities.data_processing_functions import *
from utilities import support_functions as sf


### Read and format TRAINING data ####
training_ids = pd.read_csv('C:/Users/john.hife/Documents/workspaces/TSA Kaggle/data/training_ids.csv')
training_ids = training_ids['id']

dir = 'C:/Users/john.hife/Documents/workspaces/TSA Kaggle/data/stage1_aps/'

#test if images at different angles
zone1_angles = [0,2,6,8,10,12,14]
data = ir.read_data('C:/Users/john.hife/Documents/workspaces/TSA Kaggle/data/stage1_aps/0a27d19c6ec397661b09f7d5998e0b14.aps')
plot_image(data, 11)

plot_images(data)
image_list = get_image_list(data, zone_num=1, zone_angles = zone1_angles)
plot_image(image_list[0])
plot_image(image_list[1])

#test aggregation of images at different angles
max_x_width = get_max_width(image_list, dim=0)
max_y_width = get_max_width(image_list, dim=1)
zone1_aggregated = aggregate_zone_images(image_list, max_x_width, max_y_width)
plot_image(zone1_aggregated)

#test dataset of zone images
#zone_dataset = create_zone_dataset(dir, training_ids,zone_num=1, zone_angles=zone1_angles)
zone_1_dataset = create_zone_dataset(dir, training_ids,zone_num=1, zone_angles=zone1_angles)
zone_1_dataset = zone_1_dataset.reshape(zone_1_dataset.shape[0], zone_1_dataset.shape[1], zone_1_dataset.shape[2], 3)
plot_image(zone_1_dataset[0,:,:,:])

### Read Labels ###
zone1_labels = pd.read_csv('C:/Users/john.hife/Documents/workspaces/TSA Kaggle/data/ZONE_LABELS/ZONE1_LABELS.csv')
zone1_labels = np.array(zone1_labels['label'])[0:zone_1_dataset.shape[0]]
zone1_labels = np_utils.to_categorical(zone1_labels, 2)


### Zone 1 Modeling ###

zone1_model = Sequential()

#input_shape should be identical to the shape of the training set
zone1_model.add(Convolution2D(32, (3, 3), activation = 'elu', input_shape = (zone_1_dataset.shape[1], zone_1_dataset.shape[2],1)))
zone1_model.add(Dropout(0.10))
zone1_model.add(MaxPooling2D(pool_size = (2,2)))
#zone_model.add(BatchNormalization())
zone1_model.add(Convolution2D(32, (3, 3), activation = 'elu'))
#zone_model.add(BatchNormalization())
zone1_model.add(Dropout(0.10))
zone1_model.add(MaxPooling2D(pool_size = (2,2)))
zone1_model.add(Flatten())
#zone_model.add(Dense(2, activation = 'softmax'))
zone1_model.add(Dense(1, activation='sigmoid'))



#Compile zone_model
zone1_model.compile(#loss = 'mean_squared_logarithmic_error',
    loss='binary_crossentropy',
    optimizer = 'adam',
    metrics = ['accuracy'])


class_weight = {0 : 1, 1: 10}

#Fit zone_model
zone1_model.fit(zone_1_dataset, zone1_labels[:,1],
          batch_size = 5, epochs = 50, verbose = 1, class_weight=class_weight)

#### Save and load model #####
#zone_model.save('C:/Users/kristopher.patton/Desktop/TSA_Kaggle/Models/zone1_model_7_26.h5')
#zone_model = load_model('C:/Users/kristopher.patton/Desktop/TSA_Kaggle/Models/zone1_model_7_26.h5')
zone1_train_predictions = zone1_model.predict(zone_1_dataset, batch_size = 1, verbose = 1)
max(zone1_train_predictions)
min(zone1_train_predictions)



zone1_model = load_model('C:/Users/john.hife/Documents/workspaces/TSA Kaggle/data/zone1_model_8-2-2017.h5')

### Read and format TESTING data ###
testing_ids = pd.read_csv('C:/Users/john.hife/Documents/workspaces/TSA Kaggle/data/testing_ids.csv')
testing_ids = testing_ids['id']

zone_1_testset = create_zone_dataset(dir, testing_ids,zone_num=1, zone_angles=zone1_angles)
zone_1_testset = zone_1_testset.reshape(zone_1_testset.shape[0], zone_1_testset.shape[1], zone_1_testset.shape[2], 1)


#### Zone 1 Predictions ####
zone1_predictions = zone1_model.predict(zone_1_testset, batch_size = 20, verbose = 1)
zone1_predictions

len(zone1_predictions)
max(zone1_predictions)

preds = pd.concat((pd.DataFrame(testing_ids, dtype=str), pd.DataFrame(zone1_predictions)))
pd.DataFrame(testing_ids, zone1_predictions)
pd.DataFrame(zone1_predictions).to_csv('zone_predictions.csv')

#zone1_scores = sf.prediction_dataframe(testing_ids, zone_predictions, 'Zone1')
#zone1_scores = sf.default_prediction_dataframe(testing_ids, 0.07, 'Zone1')
#zone1_scores.to_csv('test1.csv')





zone1_angles = [0,2,6,8,10,12,14]

data = ir.read_data('C:/Users/john.hife/Documents/workspaces/TSA Kaggle/data/stage1_aps/49c3fc4b14948ab097a3462bb825e2f0.aps')
data = ir.read_data('C:/Users/john.hife/Documents/workspaces/TSA Kaggle/data/stage1_aps/1cb13f156bd436222447dd658180bd96.aps')
data = ir.read_data('C:/Users/john.hife/Documents/workspaces/TSA Kaggle/data/stage1_aps/b26e79a465ebac796ceb022cb998c236.aps')
data = ir.read_data('C:/Users/john.hife/Documents/workspaces/TSA Kaggle/data/stage1_aps/adc28b596accd5039620bfef68feacf3.aps')

data = ir.read_data('C:/Users/john.hife/Documents/workspaces/TSA Kaggle/data/stage1_aps/ce8ab5548464940a8a14a813a75500d0.aps')
data = ir.read_data('C:/Users/john.hife/Documents/workspaces/TSA Kaggle/data/stage1_aps/104e284349ffe68378745fc8c5638eab.aps')
data = ir.read_data('C:/Users/john.hife/Documents/workspaces/TSA Kaggle/data/stage1_aps/6ec5ce5500d9ce14887e64f47486ad5f.aps')
data = ir.read_data('C:/Users/john.hife/Documents/workspaces/TSA Kaggle/data/stage1_aps/5b900a35f3e9591914ea328cacb73dfa.aps')


plot_images(data)
plot_image(data,0)





