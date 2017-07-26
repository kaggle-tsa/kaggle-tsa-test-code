from utilities import image_reader as ir
from utilities import support_functions as sf
import numpy as np
import pandas as pd
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D
from keras.models import load_model


### Read and format TRAINING data ####
training_ids = pd.read_csv('C:/Users/kristopher.patton/Desktop/TSA_Kaggle/Data/training_ids.csv')
training_ids = training_ids['id']
dir = 'C:/Users/kristopher.patton/Desktop/TSA_Kaggle/Data/stage1_aps/'
training_images = ir.read_batch_images(dir, training_ids)

# Add depth of 1 to the image...
training_images = training_images.reshape(training_images.shape[0], training_images.shape[1], training_images.shape[2], 1)

# Convert data type to float32
training_images = training_images.astype('float32')

### Read and format TESTING data ###
testing_ids = pd.read_csv('C:/Users/kristopher.patton/Desktop/TSA_Kaggle/Data/testing_ids.csv')
testing_ids = testing_ids['id']
dir = 'C:/Users/kristopher.patton/Desktop/TSA_Kaggle/Data/stage1_aps/'
testing_images = ir.read_batch_images(dir, testing_ids)


# Formatting
testing_images = testing_images.reshape(testing_images.shape[0], testing_images.shape[1], testing_images.shape[2], 1)
testing_images = testing_images.astype('float32')

### Read Labels ###
zone1_labels = pd.read_csv('C:/Users/kristopher.patton/Desktop/TSA_Kaggle/Data/ZONE1_LABELS.csv')
zone1_labels = np.array(zone1_labels['label'])[0:training_images.shape[0]]
zone1_labels = np_utils.to_categorical(zone1_labels, 2)



### Zone 1 Modeling ###

zone1_model = Sequential()

#input_shape should be identical to the shape of the training set
## 32 == number of convolution filters to use
## 3 == number of rows in each convolution kernel
## 3 == number of columns in each convolution kernel
zone1_model.add(Convolution2D(32, (3, 3), activation = 'relu', input_shape = (training_images.shape[1], training_images.shape[2], 1)))

#Add an additional conlution layer
#zone1_model.add(Convolution2D(32, (3, 3), activation = 'relu'))

#Apply max poooling
##this takes the maxium of the four values in the 2x2 filter
zone1_model.add(MaxPooling2D(pool_size = (2,2)))

#Add Dropout layer
##Regularizes the zone1_model to prevent overfitting
##Extreme version of bagging, each ensemble variant is trained with a different subsample of the input data
zone1_model.add(Dropout(0.25))


#Must flatten (make 1-dimensional) the weights of convolution layer prior to passing through the dense layer
zone1_model.add(Flatten())

#Add Desne layer
## 128 == output size of 128
## zone1_model.add(Dense(128, activation = 'relu'))

#Add droput layer
zone1_model.add(Dropout(0.5))

#Add Desne layer
## 2 == 2 classes in final output layer
zone1_model.add(Dense(2, activation = 'softmax'))


#Compile zone1_model
zone1_model.compile(loss = 'mean_squared_logarithmic_error',
              optimizer = 'adam',
              metrics = ['accuracy'])

#Fit zone1_model
zone1_model.fit(training_images, zone1_labels,
          batch_size = 1, epochs = 1, verbose = 1)


#### Save and load model #####
zone1_model.save('C:/Users/kristopher.patton/Desktop/TSA_Kaggle/Models/zone1_model_7_26.h5')
zone1_model = load_model('C:/Users/kristopher.patton/Desktop/TSA_Kaggle/Models/zone1_model_7_26.h5')


#### Zone 1 Predictions ####
zone1_predictions = zone1_model.predict(testing_images, batch_size = 1, verbose = 1)
zone1_scores = sf.prediction_dataframe(testing_ids, zone1_predictions, 'Zone1')
#zone1_scores = sf.default_prediction_dataframe(testing_ids, 0.07, 'Zone1')


#### Zone 2 Predictions ####
zone2_scores = sf.default_prediction_dataframe(testing_ids, 0.07, 'Zone2')

#### Zone 3 Predictions ####
zone3_scores = sf.default_prediction_dataframe(testing_ids, 0.06, 'Zone3')

#### Zone 4 Predictions ####
zone4_scores = sf.default_prediction_dataframe(testing_ids, 0.06, 'Zone4')

#### Zone 5 Predictions ####
zone5_scores = sf.default_prediction_dataframe(testing_ids, 0.06, 'Zone5')

#### Zone 6 Predictions ####
zone6_scores = sf.default_prediction_dataframe(testing_ids, 0.06, 'Zone6')

#### Zone 7 Predictions ####
zone7_scores = sf.default_prediction_dataframe(testing_ids, 0.05, 'Zone7')

#### Zone 8 Predictions ####
zone8_scores = sf.default_prediction_dataframe(testing_ids, 0.07, 'Zone8')

#### Zone 9 Predictions ####
zone9_scores = sf.default_prediction_dataframe(testing_ids, 0.05, 'Zone9')

#### Zone 10 Predictions ####
zone10_scores = sf.default_prediction_dataframe(testing_ids, 0.05, 'Zone10')

#### Zone 11 Predictions ####
zone11_scores = sf.default_prediction_dataframe(testing_ids, 0.06, 'Zone11')

#### Zone 12 Predictions ####
zone12_scores = sf.default_prediction_dataframe(testing_ids, 0.05, 'Zone12')

#### Zone 13 Predictions ####
zone13_scores = sf.default_prediction_dataframe(testing_ids, 0.06, 'Zone13')

#### Zone 14 Predictions ####
zone14_scores = sf.default_prediction_dataframe(testing_ids, 0.07, 'Zone14')

#### Zone 15 Predictions ####
zone15_scores = sf.default_prediction_dataframe(testing_ids, 0.06, 'Zone15')

#### Zone 16 Predictions ####
zone16_scores = sf.default_prediction_dataframe(testing_ids, 0.06, 'Zone16')

#### Zone 17 Predictions ####
zone17_scores = sf.default_prediction_dataframe(testing_ids, 0.05, 'Zone17')



### Combine predictions from all zones ###
frames = [zone1_scores,
                   zone2_scores,
                   zone3_scores,
                   zone4_scores,
                   zone5_scores,
                   zone6_scores,
                   zone7_scores,
                   zone8_scores,
                   zone9_scores,
                   zone10_scores,
                   zone11_scores,
                   zone12_scores,
                   zone13_scores,
                   zone14_scores,
                   zone15_scores,
                   zone16_scores,
                   zone17_scores]
all_zone_scores = pd.concat(frames)


### Create subsimssion datatset... ###
submission_scores = pd.read_csv('C:/Users/kristopher.patton/Desktop/TSA_Kaggle/Data/stage1_sample_submission.csv')

sf.write_scores_to_csv(submission_scores, all_zone_scores, 'C:/Users/kristopher.patton/Desktop/TSA_Kaggle/Data/submission_scores.csv')












