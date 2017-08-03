from utilities import image_reader as ir
import numpy as np
import pandas as pd
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D
from keras.layers.normalization import BatchNormalization
from keras.models import load_model
from utilities.data_processing_functions import *
import random

#<><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><>
data_dir = 'C:/Users/john.hife/Documents/workspaces/TSA Kaggle/data/'
zone_num = 1
zone_angles = [0,2,6,8,10,12,14]
aps_dir = data_dir + 'stage1_aps/'
labels_file = data_dir + 'ZONE_LABELS/' + 'ZONE1_LABELS.csv'
#model_file_name = 'zone1_model_8-2-2017_training.h5'
model_file_name = 'zone1_model_8-2-2017.h5'
#<><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><>

### read training ids
zone_labels = pd.read_csv(labels_file)
sum(zone_labels['label'] == 0)
sum(zone_labels['label'] == 1)

#partition labeled data into train/test
random.seed(12345)
training_ids, testing_ids = create_train_test(zone_labels, test_pct=0.15)
training_ids.shape
testing_ids.shape

#create zone data set
zone_dataset = create_zone_dataset(aps_dir, training_ids['id'], zone_num, zone_angles)
zone_dataset = zone_dataset.reshape(zone_dataset.shape[0], zone_dataset.shape[1], zone_dataset.shape[2], 1)
#plot_image(zone_dataset[0,:,:,:])

training_labels = np.array(training_ids['label'])[0:zone_dataset.shape[0]]
training_labels = np_utils.to_categorical(training_labels, 2)


### define model ###
zone_model = Sequential()
zone_model.add(Convolution2D(32, (3, 3), activation ='elu', input_shape = (zone_dataset.shape[1], zone_dataset.shape[2], 1)))
zone_model.add(Dropout(0.10))
zone_model.add(MaxPooling2D(pool_size = (2, 2)))
#zone_model.add(BatchNormalization())
zone_model.add(Convolution2D(32, (3, 3), activation ='elu'))
#zone_model.add(BatchNormalization())
zone_model.add(Dropout(0.10))
zone_model.add(MaxPooling2D(pool_size = (2, 2)))
zone_model.add(Flatten())
#zone_model.add(Dense(2, activation = 'softmax'))
zone_model.add(Dense(1, activation='sigmoid'))

#Compile model
zone_model.compile(#loss = 'mean_squared_logarithmic_error',
    loss='binary_crossentropy',
    optimizer = 'adam',
    metrics = ['accuracy'])

#Fit zone_model
class_weight = {0:1, 1:10}
zone_model.fit(zone_dataset, training_labels[:, 1],
               batch_size = 30, epochs = 150, verbose = 1, class_weight=class_weight)

train_preds = zone_model.predict(zone_dataset, batch_size = 1, verbose = 1)
#max(train_preds)
#min(train_preds)

#zone_model.save(model_file_name)
#zone_model = load_model(model_file_name)

#create test set
zone_testset = create_zone_dataset(aps_dir, testing_ids['id'], zone_num, zone_angles)
zone_testset = zone_testset.reshape(zone_testset.shape[0], zone_testset.shape[1], zone_testset.shape[2], 1)

#compute probabilities for test set
preds = zone_model.predict(zone_testset, batch_size=1, verbose = 1)
max(preds)
min(preds)

zone_predictions = pd.concat([testing_ids.reset_index(drop=True),
                              pd.DataFrame(preds, columns=['prob'])], axis=1)
zone_predictions.to_csv('zone_' + str(zone_num) + '_predictions.csv', index=False)

from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_fscore_support
def plotROC(labels, probs):
    false_positive_rate, true_positive_rate, thresholds = roc_curve(labels, probs)
    roc_auc = auc(false_positive_rate, true_positive_rate)
    plt.title('Receiver Operating Characteristic')
    plt.plot(false_positive_rate, true_positive_rate, 'b',label='AUC = %0.2f'% roc_auc)
    plt.legend(loc='lower right')
    plt.plot([0,1],[0,1],'r--')
    plt.xlim([-0.1,1.2])
    plt.ylim([-0.1,1.2])
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    plt.show()

threshold = 0.5
area_under_roc = roc_auc_score(zone_predictions['label'], zone_predictions['prob'])
plotROC(zone_predictions['label'], zone_predictions['prob'])
precision_recall_fscore_support(zone_predictions['label'], np.where(zone_predictions['prob'] > threshold, 1,0))
precision, recall, fscore, support = precision_recall_fscore_support(zone_predictions['label'], np.where(zone_predictions['prob'] > threshold, 1,0), average='binary')
print('AUC: ' + str(area_under_roc))
print('Precision: ' + str(precision))
print('Recall: ' + str(recall))
print('FScore: ' + str(fscore))

# data = ir.read_data('C:/Users/john.hife/Documents/workspaces/TSA Kaggle/data/stage1_aps/49c3fc4b14948ab097a3462bb825e2f0.aps')
# data = ir.read_data('C:/Users/john.hife/Documents/workspaces/TSA Kaggle/data/stage1_aps/1cb13f156bd436222447dd658180bd96.aps')
# data = ir.read_data('C:/Users/john.hife/Documents/workspaces/TSA Kaggle/data/stage1_aps/b26e79a465ebac796ceb022cb998c236.aps')
# data = ir.read_data('C:/Users/john.hife/Documents/workspaces/TSA Kaggle/data/stage1_aps/adc28b596accd5039620bfef68feacf3.aps')
# data = ir.read_data('C:/Users/john.hife/Documents/workspaces/TSA Kaggle/data/stage1_aps/ce8ab5548464940a8a14a813a75500d0.aps')
# data = ir.read_data('C:/Users/john.hife/Documents/workspaces/TSA Kaggle/data/stage1_aps/104e284349ffe68378745fc8c5638eab.aps')
# data = ir.read_data('C:/Users/john.hife/Documents/workspaces/TSA Kaggle/data/stage1_aps/6ec5ce5500d9ce14887e64f47486ad5f.aps')
# data = ir.read_data('C:/Users/john.hife/Documents/workspaces/TSA Kaggle/data/stage1_aps/5b900a35f3e9591914ea328cacb73dfa.aps')
# plot_images(data)
# plot_image(data,0)





