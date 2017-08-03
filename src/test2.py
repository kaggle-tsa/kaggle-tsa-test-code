import keras
from keras import optimizers
from keras.models import Sequential, Model
from keras.layers import Dropout, Flatten, Dense, GlobalAveragePooling2D
from keras.utils import np_utils


from utilities.data_processing_functions import *
from keras.utils import plot_model
#import tensorflow as tf

#create training data set
dir = 'C:/Users/john.hife/Documents/workspaces/TSA Kaggle/data/stage1_aps/'
training_ids = pd.read_csv('C:/Users/john.hife/Documents/workspaces/TSA Kaggle/data/training_ids.csv')
training_ids = training_ids['id']
zone1_angles = [0,2,6,8,10,12,14]
#zone_dataset = create_zone_dataset(dir, training_ids,zone_num=1, zone_angles=zone1_angles)
#zone_dataset = zone_dataset.reshape(zone_dataset.shape[0], zone_dataset.shape[1], zone_dataset.shape[2], 3)
zone_1_dataset = create_zone_dataset_rgb(dir, training_ids,zone_num=1, zone_angles=zone1_angles)
zone1_labels = pd.read_csv('C:/Users/john.hife/Documents/workspaces/TSA Kaggle/data/ZONE_LABELS/ZONE1_LABELS.csv')
zone1_labels = np.array(zone1_labels['label'])[0:zone_1_dataset.shape[0]]
zone1_labels = np_utils.to_categorical(zone1_labels, 2)

#pretrained neural network for transfer learning
model = keras.applications.xception.Xception(include_top=False, weights='imagenet',
                                             input_shape=(1750, 140, 3))
model.summary()
model.layers.__sizeof__()
#remove top layers of trained model
# num_layers_to_remove = 5
# for i in range(0,num_layers_to_remove):
#     model.layers.pop() # Get rid of the classification layer
#     #model.outputs = [model.layers[-1].output]
#     model.inputs = [model.layers[-1].input]
#     model.layers[-1].outbound_nodes = []

#specify layers that should not be retrained
num_layers_to_freeze = 1224
for layer in model.layers[:num_layers_to_freeze]:
    layer.trainable = False

#specify top and output layers
x = model.output
x = Flatten()(x)
x = Dense(1024, activation="relu")(x)
x = Dropout(0.5)(x)
x = Dense(1024, activation="relu")(x)
predictions = Dense(2, activation="softmax")(x)

#combine input and outoput layers
model_final = Model(input = model.input, output = predictions)
model_final.compile(loss = "binary_crossentropy", optimizer = optimizers.SGD(lr=0.0001, momentum=0.9), metrics=["accuracy"])
model_final.summary()
plot_model(model_final, to_file='model.png')


#Fit zone_model
class_weight = {0 : 1, 1: 10}
model_final.fit(zone_1_dataset, zone1_labels,
          batch_size = 1, epochs = 50, verbose = 1, class_weight=class_weight)
