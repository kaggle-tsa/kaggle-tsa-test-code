from utilities import image_reader as ir
import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot
import matplotlib.animation
import math


#input: aps data, angle, and zone number
#output: slice corresponding to that zone for that angle
#if zone is not viewable from that angle returns None
def get_zone_slice(data, i, zone_num, buffer=0):

    #hard coded pixel regions for a given angle and body zone
    zone_pixels = {
        0:{
            1: np.index_exp[60:200,400:520,0], 2: np.index_exp[50:240,480:660,0], 3: np.index_exp[310:480,400:520,0],
            4: np.index_exp[270:480,480:660,0], 5: np.index_exp[200:320,400:460,0], 6: np.index_exp[120:260,300:400,0],
            7: np.index_exp[250:390,300:400,0], 8: np.index_exp[100:240,180:280,0], 9: np.index_exp[200:300,180:300,0],
            10: np.index_exp[260:400,180:300,0], 11: np.index_exp[100:240,130:190,0], 12: np.index_exp[260:400,130:190,0],
            13: np.index_exp[100:240,70:130,0], 14: np.index_exp[260:400,70:130,0], 15: np.index_exp[110:210,0:70,0],
            16: np.index_exp[260:400,0:70,0], 17: None
        },
        2: {
            1: np.index_exp[100:230,400:520,2], 2: np.index_exp[50:260,480:660,2], 3: np.index_exp[310:480,400:520,2],
            4: np.index_exp[270:480,480:660,2], 5: np.index_exp[150:400,400:460,2], 6: None,
            7: np.index_exp[150:400,300:400,2], 8: np.index_exp[120:230,180:280,2], 9: np.index_exp[210:290,180:300,2],
            10: np.index_exp[240:400,180:300,2], 11: np.index_exp[120:280,130:190,2], 12: np.index_exp[240:400,130:190,2],
            13: np.index_exp[120:280,70:130,2], 14: np.index_exp[240:400,70:130,2], 15: np.index_exp[120:280,0:70,2],
            16: np.index_exp[240:420,0:70,2], 17: None
        },
        4:{
            1: None, 2: None, 3: np.index_exp[250:350,420:520,4],
            4: np.index_exp[220:350,500:650,4], 5: np.index_exp[200:275,400:480,4], 6: None,
            7: np.index_exp[100:380,300:400,4], 8: None, 9: None,
            10: np.index_exp[140:400,180:300,4], 11: None, 12: np.index_exp[180:380,130:190,4],
            13: None, 14: np.index_exp[200:380,60:130,4],15: np.index_exp[110:210,0:70,4],
            16: np.index_exp[200:380,0:70,4], 17: np.index_exp[300:400,400:480,4]
        },
        6:{
            1: np.index_exp[320:450,400:520,6], 2: np.index_exp[280:450,480:660,6], 3: np.index_exp[80:270,400:520,6],
            4: np.index_exp[80:270,480:660,6], 5: None, 6: np.index_exp[290:400,300:400,6],
            7: np.index_exp[140:310,300:400,6], 8: np.index_exp[290:420,180:280,6], 9: np.index_exp[260:310,180:300,6],
            10: np.index_exp[160:300,180:300,6], 11: np.index_exp[250:420,130:190,6], 12: np.index_exp[180:280,130:190,6],
            13: np.index_exp[260:420,70:130,6], 14: np.index_exp[180:280,70:130,6], 15: np.index_exp[300:420,0:70,6],
            16: np.index_exp[150:290,0:70,6], 17: np.index_exp[170:420,400:460,6]
        },

        8:{
            1: np.index_exp[310:480,400:520,8],2: np.index_exp[270:480,480:660,8], 3: np.index_exp[60:200,400:520,8],
            4: np.index_exp[50:240,480:660,8], 5: None, 6: np.index_exp[250:390,300:400,8],
            7: np.index_exp[120:260,300:400,8],8: np.index_exp[260:400,180:300,8], 9: np.index_exp[200:300,180:300,8],
            10: np.index_exp[100:240,180:280,8], 11: np.index_exp[260:400,130:190,8], 12: np.index_exp[100:240,130:190,8],
            13: np.index_exp[260:400,70:130,8], 14: np.index_exp[100:240,70:130,8], 15: np.index_exp[260:400,0:70,8],
            16: np.index_exp[110:210,0:70,8], 17: np.index_exp[180:320,400:460,8]
        },

        10:{
            1: np.index_exp[250:480,400:520,10], 2: np.index_exp[230:480,480:660,10], 3: np.index_exp[80:200,400:520,10],
            4: np.index_exp[80:270,480:660,10], 5: None, 6: np.index_exp[180:400,300:400, 10],
            7: np.index_exp[140:220,300:400,10], 8: np.index_exp[220:350,180:280,10], 9: np.index_exp[190:230,180:300,10],
            10: np.index_exp[150:230,180:300,10], 11: np.index_exp[220:350,130:190,10], 12: np.index_exp[120:250,130:190,10],
            13: np.index_exp[220:350,70:130,10], 14: np.index_exp[90:250,70:130,10], 15: np.index_exp[220:350,0:70,10],
            16: np.index_exp[90:250,0:70,10], 17: np.index_exp[100:290,400:460,10]
        },

        12:{
            1: np.index_exp[200:280,420:500,12], 2: np.index_exp[200:280,500:650,12],3: None,
            4: None, 5: np.index_exp[240:300,380:450,12], 6: np.index_exp[140:340,300:400,12],
            7: None, 8: np.index_exp[140:320,180:300,12], 9: None,
            10: np.index_exp[100:240,180:280,12], 11: np.index_exp[140:280,130:190,12], 12: None,
            13: np.index_exp[140:270,60:130,12], 14: None, 15: np.index_exp[100:300,0:70,12],
            16: None,17: np.index_exp[140:200,400:600,12]
        },

        14: {
            1: np.index_exp[30:220,400:520,14], 2: np.index_exp[30:250,480:660,14], 3: np.index_exp[270:450,400:520,14],
            4: np.index_exp[240:450,480:660,14], 5: np.index_exp[100:400,400:460,14], 6: np.index_exp[100:400,300:400,14],
            7: None, 8: np.index_exp[80:270,180:280, 14], 9: np.index_exp[250:310,180:300,14],
            10: np.index_exp[270:370,180:300,14], 11: np.index_exp[80:250,130:190,14], 12: np.index_exp[220:350,130:190,14],
            13: np.index_exp[80:250,70:130,14], 14: np.index_exp[220:350,70:130,14], 15: np.index_exp[80:240,0:70,14],
            16: np.index_exp[220:400,0:70,14], 17: None
        }
    }

    slice = zone_pixels[i][zone_num]
    # return data[slice]
    if slice != None:
        slice_buffer = np.index_exp[max((slice[0].start-buffer),0):min((slice[0].stop+buffer),512),
                       max((slice[1].start-buffer),0):min((slice[1].stop+buffer),660), i]
        return slice_buffer
    else:
        return None

#input: aps data, angle, and zone number
#output: subset of aps data corresponding to that zone for that angle
#if zone is not viewable from that angle returns None
def get_zone_data(data, i, zone_num, buffer=0):
    slice = get_zone_slice(data, i, zone_num, buffer)
    if slice != None:
        return data[slice]
    else:
        return None


def plot_image(data, i=0):
    fig = matplotlib.pyplot.figure(figsize=(16, 16))
    ax = fig.add_subplot(111)
    if len(data.shape) == 3:
        ax.imshow(np.flipud(data[:, :, i].transpose()), cmap='viridis')
    else:
        ax.imshow(np.flipud(data[:, :].transpose()), cmap='viridis')

def plot_images(data):
    fig = matplotlib.pyplot.figure(figsize = (16,16))
    ax = fig.add_subplot(111)
    def animate(i):
        im = ax.imshow(np.flipud(data[:,:,i].transpose()), cmap = 'viridis')
        return [im]
    return matplotlib.animation.FuncAnimation(fig, animate, frames=range(0,data.shape[2]), interval=200, blit=True)



def get_image_list(data, zone_num, zone_angles):
    image_list = []
    for i in range(0,len(zone_angles)):
        image_list.append(get_zone_data(data, zone_angles[i], zone_num, buffer=10))
        #plot_image(image_list[i])
    return image_list

def get_max_width(image_list, dim):
    widths = []
    for i in range(0,len(image_list)):
        widths.append(image_list[i].shape[dim])
    return max(widths)

def aggregate_zone_images(image_list, max_x_width, max_y_width):
    a = (max_x_width - image_list[0].shape[0])/2
    b = (max_y_width - image_list[0].shape[1])/2
    npad = ((math.floor(a),math.ceil(a)), (math.floor(b), math.ceil(b)))
    zone_aggregated = np.pad(image_list[0], pad_width=npad, mode='constant', constant_values=0)
    for i in range(1,len(image_list)):
        a = (max_x_width - image_list[i].shape[0]) / 2
        b = (max_y_width - image_list[i].shape[1]) / 2
        npad = ((math.floor(a), math.ceil(a)), (math.floor(b), math.ceil(b)))
        zone_aggregated = np.row_stack((zone_aggregated , np.pad(image_list[i],
                                                                 pad_width=npad,
                                                                 mode='constant',
                                                                 constant_values=0)))
    return zone_aggregated


def create_zone_dataset(path, ids, zone_num, zone_angles):

    #n = 10
    n = len(ids)
    zone_dataset = None
    for i in range(n):
        id = ids.iloc[i]
        image_path = path + id + '.aps'
        data = ir.read_data(image_path)

        image_list = get_image_list(data, zone_num, zone_angles)
        max_x_width = get_max_width(image_list, dim=0)
        max_y_width = get_max_width(image_list, dim=1)
        aggregated_image = aggregate_zone_images(image_list, max_x_width, max_y_width)

        if zone_dataset is None:
            zone_dataset = np.empty((n, aggregated_image.shape[0], aggregated_image.shape[1]))

        zone_dataset[i, :, :] = aggregated_image

    return zone_dataset


def create_zone_dataset_rgb(path, ids, zone_num, zone_angles):

    #n = 10
    n = len(ids)
    zone_dataset = None
    for i in range(n):
        id = ids[i]
        image_path = path + id + '.aps'
        data = ir.read_data(image_path)

        image_list = get_image_list(data, zone_num, zone_angles)
        max_x_width = get_max_width(image_list, dim=0)
        max_y_width = get_max_width(image_list, dim=1)
        aggregated_image = aggregate_zone_images(image_list, max_x_width, max_y_width)
        aggregated_image = aggregated_image.reshape(aggregated_image.shape[0], aggregated_image.shape[1], 1)

        if zone_dataset is None:
            zone_dataset = np.empty((n, aggregated_image.shape[0], aggregated_image.shape[1], 3))

        zone_dataset[i, :, :, :0] = aggregated_image
        zone_dataset[i, :, :, :1] = aggregated_image
        zone_dataset[i, :, :, :2] = aggregated_image

    return zone_dataset


def create_train_test(labels, test_pct):
    temp = labels[labels['label'] == 0]
    test_rows = temp.sample(math.floor(temp.shape[0] * test_pct))
    temp = labels[labels['label'] == 1]
    test_rows = test_rows.append(temp.sample(math.floor(temp.shape[0] * test_pct)))
    train_rows = labels.loc[~labels.index.isin(test_rows.index)]
    # random.seed(12345)
    # training_ids, testing_ids = create_train_test(zone_labels, test_pct=0.15)
    # training_ids.shape
    # sum(training_ids['label'] == 0)
    # sum(training_ids['label'] == 1)
    # testing_ids.shape
    # sum(testing_ids['label'] == 0)
    # sum(testing_ids['label'] == 1)
    # training_ids[0:5]
    # testing_ids[0:5]
    # set.intersection(set(training_ids['id']), set(testing_ids['id']))

    return train_rows, test_rows
