from utilities.ZoneCoordinates import ZoneCoordinates
from utilities.data_processing_functions import *
import matplotlib.patches as patches
from utilities import image_reader as ir


def draw_patch(zoneCoordinates):
    x = zoneCoordinates.c1
    width = zoneCoordinates.c2-x
    y = zoneCoordinates.c4
    height = y - zoneCoordinates.c3
    return patches.Rectangle((x,660-y),width, height, linewidth=1, edgecolor='r', facecolor='none')

def plot_image_with_box_from_slice(data, angle, zones, buffer=0):
    fig = matplotlib.pyplot.figure(figsize=(16, 16))
    ax = fig.add_subplot(111)
    # plot_image(data[110:210,0:70,:], 0) # ~ zone 15
    # plot_image(data[260:400,70:130,:], 0) # ~ zone 14
    if len(data.shape) == 3:
        for zone in zones:
            print(type(zone))
            try:
                angleZone = get_zone_slice(data, angle, zone, buffer)
                zoneCoords = ZoneCoordinates(angleZone[0].start, angleZone[0].stop, angleZone[1].start, angleZone[1].stop, zone)
                ax.add_patch(draw_patch(zoneCoords))
            except TypeError:
                continue
        ax.imshow(np.flipud(data[:, :, angle].transpose()), cmap='viridis')
    else:
        ax.imshow(np.flipud(data[:, :].transpose()), cmap='viridis')

#data = ir.read_data('C://Users//christine.ibaraki//Desktop//0a27d19c6ec397661b09f7d5998e0b14.aps')
data = ir.read_data('C:/Users/john.hife/Documents/workspaces/TSA Kaggle/data/stage1_aps/0a27d19c6ec397661b09f7d5998e0b14.aps')

plot_image(get_zone_data(data, 0, 4, buffer=0))
plot_image(get_zone_data(data, 0, 4, buffer=20))

zones = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17]
plot_image_with_box_from_slice(data, 0, zones)
plot_image_with_box_from_slice(data, 2, zones)
plot_image_with_box_from_slice(data, 4, zones)
plot_image_with_box_from_slice(data, 6, zones)
plot_image_with_box_from_slice(data, 8, zones)
plot_image_with_box_from_slice(data, 10, zones)
plot_image_with_box_from_slice(data, 12, zones)
plot_image_with_box_from_slice(data, 14, zones)
