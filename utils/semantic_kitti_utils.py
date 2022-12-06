import numpy as np


class LabelDataConverter:
    """Convert .label binary data to instance id and rgb"""

    def __init__(self, labelscan):

        self.convertdata(labelscan)

    def convertdata(self, labelscan):

        self.semantic_id = []
        self.rgb_id = []
        self.instance_id = []
        self.rgb_arr_id = []

        for counting in range(len(labelscan)):

            sem_id = int(labelscan[counting]) & 0xFFFF  # lower 16 bit
            rgb, rgb_arr = self.get_sem_rgb(sem_id)
            instance_id = int(labelscan[counting]) >> 16  # higher 16 bit
            # rgb = self.get_random_rgb(instance_id)

            # print("Sem label:", sem_id, "Ins label:", instance_id, "Color:", hex(rgb))
            # print(hex(rgb))
            # instance label is given in each semantic label

            self.semantic_id.append(sem_id)
            self.rgb_id.append(rgb)
            self.rgb_arr_id.append(rgb_arr)
            self.instance_id.append(instance_id)


def get_sem_rgb(sem_id):
    RGB_id = 0
    RGB_id_array = [0, 0, 0]
    # not used
    if sem_id == 0:  # unlabled
        RGB_id = 0x000000
        RGB_id_array = [0, 0, 0]
    elif sem_id == 1:  # outlier
        RGB_id = 0xFF0000
        RGB_id_array = [255, 0, 0]

    # instance
    elif sem_id == 10:  # car
        RGB_id = 0x6496F5
        RGB_id_array = [100, 150, 245]
    elif sem_id == 11:  # bicycle
        RGB_id = 0x64E6F5
        RGB_id_array = [100, 230, 245]
    elif sem_id == 13:  # bus
        RGB_id = 0x6450FA
        RGB_id_array = [100, 80, 250]
    elif sem_id == 15:  # motorcycle
        RGB_id = 0x1E3C96
        RGB_id_array = [30, 60, 150]
    elif sem_id == 16:  # on-rails
        RGB_id = 0x0000FF
        RGB_id_array = [0, 0, 255]
    elif sem_id == 18:  # truck
        RGB_id = 0x501EB4
        RGB_id_array = [80, 30, 180]
    elif sem_id == 20:  # other-vehicle
        RGB_id = 0x0000FF
        RGB_id_array = [0, 0, 255]
    elif sem_id == 30:  # person
        RGB_id = 0xFF1E1E
        RGB_id_array = [255, 30, 30]
    elif sem_id == 31:  # bicyclist
        RGB_id = 0xFF28C8
        RGB_id_array = [255, 40, 200]
    elif sem_id == 32:  # motorcyclist
        RGB_id = 0x961E5A
        RGB_id_array = [150, 30, 90]

    # background

    elif sem_id == 40:  # road
        RGB_id = 0xFF00FF
        RGB_id_array = [255, 0, 255]
    elif sem_id == 44:  # parking
        RGB_id = 0xFF96FF
        RGB_id_array = [255, 150, 255]
    elif sem_id == 48:  # sidewalk
        RGB_id = 0x4B004B
        RGB_id_array = [75, 0, 75]
    elif sem_id == 49:  # other-ground
        RGB_id = 0xAF004B
        RGB_id_array = [175, 0, 75]
    elif sem_id == 50:  # building
        RGB_id = 0xFFC800
        RGB_id_array = [255, 200, 0]
    elif sem_id == 51:  # fence
        RGB_id = 0xFF7832
        RGB_id_array = [255, 120, 50]
    elif sem_id == 52:  # other-structure
        RGB_id = 0xFF9600
        RGB_id_array = [255, 150, 0]
    elif sem_id == 60:  # lane-marking
        RGB_id = 0x96FFAA
        RGB_id_array = [150, 255, 170]
    elif sem_id == 70:  # vegetation
        RGB_id = 0x00AF00
        RGB_id_array = [0, 175, 0]
    elif sem_id == 71:  # trunk
        RGB_id = 0x873C00
        RGB_id_array = [135, 60, 0]
    elif sem_id == 72:  # terrain
        RGB_id = 0x96F050
        RGB_id_array = [150, 240, 80]
    elif sem_id == 80:  # pole
        RGB_id = 0xFFF096
        RGB_id_array = [255, 240, 150]
    elif sem_id == 81:  # traffic-sign
        RGB_id = 0xFF0000
        RGB_id_array = [255, 0, 0]
    elif sem_id == 99:  # other-object
        RGB_id = 0x32FFFF
        RGB_id_array = [50, 255, 255]

    # dynamic objects
    elif sem_id == 252:  # moving car
        RGB_id = 0x6496F5
        RGB_id_array = [100, 150, 245]
    elif sem_id == 253:  # moving bicycle
        RGB_id = 0xFF28C8
        RGB_id_array = [255, 40, 200]
    elif sem_id == 254:  # moving person
        RGB_id = 0xFF1E1E
        RGB_id_array = [255, 30, 30]
    elif sem_id == 255:  # moving motorcyle
        RGB_id = 0x961E5A
        RGB_id_array = [150, 30, 90]
    elif sem_id == 256:  # moving on-rails(tram)
        RGB_id = 0x0000FF
        RGB_id_array = [0, 0, 255]
    elif sem_id == 257:  # moving bus
        RGB_id = 0x6450FA
        RGB_id_array = [100, 80, 250]
    elif sem_id == 258:  # moving truck
        RGB_id = 0x501EB4
        RGB_id_array = [80, 30, 180]
    elif sem_id == 259:  # moving other vehicles
        RGB_id = 0x0000FF
        RGB_id_array = [0, 0, 255]
    else:
        RGB_id = 0x000000
        RGB_id_array = [0, 0, 0]

    return RGB_id, RGB_id_array


def get_random_rgb(n):
    n = ((n ^ n >> 15) * 2246822519) & 0xFFFFFFFF
    n = ((n ^ n >> 13) * 3266489917) & 0xFFFFFFFF
    n = (n ^ n >> 16) >> 8
    print(n)
    return tuple(n.to_bytes(3, "big"))


# semantic_mapping = {  # bgr
#     0:  [0, 0, 0],          # "unlabeled", and others ignored
#     1:  [245, 150, 100],    # "car"
#     2:  [245, 230, 100],    # "bicycle"
#     3:  [150, 60, 30],      # "motorcycle"
#     4:  [180, 30, 80],      # "truck"
#     5:  [255, 0, 0],        # "other-vehicle"
#     6:  [30, 30, 255],      # "person"
#     7:  [200, 40, 255],     # "bicyclist"
#     8:  [90, 30, 150],      # "motorcyclist"
#     9:  [255, 0, 255],      # "road"
#     10: [255, 150, 255],    # "parking"
#     11: [75, 0, 75],        # "sidewalk"
#     12: [75, 0, 175],       # "other-ground"
#     13: [0, 200, 255],      # "building"
#     14: [50, 120, 255],     # "fence"
#     15: [0, 175, 0],        # "vegetation"
#     16: [0, 60, 135],       # "trunk"
#     17: [80, 240, 150],     # "terrain"
#     18: [150, 240, 255],    # "pole"
#     19: [0, 0, 255]         # "traffic-sign"
#     }
