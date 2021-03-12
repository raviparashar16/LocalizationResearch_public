"""
This is an implementation of the weighted centroid localization algorithm.
"""

__author__ = 'Ravi Parashar'
__version__ = '1.0'

wifi_ap = 0, 0
path_locations = []
RSS_measurements = []
weights = []
curr_locs_x = []
curr_locs_y = []
g_const = 2

with open("Dataset1.datalog", "r") as dataset_file:
    arr = []
    # skips header
    next(dataset_file)
    # data moved from dataset into array
    for line in dataset_file:
        attributes = []
        attributes = line.split()
        arr.append(attributes)
    curr = 0
    # values added to path location array and RSS measurement arrays
    while curr < arr.__len__() - 1:
        pos = (arr[curr][3], arr[curr][4])
        path_locations.append(pos)
        rss_meas = float(arr[curr][14])
        RSS_measurements.append(rss_meas)
        curr += 1

curr = 0
while curr < path_locations.__len__():
    curr_loc_x, curr_loc_y = path_locations[curr]
    # x locations thus far
    curr_locs_x.append(float(curr_loc_x))
    # y locations thus far
    curr_locs_y.append(float(curr_loc_y))
    # weight added to weights array for weights thus far
    weight = ((10 ** (RSS_measurements[curr] / 10)) ** g_const) ** 0.5
    weights.append(weight)
    inner_curr = 0
    sum_x = 0
    sum_y = 0
    total_weight = sum(weights)
    while inner_curr < weights.__len__():
        # x and y coordinates weighted by RSS measurement at current location divided by total RSS measurements so far
        weightage = weights[inner_curr] / total_weight
        sum_x += weightage * curr_locs_x[inner_curr]
        sum_y += weightage * curr_locs_y[inner_curr]
        inner_curr += 1
    wifi_ap = sum_x, sum_y
    # outputs final estimate of wifi AP location
    if curr == path_locations.__len__() - 1:
        print("AP estimate: " + str(wifi_ap))
    curr += 1
