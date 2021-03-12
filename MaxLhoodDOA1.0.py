"""
This is an implementation of a non-deterministic maximum likelihood localization approach.
It uses the cumulative error b/w estimated DOA and actual DOA for candidate points to estimate the Wi-Fi AP location.
"""

__author__ = 'Ravi Parashar'
__version__ = '1.0'

import random
import math
import statistics as stat
import time

counter = 0
locs = []
# runs program 100 times for dataset to get average distance error for AP location estimate
while counter < 1:
    # wifi AP location set to 0 initially
    APx = 0
    APy = 0
    wifi_ap = (APx, APy)
    # lists created for path locations, angles, random samples, and the probabilities of these samples
    path_locations = []
    RSS_x_grad = []
    RSS_y_grad = []
    x_vals = []
    y_vals = []
    z_vals = []
    w_vals = []
    std_k = 1
    # previous amount of path locations included for avg estimated doa
    prev_sample_size = 100
    # range in which particles can be randomly generated
    possible_x = list(range(-3, 11))
    possible_y = list(range(-8, 15))
    # number of particles generated for each path location
    num_particles = 400

    # function to convert quaternion value to degrees in radians
    def quaternion_to_yaw(x, y, z, w):
        t0 = 2.0 * (w * z + x * y)
        t1 = 1.0 - 2.0 * (y * y + z * z)
        yaw = math.atan2(t0, t1)
        return yaw


    # points from path data set
    with open("./robot_paths/Dataset1.datalog", "r") as dataset_file:
        arr = []
        next(dataset_file)
        # data moved from dataset into array
        for line in dataset_file:
            attributes = []
            attributes = line.split()
            arr.append(attributes)
        curr = 0
        # values added to x,y,z,w arrays, path location array, and RSS y gradient and RSS x gradient arrays
        while curr < arr.__len__() - 1:
            pos = (arr[curr][3], arr[curr][4])
            RSS_y_grad.append(((float(arr[curr][10]) - float(arr[curr][11])) +
                               (float(arr[curr][12]) - float(arr[curr][13]))) / 0.29)
            RSS_x_grad.append(((float(arr[curr][11]) - float(arr[curr][13])) +
                               (float(arr[curr][10]) - float(arr[curr][12]))) / 0.19)
            path_locations.append(pos)
            x_vals.append(arr[curr][5])
            y_vals.append(arr[curr][6])
            z_vals.append(arr[curr][7])
            w_vals.append(arr[curr][8])
            curr += 1

    # previous amount of path locations included in included in Gaussian probability calculation
    sample_size = path_locations.__len__() - prev_sample_size - 1

    # function to calculate gaussian probability of a point being wifi AP
    def point_error(point, path_loc):
        # coordinates for randomly generated particle
        p_x_coor, p_y_coor = point
        curr = path_loc - sample_size
        error = 0
        while curr < path_loc:
            inner_curr = curr - prev_sample_size
            est_sin_sum = 0
            est_cos_sum = 0
            starting_curr = inner_curr
            weight_sum = 0
            # average estimated DoA calculated
            while inner_curr < curr:
                estimated_grad = math.atan2(float(RSS_y_grad[inner_curr]), float(RSS_x_grad[inner_curr]))
                quat = quaternion_to_yaw(float(x_vals[inner_curr]), float(y_vals[inner_curr]), float(z_vals[inner_curr]),
                                         float(w_vals[inner_curr]))
                estimated_grad += quat
                # DoA made circular
                if estimated_grad > math.pi:
                    estimated_grad = -2 * math.pi + estimated_grad
                elif estimated_grad < -math.pi:
                    estimated_grad = math.pi - abs(-math.pi - estimated_grad)
                weight = 0.99 ** (inner_curr - starting_curr)
                weight_sum += weight
                estimated_grad = weight * estimated_grad
                est_sin_sum += math.sin(estimated_grad)
                est_cos_sum += math.cos(estimated_grad)
                inner_curr += 1
            avg_est_sin = est_sin_sum / weight_sum
            avg_est_cos = est_cos_sum / weight_sum
            avg_grad = math.atan2(avg_est_sin, avg_est_cos)
            # actual gradient calculated for past sample_size points
            # avg estimated DoA used from above loop
            r_x_coor, r_y_coor = path_locations[curr]
            r_x_coor = float(r_x_coor)
            r_y_coor = float(r_y_coor)
            actual_grad = math.atan2(p_y_coor - r_y_coor, p_x_coor - r_x_coor)
            error += abs(actual_grad - avg_grad)
            curr += 1
        return error


    # loop starts from path location at sample_size + prev_sample_size
    curr = prev_sample_size + sample_size
    particle_errors = []
    particles = []
    # initial random particles generated in possible range with uniform probability
    for x in range(num_particles):
        particle_x = random.choice(possible_x)
        particle_y = random.choice(possible_y)
        particle = particle_x, particle_y
        particles.append(particle)
    # unique particles
    particles = list(set(particles))
    # cumulative error for each particle initialized as 0
    for part in particles:
        particle_errors.append(0)
    # loops through each path location
    while curr < path_locations.__len__():
        curr_inner = 0
        # cumulative error calculated for each particle
        while curr_inner < particles.__len__():
            particle_errors[curr_inner] += point_error(particles[curr_inner], curr)
            curr_inner += 1
        # at the last path location, the particle with the least cumulative error is chosen as the Wi-Fi AP
        if curr == path_locations.__len__() - 1:
            curr_inner = 0
            max_point = 0, 0
            least_error = float('inf')
            while curr_inner < particles.__len__():
                if particle_errors[curr_inner] < least_error:
                    least_error = particle_errors[curr_inner]
                    max_point = particles[curr_inner]
                curr_inner += 1
            wifi_ap = max_point
            locs.append(wifi_ap)
            print("Wi-Fi AP estimate is: " + str(wifi_ap))
        curr += 1
    counter += 1
# all 100 wifi AP estimates printed
print(locs)
err = []
error = 0
for loc in locs:
    x, y = loc
    curr_error = (math.sqrt((9 - x) ** 2 + (0 - y) ** 2))
    curr_error1 = curr_error ** 2
    error += curr_error1
    err.append(curr_error)
print(math.sqrt(error / locs.__len__()))
print(stat.stdev(err))
