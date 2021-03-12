"""
This is an implementation of a particle filtering algorithm which calculates DOA from RSS measurements and uses this to
calculate probabilities for candidate APs. It assumes a Gaussian distribution of noise.
Gaussian probability calculated for unique particles. When one particle left, program finishes.
Multithreading used for particle probability calculations.
"""

__author__ = 'Ravi Parashar'
__version__ = '2.8'

import random
import math
import statistics as stat
from threading import Thread

counter = 0
locs = []
# runs program 100 times for dataset to get average distance error for AP location estimate
while counter < 100:
    # wifi AP location set to 0 initially
    APx = 0
    APy = 0
    wifi_ap = (APx, APy)
    # lists created for path locations, angles, random samples, and the probabilities of these samples
    path_locations = []
    max_points = []
    max_point_probs = []
    RSS_x_grad = []
    RSS_y_grad = []
    unique_particle_probs = []
    x_vals = []
    y_vals = []
    z_vals = []
    w_vals = []
    std_k = 1
    # previous amount of path locations included for avg estimated doa
    prev_sample_size = 100
    # previous amount of path locations included in included in Gaussian probability calculation
    sample_size = 20
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

    # function to calculate gaussian probability of a point being wifi AP
    def gauss_prob(point, path_loc):
        # coordinates for randomly generated particle
        p_x_coor, p_y_coor = point
        prod_sum = 1
        curr = path_loc - sample_size
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
            # actual DoA from candidate point to current location calculated for past sample_size points (using loop)
            r_x_coor, r_y_coor = path_locations[curr]
            r_x_coor = float(r_x_coor)
            r_y_coor = float(r_y_coor)
            actual_grad = math.atan2(p_y_coor - r_y_coor, p_x_coor - r_x_coor)
            # avg estimated DoA used from above loop
            error = abs(actual_grad - avg_grad)
            # guass probability formula
            prod_sum *= ((1 / (std_k * (math.sqrt(2 * math.pi)))) * (math.e ** (-((error ** 2) / (2 * (std_k ** 2))))))
            curr += 1
        return point, prod_sum


    # loop starts from path location at sample_size + prev_sample_size
    curr = prev_sample_size + sample_size
    particle_probs = []
    particles = []
    particles_x = []
    particles_y = []
    range_probs = []
    rands = []
    # initial random particles generated in possible range with uniform probability
    for x in range(num_particles):
        particle_x = random.choice(possible_x)
        particle_y = random.choice(possible_y)
        particles_x.append(particle_x)
        particles_y.append(particle_y)
        particle = particle_x, particle_y
        particles.append(particle)
    while curr < path_locations.__len__():
        curr_inner = 0
        maxPoint = particles[0]
        maxPoint_prob = -1
        particle_prob_sum = 0
        # gaussian probability calculated for each particle and most probable AP location becomes particle with greatest
        # probability
        unique_particles = list(set(particles))
        if len(unique_particles) == 1:
            locs.append(wifi_ap)
            print("Program exited early because only one particle left. Wi-Fi AP estimate is: " + str(wifi_ap))
            break
        unique_particle_probs = []
        threads = []

        # function for threads to call
        def call_gauss_prob(point, path_loc):
            unique_particle_probs.append(gauss_prob(point, path_loc))

        # creates a thread to call gauss prob function for each unique particle
        for parts in unique_particles:
            t = Thread(target=call_gauss_prob, args=(parts, curr))
            t.start()
            threads.append(t)
        for t in threads:
            t.join()
        ucount = 0
        maxPoint = wifi_ap
        maxPoint_prob = 0
        # point with maximum probability obtained
        while ucount < unique_particle_probs.__len__():
            un_part, un_part_prob = unique_particle_probs[ucount]
            if un_part_prob > maxPoint_prob:
                maxPoint = un_part
                maxPoint_prob = un_part_prob
            ucount += 1
        # adds probabilities of unique particles to list containing probabilities for particles at same location
        for part in particles:
            num = 0
            found = False
            while num < unique_particle_probs.__len__():
                un_part, un_part_prob = unique_particle_probs[num]
                if part == un_part:
                    particle_probs.append(un_part_prob)
                    particle_prob_sum += un_part_prob
                    found = True
                num += 1
            if not found:
                print("Not found!")
        curr_inner = 0
        is_zero = False
        # loop normalizes the probabilities for the particles
        while curr_inner < particles.__len__():
            if particle_prob_sum > 0:
                particle_probs[curr_inner] /= particle_prob_sum
            else:
                is_zero = True
                break
            curr_inner += 1
        # if the sum of the probabilities of the particles is zero, must redo the whole loop
        if is_zero:
            break
        curr_sum = 0
        # upper bounds put in an array by adding particle probabilities
        for i in particle_probs:
            curr_sum += i
            range_probs.append(curr_sum)
        # random numbers generated in uniform distribution form 0 to 1
        for x in range(num_particles):
            rands.append(random.uniform(0, 1))
        count = 0
        new_particles = []
        # new particles generated based on the range in range_probs[] in which random numbers in rands[] fall
        # and the corresponding particle from the set of old particles
        while count < rands.__len__():
            count1 = 0
            while count1 < range_probs.__len__():
                lower = 0
                if count1 > 0:
                    lower = range_probs[count1 - 1]
                upper = range_probs[count1]
                if count1 == range_probs.__len__() - 1:
                    upper = 1
                if lower < rands[count] <= upper:
                    break
                count1 += 1
            new_particles.append(particles[count1])
            part_x, part_y = particles[count1]
            particles_x.append(part_x)
            particles_y.append(part_y)
            count += 1
        # particles array set equal to the new particles generated
        particles = new_particles[:]
        # particle probabilities, random numbers, and ranges reset
        particle_probs = []
        unique_particle_probs = []
        rands = []
        range_probs = []
        max_points.append(maxPoint)
        APx, APy = maxPoint
        wifi_ap = APx, APy
        # if counter == 0:
        # print(wifi_ap)
        max_point_probs.append(maxPoint_prob)
        # if last particle, wifi_ap estimate added to locs[]
        if curr == path_locations.__len__() - 1:
            locs.append(wifi_ap)
            print("Wi-Fi AP estimate is: " + str(wifi_ap))
        curr += 1
    counter += 1
err = []
error = 0
for loc in locs:
    x, y = loc
    curr_error = (math.sqrt((9 - x) ** 2 + (0 - y) ** 2))
    curr_error1 = curr_error ** 2
    error += curr_error1
    err.append(curr_error)
print("rmse: " + str(math.sqrt(error / locs.__len__())))
print("std dev: " + str(stat.stdev(err)))
