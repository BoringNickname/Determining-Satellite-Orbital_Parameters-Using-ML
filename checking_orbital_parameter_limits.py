#%%
file1 = open('amateur.txt', 'r')
Lines = file1.readlines()

inclination, right_ascention, eccentricity, arg_of_perigee, mean_anomaly, mean_motion = [],[],[],[],[],[]
for i,line in enumerate(Lines):
    if (i+1)%3==0:
        second_line = line.strip().split()
        inclination.append(float(second_line[2]))
        right_ascention.append(float(second_line[3]))
        eccentricity.append(float(second_line[4][:1]+'.'+second_line[4][1:]))
        arg_of_perigee.append(float(second_line[5]))
        mean_anomaly.append(float(second_line[6]))
        mean_motion.append(float(second_line[7]))

import matplotlib.pyplot as plt
import numpy as np 

fig, ax = plt.subplots(2,3, figsize= (20,10))
ax[0][0].hist(inclination)
ax[0][0].set_xlabel('inclination value')
ax[0][0].set_ylabel('No. of occurences')
ax[0][0].set_title('inclination values distribution for Glasgow\'s latitude')

ax[0][1].hist(right_ascention)
ax[0][1].set_xlabel('right ascention value')
ax[0][1].set_ylabel('No. of occurences')
ax[0][1].set_title('right ascention values distribution for Glasgow\'s latitude')

ax[0][2].hist(eccentricity)
ax[0][2].set_xlabel('eccentricity value')
ax[0][2].set_ylabel('No. of occurences')
ax[0][2].set_title('eccentricity values distribution for Glasgow\'s latitude')

ax[1][0].hist(arg_of_perigee)
ax[1][0].set_xlabel('argument of perigee value')
ax[1][0].set_ylabel('No. of occurences')
ax[1][0].set_title('argument of perigee values distribution for Glasgow\'s latitude')

ax[1][1].hist(mean_anomaly)
ax[1][1].set_xlabel('mean anomaly value')
ax[1][1].set_ylabel('No. of occurences')
ax[1][1].set_title('mean anomaly values distribution for Glasgow\'s latitude')

ax[1][2].hist(mean_motion)
ax[1][2].set_xlabel('mean motion value')
ax[1][2].set_ylabel('No. of occurences')
ax[1][2].set_title('mean motion  values distribution for Glasgow\'s latitude')

plt.savefig('./range_of_orbital_parameters/parameters_combined.png')
plt.show()
# %%