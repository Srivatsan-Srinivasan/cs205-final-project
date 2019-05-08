import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

shared_times = pd.read_csv('shared_memory_times.csv')

shared_times_2224 = shared_times[shared_times['# buses'] == 2224]
shared_times_189 = shared_times[shared_times['# buses'] == 189]

fig,ax = plt.subplots(2,1)
ax[0].plot(shared_times_2224[shared_times_2224['# cont']==6]['# processors'], shared_times_2224[shared_times_2224['# cont']==6]['Speed up'],
	label = '6 contingencies', color = 'red')
ax[0].plot(shared_times_2224[shared_times_2224['# cont']==5]['# processors'], shared_times_2224[shared_times_2224['# cont']==5]['Speed up'],
	label = '5 contingencies', color = 'blue')
ax[0].plot(shared_times_2224[shared_times_2224['# cont']==4]['# processors'], shared_times_2224[shared_times_2224['# cont']==4]['Speed up'],
	label = '4 contingencies', color = 'green')
ax[0].plot(shared_times_2224[shared_times_2224['# cont']==3]['# processors'], shared_times_2224[shared_times_2224['# cont']==3]['Speed up'],
	label = '3 contingencies', color = 'black')
ax[0].plot(shared_times_2224[shared_times_2224['# cont']==2]['# processors'], shared_times_2224[shared_times_2224['# cont']==2]['Speed up'],
	label = '2 contingencies', color = 'orange')
ax[0].plot(shared_times_2224[shared_times_2224['# cont']==1]['# processors'], shared_times_2224[shared_times_2224['# cont']==1]['Speed up'],
	label = '1 contingencies', color = 'purple')
ax[0].set_xlabel('number of cores')
ax[0].set_ylabel('speedup')
ax[0].set_title('Speedup vs Number of Cores for 2224 Buses')
ax[0].legend()

ax[1].plot(shared_times_189[shared_times_189['# cont']==6]['# processors'], shared_times_189[shared_times_189['# cont']==6]['Speed up'],
	label = '6 contingencies', color = 'red')
ax[1].plot(shared_times_189[shared_times_189['# cont']==5]['# processors'], shared_times_189[shared_times_189['# cont']==5]['Speed up'],
	label = '5 contingencies', color = 'blue')
ax[1].plot(shared_times_189[shared_times_189['# cont']==4]['# processors'], shared_times_189[shared_times_189['# cont']==4]['Speed up'],
	label = '4 contingencies', color = 'green')
ax[1].plot(shared_times_189[shared_times_189['# cont']==3]['# processors'], shared_times_189[shared_times_189['# cont']==3]['Speed up'],
	label = '3 contingencies', color = 'black')
ax[1].plot(shared_times_189[shared_times_189['# cont']==2]['# processors'], shared_times_189[shared_times_189['# cont']==2]['Speed up'],
	label = '2 contingencies', color = 'orange')
ax[1].plot(shared_times_189[shared_times_189['# cont']==1]['# processors'], shared_times_189[shared_times_189['# cont']==1]['Speed up'],
	label = '1 contingencies', color = 'purple')
ax[1].set_xlabel('number of cores')
ax[1].set_ylabel('speedup')
ax[1].set_title('Speedup vs Number of Cores for 189 Buses')
ax[1].legend()


plt.show()


