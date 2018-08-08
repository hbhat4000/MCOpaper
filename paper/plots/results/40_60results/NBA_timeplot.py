import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

rawdata = {'Training Size': [40,60],
    'Naive CTMC': [61,62],
    'Fixed CTMC': [154,165]}

df = pd.DataFrame(rawdata, columns = ['Training Size','Naive CTMC', 'Fixed CTMC'])
print(df)

# Setting the positions and width for the bars
pos = list(range(len(df['Naive CTMC'])))
width = 0.25

# Plotting the bars
fig, ax = plt.subplots(figsize = (10,5))

# Create a bar with Our_model data,
# in position pos
plt.bar(pos,
    #using df['Our_model'],
    df['Naive CTMC'],
    # of width,
    width,
    # with alpha 0.5
    alpha = 0.5,
    # with color 
    color = '#EE3224',
    # with label the first value in Data
    label = df['Training Size'][0])

# Create a bar with HMMs data,
# in a position pos + some width buffer,
plt.bar([p + width for p in pos],
    #using df['HMMS'] data,
    df['Fixed CTMC'],
    # of width
    width,
    # with alpha 0.5
    alpha = 0.5,
    # with color
    color = '#FFC222',
    # with label the second value in Data
    label = df['Training Size'][1])

# Set the y axis label 
ax.set_ylabel('Time (mins) in log scale')

# Set the chart's title
ax.set_title('System training time naive vs. fixed CTMC')

# Set the position of the x ticks
ax.set_xticks([p + 1.5 * width for p in pos])

# Set the labels for the x ticks
ax.set_xticklabels(df['Training Size'])

# Setting the x-axis and y-axis limits
plt.xlim(min(pos)-width, max(pos)+width*4)
#plt.ylim([0, max(df['Our model'] + df['HMMS'])] )
plt.yscale('log')
# Adding the legend and showing the plot
plt.legend(['Naive CTMC','Fixed CTMC'], loc = 'best')
plt.rc('font',size = 14)
plt.grid()
plt.savefig('NBAtraintime.eps',format = 'eps',dpi = 1200)
