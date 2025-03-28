import os
import matplotlib.pyplot as plt

labelPath = '/Users/wheausti/Documents/STLDP/HLB/Training_Dataset/labels'

labelFiles = os.listdir(labelPath)
labelFiles = [f for f in labelFiles if f.endswith('.txt')]
nLabels = len(labelFiles)

labelCount = {'0':0,'1':0,'2':0,'3':0,'4':0,'5':0,'6':0,'7':0}

for file in labelFiles:
    with open(os.path.join(labelPath, file), 'r') as f:
        lines = f.readlines()
    for line in lines:
        line = line.strip()
        fields = line.split(' ')
        if fields[0] in labelCount.keys():
            labelCount[fields[0]] += 1
        else:
            print(f'Label {fields[0]} not in labelCount')

# Plot the label counts as a pie chart
labels = list(labelCount.keys())
sizes = list(labelCount.values())
sizes = [count/nLabels*100 for count in sizes]  # Convert to percentage
fig, ax = plt.subplots()
ax.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=90)
ax.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.

# Show the plot
plt.title('Label Distribution')
plt.show()
