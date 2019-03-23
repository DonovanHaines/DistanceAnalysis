import tkinter as tk
from tkinter.filedialog import askopenfilename
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA


filetoload = "temp"
while filetoload != "":
    filetoload = askopenfilename()
    print("Loading file ")
    print(filetoload)
    if 'data_matrix' in locals() and filetoload != "" : 
        print("Adding data to existing dataframe")
        data_matrix = data_matrix.append(pd.read_csv(filetoload, delim_whitespace=True, header=None, 
                          names=['frame','Tyr51OH-SubO1', 'Arg47CZ-SubC', 'Gln73N-SubC', 
                                 'Tyr51OH-SubC', 'HEME471FE-SubC1']))
    elif filetoload != "":
        print("starting new dataframe")
        data_matrix = pd.read_csv(filetoload, delim_whitespace=True, header=None, 
                          names=['frame','Tyr51OH-SubO1', 'Arg47CZ-SubC', 'Gln73N-SubC', 
                                 'Tyr51OH-SubC', 'HEME471FE-SubC1'])

    print(data_matrix)
    print(data_matrix.shape)
print("Final matrix")
print(data_matrix)
print(data_matrix.shape)
time_conversion = 250 * 2 / 1000 / 1000 # (250 2 fs steps convert to ns)

data_matrix['frame']=range(0, len(data_matrix))

data_matrix.insert(0, "Time", data_matrix['frame'] * time_conversion)
data_matrix.reset_index()

print(data_matrix)


plt.plot(data_matrix["Time"], data_matrix['Tyr51OH-SubO1'])
plt.plot(data_matrix["Time"], data_matrix['Arg47CZ-SubC'])
plt.plot(data_matrix["Time"], data_matrix['Gln73N-SubC'])
plt.plot(data_matrix["Time"], data_matrix['Tyr51OH-SubC'])
plt.plot(data_matrix["Time"], data_matrix['HEME471FE-SubC1'])

plt.legend()

plt.show()

print("Starting PCA Analysis")

features = ['Tyr51OH-SubO1', 'Arg47CZ-SubC', 'Gln73N-SubC', 
                                 'Tyr51OH-SubC', 'HEME471FE-SubC1']
x = data_matrix.loc[:, features].values
x = StandardScaler().fit_transform(x)
print("Transform scaling done. Starting PCA.")
pca = PCA(n_components=4)
print("PCA done")
principalComponents = pca.fit_transform(x)
print("principalComponents")
print(principalComponents)
print(principalComponents.shape)

principalDf = pd.DataFrame(data = principalComponents, columns = ['PC1', 'PC2', 
                                                                  'PC3', 'PC4'])

print("principalDf")
print(principalDf)
print(principalDf.shape)
print(principalDf.iloc[199998:200007, :] )

f, axarr = plt.subplots(4,1, sharex=True)
f.suptitle('Principal Components')
axarr[0].plot(principalDf['PC1'])
axarr[1].plot(principalDf['PC2'], color='r')
axarr[2].plot(principalDf['PC3'], color='g')
axarr[3].plot(principalDf['PC4'], color='cyan')
#axarr[4].plot(principalDf['PC5'])
f.legend()
plt.show()

print("Done. Thank-you!")



# Now for loadings

loadings = pca.components_.T * np.sqrt(pca.explained_variance_)
print(loadings)
#ax = loadings.plt(kind='bar', title = "Loadings")
#ax.set_xlabel("Frame")
#ax.set_ylabel("Loading")
index = np.arange(5)
bar_width=1.0/6.0
plt.bar(index, loadings[:,0], bar_width, label="PC1")
plt.bar(index + bar_width, loadings[:,1], bar_width, label="PC2")
plt.bar(index + bar_width*2, loadings[:,2], bar_width, label="PC3")
plt.bar(index+ bar_width *3, loadings[:,3], bar_width, label="PC4")
plt.xticks(index + bar_width, features)
plt.show()

#Now find the frame with the max and min for each PC
print("Finding maximum and minimum frame for each PC...")

def find_min_idx(x): #from https://stackoverflow.com/questions/30180241/numpy-get-the-column-and-row-index-of-the-minimum-value-of-a-2d-array
    k = x.argmin()
    ncol = x.shape[1]
    return k/ncol, k%ncol

for i in arange(pca.components_):
    print("For PC{0}:".format(i))
    indexmin=find_min_idx(principalDf['PC{0}'.format(i)])
    print("    max is {0} at frame {1} (time {2} ns)".format(
    principalDf['PC{0}'.format(i)].argmin(),
    indexmin,
    indexmin*time_conversion))
