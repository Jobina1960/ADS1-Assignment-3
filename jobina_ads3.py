import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import scipy.optimize as opt
from sklearn import cluster 
import err_ranges as err
#defining file read function
def readfile(file_name):
    """
        Parameters
    ----------
    file_name : string
        full address of the file to be read.

    Returns
    -------
    d : dataframe
        input data as dataframe.
    d_trans : dataframe
        Transpose of the input dataframe.

    """
    
    #reading the file
    d = pd.read_excel(file_name)
    
    #removing unwanted columns
    d = d.drop(['Series Name', 'Series Code','Country Code'], axis = 1)
    
    #taking transpose
    d_trans = d.transpose()
    d_trans = d_trans.iloc[1:31,:]
    d_trans = d_trans.reset_index()
    d_trans = d_trans.rename(columns = {"index":"years", 0:"Brazil", 1:"Italy"})
    
    d_trans = d_trans.dropna()
    #Renaming the header for transposed dataframe
    print(d_trans)
    d_trans["years"] = d_trans["years"].str[:4]
    d_trans["years"] = pd.to_numeric(d_trans["years"])
    d_trans["Brazil"] = pd.to_numeric(d_trans["Brazil"])
    d_trans["Italy"] = pd.to_numeric(d_trans["Italy"])
    print(d_trans)
    print(d_trans.dtypes)
    return d, d_trans

def exponential(t, n0, g):
    """Calculates exponential function with scale factor n0 and growth rate g."""
    t = t - 1960.0
    f = n0 * np.exp(g*t)
    return f   

# Reading the excel file 
population, population_trans = readfile(
    "C:/Users/babur/Desktop/Assignment 3/population.xlsx")
methane, methane_trans = readfile(
    "C:/Users/babur/Desktop/Assignment 3/agr_methane.xlsx")
agri_land, agri_land_trans = readfile(
    "C:/Users/babur/Desktop/Assignment 3/agri_land.xlsx")
methane_trans["NBrazil"] = methane_trans["Brazil"]/methane_trans["Brazil"].abs().max()
param,cv = opt.curve_fit(exponential,methane_trans["years"],methane_trans["NBrazil"],p0=[4e8,0.02])
methane_trans["fit"] = exponential(methane_trans["years"],*param)
plt.figure()
plt.title("Brazil ")
plt.plot(methane_trans["years"],methane_trans["NBrazil"],label="data")
plt.plot(methane_trans["years"],methane_trans["fit"],label="fit")
sigma = np.sqrt(np.diag(cv))
low,up = err.err_ranges(methane_trans["years"],exponential,param,sigma)
plt.fill_between(methane_trans["years"],low,up,alpha=0.5)
plt.legend()
plt.show()
plt.figure()
plt.title("Brazil")
plt.plot(methane_trans["years"],methane_trans["NBrazil"],label="data")
pred = np.arange(1990,2040)
pred_ = exponential(pred,*param)
plt.plot(pred,pred_,label="pred")
plt.legend()
plt.show()

methane_trans["NItaly"] = methane_trans["Italy"]/methane_trans["Italy"].abs().max()
print(methane_trans)
param,cv = opt.curve_fit(exponential,methane_trans["years"],methane_trans["NItaly"],p0=[4e8,0.02])
methane_trans["fit"] = exponential(methane_trans["years"],*param)
plt.figure()
plt.title("Italy ")
plt.plot(methane_trans["years"],methane_trans["NItaly"],label="data")
plt.plot(methane_trans["years"],methane_trans["fit"],label="fit")
low,up = err.err_ranges(methane_trans["years"],exponential,param,sigma)
plt.fill_between(methane_trans["years"],low,up,alpha=0.5)
plt.legend()
plt.show()
plt.figure()
plt.title("Italy")
plt.plot(methane_trans["years"],methane_trans["NItaly"],label="data")
pred = np.arange(1990,2040)
pred_ = exponential(pred,*param)
plt.plot(pred,pred_,label="pred")
plt.legend()
plt.show()


Brazil = pd.DataFrame()
Brazil["methane"] = methane_trans["Brazil"]
Brazil["agri_land"] = agri_land_trans["Brazil"]

 
km = cluster.KMeans(n_clusters=2).fit(Brazil)
label = km.labels_
plt.figure()
plt.title("Brazil")
plt.scatter(Brazil["methane"],Brazil["agri_land"],c=label,cmap="jet")
plt.xlabel("methane")
plt.ylabel("agri_land")
c = km.cluster_centers_
for s in range(2):
    xc,yc = c[s,:]
    plt.plot(xc,yc,"dk",markersize=15)


Italy = pd.DataFrame()
Italy["methane"] = methane_trans["Italy"]
Italy["agri_land"] = agri_land_trans["Italy"]


km = cluster.KMeans(n_clusters=2).fit(Italy)
label = km.labels_
plt.figure()
plt.title("Italy")
plt.scatter(Italy["methane"],Italy["agri_land"],c=label,cmap="jet")
plt.xlabel("methane")
plt.ylabel("agri_land")
c = km.cluster_centers_
for s in range(2):
    xc,yc = c[s,:]
    plt.plot(xc,yc,"dk",markersize=15)
