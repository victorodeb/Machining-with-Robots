# -*- coding: utf-8 -*-
"""
Created on Thu Jul 23 13:07:09 2020

@author: odebvict
"""

import pandas as pd



import os
pd.set_option("display.max_colwidth", 10000)
import re


import numpy as np
np.set_printoptions(suppress=True)
import pickle

from sklearn.model_selection import train_test_split
from sklearn.svm import SVC


import matplotlib.pyplot as plt

from matplotlib.pyplot import figure
import matplotlib.colors
import timeit
from scipy.fftpack import fft,  rfft, irfft
from scipy.signal import butter, lfilter#, freqz
from collections import namedtuple#, OrderedDict
color = namedtuple('RGB','red, green, blue')   
class RGB(color):
    def hex_format(self):
        return '#{:02X}{:02X}{:02X}'.format(self.red,self.green,self.blue) 
    
  
    
iwf1 = (0/255, 0/255, 0/255)
iwf3 = (159/255,182/255,196/255)   #IWF gray
iwf4 = (125/255,102/255,102/255)   #IWF brown
iwf5 = (153/255,0/255,0/255)       #IWF red
ccm4 = (25/255,50/255,120/255)     #CCM azul escuro
ccm7 = (255/255,100/255,0/255)     #orange
ccm2 = color(85/255,85/255,85/255) #CCM cinza escuro
ccm3 = color(210/255,210/255,210/255)  #CCM cinza claro
ccm5 = color(20/255,80/255,200/255) #CCM azul médio
ccm6 = color(170/255,200/255,230/255) #CCM azul claro
ccm8 = color(190/255,0/255,0/255) #CCM vermelhor



start = timeit.default_timer()

#Arquivos de dados

dados_programado = "TI_20200821_5_05.src"

#Carrega modelo ML - pasta EVER
filename =  "E:\0000_CCM\01_Ensaios_Doutorado_UsinagemIR\
    Machine_Learning_Victor\Programas\Machine Learning\MODELOSmodel_20200720_POS.sav"

#Carrega Modelo ML - pasta Victor
filename = r"C:\Users\Victor Andre\Google Drive\Machine_Learning_Victor\Programas\Machine Learning\MODELOS\model_20200720_POS.sav"


#Carrega modelo ML 
# filename =  "MODELOSmodel_20200720_POS.sav"



modelo_ML = pickle.load(open(filename,'rb'))



#%%




#%%
df2 = pd.read_fwf(dados_programado, header = None)
#df2 = df2[df2.iloc[:, 0]]
df2.head()
df2.info()

#%%
teste = df2.to_string()
teste = teste.split()

count = 0
typ = []
x = []
y = []
z = []
oa = []
ob = []
oc = []


for i in range(teste.index(";USINAGEM"),len(teste)):
    if teste[i]  == '{X':
        x.append(teste[i +1].replace(",Y",""))
        typ.append(teste[i - 1])
    elif re.search('Y$',teste[i]):
        y.append(teste[i +1].replace(",Z",""))
    elif re.search('Z$',teste[i]):
        z.append(teste[i +1].replace(",A",""))
    elif re.search('A$',teste[i]):
        oa.append(teste[i +1].replace(",B",""))
    elif re.search('B$',teste[i]):
        ob.append(teste[i +1].replace(",C",""))
    elif re.search('C$',teste[i]):
        oc.append(teste[i +1].replace(",E1",""))
    #i = i + 1

count = 0
typ = []
x = []
y = []
z = []
oa = []
ob = []
oc = []
cdis = []
vel = []

for i in range(teste.index(";USINAGEM"),len(teste)):
    if teste[i] == '$APO.CDIS':
        c = teste[i +2]
    if teste[i] == '$VEL.CP':
        v = teste[i +2]
    if teste[i]  == '{X':
        cdis.append(c)
        vel.append(v)
        x.append(teste[i +1].replace(",Y",""))
        typ.append(teste[i - 1])
    if re.search('Y$',teste[i]):
        y.append(teste[i +1].replace(",Z",""))
    if re.search('Z$',teste[i]):
        z.append(teste[i +1].replace(",A",""))
    if re.search('A$',teste[i]):
        oa.append(teste[i +1].replace(",B",""))
    if re.search('B$',teste[i]):
        ob.append(teste[i +1].replace(",C",""))
    if re.search('C$',teste[i]):
        oc.append(teste[i +1].replace(",S",""))    
    #i = i + 1
#%%
typ = list(map(str,typ))
typ.pop()    
    
cdis = list(map(float,cdis))
cdis.pop()

vel = list(map(float,vel))
vel.pop()

x = list(map(float,x))
x.pop()

y = list(map(float,y))
y.pop()

z = list(map(float,z))
z.pop()

oa = list(map(float,oa))
oa.pop()

ob = list(map(float,ob))
ob.pop()

oc = list(map(float,oc))
oc.pop()

robot = [typ,cdis,vel,x,y,z,oa,ob,oc]

correct_df = pd.DataFrame.from_records(robot)
correct_df_transposed = correct_df.transpose()
correct_df_transposed.columns = ["TYPE","CDIS","VEL","PosX","PosY","PosZ","OriA","OriB","OriC"]
correct_df_transposed.head()

correct_df_transposed["CDIS"] = correct_df_transposed["CDIS"].astype(float)
correct_df_transposed["VEL"] = correct_df_transposed["VEL"].astype(float)
correct_df_transposed["PosX"] = correct_df_transposed["PosX"].astype(float)
correct_df_transposed["PosY"] = correct_df_transposed["PosY"].astype(float)
correct_df_transposed["PosZ"] = correct_df_transposed["PosZ"].astype(float)
correct_df_transposed["OriA"] = correct_df_transposed["OriA"].astype(float)
correct_df_transposed["OriB"] = correct_df_transposed["OriB"].astype(float)
correct_df_transposed["OriC"] = correct_df_transposed["OriC"].astype(float)




correct_df_transposed["PosX"] = round(correct_df_transposed["PosX"],3)
correct_df_transposed["PosY"] = round(correct_df_transposed["PosY"],3)
correct_df_transposed["PosZ"] = round(correct_df_transposed["PosZ"],3)



# In[5]:

X = correct_df_transposed[["PosX","PosY","PosZ"]]
X.columns = ["POSX","POSY","POSZ"]

#%%

y_pred = modelo_ML.predict(X)



   # In[6]:
#df3 = df2[(df2["Error"] > 0.05) & (df2["Error"] < 2)]

# df3 = df3[df3["Error"] > 0.05]

df = correct_df_transposed
df["Error"]  = [y_pred[i] for i in range(0,len(y_pred))]



# In[40]:


input_file = "TI_20200821_5_05.src"
output_file = input_file.strip(".src") + "_Model04.src"


# In[42]:


def create_file(output_file, input_file):
    if not os.path.exists('./' + output_file):
       
        f = open(output_file, "x")   # create file if doesnt exist
        
    else:
        f = open(output_file,"r+")
        f.truncate(0)
    
    
    
    
    #read input file
    fin = open(input_file, "rt")
    #read file contents to string
    data = fin.read()
    #write data to the file
    f.write(data)
    f.close()
    fin.close()


# In[45]:


create_file(output_file, input_file)


# In[ ]:

#essa parte faz mudar fácil


def altera(base_program, file):
    

    #read input file
    fin = open(file, "rt")
    #read file contents to string
    data = fin.read()

    for value in range(0, len(base_program)-1):
        
        
        #a = df.Vector[value]
    
        new_vel = base_program.VEL[value]*0.8 #80% da velocidade
        
        #print(vel)
    
        vel_m = "$VEL.CP = " + str(new_vel)
        
    
        
        b1 = base_program.PosX[value]
        b2 = base_program.PosY[value]
        b3 = base_program.PosZ[value]
        
        
        ax =  str(b1)
        ay =  str(b2)
        az =  str(b3)
        ax = re.sub('\.0(?!\d)','.000', ax)
        ay = re.sub('\.0(?!\d)','.000', ay)
        az = re.sub('\.0(?!\d)','.000', az)
        
        
        ax = re.sub('\d+\.\d(?!\d)',ax+"00", ax)
        ay = re.sub('\d+\.\d(?!\d)',ay+"00", ay)
        az = re.sub('\d+\.\d(?!\d)',az+"00", az)
        
        ax = re.sub('\d+\.\d\d(?!\d)',ax+"0", ax)
        ay = re.sub('\d+\.\d\d(?!\d)',ay+"0", ay)
        az = re.sub('\d+\.\d\d(?!\d)',az+"0", az)        
        
        
        ax = re.sub('-+','-', ax)
        ay = re.sub('-+','-', ay)
        az = re.sub('-+','-', az)
        
        aa = "{X "+ ax +",Y "+ ay +",Z "+ az +",A"
        
        #replace all occurrences of the required string
        data = data.replace("LIN " + aa, vel_m+"\n"+"LIN " +aa)
        

    
        vel = base_program.VEL[value+1] #volta a velocidade do prox ponto
        vel_a = "$VEL.CP = " + str(vel)
        
        
        
        
        
        
        

        #print(pos+1)
        print("st")
        nx = str(base_program.PosX[value+1])
        ny = str(base_program.PosY[value+1])
        nz = str(base_program.PosZ[value+1])
        nx = re.sub('\.0(?!\d)','.000', nx)
        ny = re.sub('\.0(?!\d)','.000', ny)
        nz = re.sub('\.0(?!\d)','.000', nz)
        
        
        nx = re.sub('\d+\.\d(?!\d)',nx+"00", nx)
        ny = re.sub('\d+\.\d(?!\d)',ny+"00", ny)
        nz = re.sub('\d+\.\d(?!\d)',nz+"00", nz)
        
        nx = re.sub('\d+\.\d\d(?!\d)',nx+"0", nx)
        ny = re.sub('\d+\.\d\d(?!\d)',ny+"0", ny)
        nz = re.sub('\d+\.\d\d(?!\d)',nz+"0", nz)        
        
        
        nx = re.sub('-+','-', nx)
        ny = re.sub('-+','-', ny)
        nz = re.sub('-+','-', nz)
        print(nx)
        print(ny)
        print(nz)
        

        na = "{X "+ nx +",Y "+ ny +",Z "+ nz +",A"

        data = data.replace("LIN " + na, vel_a+"\n"+"LIN " +na)
        
        
        data = data.replace(vel_m+"\n"+vel_a, vel_m)
        data = data.replace(vel_a+"\n"+vel_m, vel_m)


    #data = data.replace(r'(?<=\$VEL.CP = \S+\n)\$VEL.CP = \S+\n',"")
    #data = re.sub("(?<=\$VEL.CP = 0.034\n)\$VEL.CP = \S+\n", "", data)

    #close the input file
    fin.close()
    #open the input file in write mode
    fin = open(file, "wt")
    #overrite the input file with the resulting data
    fin.write(data)
    #close the file
    fin.close()



# In[ ]:

altera(df, output_file)



# In[46]:

#time
stop = timeit.default_timer()

print('Time: ', stop - start)  