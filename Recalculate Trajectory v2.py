# -*- coding: utf-8 -*-
"""
Created on Thu Jun 25 09:40:09 2020

@author: grisever
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
from mpl_toolkits import mplot3d
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


from sklearn.svm import SVC

start = timeit.default_timer()

#Arquivos de dados

dados_robo   = "IR_2total.csv"
dados_programado = "TI_20200821_1_05.csv"

#Carrega modelo ML - pasta EVER
filename =  "C:/Users/Victor Andre/Google Drive/Machine_Learning_Victor/Programas/Machine Learning/MODELOS/model_20200720_IR_F1_0_2.sav"

#Carrega modelo ML 
# filename =  "MODELOSmodel_20200720_POS.sav"



modelo_ML = pickle.load(open(filename,'rb'))



#%%


df = pd.read_csv(dados_robo)

programado = pd.read_csv(dados_programado)

programado["PosX"] = round(programado["PosX"],3)
programado["PosY"] = round(programado["PosY"],3)
programado["PosZ"] = round(programado["PosZ"],3)


# In[4]:


df2 = df.drop(['Unnamed: 0', 'Vector', 'Velocity', 'Error', 'Error_oria', 'Error_orib', 'Error_oric'], axis=1)

programado2 = programado.drop(['Unnamed: 0'], axis=1)


# In[5]:


y_pred = modelo_ML.predict(df2)



   # In[6]:
#df3 = df2[(df2["Error"] > 0.05) & (df2["Error"] < 2)]

# df3 = df3[df3["Error"] > 0.05]
df2["Error"]  = [y_pred[i] for i in range(0,len(y_pred))]




# In[9]:


df3 = df[['Vector']]


# In[11]:


s = df3["Vector"].value_counts()




# In[13]:


teste  = pd.DataFrame(data=s)



# In[15]:


teste = teste[teste["Vector"] > 10]


# In[16]:


indexNamesArr = teste.index.values
listOfRowIndexLabels = list(indexNamesArr)



# In[18]:


def reformulate_list(l):
    li = []
    for item in l:
        item = re.sub(" \d+]","]",item)
        item = item.replace('[','[ ')
        item = re.sub('\s+',' ', item)
        item = item.replace('[','{X')
        item = item.replace(' ]',']')
        item = item.replace(']',',A')
        item = re.sub('(?<=[^X]) (?=\S+ )',',Y ', item)
        item = re.sub('(?<=[^XY]) (?=\S+,)',',Z ', item)
        item = re.sub('\.0(?=,)','.000', item)
        
        
        
        li.append(item)
        
        
        
    return li


# In[19]:


new_List = reformulate_list(listOfRowIndexLabels)



# In[23]:
list_to_create_df = [new_List]

df_values_to_change = pd.DataFrame.from_records(list_to_create_df)
df_values_to_change_transposed = df_values_to_change.transpose()
df_values_to_change_transposed.columns = ["Vector"]



#%%



# a = df_values_to_change_transposed.Vector[0]
# posx = re.search('(?<=X )\S+(?=,)', a)
# posy = re.search('(?<=Y )\S+(?=,)', a)
# posz = re.search('(?<=Z )\S+(?=,)', a)

# posx = posx.group(0)
# posy = posy.group(0)
# posz = posz.group(0)

# posx = float(posx)
# posy = float(posy)
# posz = float(posz)

# for i in range(0,len(programado2)):
#     if programado2.PosX[i] == posx and programado2.PosY[i] == posy and programado2.PosZ[i] == posz:
#         v = programado2.VEL[i]
#         break
    
    
    

# for i in range(0,len(programado2)):
#     if programado2.PosZ[i] == round(-0.5,3):
#         # print("Aqui")
    
#%%

def get_velocity(df, base_program):
    k = 0
    df["Velocity"] = "No data"
    
    while k < len(df):
        a = df.Vector[k]
        
        posx = re.search('(?<=X )\S+(?=,)', a)
        posy = re.search('(?<=Y )\S+(?=,)', a)
        posz = re.search('(?<=Z )\S+(?=,)', a)
        
        posx = posx.group(0)
        posy = posy.group(0)
        posz = posz.group(0)
        
        posx = float(posx)
        print(posx)
        posy = float(posy)
        #print(posy)
        posz = float(posz)
        #print(posz)
        
        for i in range(0,len(base_program)):
            if base_program.PosX[i] == posx and base_program.PosY[i] == posy and base_program.PosZ[i] == posz:
                df.Velocity[k] = base_program.VEL[i]
                break
            
        k = k + 1
    

    return df
     
    
#%%


new_df = get_velocity(df_values_to_change_transposed, programado2)



# In[40]:


input_file =  "TI_20200821_1_05.src"
output_file = input_file.strip(".src") + "_Model03.src"


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


def altera(df, base_program, file):
    
    df = df.reset_index(drop=True)

    #read input file
    fin = open(file, "rt")
    #read file contents to string
    data = fin.read()

    for value in range(0, len(df)):
        
        
        a = df.Vector[value]
    
        new_vel = df.Velocity[value]*0.8 #80% da velocidade
        
        #print(vel)
    
        vel_m = "$VEL.CP = " + str(new_vel)
        
    
        

        b = a.split()
        b1 = float(b[1].replace(",Y",""))
        b2 = float(b[2].replace(",Z",""))
        b3 = float(b[3].replace(",A",""))
        
        
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
        

        for i in range(0,len(base_program)-1):
            if base_program.PosX[i] == b1 and base_program.PosY[i] == b2 and base_program.PosZ[i] == b3:
                pos = i
                print("step")
                
        vel = base_program.VEL[pos+1] #volta a velocidade do prox ponto
        vel_a = "$VEL.CP = " + str(vel)
        
        
        
        
        
        
        

        #print(pos+1)
        print("st")
        nx = str(base_program.PosX[pos+1])
        ny = str(base_program.PosY[pos+1])
        nz = str(base_program.PosZ[pos+1])
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

altera(new_df, programado2, output_file)



# In[46]:

#time
stop = timeit.default_timer()

print('Time: ', stop - start)  

#%%
