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

start = timeit.default_timer()

#Arquivos de dados
#dados_sensor = "SENSOR_.txt"
dados_robo   = "IR_2.txt"
dados_programado = "TROCO_05_Model02.src"
leica = "Leica_.txt"

# #Valores maximos de erros para calcular as tabelas de erros
# erro_robo = 0.05
# erro_leica = 0.05

#%%
df_robo =  pd.read_csv(dados_robo, delimiter=r'"(.*?)"',header = None,
                   usecols=[3,5,7,9,11,13,15,17,19,21,23,25,27,29,31,33,35,37,
                            39,41,43,45,47,49,51,53,55,57,59,61,63,65])
df_robo.columns = ['POSX','POSY','POSZ','ORIA','ORIB','ORIC','ORIS','ORIT','CURA1',
               'CURA2','CURA3','CURA4','CURA5','CURA6','CURE1','CURE2','TQA1',
               'TQA2','TQA3','TQA4','TQA5','TQA6','TQE1','TQE2','AXA1','AXA2',
               'AXA3','AXA4','AXA5','AXA6','AXE1','AXE2']

#Adjust Time
dft = pd.read_csv(dados_robo, sep=':', header = None)
dft = pd.concat([dft[[0]], dft[1].str.split(' ', expand=True)], axis=1)
dft.drop(dft.iloc[:,2:41], axis=1, inplace=True)
dft.columns = ['MINUTE','TIME']
dft['MINUTE'] = (dft['MINUTE']-dft['MINUTE'][0])*60
dft['TIME'] = dft['TIME'].astype(float)
dft['TIME'] = dft['MINUTE'] + dft['TIME']
##### T0, initial time. Has to be the same in both dataframes. NAO ALTERAR-----
t0 = dft['TIME'][1]
####
dft['TIME'] = dft['TIME'] - t0
dft= dft[dft.columns[~dft.columns.isin(['MINUTE'])]]
df_robo['TIME'] = dft['TIME']
df_robo = df_robo[df_robo["TIME"] >= 0]
df_robo["TIME"] = df_robo["TIME"].round(4)
df_robo.head()
df_IR = df_robo
df_IR.info()

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

correct_df_transposed.info()


correct_df_transposed.head()


#%%
def distance(a,b,p):
    #Calcula a distancia entre um ponto e um vetor

    ap = p - a
    ab = b - a

    cp = np.cross(ap,ab)

    d = np.sqrt(np.power(cp[0],2) + np.power(cp[1],2) + np.power(cp[2],2))
    d2 = np.sqrt(np.power(ab[0],2) + np.power(ab[1],2) + np.power(ab[2],2))

    return d/d2

def proj(n,p,a):
    h = p - a
    
    den = np.dot(h,n)
    
    num = np.power(n[0],2) + np.power(n[1],2) + np.power(n[2],2)
    
    t = den/num
    
    
    x_p = a[0] + n[0]*t   
    y_p = a[1] + n[1]*t   
    z_p = a[2] + n[2]*t   
    
    return x_p, y_p, z_p

def myfunc(j,a,b,d1):    
    
   
    p = np.array((d1.POSX[j],d1.POSY[j],d1.POSZ[j]))
    k = distance(a,b,p)
    
    return k

def myfunc2(j,n,a,d1):
   
    p = np.array((d1.POSX[j],d1.POSY[j],d1.POSZ[j]))
            
    x_p, y_p, z_p = proj(n,p,a)
    
    return x_p


def myfunc3(j,n,a,d1):
   
    p = np.array((d1.POSX[j],d1.POSY[j],d1.POSZ[j]))
            
    x_p, y_p, z_p = proj(n,p,a)
    
    return y_p

def myfunc4(j,n,a,d1):
   
    p = np.array((d1.POSX[j],d1.POSY[j],d1.POSZ[j]))
            
    x_p, y_p, z_p = proj(n,p,a)
    
    return z_p


def myfuncA(j,oria,d1):    
    
   
    erro_a = abs(d1.ORIA[j]) - abs(oria)
    
    return erro_a

def myfuncB(j,orib,d1):    
    
   
    erro_b = abs(d1.ORIB[j]) - abs(orib)
    
    return erro_b

def myfuncC(j,oric,d1):    
    
   
    erro_c = abs(d1.ORIC[j]) - abs(oric)
    
    return erro_c


def myfuncVEL(j, df_errors):
    if j == 0 or j == len(df_errors):
        v = 0
    else:
        X = np.power(df_errors.POSX[j] - df_errors.POSX[j-1],2)
        Y = np.power(df_errors.POSY[j] - df_errors.POSY[j-1],2)
        Z = np.power(df_errors.POSZ[j] - df_errors.POSZ[j-1],2)
        
        d = np.sqrt(X+Y+Z)
        t = df_errors.TIME[j] - df_errors.TIME[j-1]
        
        v = d/t
        
    return v

#%%

# te = np.array((correct_df_transposed.PosX[1],correct_df_transposed.PosY[1],correct_df_transposed.PosZ[1]))

# print("["+str(te[0]) + " "+ str(te[1]) + " "+ str(te[2]) + "]")


#%%
LIMXY = 0.5
LIMZ  = 0.5
LIMPTP = 11

def calcula(base_program, df):
    print(df_IR.POSX[0]) 
    print(df.POSX[0]) 
    i = 0
    df["Error"] = "No data"
    df["X_Proj"] = "No data"
    df["Y_Proj"] = "No data"
    df["Z_Proj"] = "No data"
    df["Vector"] = "No data"
    df["VEL"] = "No data"
    df["Error_oria"] = "No data"
    df["Error_orib"] = "No data"
    df["Error_oric"] = "No data"
    df["Velocity"] = "No data"
    df["Error_X"] = "No data"
    df["Error_Y"] = "No data"
    df["Error_Z"] = "No data"
    
    df_errors = pd.DataFrame
    f = 0
    
    change_list = []

    pos = 0
    
    #pega os pontos de troca
                 
    
    for j in range(0,len(df)):
        pos_atual = j
        
        if pos_atual >= len(base_program) - 1:
            break     
        
        if base_program.TYPE[pos_atual] == "LIN":
            print("LIN")
            x0 = base_program.PosX[pos_atual] 
            y0 = base_program.PosY[pos_atual] 
            z0 = base_program.PosZ[pos_atual] 

            #calcula os limites inferiores baseados no C_DIS
            lix = x0 - base_program.CDIS[pos_atual +1] - LIMXY
            liy = y0 - base_program.CDIS[pos_atual +1] - LIMXY
            liz = z0 - base_program.CDIS[pos_atual +1] - LIMZ

            #calcula os limites superiores baseados no C_DIS
            lsx = x0 + base_program.CDIS[pos_atual +1] + LIMXY
            lsy = y0 + base_program.CDIS[pos_atual +1] + LIMXY
            lsz = z0 + base_program.CDIS[pos_atual +1] + LIMZ
            
        
        else:
            print("PTP")
            x0 = base_program.PosX[pos_atual] 
            y0 = base_program.PosY[pos_atual] 
            z0 = base_program.PosZ[pos_atual] 

            #calcula os limites inferiores baseados no C_DIS
            lix = x0 - base_program.CDIS[pos_atual +1] - LIMPTP
            liy = y0 - base_program.CDIS[pos_atual +1] - LIMPTP
            liz = z0 - base_program.CDIS[pos_atual +1] - LIMPTP

            #calcula os limites superiores baseados no C_DIS
            lsx = x0 + base_program.CDIS[pos_atual +1] + LIMPTP
            lsy = y0 + base_program.CDIS[pos_atual +1] + LIMPTP
            lsz = z0 + base_program.CDIS[pos_atual +1] + LIMPTP
    

        # print(lsx)

        flag = 0
        while flag != 1 and pos < len(df)-1:
            # print("EGM")
            if df.POSX[pos] >= lix and df.POSX[pos] <= lsx:     
        
                if df.POSY[pos] >= liy and df.POSY[pos] <= lsy:
                    
                    if df.POSZ[pos] >= liz and df.POSZ[pos] <= lsz:
                        achou = pos
                        # print("Oi Ever")
                        print(achou)
                        change_list.append(achou)
                        pos = achou
                        flag = 1
                
            pos = pos + 1
    #dividi de acordo com os pontos programados 
    while(i <len(change_list)-1):
        
        print(len(change_list))
        
        start = change_list[i]
        end = change_list[i+1]

        #novo dataframe
        d1 = df[(df.index >= start) & (df.index < end)]
        #reset index
        d1 = d1.reset_index(drop=True) 
        d1 = d1.drop([0])
        # valores da reta/vetor
       
        a = np.array((base_program.PosX[i],base_program.PosY[i],base_program.PosZ[i]))
        b = np.array((base_program.PosX[i+1],base_program.PosY[i+1],base_program.PosZ[i+1]))
        n = b - a
        
        oria = base_program.OriA[i]
        orib = base_program.OriB[i]
        oric = base_program.OriC[i]
        vel  = base_program.VEL[i]
            
        d1 = d1.reset_index(drop=True)
        
        if len(d1) != 0:
            print(i)
            # print(d1.index[0])
            # print(d1.POSX[0])
            # print(d1.POSY[0])
            # print(d1.POSZ[0])


            # d1["Error_X"] = [base_program.PosX[i] - df_IR.POSX[i] for j in range(0,len(d1))]  
            # d1["Error_Y"] = [base_program.PosY[i] - df_IR.POSY[i] for j in range(0,len(d1))]
            # d1["Error_Z"] = [base_program.PosZ[i] - df_IR.POSZ[i] for j in range(0,len(d1))]
            
          
         
            d1["Error"]  = [myfunc(j,a,b,d1) for j in range(0,len(d1))]  
            d1["X_Proj"] = [myfunc2(j,n,a,d1) for j in range(0,len(d1))]
            d1["Y_Proj"] = [myfunc3(j,n,a,d1) for j in range(0,len(d1))]
            d1["Z_Proj"] = [myfunc4(j,n,a,d1) for j in range(0,len(d1))]
            d1["Vector"] = ["["+str(b[0]) + " "+ str(b[1]) + " "+ str(b[2]) +  " " + str(i) + "]" for j in range(0,len(d1))]
            d1["VEL"]    = [vel for j in range(0,len(d1))]
            
            d1["Error_oria"] = [myfuncA(j,oria,d1) for j in range(0,len(d1))]
            d1["Error_orib"] = [myfuncB(j,orib,d1) for j in range(0,len(d1))]
            d1["Error_oric"] = [myfuncC(j,oric,d1) for j in range(0,len(d1))]
            
            
            if i == 0 or f==0:
                df_errors = d1
                f = 1
            else:
                df_errors = df_errors.append(d1, ignore_index=True)       
            
                
            
        i = i + 1
    
    
    df_errors["Velocity"] = [myfuncVEL(j,df_errors) for j in range(0,len(df_errors))]        
    df_errors["Velocity"] = df_errors["Velocity"]/16.85
    print(df_errors["Velocity"])
    
    # df_errors["Error_X"] = [ correct_df_transposed["PosX"] -d1["X_Proj"] ]
    # df_errors["Error_Y"] = [ correct_df_transposed["PosY"] -d1["Y_Proj"] ]
    # df_errors["Error_Z"] = [ correct_df_transposed["PosZ"] -d1["Z_Proj"] ]
    
    return df_errors  
   
#%%

df_IR = df_IR.reset_index(drop=True)

df_errors = calcula(correct_df_transposed,df_IR)



#%%
df_errors["VEL"] = df_errors["VEL"] * 60



#%%


#%%
# df_errors['Error2'] = "No data"

df_errors["Error_X"] = [(df_errors['POSX'][j] - df_errors['X_Proj'][j]) for j in range(0,len(df_errors))]  
df_errors["Error_Y"] = [(df_errors['POSY'][j] - df_errors['Y_Proj'][j]) for j in range(0,len(df_errors))]  
df_errors["Error_Z"] = [(df_errors['POSZ'][j] - df_errors['Z_Proj'][j]) for j in range(0,len(df_errors))]  

# df_errors['Error2'] = [(np.sqrt((df_errors['Error_X'][j]**2)+ (df_errors['Error_Y'][j]**2)+ (df_errors['Error_Z'][j]**2)))
                       # for j in range(0,len(df_errors))] 


df_T= df_errors[['TIME','POSX', 'POSY', 'POSZ', 'X_Proj', 'Y_Proj', 'Z_Proj', 'Error','Error_X','Error_Y','Error_Z' ,'Vector']] #'Error2',


#%%  LEICA


df_leica = pd.read_csv(leica, parse_dates = True,names=["POSX","POSY","POSZ"] , sep=',', skiprows=1, decimal='.', encoding='utf-8')#,encoding='utf-8') #header=None,header=0
# df_leicaerror = df_leica
# df_leica["POSX"] = df_leica["POSX"]*-1
# df_leica["POSY"] = df_leica["POSY"]+0.65
# df_leica["POSZ"] = df_leica["POSZ"]- 0.5



#%%

def calcula(base_program, df): 
    print(df.POSX[0]) 
    i = 0
    df["Error"] = "No data"
    df["X_Proj"] = "No data"
    df["Y_Proj"] = "No data"
    df["Z_Proj"] = "No data"
    df["Vector"] = "No data"
    df["Error_X"] = "No data"
    df["Error_Y"] = "No data"
    df["Error_Z"] = "No data"
    
    df_errorsL = pd.DataFrame
    f = 0
    
    change_list = []

    pos = 0
    
    #pega os pontos de troca
    
    LIMXY = 0.5
    LIMZ = 0.5           
    
    for j in range(0,len(df)):
        pos_atual = j
        
        if pos_atual >= len(base_program) - 1:
            break
        
            
        
        
        
        if base_program.TYPE[pos_atual] == "LIN":
            print("LIN")
            x0 = base_program.PosX[pos_atual] 
            y0 = base_program.PosY[pos_atual] 
            z0 = base_program.PosZ[pos_atual] 

            #calcula os limites inferiores baseados no C_DIS
            lix = x0 - base_program.CDIS[pos_atual +1] - LIMXY
            liy = y0 - base_program.CDIS[pos_atual +1] - LIMXY
            liz = z0 - base_program.CDIS[pos_atual +1] - LIMZ

            #calcula os limites superiores baseados no C_DIS
            lsx = x0 + base_program.CDIS[pos_atual +1] + LIMXY
            lsy = y0 + base_program.CDIS[pos_atual +1] + LIMXY
            lsz = z0 + base_program.CDIS[pos_atual +1] + LIMZ
            
        
        else:
            print("PTP")
            x0 = base_program.PosX[pos_atual] 
            y0 = base_program.PosY[pos_atual] 
            z0 = base_program.PosZ[pos_atual] 

            #calcula os limites inferiores baseados no C_DIS
            lix = x0 - base_program.CDIS[pos_atual +1] - LIMPTP
            liy = y0 - base_program.CDIS[pos_atual +1] - LIMPTP
            liz = z0 - base_program.CDIS[pos_atual +1] - LIMPTP

            #calcula os limites superiores baseados no C_DIS
            lsx = x0 + base_program.CDIS[pos_atual +1] + LIMPTP
            lsy = y0 + base_program.CDIS[pos_atual +1] + LIMPTP
            lsz = z0 + base_program.CDIS[pos_atual +1] + LIMPTP
    

        print(lsx)

        flag = 0
        while flag != 1 and pos < len(df)-1:
            # print("EGM")
            if df.POSX[pos] >= lix and df.POSX[pos] <= lsx:     
        
                if df.POSY[pos] >= liy and df.POSY[pos] <= lsy:
                    
                    if df.POSZ[pos] >= liz and df.POSZ[pos] <= lsz:
                        achou = pos
                        # print("Oi Ever")
                        # print(achou)
                        change_list.append(achou)
                        pos = achou
                        flag = 1
                
            pos = pos + 1
    #dividi de acordo com os pontos programados 
    while(i <len(change_list)-1):
        
        print(len(change_list))
        
        start = change_list[i]
        end = change_list[i+1]

        #novo dataframe
        d1 = df[(df.index >= start) & (df.index < end)]
        #reset index
        d1 = d1.reset_index(drop=True) 
 
        # valores da reta/vetor
       
        a = np.array((base_program.PosX[i],base_program.PosY[i],base_program.PosZ[i]))
        b = np.array((base_program.PosX[i+1],base_program.PosY[i+1],base_program.PosZ[i+1]))
        n = b - a
            
        d1 = d1.reset_index(drop=True)
        
        if len(d1) != 0:
            print(i)
            # print(d1.index[0])
            # print(d1.POSX[0])
            # print(d1.POSY[0])
            # print(d1.POSZ[0])

            
            d1["Error"]  = [myfunc(j,a,b,d1) for j in range(0,len(d1))]  
            d1["X_Proj"] = [myfunc2(j,n,a,d1) for j in range(0,len(d1))]
            d1["Y_Proj"] = [myfunc3(j,n,a,d1) for j in range(0,len(d1))]
            d1["Z_Proj"] = [myfunc4(j,n,a,d1) for j in range(0,len(d1))]
            d1["Vector"] = ["["+str(b[0]) + " "+ str(b[1]) + " "+ str(b[2]) + "]" for j in range(0,len(d1))]
            
            if i == 0 or f==0:
                df_errorsL = d1
                f = 1
            else:
                df_errorsL = df_errorsL.append(d1, ignore_index=True)       
            
                
            
        i = i + 1
    
    
    return df_errorsL  

   
#%%

df_leica = df_leica.reset_index(drop=True)

df_errorsL = calcula(correct_df_transposed,df_leica)



#%%

df_errorsL["Error_X"] = [(df_errorsL['POSX'][j] - df_errorsL['X_Proj'][j]) for j in range(0,len(df_errorsL))]  
df_errorsL["Error_Y"] = [(df_errorsL['POSY'][j] - df_errorsL['Y_Proj'][j]) for j in range(0,len(df_errorsL))]  
df_errorsL["Error_Z"] = [(df_errorsL['POSZ'][j] - df_errorsL['Z_Proj'][j]) for j in range(0,len(df_errorsL))]  



#%%


ax = df_T.plot.scatter("X_Proj","Y_Proj" ,color=iwf3, label='Proj')
# ax.set_title('Trajetória Programada e ROBÔ para POSX x POSZ')
df_T.plot.scatter("POSX", "POSY",ax=ax,color=ccm5, label='IR')
correct_df_transposed.plot.scatter("PosX","PosY", color=iwf1, ax=ax , label='PROG')
df_leica.plot.scatter("POSX","POSY", color=iwf5, ax=ax , label='LEICA')



#%%
# CorteZ = 953

# df_leica = df_leica[df_leica["POSZ"] < CorteZ]

# df_T = df_T[df_T["POSZ"] < CorteZ]

# correct_df_transposed = correct_df_transposed[correct_df_transposed["PosZ"] < CorteZ]

# df_errors = df_errors[df_errors["POSZ"] < CorteZ]

#%%

# a = np.array([[0,0.5,1,1,2,3.5,4,4.5,5]])
a = np.array([[0,0.25,0.5,0.75,0.9,1,1.25,1.5,1.75,2]])
# a = np.array([[-200,-180,-150,-100,0 ,100,150,180,200]])

fig = plt.figure()
#mycmap = matplotlib.colors.ListedColormap(['yellow','orange','red','darkred'])
mycmap = matplotlib.colors.ListedColormap(['lightgreen','green',ccm6,iwf3,ccm5,ccm4,iwf5,ccm7,ccm8])
norm = matplotlib.colors.BoundaryNorm(a[0], len(a[0])-1)

ax = fig.add_subplot(111, projection='3d')
cax = ax.scatter(df_errors["POSX"], df_errors["POSY"], df_errors["POSZ"], c=df_errors["Velocity"],  cmap=mycmap, norm = norm)
plt.axis(aspect='equal')
plt.title('Velocidade')
ax.w_xaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
ax.w_yaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
ax.w_zaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
# plt.rcParams['grid.color'] = "deeppink"
matplotlib.pyplot.grid(color=iwf1, linestyle='-', linewidth=2)
cbar = fig.colorbar(cax, ticks=a)#[np.min(Vel1),50, 100, 150, 200, 250, 300, np.max(Vel1)])
cbar.ax.set_yticklabels( [np.min(a),np.max(a)], minor=True )   #**kwargs)  # vertically oriented colorbar
# ax.set_zlim3d(80,130)

plt.savefig(dados_robo.strip(".txt") + "Velocidade.jpg")

#%%



a = np.array([[0,0.05,0.07,0.1,0.15,0.25,0.5,0.75,1]])

a = np.array([[np.min(df_errors["Error"]),np.max(df_errors["Error"])*2/9,np.max(df_errors["Error"])*1/3,
                np.max(df_errors["Error"])*4/9,np.max(df_errors["Error"])/2,np.max(df_errors["Error"])*2/3,
                np.max(df_errors["Error"])*7/9,np.max(df_errors["Error"])*8/9,np.max(df_errors["Error"])]])

fig = plt.figure(num=None, figsize=(10, 8), dpi=80, facecolor='w', edgecolor='k')
#mycmap = matplotlib.colors.ListedColormap(['yellow','orange','red','darkred'])
mycmap = matplotlib.colors.ListedColormap(['lightgreen','green',iwf3,ccm7,ccm8,ccm5,ccm4,iwf5,ccm6])
norm = matplotlib.colors.BoundaryNorm(a[0], len(a[0])-1)

ax = fig.add_subplot(111, projection='3d')
cax = ax.scatter(df_errors["POSX"], df_errors["POSY"], df_errors["POSZ"], c=df_errors["Error"],  cmap=mycmap, norm = norm)
plt.axis(aspect='equal')
plt.title('Erro IR')
ax.w_xaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
ax.w_yaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
ax.w_zaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
plt.rcParams['grid.color'] = iwf1
matplotlib.pyplot.grid(color=iwf1, linestyle='-', linewidth=2)
cbar = fig.colorbar(cax, ticks=a)#[np.min(Vel1),50, 100, 150, 200, 250, 300, np.max(Vel1)])
cbar.ax.set_yticklabels( [np.min(a),np.max(a)], minor=True )   #**kwargs)  # vertically oriented colorbar
# ax.set_zlim3d(125,126)

plt.show()

plt.savefig(dados_robo.strip(".txt") + "Erro IR.jpg")

#%%
a = np.array([[0,0.05,0.07,0.1,0.15,0.25,0.5,0.75,1]])

# a = np.array([[np.min(df_errorsL["Error"]),np.max(df_errorsL["Error"])*2/9,np.max(df_errorsL["Error"])*1/3,
#                np.max(df_errorsL["Error"])*4/9,np.max(df_errorsL["Error"])/2,np.max(df_errorsL["Error"])*2/3,
#                np.max(df_errorsL["Error"])*7/9,np.max(df_errorsL["Error"])*8/9,np.max(df_errorsL["Error"])]])


fig = plt.figure(num=None, figsize=(10, 8), dpi=80, facecolor='w', edgecolor='k')
#mycmap = matplotlib.colors.ListedColormap(['yellow','orange','red','darkred'])
mycmap = matplotlib.colors.ListedColormap(['lightgreen','green',iwf3,ccm7,ccm8,ccm5,ccm4,iwf5,ccm6])
norm = matplotlib.colors.BoundaryNorm(a[0], len(a[0])-1)

ax = fig.add_subplot(111, projection='3d')
cax = ax.scatter(df_errorsL["POSX"], df_errorsL["POSY"], df_errorsL["POSZ"], c=df_errorsL["Error"],  cmap=mycmap, norm = norm)
plt.axis(aspect='equal')
plt.title('Erro Leica')
# plt.set_title('Erros Leica')
ax.w_xaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
ax.w_yaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
ax.w_zaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
plt.rcParams['grid.color'] = iwf1
matplotlib.pyplot.grid(color=iwf1, linestyle='-', linewidth=2)
cbar = fig.colorbar(cax, ticks=a)#[np.min(Vel1),50, 100, 150, 200, 250, 300, np.max(Vel1)])
cbar.ax.set_yticklabels( [np.min(a),np.max(a)], minor=True )   #**kwargs)  # vertically oriented colorbar
# ax.set_zlim3d(125,126)
plt.show()

plt.savefig(dados_robo.strip(".txt") + "Erro Leica.jpg")

#%%

figure(num=None, figsize=(10, 8), dpi=80, facecolor='w', edgecolor='k')

ax = plt.axes(projection='3d')


 # Data for a three-dimensional line
zline = df_errors["POSZ"]
xline = df_errors["POSX"]
yline = df_errors["POSY"]
ax.plot3D(xline, yline, zline, color=iwf4,label='IR')
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('z');

 # Data for three-dimensional scattered points
zdata = df_errors["Z_Proj"]
xdata = df_errors["X_Proj"]
ydata = df_errors["Y_Proj"]
ax.plot3D(xdata, ydata, zdata, color=ccm5, label="Proj");

zdata = correct_df_transposed["PosZ"]
xdata = correct_df_transposed["PosX"]
ydata = correct_df_transposed["PosY"]
ax.plot3D(xdata, ydata, zdata,color=iwf1, label="Prog");
# ax.set_zlim3d(125,126)

zdata = df_leica["POSZ"]
xdata = df_leica["POSX"]
ydata = df_leica["POSY"]
ax.plot3D(xdata, ydata, zdata,color=ccm7, label="LEICA");

# ax.set_zlim3d(75,90)
# ax.set_ylim3d(-100,100)
# ax.set_xlim3d(-50,-60)
matplotlib.pyplot.grid(color=iwf1, linestyle='-', linewidth=2)
# ax.set_title('Trajetória Programada x Trajetória ROBÔ')
leg = ax.legend()
# ax.set_xlim3d(1875,1885)
# ax.set_zlim3d(900,905)

plt.show()

plt.savefig(dados_robo.strip(".txt") + "Trajetoria Todos.jpg")

#%%

#figure(num=None, figsize=(10, 6), dpi=80, facecolor='w', edgecolor='k')
#plt.plot(df_errors['VEL'], color=iwf1, label='VEL')
#plt.plot(df_errors['Velocity'], color=iwf5, label='Velocity' )
# plt.plot(df_errors['POSX'], color=iwf3, label='PX')
#plt.legend(loc='upper left')
# plt.xlim([0,xl])
# plt.ylim([0,yl/5])
#plt.show()

#%%

figure(num=None, figsize=(10, 6), dpi=80, facecolor='w', edgecolor='k')
plt.plot(df_errors['Error'], color=iwf1, label='erro IR')
plt.plot(df_errors['Error_X'], color=ccm5, label='erro X')
plt.plot(df_errors['Error_Y'], color=iwf4, label='erro Y')
plt.plot(df_errors['Error_Z'], color=ccm7, label='erro Z')
plt.legend(loc='upper left')
# plt.xlim([0,xl])
# plt.ylim([0,yl/5])
plt.show()


figure(num=None, figsize=(10, 6), dpi=80, facecolor='w', edgecolor='k')
plt.plot(df_errorsL['Error'], color=iwf1, label='erro Leica')
plt.plot(df_errorsL['Error_X'], color=ccm5, label='erro X')
plt.plot(df_errorsL['Error_Y'], color=iwf4, label='erro Y')
plt.plot(df_errorsL['Error_Z'], color=ccm7, label='erro Z')
plt.legend(loc='upper left')
# plt.xlim([0,xl])
# plt.ylim([0,yl/5])
plt.show()

plt.savefig(dados_robo.strip(".txt") + "Erros Todos.jpg")

#%%

# leicaX= butter_lowpass_filter( df_leica["POSX"], cutoff, fsS, order)
# leicaY= butter_lowpass_filter( df_leica["POSY"], cutoff, fsS, order)

# irX= butter_lowpass_filter( df_T["POSX"], cutoff, fsS, order)
# irY= butter_lowpass_filter( df_T["POSY"], cutoff, fsS, order)

# ff_IR= pd.DataFrame([butter_lowpass_filter( df_T["POSX"], cutoff, fsS, order),
#                       butter_lowpass_filter( df_T["POSY"], cutoff, fsS, order),
#                       butter_lowpass_filter( df_T["POSZ"], cutoff, fsS, order),])
# ff_IR = ff_IR.transpose()
# ff_IR.columns = ['POSX', 'POSY','POSZ']


# ff_L= pd.DataFrame([butter_lowpass_filter( df_leica["POSX"], cutoff, fsS, order),
#                       butter_lowpass_filter( df_leica["POSY"], cutoff, fsS, order),
#                       butter_lowpass_filter( df_leica["POSZ"], cutoff, fsS, order),])
# ff_L = ff_L.transpose()
# ff_L.columns = ['POSX', 'POSY','POSZ']




# # ax = df_T.plot("X_Proj","Y_Proj" ,color=iwf3, label='Proj')
# ax = ff_IR.plot("POSX","POSY" ,color=iwf3, label='Proj')
# # ax.set_title('Trajetória Programada e ROBÔ para POSX x POSZ')
# df_T.plot("POSX", "POSY",ax=ax,color=ccm5, label='IR')
# correct_df_transposed.plot("PosX","PosY", color=iwf1, ax=ax , label='PROG')
# # df_leica.plot("POSX","POSY", color=iwf5, ax=ax , label='LEICA')
# ff_L.plot("POSX","POSY" ,color=iwf5, ax=ax , label='LEICA')



#%%

df_ISO = pd.DataFrame
tol_a = 0.1   #tolerancia altera
tol_e = 0.1    #tolerancia estatico


# Ponto 1: X 1005.18 Y 1000 Z 754.75
dfc =  df_errors[(df_errors.POSX >= 1005.18 - tol_a) & (df_errors.POSX <= 1005.18 + tol_a) &
                 (df_errors.POSY >= 1000 - tol_e) & (df_errors.POSY <= 1000 + tol_e) &
                 (df_errors.POSZ >= 754.75 - tol_a) & (df_errors.POSZ <= 754.75 + tol_a)
                 ]

dfc = dfc.reset_index()

df_ISO = dfc[dfc.index == 0]

# Ponto 2: X 1204.46 Y 1000 Z 937.39
dfc =  df_errors[(df_errors.POSX >= 1204.46 - tol_a) & (df_errors.POSX <= 1204.46 + tol_a) &
                 (df_errors.POSY >= 1000 - tol_e) & (df_errors.POSY <= 1000 + tol_e) &
                 (df_errors.POSZ >= 937.39 - tol_a) & (df_errors.POSZ <= 937.39 + tol_a)
                 ]

dfc = dfc.reset_index()
dfc = dfc[dfc.index == 0]

df_ISO = df_ISO.append(dfc)

# Ponto 3: X 1499.35 Y 1000 Z 1207.67
dfc =  df_errors[(df_errors.POSX >= 1499.35 - tol_a) & (df_errors.POSX <= 1499.35 + tol_a) &
                 (df_errors.POSY >= 1000 - tol_e) & (df_errors.POSY <= 1000 + tol_e) &
                 (df_errors.POSZ >= 1207.67 - tol_a) & (df_errors.POSZ <= 1207.67 + tol_a)
                 ]

dfc = dfc.reset_index()
dfc = dfc[dfc.index == 0]

df_ISO = df_ISO.append(dfc)

# Ponto 4: X 1600 Y 744.48 Z 1300
dfc =  df_errors[(df_errors.POSX >= 1600 - tol_a) & (df_errors.POSX <= 1600 + tol_a) &
                 (df_errors.POSY >= 744.48 - tol_e) & (df_errors.POSY <= 744.48 + tol_e) &
                 (df_errors.POSZ >= 1300 - tol_a) & (df_errors.POSZ <= 1300 + tol_a)
                 ]

dfc = dfc.reset_index()
dfc = dfc[dfc.index == 0]

df_ISO = df_ISO.append(dfc)

# Ponto 5: X 1600 Y 344.47 Z 1300
dfc =  df_errors[(df_errors.POSX >= 1600 - tol_a) & (df_errors.POSX <= 1600 + tol_a) &
                 (df_errors.POSY >= 344.47 - tol_e) & (df_errors.POSY <= 344.47 + tol_e) &
                 (df_errors.POSZ >= 1300 - tol_a) & (df_errors.POSZ <= 1300 + tol_a)
                 ]

dfc = dfc.reset_index()
dfc = dfc[dfc.index == 0]

df_ISO = df_ISO.append(dfc)

# Ponto 6: X 1600 Y -455.56 Z 1300
dfc =  df_errors[(df_errors.POSX >= 1600 - tol_a) & (df_errors.POSX <= 1600 + tol_a) &
                 (df_errors.POSY >= -455.56 - tol_e) & (df_errors.POSY <= -455.56 + tol_e) &
                 (df_errors.POSZ >= 1300 - tol_a) & (df_errors.POSZ <= 1300 + tol_a)
                 ]

dfc = dfc.reset_index()
dfc = dfc[dfc.index == 0]

df_ISO = df_ISO.append(dfc)

# Ponto 7: X 1600 Y -700 Z 1153.14
dfc =  df_errors[(df_errors.POSX >= 1600 - tol_a) & (df_errors.POSX <= 1600 + tol_a) &
                 (df_errors.POSY >= -700 - tol_e) & (df_errors.POSY <= -700 + tol_e) &
                 (df_errors.POSZ >= 1153.14 - tol_a) & (df_errors.POSZ <= 1153.14 + tol_a)
                 ]

dfc = dfc.reset_index()
dfc = dfc[dfc.index == 0]

df_ISO = df_ISO.append(dfc)

# Ponto 8: X 1600 Y -700 Z 753.12
dfc =  df_errors[(df_errors.POSX >= 1600 - tol_a) & (df_errors.POSX <= 1600 + tol_a) &
                 (df_errors.POSY >= -700 - tol_e) & (df_errors.POSY <= -700 + tol_e) &
                 (df_errors.POSZ >= 753.12 - tol_a) & (df_errors.POSZ <= 753.12 + tol_a)
                 ]

dfc = dfc.reset_index()
dfc = dfc[dfc.index == 0]

df_ISO = df_ISO.append(dfc)

# Ponto 9: X 1600 Y -700 Z 648.32
dfc =  df_errors[(df_errors.POSX >= 1600 - tol_a) & (df_errors.POSX <= 1600 + tol_a) &
                 (df_errors.POSY >= -700 - tol_e) & (df_errors.POSY <= -700 + tol_e) &
                 (df_errors.POSZ >= 648.32 - tol_a) & (df_errors.POSZ <= 648.32 + tol_a)
                 ]

dfc = dfc.reset_index()
dfc = dfc[dfc.index == 0]

df_ISO = df_ISO.append(dfc)

# Ponto 10: X 1481.1 Y -700 Z 600
dfc =  df_errors[(df_errors.POSX >= 1481.1 - tol_a) & (df_errors.POSX <= 1481.1 + tol_a) &
                 (df_errors.POSY >= -700 - tol_e) & (df_errors.POSY <= -700 + tol_e) &
                 (df_errors.POSZ >= 600 - tol_a) & (df_errors.POSZ <= 600 + tol_a)
                 ]

dfc = dfc.reset_index()
dfc = dfc[dfc.index == 0]

df_ISO = df_ISO.append(dfc)

# Ponto 11: X 1361.1 Y -700 Z 600
dfc =  df_errors[(df_errors.POSX >= 1361.1 - tol_a) & (df_errors.POSX <= 1361.1 + tol_a) &
                 (df_errors.POSY >= -700 - tol_e) & (df_errors.POSY <= -700 + tol_e) &
                 (df_errors.POSZ >= 600 - tol_a) & (df_errors.POSZ <= 600 + tol_a)
                 ]

dfc = dfc.reset_index()
dfc = dfc[dfc.index == 0]

df_ISO = df_ISO.append(dfc)

# Ponto 12: X 1182.69 Y -700 Z 600
dfc =  df_errors[(df_errors.POSX >= 1182.69 - tol_a) & (df_errors.POSX <= 1182.69 + tol_a) &
                 (df_errors.POSY >= -700 - tol_e) & (df_errors.POSY <= -700 + tol_e) &
                 (df_errors.POSZ >= 600 - tol_a) & (df_errors.POSZ <= 600 + tol_a)
                 ]

dfc = dfc.reset_index()
dfc = dfc[dfc.index == 0]

df_ISO = df_ISO.append(dfc)

# Ponto 13: X 1008.97 Y -671.36 Z 603.6
dfc =  df_errors[(df_errors.POSX >= 1008.97 - tol_a) & (df_errors.POSX <= 1008.97 + tol_a) &
                 (df_errors.POSY >= -671.36 - tol_e) & (df_errors.POSY <= -671.36 + tol_e) &
                 (df_errors.POSZ >= 603.6 - tol_a) & (df_errors.POSZ <= 603.6 + tol_a)
                 ]

dfc = dfc.reset_index()
dfc = dfc[dfc.index == 0]

df_ISO = df_ISO.append(dfc)

# Ponto 14: X 1245.88 Y 86.84 Z 698.32
dfc =  df_errors[(df_errors.POSX >= 1245.88 - tol_a) & (df_errors.POSX <= 1245.88 + tol_a) &
                 (df_errors.POSY >= 86.84 - tol_e) & (df_errors.POSY <= 86.84 + tol_e) &
                 (df_errors.POSZ >= 698.32 - tol_a) & (df_errors.POSZ <= 698.32 + tol_a)
                 ]

dfc = dfc.reset_index()
dfc = dfc[dfc.index == 0]

df_ISO = df_ISO.append(dfc)

# Ponto 15: X 1482.82 Y 845.09 Z 793.09
dfc =  df_errors[(df_errors.POSX >= 1482.82 - tol_a) & (df_errors.POSX <= 1482.82 + tol_a) &
                 (df_errors.POSY >= 845.09 - tol_e) & (df_errors.POSY <= 845.09 + tol_e) &
                 (df_errors.POSZ >= 793.09 - tol_a) & (df_errors.POSZ <= 793.09 + tol_a)
                 ]

dfc = dfc.reset_index()
dfc = dfc[dfc.index == 0]

df_ISO = df_ISO.append(dfc)

df_ISO.to_excel(dados_robo.strip(".txt") + "dadosISO.xlsx")


#%%

#LEICA


df_ISO_L = pd.DataFrame
tol_a = 0.1   #tolerancia altera
tol_e = 0.1    #tolerancia estatico


# Ponto 1: X 1005.18 Y 1000 Z 754.75
dfc =  df_errorsL[(df_errorsL.POSX >= 1005.18 - tol_a) & (df_errorsL.POSX <= 1005.18 + tol_a) &
                 (df_errorsL.POSY >= 1000 - tol_e) & (df_errorsL.POSY <= 1000 + tol_e) &
                 (df_errorsL.POSZ >= 754.75 - tol_a) & (df_errorsL.POSZ <= 754.75 + tol_a)
                 ]

dfc = dfc.reset_index()

df_ISO_L = dfc[dfc.index == 0]

# Ponto 2: X 1204.46 Y 1000 Z 937.39
dfc =  df_errorsL[(df_errorsL.POSX >= 1204.46 - tol_a) & (df_errorsL.POSX <= 1204.46 + tol_a) &
                 (df_errorsL.POSY >= 1000 - tol_e) & (df_errorsL.POSY <= 1000 + tol_e) &
                 (df_errorsL.POSZ >= 937.39 - tol_a) & (df_errorsL.POSZ <= 937.39 + tol_a)
                 ]

dfc = dfc.reset_index()
dfc = dfc[dfc.index == 0]

df_ISO_L = df_ISO_L.append(dfc)

# Ponto 3: X 1499.35 Y 1000 Z 1207.67
dfc =  df_errorsL[(df_errorsL.POSX >= 1499.35 - tol_a) & (df_errorsL.POSX <= 1499.35 + tol_a) &
                 (df_errorsL.POSY >= 1000 - tol_e) & (df_errorsL.POSY <= 1000 + tol_e) &
                 (df_errorsL.POSZ >= 1207.67 - tol_a) & (df_errorsL.POSZ <= 1207.67 + tol_a)
                 ]

dfc = dfc.reset_index()
dfc = dfc[dfc.index == 0]

df_ISO_L = df_ISO_L.append(dfc)

# Ponto 4: X 1600 Y 744.48 Z 1300
dfc =  df_errorsL[(df_errorsL.POSX >= 1600 - tol_a) & (df_errorsL.POSX <= 1600 + tol_a) &
                 (df_errorsL.POSY >= 744.48 - tol_e) & (df_errorsL.POSY <= 744.48 + tol_e) &
                 (df_errorsL.POSZ >= 1300 - tol_a) & (df_errorsL.POSZ <= 1300 + tol_a)
                 ]

dfc = dfc.reset_index()
dfc = dfc[dfc.index == 0]

df_ISO_L = df_ISO_L.append(dfc)

# Ponto 5: X 1600 Y 344.47 Z 1300
dfc =  df_errorsL[(df_errorsL.POSX >= 1600 - tol_a) & (df_errorsL.POSX <= 1600 + tol_a) &
                 (df_errorsL.POSY >= 344.47 - tol_e) & (df_errorsL.POSY <= 344.47 + tol_e) &
                 (df_errorsL.POSZ >= 1300 - tol_a) & (df_errorsL.POSZ <= 1300 + tol_a)
                 ]

dfc = dfc.reset_index()
dfc = dfc[dfc.index == 0]

df_ISO_L = df_ISO_L.append(dfc)

# Ponto 6: X 1600 Y -455.56 Z 1300
dfc =  df_errorsL[(df_errorsL.POSX >= 1600 - tol_a) & (df_errorsL.POSX <= 1600 + tol_a) &
                 (df_errorsL.POSY >= -455.56 - tol_e) & (df_errorsL.POSY <= -455.56 + tol_e) &
                 (df_errorsL.POSZ >= 1300 - tol_a) & (df_errorsL.POSZ <= 1300 + tol_a)
                 ]

dfc = dfc.reset_index()
dfc = dfc[dfc.index == 0]

df_ISO_L = df_ISO_L.append(dfc)

# Ponto 7: X 1600 Y -700 Z 1153.14
dfc =  df_errorsL[(df_errorsL.POSX >= 1600 - tol_a) & (df_errorsL.POSX <= 1600 + tol_a) &
                 (df_errorsL.POSY >= -700 - tol_e) & (df_errorsL.POSY <= -700 + tol_e) &
                 (df_errorsL.POSZ >= 1153.14 - tol_a) & (df_errorsL.POSZ <= 1153.14 + tol_a)
                 ]

dfc = dfc.reset_index()
dfc = dfc[dfc.index == 0]

df_ISO_L = df_ISO_L.append(dfc)

# Ponto 8: X 1600 Y -700 Z 753.12
dfc =  df_errorsL[(df_errorsL.POSX >= 1600 - tol_a) & (df_errorsL.POSX <= 1600 + tol_a) &
                 (df_errorsL.POSY >= -700 - tol_e) & (df_errorsL.POSY <= -700 + tol_e) &
                 (df_errorsL.POSZ >= 753.12 - tol_a) & (df_errorsL.POSZ <= 753.12 + tol_a)
                 ]

dfc = dfc.reset_index()
dfc = dfc[dfc.index == 0]

df_ISO_L = df_ISO_L.append(dfc)

# Ponto 9: X 1600 Y -700 Z 648.32
dfc =  df_errorsL[(df_errorsL.POSX >= 1600 - tol_a) & (df_errorsL.POSX <= 1600 + tol_a) &
                 (df_errorsL.POSY >= -700 - tol_e) & (df_errorsL.POSY <= -700 + tol_e) &
                 (df_errorsL.POSZ >= 648.32 - tol_a) & (df_errorsL.POSZ <= 648.32 + tol_a)
                 ]

dfc = dfc.reset_index()
dfc = dfc[dfc.index == 0]

df_ISO_L = df_ISO_L.append(dfc)

# Ponto 10: X 1481.1 Y -700 Z 600
dfc =  df_errorsL[(df_errorsL.POSX >= 1481.1 - tol_a) & (df_errorsL.POSX <= 1481.1 + tol_a) &
                 (df_errorsL.POSY >= -700 - tol_e) & (df_errorsL.POSY <= -700 + tol_e) &
                 (df_errorsL.POSZ >= 600 - tol_a) & (df_errorsL.POSZ <= 600 + tol_a)
                 ]

dfc = dfc.reset_index()
dfc = dfc[dfc.index == 0]

df_ISO_L = df_ISO_L.append(dfc)

# Ponto 11: X 1361.1 Y -700 Z 600
dfc =  df_errorsL[(df_errorsL.POSX >= 1361.1 - tol_a) & (df_errorsL.POSX <= 1361.1 + tol_a) &
                 (df_errorsL.POSY >= -700 - tol_e) & (df_errorsL.POSY <= -700 + tol_e) &
                 (df_errorsL.POSZ >= 600 - tol_a) & (df_errorsL.POSZ <= 600 + tol_a)
                 ]

dfc = dfc.reset_index()
dfc = dfc[dfc.index == 0]

df_ISO_L = df_ISO_L.append(dfc)

# Ponto 12: X 1182.69 Y -700 Z 600
dfc =  df_errorsL[(df_errorsL.POSX >= 1182.69 - tol_a) & (df_errorsL.POSX <= 1182.69 + tol_a) &
                 (df_errorsL.POSY >= -700 - tol_e) & (df_errorsL.POSY <= -700 + tol_e) &
                 (df_errorsL.POSZ >= 600 - tol_a) & (df_errorsL.POSZ <= 600 + tol_a)
                 ]

dfc = dfc.reset_index()
dfc = dfc[dfc.index == 0]

df_ISO_L = df_ISO_L.append(dfc)

# Ponto 13: X 1008.97 Y -671.36 Z 603.6
dfc =  df_errorsL[(df_errorsL.POSX >= 1008.97 - tol_a) & (df_errorsL.POSX <= 1008.97 + tol_a) &
                 (df_errorsL.POSY >= -671.36 - tol_e) & (df_errorsL.POSY <= -671.36 + tol_e) &
                 (df_errorsL.POSZ >= 603.6 - tol_a) & (df_errorsL.POSZ <= 603.6 + tol_a)
                 ]

dfc = dfc.reset_index()
dfc = dfc[dfc.index == 0]

df_ISO_L = df_ISO_L.append(dfc)

# Ponto 14: X 1245.88 Y 86.84 Z 698.32
dfc =  df_errorsL[(df_errorsL.POSX >= 1245.88 - tol_a) & (df_errorsL.POSX <= 1245.88 + tol_a) &
                 (df_errorsL.POSY >= 86.84 - tol_e) & (df_errorsL.POSY <= 86.84 + tol_e) &
                 (df_errorsL.POSZ >= 698.32 - tol_a) & (df_errorsL.POSZ <= 698.32 + tol_a)
                 ]

dfc = dfc.reset_index()
dfc = dfc[dfc.index == 0]

df_ISO_L = df_ISO_L.append(dfc)

# Ponto 15: X 1482.82 Y 845.09 Z 793.09
dfc =  df_errorsL[(df_errorsL.POSX >= 1482.82 - tol_a) & (df_errorsL.POSX <= 1482.82 + tol_a) &
                 (df_errorsL.POSY >= 845.09 - tol_e) & (df_errorsL.POSY <= 845.09 + tol_e) &
                 (df_errorsL.POSZ >= 793.09 - tol_a) & (df_errorsL.POSZ <= 793.09 + tol_a)
                 ]

dfc = dfc.reset_index()
dfc = dfc[dfc.index == 0]

df_ISO_L = df_ISO_L.append(dfc)

df_ISO_L.to_excel(dados_robo.strip(".txt") + "dadosISO.xlsx")





#%%

df_errors["Error"].mean()

df_errors["Error_X"].mean()
df_errors["Error_Y"].mean()
df_errors["Error_Z"].mean()


df_errorsL["Error"].mean()

df_errorsL["Error_X"].mean()
df_errorsL["Error_Y"].mean()
df_errorsL["Error_Z"].mean()

#%%



#Salvar os dados em outros arquivos

df_errors.to_csv(dados_robo.strip(".txt") + "total.csv")

df_errorsL.to_csv(dados_robo.strip(".txt") + "totalleica.csv")





#%%


#df = pd.read_csv(dados_robo.strip(".txt") + "total.csv")


# In[3]:


#df.head()


# In[4]:


#df2 = df.drop(['Unnamed: 0'], axis=1)


# In[5]:


 # df3 = df2[df2["Error"] < 0.15]



   # In[6]:
#df3 = df2[(df2["Error"] > 0.05) & (df2["Error"] < 2)]

# df3 = df3[df3["Error"] > 0.05]


# In[7]:




# In[9]:


#df4 = df3[['Vector']]


# In[11]:


#s = df4["Vector"].value_counts()




# In[13]:


#teste  = pd.DataFrame(data=s)




# In[15]:


#teste = teste[teste["Vector"] > 10]


# In[16]:


#indexNamesArr = teste.index.values
#listOfRowIndexLabels = list(indexNamesArr)


# In[17]:


#listOfRowIndexLabels

# In[18]:


#def reformulate_list(l):
#    li = []
#    for item in l:
#        item = item.replace('[','[ ')
#        item = re.sub('\s+',' ', item)
#        item = item.replace('[','{X')
#        item = item.replace(' ]',']')
 #       item = item.replace(']',',A')
  #      item = re.sub('(?<=[^X]) (?=\S+ )',',Y ', item)
#        item = re.sub('(?<=[^XY]) (?=\S+,)',',Z ', item)
#        item = re.sub('\.0(?=,)','.000', item)
#        li.append(item)
        
#    return li


# In[19]:


#new_List = reformulate_list(listOfRowIndexLabels)



# In[23]:
#list_to_create_df = [new_List]

#df_values_to_change = pd.DataFrame.from_records(list_to_create_df)
#df_values_to_change_transposed = df_values_to_change.transpose()
#df_values_to_change_transposed.columns = ["Vector"]



    
#%%

#def get_velocity(df, base_program):
#    k = 0
#    df["Velocity"] = "No data"
    
#    while k < len(df):
#        a = df.Vector[k]
        
#        posx = re.search('(?<=X )\S+(?=,)', a)
#        posy = re.search('(?<=Y )\S+(?=,)', a)
#        posz = re.search('(?<=Z )\S+(?=,)', a)
        
#        posx = posx.group(0)
#        posy = posy.group(0)
#        posz = posz.group(0)
        
#        posx = float(posx)
#        posy = float(posy)
#        posz = float(posz)
        
#        for i in range(0,len(base_program)):
#            if base_program.PosX[i] == posx and base_program.PosY[i] == posy and base_program.PosZ[i] == posz:
#                df.Velocity[k] = base_program.VEL[i]
#                break
            
#        k = k + 1
        
#    return df
     
    
#%%
    
#new_df = get_velocity(df_values_to_change_transposed, correct_df_transposed)



# In[40]:


#input_file = dados_programado
#output_file = input_file.strip(".src") + "_New.src"


# In[42]:


#def create_file(output_file, input_file):
#    if not os.path.exists('./' + output_file):
       
#        f = open(output_file, "x")   # create file if doesnt exist
        
#    else:
#        f = open(output_file,"r+")
#        f.truncate(0)
    
    
    
    
#    #read input file
#    fin = open(input_file, "rt")
#    #read file contents to string
#    data = fin.read()
#    #write data to the file
#    f.write(data)
#    f.close()
#    fin.close()


# In[45]:


#create_file(output_file, input_file)


# In[ ]:

#essa parte faz mudar fácil


#def altera(df, base_program, file):
    
#    df = df.reset_index(drop=True)

    #read input file
#    fin = open(file, "rt")
    #read file contents to string
#    data = fin.read()

#    for value in range(0, len(df)):
        
        
#        a = df.Vector[value]
    
#        new_vel = df.Velocity[value]*0.8 #80% da velocidade
        
        #print(vel)
    
#        vel_m = "$VEL.CP = " + str(new_vel)
        
    
        

#        b = a.split()
#        b1 = float(b[1].replace(",Y",""))
#        b2 = float(b[2].replace(",Z",""))
#        b3 = float(b[3].replace(",A",""))
        
        
#        ax =  str(b1)
#        ay =  str(b2)
#        az =  str(b3)
#        ax = re.sub('\.0(?!\d)','.000', ax)
#        ay = re.sub('\.0(?!\d)','.000', ay)
#        az = re.sub('\.0(?!\d)','.000', az)
        
        
#        ax = re.sub('\d+\.\d(?!\d)',ax+"00", ax)
#        ay = re.sub('\d+\.\d(?!\d)',ay+"00", ay)
#        az = re.sub('\d+\.\d(?!\d)',az+"00", az)
        
#        ax = re.sub('\d+\.\d\d(?!\d)',ax+"0", ax)
#        ay = re.sub('\d+\.\d\d(?!\d)',ay+"0", ay)
#        az = re.sub('\d+\.\d\d(?!\d)',az+"0", az)        
        
        
#        ax = re.sub('-+','-', ax)
#        ay = re.sub('-+','-', ay)
#        az = re.sub('-+','-', az)
        
#        aa = "{X "+ ax +",Y "+ ay +",Z "+ az +",A"
        
#        #replace all occurrences of the required string
#        data = data.replace("LIN " + aa, vel_m+"\n"+"LIN " +aa)
        

#        for i in range(0,len(base_program)-1):
#            if base_program.PosX[i] == b1 and base_program.PosY[i] == b2 and base_program.PosZ[i] == b3:
#                pos = i
#                print("step")
                
#        vel = base_program.VEL[pos+1] #volta a velocidade do prox ponto
#        vel_a = "$VEL.CP = " + str(vel)
        
        
        
        
        
        
        

#        #print(pos+1)
#        print("st")
#        nx = str(base_program.PosX[pos+1])
#        ny = str(base_program.PosY[pos+1])
#        nz = str(base_program.PosZ[pos+1])
#        nx = re.sub('\.0(?!\d)','.000', nx)
#        ny = re.sub('\.0(?!\d)','.000', ny)
#        nz = re.sub('\.0(?!\d)','.000', nz)
        
        
#        nx = re.sub('\d+\.\d(?!\d)',nx+"00", nx)
#        ny = re.sub('\d+\.\d(?!\d)',ny+"00", ny)
#        nz = re.sub('\d+\.\d(?!\d)',nz+"00", nz)
        
#        nx = re.sub('\d+\.\d\d(?!\d)',nx+"0", nx)
#        ny = re.sub('\d+\.\d\d(?!\d)',ny+"0", ny)
#        nz = re.sub('\d+\.\d\d(?!\d)',nz+"0", nz)        
        
        
#        nx = re.sub('-+','-', nx)
#        ny = re.sub('-+','-', ny)
#        nz = re.sub('-+','-', nz)
#        print(nx)
#        print(ny)
#        print(nz)
        

#        na = "{X "+ nx +",Y "+ ny +",Z "+ nz +",A"

#        data = data.replace("LIN " + na, vel_a+"\n"+"LIN " +na)
        
        
#        #data = data.replace(vel_m+"\n"+vel_a, vel_m)
#        #data = data.replace(vel_a+"\n"+vel_m, vel_m)


#    #data = data.replace(r'(?<=\$VEL.CP = \S+\n)\$VEL.CP = \S+\n',"")
#    #data = re.sub("(?<=\$VEL.CP = 0.034\n)\$VEL.CP = \S+\n", "", data)

#    #close the input file
#    fin.close()
#    #open the input file in write mode
#    fin = open(file, "wt")
#    #overrite the input file with the resulting data
#    fin.write(data)
#    #close the file
#    fin.close()



# In[ ]:

#altera(new_df, correct_df_transposed, output_file)



# In[46]:

#time
stop = timeit.default_timer()

print('Time: ', stop - start)  

#%%
