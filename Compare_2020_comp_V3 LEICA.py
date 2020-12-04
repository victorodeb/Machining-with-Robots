# -*- coding: utf-8 -*-
"""
Created on Fri Jul 17 10:10:54 2020

@author: Victor Andre
"""

import pandas as pd

pd.set_option("display.max_colwidth", 10000)
import re

import matplotlib
import numpy as np
np.set_printoptions(suppress=True)
from mpl_toolkits import mplot3d

import timeit


start = timeit.default_timer()

#Arquivos de dados

dados_programado = "Test_20200803.src"
dados_leica = "Leica_.txt"


#%%

dfleica = pd.read_csv(dados_leica, names=["POSX","POSY","POSZ"])
dfleica = dfleica.drop(dfleica.index[0])
dfleica = dfleica.astype("float")
dfleica.head()

dfleica["POSZ"] = dfleica["POSZ"] 

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
    n = np.sqrt(np.power(ab[0],2) + np.power(ab[1],2) + np.power(ab[2],2))

    return d/n

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


def myfuncVEL(j, df_errorsL):
    if j == 0 or j == len(df_errorsL):
        v = 0
    else:
        X = np.power(df_errorsL.POSX[j] - df_errorsL.POSX[j-1],2)
        Y = np.power(df_errorsL.POSY[j] - df_errorsL.POSY[j-1],2)
        Z = np.power(df_errorsL.POSZ[j] - df_errorsL.POSZ[j-1],2)
        
        d = np.sqrt(X+Y+Z)
        t = df_errorsL.TIME[j] - df_errorsL.TIME[j-1]
        
        v = d/t
        
    return v

#

# te = np.array((correct_df_transposed.PosX[1],correct_df_transposed.PosY[1],correct_df_transposed.PosZ[1]))

# print("["+str(te[0]) + " "+ str(te[1]) + " "+ str(te[2]) + "]")


#%%
LIMXY = 0.5
LIMZ  = 0.5
LIMPTP = 11

def calcula(base_program, df): 
    print(df.POSX[0]) 
    i = 0
    df["Error"] = "No data"
    df["X_Proj"] = "No data"
    df["Y_Proj"] = "No data"
    df["Z_Proj"] = "No data"
    df["Vector"] = "No data"

    df_errorsL = pd.DataFrame
    f = 0
    
    change_list = []

    pos = 0
    
    #pega os pontos de troca
                 
    
    for j in range(0,len(df)):
        pos_atual = j
        
        if pos_atual >= len(base_program) - 1:
            break
        
            
        
        
        
        if base_program.TYPE[pos_atual] == "LIN":
            print("hey")
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
            print("nao era ")
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
            print("EGM")
            if df.POSX[pos] >= lix and df.POSX[pos] <= lsx:     
        
                if df.POSY[pos] >= liy and df.POSY[pos] <= lsy:
                    
                    if df.POSZ[pos] >= liz and df.POSZ[pos] <= lsz:
                        achou = pos
                        print("Oi Ever")
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
 
        # valores da reta/vetor
       
        a = np.array((base_program.PosX[i],base_program.PosY[i],base_program.PosZ[i]))
        b = np.array((base_program.PosX[i+1],base_program.PosY[i+1],base_program.PosZ[i+1]))
        n = b - a
            
        d1 = d1.reset_index(drop=True)
        
        if len(d1) != 0:
            print(i)
            print(d1.index[0])
            print(d1.POSX[0])
            print(d1.POSY[0])
            print(d1.POSZ[0])

            
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

#Salvar os dados em outros arquivos

df_errorsL.to_csv(dados_leica.strip(".txt") + "TOTAL.csv")


#%%




from matplotlib.pyplot import figure
from matplotlib import pyplot as plt

%matplotlib qt

figure(num=None, figsize=(10, 8), dpi=80, facecolor='w', edgecolor='k')

ax = plt.axes(projection='3d')


 # Data for a three-dimensional line
zline = correct_df_transposed["PosZ"]
xline = correct_df_transposed["PosX"]
yline = correct_df_transposed["PosY"]
ax.plot3D(xline, yline, zline, 'blue',label='Programado')
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('z');

 # Data for three-dimensional scattered points
zdata = dfleica["POSZ"]
xdata = dfleica["POSX"]
ydata = dfleica["POSY"]
ax.plot3D(xdata, ydata, zdata, 'orange', label="ROBÔ");

ax.set_title('Trajetória Programada x Trajetória ROBÔ')
leg = ax.legend()

ax.zlim = (120,130)





