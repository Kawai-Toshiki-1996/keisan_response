import random
import pandas as pd
import math
import numpy as np
import matplotlib.pyplot as plt
print("Ok")

ship_name='BLUE_JAY'


"""応答関数読み込む"""
a = pd.read_csv("strip_on_JWA.csv",engine='python',header=None,skiprows=1)#widly_20_Sym:shift_jis
strip =a.values
"""海象データ読み込む"""
wave=pd.read_csv("元データから計算すべき海象抽出/"+ship_name+"_JWA海象データ.csv",engine='python',header=None,skiprows=1)
WaveData=wave.values#[:,25:925]
"""csvファイルの読み込み"""
inputdata= pd.read_csv('C:/Users/kouzou/Desktop/MasterKawai/JWA/GAを用いて全探索/推定/Keras/Datasets/'+ship_name+'_by_FFT_全データ.csv', 
                       engine='python',header=None,skiprows=1,usecols=[4,5])
kn  = inputdata.values[:,0]
HDG = inputdata.values[:,1]
print(np.shape(WaveData))
print(np.shape(kn))
print(np.shape(HDG)
print(np.shape(strip))
print("Ok")

"""諸々の定義"""
n_fruid  = 1    #速度６種類
n_angle  = 36   #0~350まで36パターン 0,10,20,30,40,50,60,70,80,90,100,110,120,130,140,150,160,170,180,190,200,
n_lambda = 25   #波長40パターン
Iy = 2661319053 #mm^2*m^2
y_y = 17.612 #[m] 中立軸
Zc=76.329
kissui=15.7
interval=0.03

omegalist=[]
for i in range(n_lambda):
    omegalist.append(strip[i,1])
omegalist=np.array(omegalist)
encount_omega_2 = np.arange( 0, 2.11,interval )
print("Ok")


class Wave_Spectrum:    
    def __init__(self):
        self.SeaStateValues=np.reshape(SeaValues*(360/n_angle)*np.pi/180/2/np.pi,(25,36))#2π掛けてること忘れずに/ 10°分かけてる
    
    def Direc(self,relative_index,i):
        """2次元方向分布の値の計算"""
        return int(relative_index-i) if int(relative_index-i)>=0 else 36+int(relative_index-i)

    def Directional_wave_spectrum(self,theta_c):       
        """2次元方向分布の値の計算"""
        relative_direction = (theta_c-180) if (theta_c-180) >=0 else 360+(theta_c-180)#船体後方の角度
        relative_index=((relative_direction+5)%360)//10#175~185が180°
        relative_angle= [self.Direc(relative_index,i) for i in range(n_angle)]
        return np.array([[self.SeaStateValues[i][j] for j in relative_angle] for i in range(n_lambda)])

class Data_Handling(Wave_Spectrum):
    def __init__(self,column1,column2):
        self.column1=column1
        self.column2=column2
        self.response1=strip[:][self.column1]
        self.response2=strip[:][self.column2]
        self.Se_data=[]
        self.S_data=[]
        self.z1= 1000*9.8065/Zc/10**6 if column1 == 5 else 1  #1000*9.8065/Zc[m^3]/10^6なんじゃない？#1000*9.8065/Zc
        self.z2= 1000*9.8065/Zc/10**6 if column2 == 5 else 1  #1000*9.8065/Zc
        self.Matrix=np.array([[0 for k in range(2)] for l in range(len(encount_omega_2))], dtype = 'float64')
        #self.omega_vertex=
        self.V  =Fn*math.sqrt(349.2*9.80655)
    
    def Response_calculation(self,line1):
        X_Conjugate = strip[line1][self.column1]
        Y= strip[line1][self.column2]
        return X_Conjugate*Y #strip[line1][self.column1]*strip[line1][self.column1]#
    
    def dw_function(self,omega_number):
        return np.abs(1.0-2.0*self.V/9.80655*omegalist[omega_number]*math.cos(encount_angle*math.pi/180.0))        
    
    def To_trans_Se(self,line,w,j,switch,Eomega1):#Eomega1,2=Eomega_fore,aft
        """線形補間"""  
        Sf_re,Sa_re=self.Response_calculation(line).real*wave_values[w][j],self.Response_calculation(line-1).real*wave_values[w-1][j]
        Sf_im,Sa_im=self.Response_calculation(line).imag*wave_values[w][j],self.Response_calculation(line-1).imag*wave_values[w-1][j]
        dw= omegalist[w]-omegalist[w-1]
        if len(list_omega_j) == 0 and Eomega1 < np.max(encount_omega_2):
            self.Matrix[int(Eomega1/interval)][0] += (Sa_re+Sf_re)*np.abs(dw)/2/interval
            self.Matrix[int(Eomega1/interval)+1][0] += (Sa_re+Sf_re)*np.abs(dw)/2/interval
            self.Matrix[int(Eomega1/interval)][1] += (Sa_im+Sf_im)*np.abs(dw)/2/interval
            self.Matrix[int(Eomega1/interval)+1][1] += (Sa_im+Sf_im)*np.abs(dw)/2/interval
            
        for w_e in list_omega_j:        
            if switch == 1: #頂点のx座標が挟まれる場合
                self.Matrix[w_e][0] += (Sa_re+Sf_re)*dw/2/(omega_vertex/2-min(Eomega1,Eomega2))
                self.Matrix[w_e][1] += (Sa_im+Sf_im)*dw/2/(omega_vertex/2-min(Eomega1,Eomega2))
            elif switch == 2: #x軸との交点が挟まれる場合
                self.Matrix[w_e][0] += (Sa_re+Sf_re)*dw/2/max(Eomega1,Eomega2)
                self.Matrix[w_e][1] += (Sa_im+Sf_im)*dw/2/max(Eomega1,Eomega2)
            else:
                dwe1 = self.dw_function(w-1)
                dwe2 = self.dw_function(w)
                self.Matrix[w_e][0] +=(Eomega2-encount_omega_2[w_e])/(Eomega2-Eomega1)*Sa_re/dwe1+(encount_omega_2[w_e]-Eomega1)/(Eomega2-Eomega1)*Sf_re/dwe2
                self.Matrix[w_e][1] +=(Eomega2-encount_omega_2[w_e])/(Eomega2-Eomega1)*Sa_im/dwe1+(encount_omega_2[w_e]-Eomega1)/(Eomega2-Eomega1)*Sf_im/dwe2
            
    def final_keisan(self,f):
        F =np.zeros(len(encount_omega_2))
        for w_e in range(len(encount_omega_2)):
            F[w_e] += math.sqrt(self.Matrix[w_e][0]**2+self.Matrix[w_e][1]**2)*self.z1*self.z2
        m0=0
        for i in range(len(encount_omega_2)-1):
            m0 +=(F[i]+F[i+1])*interval/2
        f.write(str(m0)+",")

def search_omega_i(i):
    """間に入る出会い波周波数のリスト作成"""
    Eomega1=np.abs(omegalist[i-1]*(1-Fn*math.sqrt(349.2*9.80655)/9.80655*omegalist[i-1]*math.cos(encount_angle*math.pi/180)))
    Eomega2=np.abs(omegalist[i]*(1-Fn*math.sqrt(349.2*9.80655)/9.80655*omegalist[i]*math.cos(encount_angle*math.pi/180)))
    Eomega_fore = max(Eomega1,Eomega2)
    Eomega_aft =  min(Eomega1,Eomega2)
    switch=0#0は通常，１は頂点が挟まれる，２はｘとの交点が挟まれる
    if (omega_vertex - omegalist[i-1])*(omega_vertex -omegalist[i]) <= 0.000: #頂点のx座標が挟まれる場合
        Eomega_fore = omega_vertex/2 #頂点のy座標はx座標の半分だから
        switch=1
    elif (omega_vertex*2 - omegalist[i-1])*(omega_vertex*2 -omegalist[i]) <= 0: #x軸との交点が挟まれる場合
        Eomega_aft = np.abs(0.0)
        switch=2
    return np.arange(len(encount_omega_2))[np.array( (encount_omega_2-Eomega_fore)*(encount_omega_2-Eomega_aft) <= 0 ).flatten()] ,Eomega1, Eomega2,switch
	
	
"""csvファイルへ書き込み"""
f=open('m0_'+ship_name+'_on_JWA.csv',"w")
f.write("P,R,H,S")
f.write('\n')	


for seastate in range(len(WaveData)): 
    """この海象のパラメータ"""  
    theta_course=HDG[seastate]
    SeaValues=WaveData[seastate,25:925]
    """相対波向きに注意"""
    wave=Wave_Spectrum()
    
    for speed in range(1): #6フルード数ごと 0,5,10,15,20,25
        Fn    = 0.176*(kn[seastate]/20)#strip_data[n_lambda*n_angle*k][0]#フルード数
        wave_values=wave.Directional_wave_spectrum(theta_course)
        
        """データを扱うためのクラス"""
        Pitch=Data_Handling(2,2)
        Roll =Data_Handling(3,3)
        Heave=Data_Handling(4,4)
        Load =Data_Handling(5,5)#zを乗算するところのif分で列番号直すの忘れずに！
        
        for j,encount_angle in enumerate(np.arange(0,360,360/n_angle)): #36の角度          
            omega_vertex  = 9.80655/(2*Fn*math.sqrt(349.2*9.80655)*math.cos(encount_angle*math.pi/180)) #頂点のx座標
            
            for w in range(1,len(omegalist)): #25周波数ごと n_lambda
                line = w+n_lambda*j+n_lambda*n_angle*speed    #行           
                list_omega_j,Eomega1, Eomega2,switch = search_omega_i(w)
                """それぞれの応答で変換実行"""
                Pitch.To_trans_Se(line,w,j,switch,Eomega1)#注意！！複素数ではなく純虚数対応になってるよ
                Roll.To_trans_Se(line,w,j,switch,Eomega1)
                Heave.To_trans_Se(line,w,j,switch,Eomega1)
                Load.To_trans_Se(line,w,j,switch,Eomega1)

        """最終アウトプット"""        
        Pitch_ver3 =Pitch.final_keisan(f)
        Roll_ver3 =Roll.final_keisan(f)
        Heave_ver3=Heave.final_keisan(f)
        Load_ver3 =Load.final_keisan(f)
        #Pitch_Roll.X_final_keisan()
        #Roll_Heave.X_final_keisan()
        #Heave_Pitch.X_final_keisan()
        #Load_Roll.X_final_keisan()
        #Load_Heave.X_final_keisan()
        #Load_Pitch.X_final_keisan()
        
        f.write('\n')
    print(seastate+1)
print("End")
f.close()