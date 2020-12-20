"""応答関数読み込む"""
a = pd.read_csv("20knot_wide_ver2.csv", encoding="shift_jis")#widly_20_Sym:shift_jis
strip =a.values
print(strip[:,13])
strip2D=np.reshape(strip[:,13],(72,80)).T
print(strip2D)


"""諸々の定義"""
n_fruid  = 1    #速度６種類
n_angle  = 72   #0~350まで36パターン 0,10,20,30,40,50,60,70,80,90,100,110,120,130,140,150,160,170,180,190,200,
n_lambda = 80   #波長40パターン
Iy = 2661319053 #mm^2*m^2
y_y = 17.612 #[m] 中立軸
Zc=76.329#82.211[m^3]をなんじゃない？
kissui=15.7
interval=0.03

omegalist=[]
for i in range(n_lambda):
    omegalist.append(strip[i][3])
omegalist=np.array(omegalist)
encount_omega_2 = np.arange( 0, 2.11,0.03 )


class Wave_Spectrum:    
    def __init__(self):
        self.Hv_1    =Hv_1
        self.theta_1 =theta_1
        self.lambda_1=lambda_1
        self.omega_1 =omega_1
        #self.S_max1  =0
        self.Hv_2    =Hv_2
        self.theta_2 =theta_2
        self.lambda_2=lambda_2
        self.omega_2 =omega_2
        #self.S_max2  =0
    
    def def_S_max(self,H,omega_m):
        HL=H/(1.56*4*math.pi**2/omega_m**2)
        """波集中度s1の周波数依存"""
        if H==0:
            S_max=0
        elif HL<0.026:
            log_10=0.4444*math.log10(HL)+0.5849
            a=10**(log_10)
            S_max=12.99*a**(-2.5)     
        elif HL>=0.026:
            log10_gF=(0.1507-math.sqrt(0.1507**2+4*0.005882*0.8789+4*0.005882*math.log10(HL)))/(2*0.005882)
            gF=10**log10_gF
            gT=1.37*(1-1/(1+0.008*gF**(1/3))**5)
            S_max=12.99*gT**2.5
        return np.array([(S_max*i**5/omega_m**5) if i<=omega_m else (S_max*i**(-2.5)/omega_m**(-2.5)) for i in omegalist])#S_max
        
    def A(self,x):
        """ガンマ関数×2^(2s-1)÷π"""
        if x >= 84:
            direction_m=np.prod([2*(x-k)/(2*x-2*k-1) for k in range(int(x))])*2**(2*x-2*int(x)-1)*(math.gamma(x-int(x)+1))**2.0/(np.pi*math.gamma(2*x-2*int(x)+1))
        else:
            direction_m=2.0**(2.0*x-1.0)*(math.gamma(x+1.0))**2.0/math.pi/math.gamma(2.0*x+1.0)
        return direction_m
    
    def spectrum(self,Hv_m,lambda_m,omega_m,omega):
        """2π掛けてること忘れずに-->OK"""
        spectrum_1=0.250*((4.0*lambda_m+1.0)/4.0*omega_m**4.0)**lambda_m*Hv_1**2.0/omega**(4.0*lambda_m+1.0)/math.gamma(lambda_m)
        spectrum_2=math.exp(-(4.0*lambda_m+1.0)/4.0*(omega_m/omega)**4.0)
        return spectrum_1*spectrum_2
    
    def Direc(self,s_m,theta_m):
        """2次元方向分布の値の計算"""
        return np.array([[abs((math.cos((j*(360/n_angle)-theta_m)/2.0*math.pi/180.0)))**(2.0*s_m[i]) for j in range(n_angle)] for i in range(n_lambda)])

    def Directional_wave_spectrum(self):
        """S_maxの計算とsの関数の計算"""
        s_1 =self.def_S_max(self.Hv_1,self.omega_1)
        s_2 =self.def_S_max(self.Hv_2,self.omega_2)
        """周波数方向のスペクトルの値の計算"""
        Ochi_A_1=np.array([self.spectrum(self.Hv_1,self.lambda_1,self.omega_1,omegalist[i])*self.A(s_1[i]) for i in range(n_lambda)])
        Ochi_A_2=np.array([self.spectrum(self.Hv_2,self.lambda_2,self.omega_2,omegalist[i])*self.A(s_2[i]) for i in range(n_lambda)])
        """2次元方向分布の値の計算"""
        Direc_1=self.Direc(s_1,self.theta_1)*2*np.pi/n_angle
        Direc_2=self.Direc(s_2,self.theta_2)*2*np.pi/n_angle

        """配列の積と二峰分のエネルギーの和"""
        Spectrum_2D =np.expand_dims(Ochi_A_1,1)*Direc_1+np.expand_dims(Ochi_A_2,1)*Direc_2#np.expand_dimsで1次元配列を25行1列の配列にした後，ブロードキャスト
        return Spectrum_2D# [[Spectrum_2D[i][j] for j in range(n_angle)] for i in range(n_lambda)]


class Data_Handling():
    def __init__(self,Velocity=U,Omegalist=omegalist,N_angle=n_angle,N_lambda=n_lambda):
        self.Omegalist=Omegalist
        self.N_angle=N_angle
        self.N_lambda=N_lambda
        self.Velocity=Velocity
        
        self.Dw_matrix=np.abs(1.0-2.0*U/9.80655*np.expand_dims(self.Omegalist,1)*np.cos(np.arange(0,360,360/self.N_angle)*np.pi/180.0))
        #self.Se_matrix = Response_calculation()*wave_values/dw_matrix
        #self.S_data=[]
        #self.Matrix=np.array([[0 for k in range(2)] for l in range(len(encount_omega_2))], dtype = 'float64')
        self.Velocity  =U
    
    def Start_calculation(self,S_matrix):
        return S_matrix
    
    
    def To_trans_Se(self,line,w,j,switch,Eomega1):
        """線形補間"""
        Sf_re,Sa_re=self.Response_calculation(line).real*wave_values[w][j],self.Response_calculation(line-1).real*wave_values[w-1][j]
        Sf_im,Sa_im=self.Response_calculation(line).imag*wave_values[w][j],self.Response_calculation(line-1).imag*wave_values[w-1][j]
        dw= omegalist[w]-omegalist[w-1]
        if len(list_omega_j) == 0 and Eomega1 < np.max(encount_omega_2):
            self.Matrix[int(Eomega1/interval)][0] += (Sa_re+Sf_re)*np.abs(omegalist[w]-omegalist[w-1])/2/interval
            self.Matrix[int(Eomega1/interval)+1][0] += (Sa_re+Sf_re)*np.abs(omegalist[w]-omegalist[w-1])/2/interval
            self.Matrix[int(Eomega1/interval)][1] += (Sa_im+Sf_im)*np.abs(omegalist[w]-omegalist[w-1])/2/interval
            self.Matrix[int(Eomega1/interval)+1][1] += (Sa_im+Sf_im)*np.abs(omegalist[w]-omegalist[w-1])/2/interval
            
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
            
    def final_keisan(self):
        for w_e in np.sqrt(self.Matrix[:,0]**2+self.Matrix[:,1]**2)*self.z1*self.z2:
            f.write(str(w_e)+",")       
        return 
    
    def X_final_keisan(self):
        for w_e in self.Matrix[:,0]:
            f.write(str(w_e)+",")
        for w_e in self.Matrix[:,1]:
            f.write(str(w_e)+",")
        return        

"""ここからはクラスによらないdef関数が続く"""
def get_parameter_from_ind():
    """波高"""
    Hv_1=10*random.uniform(0,1)
    H_bottom= 0 if Hv_1<6 else (0.01*(Hv_1-6)**2.4)
    H_bupper= Hv_1**0.2+6
    Hv_2= H_bottom +(H_bupper-H_bottom)*random.uniform(0,1)
    Hs    =np.sqrt(Hv_1**2+Hv_2**2)
    
    """波周波数"""
    omega_upper =(2.5-1.6/(1+3*np.exp(-Hs+1.5)))
    omega_bottom=2*np.pi/18*np.exp(-0.005*Hs*3.28084)
    omega_1 = omega_bottom+(omega_upper-omega_bottom)*random.uniform(0,1)
    w_lower = omega_1# if omega_1>=0.6 else 0.6
    omega_2 = w_lower+(2.4-1.6/(1+2.5*np.exp(-Hs+5))-w_lower)*random.uniform(0,1)
        
    """波向"""
    theta_1  =random.random()*360
    theta_2  =random.random()*360
    
    """s"""
    s_1      =0
    s_2      =0
    
    """尖度"""
    lambda_1 =10/Hv_1 if 10/Hv_1 <=9.9 else 9.9 #0.1+9.9*random.uniform(0,1)
    lambda_1 =lambda_1*random.uniform(0,1)+0.1
    #lambda_temp=lambda_1 if lambda_1<=6 else 6
    lambda_2 =10/Hv_2 if 10/Hv_2 <=9.9 else 9.9   #0.1+(lambda_temp-0.1)*random.uniform(0,1)
    lambda_2 = lambda_2*random.uniform(0,1)+0.1
    return Hv_1,theta_1,lambda_1,omega_1,Hv_2,theta_2,lambda_2,omega_2

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

def output(Hv_1,theta_1,lambda_1,omega_1,Hv_2,theta_2,lambda_2,omega_2,f):
    f.write(str('%03.6f' %(Hv_1))+","+str('%03.6f' %(math.cos(theta_1*math.pi/180)))+","+str('%03.6f' %(math.sin(theta_1*math.pi/180)))+","+str(lambda_1)+","+str('%03.6f' %(omega_1))+",")
    f.write(str('%03.6f' %(Hv_2))+","+str('%03.6f' %(math.cos(theta_2*math.pi/180)))+","+str('%03.6f' %(math.sin(theta_2*math.pi/180)))+","+str(lambda_2)+","+str('%03.6f' %(omega_2)))
    f.write('\n')
    
def make_we_matrix():
    return np.abs(np.expand_dims(omegalist,1)*(1-np.expand_dims(omegalist,1)*U/9.80655*np.cos(np.arange(0,360,360/n_angle)*np.pi/180)))


def RAO_RAO_2D(strip,column1,column2):
    amp1=np.reshape(strip[:,column1],(72,80)).T
    phase1=np.reshape(strip[:,column1+1],(72,80)).T
    amp2=np.reshape(strip[:,column2],(72,80)).T
    phase2=np.reshape(strip[:,column2+1],(72,80)).T
    z1= 1000*9.8065/Zc/10**6 if column1 == 17 else 1
    z2= 1000*9.8065/Zc/10**6 if column2 == 17 else 1
    """2成分の応答を掛け合わせる"""
    X_Conjugate =amp1*(np.cos(-phase1*np.pi/180)+1j*np.sin(-phase1*np.pi/180))
    Y=amp2*(np.cos(phase2*np.pi/180)+1j*np.sin(phase2*np.pi/180)) 
    return X_Conjugate*Y*z1*z2


"""計算開始"""
Pitch_2D=RAO_RAO_2D(strip,13,13)
Roll_2D =RAO_RAO_2D(strip,11,11)
Heave_2D=RAO_RAO_2D( strip,9, 9)
Load_2D =RAO_RAO_2D(strip,17,17)#zを乗算するところのif分で列番号直すの忘れずに！
Pitch_Roll_2D  =RAO_RAO_2D(strip,13,11)
Roll_Heave_2D  =RAO_RAO_2D(strip,11, 9)
Heave_Pitch_2D =RAO_RAO_2D(strip, 9,13)
Load_Roll_2D   =RAO_RAO_2D(strip,17,11)
Load_Heave_2D  =RAO_RAO_2D(strip,17, 9)
Load_Pitch_2D  =RAO_RAO_2D(strip,17,13)

U=np.sqrt(349.2*9.80655)*0.176

for seastate in range(50000): 
    random.seed()
    """海象作成"""
    Hv_1,theta_1,lambda_1,omega_1,Hv_2,theta_2,lambda_2,omega_2=get_parameter_from_ind()
    wave=Wave_Spectrum()
    wave_values=wave.Directional_wave_spectrum()
    
    """2次元応答スペクトルを作成"""
    S_pitch=Pitch_2D*wave_values
    S_Roll = Roll_2D*wave_values
    S_Heave=Heave_2D*wave_values
    S_Load = Load_2D*wave_values
    S_Pitch_Roll =Pitch_Roll_2D*wave_values
    S_Roll_Heave =Roll_Heave_2D*wave_values
    S_Heave_Pitch=Heave_Pitch_2D*wave_values
    S_Load_Roll  =Load_Roll_2D*wave_values
    S_Load_Heave =Load_Heave_2D*wave_values
    S_Load_Pitch =Load_Pitch_2D*wave_values
    
    for speed in range(1): #6フルード数ごと 0,5,10,15,20,25
        Handling=Data_Handling(U,)
                
        we_matrix= make_we_matrix()

        omega_vertex_list  = 9.80655/(2*U*np.cos(np.arange(0,360,360/n_angle)*np.pi/180)) #頂点のx座標
        
        """データを扱うためのクラス"""
        Pitch=Data_Handling(13,13)
        Roll =Data_Handling(11,11)
        Heave=Data_Handling( 9, 9)
        Load =Data_Handling(17,17)#zを乗算するところのif分で列番号直すの忘れずに！
        Pitch_Roll  =Data_Handling(13,11)
        Roll_Heave  =Data_Handling(11, 9)
        Heave_Pitch =Data_Handling( 9,13)
        Load_Roll   =Data_Handling(17,11)
        Load_Heave  =Data_Handling(17, 9)
        Load_Pitch  =Data_Handling(17,13)
        
        for j,encount_angle in enumerate(np.arange(0,360,360/n_angle)):
            
            
            for w in range(1,len(omegalist)): #25周波数ごと n_lambda
                line = w+n_lambda*j+n_lambda*n_angle*speed          
                list_omega_j,Eomega1, Eomega2,switch = search_omega_i(w)
                """それぞれの応答で変換実行"""         
                Pitch.To_trans_Se(line,w,j,switch,Eomega1)#注意！！複素数ではなく純虚数対応になってるよ
                Roll.To_trans_Se(line,w,j,switch,Eomega1)
                Heave.To_trans_Se(line,w,j,switch,Eomega1)
                Load.To_trans_Se(line,w,j,switch,Eomega1)
                Pitch_Roll.To_trans_Se(line,w,j,switch,Eomega1)
                Roll_Heave.To_trans_Se(line,w,j,switch,Eomega1)
                Heave_Pitch.To_trans_Se(line,w,j,switch,Eomega1)
                Load_Roll.To_trans_Se(line,w,j,switch,Eomega1)
                Load_Heave.To_trans_Se(line,w,j,switch,Eomega1)
                Load_Pitch.To_trans_Se(line,w,j,switch,Eomega1)
        """最終アウトプット"""        
        Pitch.final_keisan()
        Roll.final_keisan()
        Heave.final_keisan()
        Load.final_keisan()
        Pitch_Roll.X_final_keisan()
        Roll_Heave.X_final_keisan()
        Heave_Pitch.X_final_keisan()
        Load_Roll.X_final_keisan()
        Load_Heave.X_final_keisan()
        Load_Pitch.X_final_keisan()
    output(Hv_1,theta_1,lambda_1,omega_1,Hv_2,theta_2,lambda_2,omega_2,f)
    print(seastate+1)
f.close()
print("end")