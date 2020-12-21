"""応答関数読み込む"""
a = pd.read_csv("20knot_wide_ver2.csv", encoding="shift_jis")#widly_20_Sym:shift_jis
strip =a.values
print(strip[:,13])


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


class Transformation:
    def __init__(self,Velocity=U,Omegalist=omegalist,N_angle=n_angle,N_lambda=n_lambda,Encounter_omega_list=encount_omega_2):
        #self.Omegalist=Omegalist
        self.N_angle=N_angle
        self.N_lambda=N_lambda
        self.Encounter_omega_list=Encounter_omega_list
        #self.Velocity=Velocity
        self.Dw_matrix=np.abs(1.0-2.0*Velocity/9.80655*np.expand_dims(Omegalist,1)*np.cos(np.arange(0,360,360/N_angle)*np.pi/180.0))
        self.We_matrix=np.abs(np.expand_dims(Omegalist,1)*(1-np.expand_dims(Omegalist,1)*Velocity/9.80655*np.cos(np.arange(0,360,360/N_angle)*np.pi/180)))   
        self.Omega_vertex_list  = 9.80655/(2*Velocity*np.cos(np.arange(0,360,360/N_angle)*np.pi/180)) #頂点のx座標
        self.Trans_matrix=self.TransMatrix()

    def Input_S_matrix(self,S_matrix):
        return S_matrix/self.Dw_matrix

    def TransMatrix(self):
        """間に入る出会い波周波数のリスト作成"""
        we_index_matrix=[]
        for column in range(self.N_angle):
            we_index_list=np.zeros((len(self.Encounter_omega_list),self.N_lambda))
            for i,we in enumerate(self.Encounter_omega_list):
                for j in range(self.N_lambda-1):
                    if (self.Encounter_omega_list[i]-self.We_matrix[j,column])*(self.Encounter_omega_list[i]-self.We_matrix[j+1,column])<=0:
                        we_index_list[i,j]=(self.We_matrix[j+1,column]-we)/(self.We_matrix[j+1,column]-self.We_matrix[j,column])
                        we_index_list[i,j+1]=(we-self.We_matrix[j,column])/(self.We_matrix[j+1,column]-self.We_matrix[j,column])
            we_index_matrix.append(we_index_list)
        return np.array(we_index_matrix)
        
    def Trans_to_Se_list(self,Se_matrix):
        Se_1D=np.zeros(len(self.Encounter_omega_list))
        for column in range(self.N_angle):
            Se_1D=Se_1D+(self.Trans_matrix[column]@Se_matrix[:,column]).flatten()
        return Se_1D

    def final_keisan(self,Se_1D):
        for values in np.abs(Se_1D):
            f.write(str(values)+",")       
        return 
    
    def X_final_keisan(self,Se_1D):
        for values in Se_1D.real:
            f.write(str(values)+",")
        for values in Se_1D.imag:
            f.write(str(values)+",")
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

def output(Hv_1,theta_1,lambda_1,omega_1,Hv_2,theta_2,lambda_2,omega_2,f):
    f.write(str('%03.6f' %(Hv_1))+","+str('%03.6f' %(math.cos(theta_1*math.pi/180)))+","+str('%03.6f' %(math.sin(theta_1*math.pi/180)))+","+str(lambda_1)+","+str('%03.6f' %(omega_1))+",")
    f.write(str('%03.6f' %(Hv_2))+","+str('%03.6f' %(math.cos(theta_2*math.pi/180)))+","+str('%03.6f' %(math.sin(theta_2*math.pi/180)))+","+str(lambda_2)+","+str('%03.6f' %(omega_2)))
    f.write('\n')


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
TransFunction=Transformation()

TransFunction.TransMatrix()

for seastate in range(50000): 
    random.seed()
    """海象作成"""
    Hv_1,theta_1,lambda_1,omega_1,Hv_2,theta_2,lambda_2,omega_2=get_parameter_from_ind()
    wave=Wave_Spectrum()
    wave_values=wave.Directional_wave_spectrum()
    
    """2次元応答スペクトル作成　から　出会い波領域へ変換"""
	Se_Pitch_2D= TransFunction.Input_S_matrix(Pitch_2D*wave_values)
	Se_Roll_2D = TransFunction.Input_S_matrix(Roll_2D*wave_values)
	Se_Heave_2D= TransFunction.Input_S_matrix(Heave_2D*wave_values)
	Se_Load_2D = TransFunction.Input_S_matrix(Load_2D*wave_values)
	Se_Pitch_Roll_2D  = TransFunction.Input_S_matrix(Pitch_Roll_2D*wave_values)
	Se_Roll_Heave_2D  = TransFunction.Input_S_matrix(Roll_Heave_2D*wave_values)
	Se_Heave_Pitch_2D = TransFunction.Input_S_matrix(Heave_Pitch_2D*wave_values)
	Se_Load_Roll_2D   = TransFunction.Input_S_matrix(Load_Roll_2D*wave_values)
	Se_Load_Heave_2D  = TransFunction.Input_S_matrix(Load_Heave_2D*wave_values)
	Se_Load_Pitch_2D  = TransFunction.Input_S_matrix(Load_Pitch_2D*wave_values)
	
	"""出会い波領域の2Dスペクトルを1Dに行列で変換"""
	Pitch_1D  =  TransFunction.Trans_to_Se_list(Se_Pitch_2D)
	Roll_1D  =  TransFunction.Trans_to_Se_list(Se_Roll_2D)
	Heave_1D  =  TransFunction.Trans_to_Se_list(Se_Heave_2D)
	Load_1D  =  TransFunction.Trans_to_Se_list(Se_Load_2D)
	Pitch_Roll_1D  =  TransFunction.Trans_to_Se_list(Se_Pitch_Roll_2D)
	Roll_Heave_1D  =  TransFunction.Trans_to_Se_list(Se_Roll_Heave_2D)
	Heave_Pitch_1D  =  TransFunction.Trans_to_Se_list(Se_Heave_Pitch_2D)
	Load_Roll_1D  =  TransFunction.Trans_to_Se_list(Se_Load_Roll_2D)
	Load_Heave_1D  =  TransFunction.Trans_to_Se_list(Se_Load_Heave_2D)
	Load_Pitch_1D  =  TransFunction.Trans_to_Se_list(Se_Load_Pitch_2D)
       
	"""最終アウトプット"""        
	TransFunction.final_keisan(Pitch_1D )
	TransFunction.final_keisan(Roll_1D)
	TransFunction.final_keisan(Heave_1D )
	TransFunction.final_keisan(Load_1D)
	TransFunction.X_final_keisan(Pitch_Roll_1D)
	TransFunction.X_final_keisan(Roll_Heave_1D )
	TransFunction.X_final_keisan(Heave_Pitch_1D)
	TransFunction.X_final_keisan(Load_Roll_1D)
	TransFunction.X_final_keisan(Load_Heave_1D)
	TransFunction.X_final_keisan(Load_Pitch_1D)
    output(Hv_1,theta_1,lambda_1,omega_1,Hv_2,theta_2,lambda_2,omega_2,f)
    print(seastate+1)
f.close()
print("end")