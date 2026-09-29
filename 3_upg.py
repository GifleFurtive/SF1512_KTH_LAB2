"""Uppgift 3 -Labb 2 SF1512: Temperatur i stav – randvärdesproblem"""

import scipy as sp
import numpy as np
import matplotlib.pyplot as plt

#vi kommer att lösa RVP - temperatur  i stav: här är RVP-problemets parametrar:
L=1 #diskretisering pâ intervall 0<x<L
K=2 #konstant i RVP
T_A=2 #Randvärden
T_B=T_A #Randvärden
N=100 #diskretisering med n steg
Q=lambda x: (50*x**3 * np.log(x+1)) #HL i ekv.

H=L/N #stegläng
I=np.array([]) #varje i = 1,...,N-1
for i in range(1,N):
	I=np.append(I,i)
X_I=np.array(H*I) #varje x_i=h*i ; i=1,...,N-1

#3.a
def a_b_c_upg(p=None):
	"""
	löser frâga 3.a: Diskretisera (4) med centraldifferenser för N = 4. Skriv upp systemmatrisen och
	högerledet med alla element
	param: p=[list] med värden pâ n,q,k,t_a,t_b om man vill lösa RVP för andra parametrar
	
	return:
	a = Matris av matrisekvationen
	b= HL för matrisekv
	"""
	if not p==None:
		if not type(p)==list:
			raise TypeError("The parameter for this function should be a list:[n,q,k,t_a,t_b]")
		n,q,k,t_a,t_b=p
		h=L/n #stegläng
		ii=np.array([]) #varje i = 1,...,N-1
		for i in range(1,n):
			ii=np.append(ii,i)
		x_i=np.array(h*ii) #varje x_i=h*i ; i=1,...,N-1
		v1=(k/h**2)*np.ones(n-2) 
		v2=(k/h**2)*(-2)*np.ones(n-1)
		v3 = (k/h**2)*np.ones(n-2)
		a = sp.sparse.diags((v1,v2,v3), offsets=(-1,0,1)) #tridiagonal gles matris A
		a=sp.sparse.csc_matrix(a) #konvertera till csc för att hantera gles matrismultiplikation
		
		b=np.array(-q(x_i)) #HL i ekv
		b[0]=b[0]-(k/h**2)*t_a #rand
		b[-1]=b[-1]-(k/h**2)*t_b #rand
		
		return a,b
		
	v1=(K/H**2)*np.ones(N-2) 
	v2=(K/H**2)*(-2)*np.ones(N-1)
	v3 = (K/H**2)*np.ones(N-2)
	a = sp.sparse.diags((v1,v2,v3), offsets=(-1,0,1)) #tridiagonal gles matris A
	a=sp.sparse.csc_matrix(a) #konvertera till csc för att hantera gles matrismultiplikation
	
	b=np.array(-Q(X_I)) #HL i ekv
	b[0]=b[0]-(K/H**2)*T_A #rand
	b[-1]=b[-1]-(K/H**2)*T_A #rand

	return a,b

def d_upg(a,b,n=N,t_a=T_A,t_b=T_B, pyplot_figur_index=0,labele=""):
	"""
	löser ekvationen Ax=b , samt plottar temperaturen som funktion av X_I, T(0.2) läses av ur plotten
	param:
	a=matris(gles)
	b=HL (array)
	
	return: losning (array) 
	"""
	losning = sp.sparse.linalg.spsolve(a,b)
	
	plt.figure(pyplot_figur_index)
	x_led = np.linspace(0, L, n + 1) #alla X jämt fördelat frân 0 till L  med N+1 punkter
	y_led = np.concatenate(([t_a], losning, [t_b])) #alla temperaturvärden vi  fick
	plt.plot(x_led,y_led,label=f"{labele}temperatur i staven för N={n}")
	plt.xlabel("x[m]")
	plt.ylabel("T(x)[°C]")
	plt.grid(True)
	plt.legend()
	return losning,x_led,y_led
	
def e_upg_konv_studie(vald_x,max_k):
	"""
	gör en konvergensstudie m hjälp av stglängshalvering. 
	Suderar hur approx temp i x=0.7 ändras när N ökar.
	Verifiera att metodens noggrannhetsordning stämmer med teori
	param:
	max_k=[int] konvergensstudiens övre gräns, stegmässigt (n)
	vald_x=int/float, vilket x (0.7) ska vi genomföra konv.studie kring.
	
	return:
	diffs=[lista] av approx felet för varje n 
	p_noggranhet = [lista] av de uppskattade noggranhetsordningar
	"""
	n=50
	maxiter=100
	it=0
	t_07=[] #eh lista
	while n<max_k and it<maxiter:
		t_losning,x_led,y_led=d_upg(*a_b_c_upg([n,Q,K,T_A,T_B]),n,T_A,T_B) #h=1/50
		t_vid_07 = np.interp(vald_x, x_led, y_led)
		print(f"N={n}: T(0.7) ≈ {t_vid_07}")
		t_07.append(t_vid_07)
		n=n*2
		it+=1
	diffs=[] #Konvergensstudie (skillnad mellan N och 2N)
	for i in range(len(t_07)-1):
		d=np.abs(t_07[i+1] - t_07[i])
		diffs.append(d)
	p_total=[]
	for i in range (len(diffs)-1):#Kvot mellan på varandra följande skillnader (ska ->4 om O(h²))
		p_noggranhet=np.log(diffs[i] / diffs[i+1]) / np.log(2)
		print(f"Kvot [{i}] = {p_noggranhet:.4f}")
		p_total.append(p_noggranhet)
	return diffs,p_total
	
def f_upg(vald_x,osak_ta=0.2,osak_tb=0.3):
	"""
	Experimentell störningsräkning, uppskattar osäkärhet i punkt vald_x
	param:
	vald_x=Vart ska vi uppskatta osäkerhet ? (int/float)
	osak_ta=float , osäkerhet i indatan för a
	osak_tb=float , ""--"" b
	
	return:
	osakerhet=värdet pâ osäkerheten
	"""
	t_x,x_led,y_led=d_upg(*a_b_c_upg([200,Q,K,T_A,T_B]),200,T_A,T_B,pyplot_figur_index=2,labele="ostörd") #plt.pyplot figur 2 
	v_0 = np.interp(vald_x, x_led, y_led) #t(07) med ostörda parametervärden
	t_x,x_led,y_led=d_upg(*a_b_c_upg([200,Q,K,2.2,2]),200,2.2,2,pyplot_figur_index=2,labele="t_a störd") #plt.pyplot figur 2 
	v_1 = np.interp(vald_x, x_led, y_led)
	t_x,x_led,y_led=d_upg(*a_b_c_upg([200,Q,K,2,2.3]),200,2,2.3,pyplot_figur_index=2, labele="t_b störd") #plt.pyplot figur 2 
	v_2 = np.interp(vald_x, x_led, y_led)
	osakerhet= np.abs(v_1-v_0)+np.abs(v_2-v_0)
	return osakerhet
	
if __name__=="__main__":
	a,b=a_b_c_upg() #inom hakparanteser: andra parametrar för vârt RVP, N,q,k,Ta,Tb
	print("Uppgift 3.a, 3.b, 3.c:\n",a.toarray(),"\n",b)
	t_x,x_led,y_led=(d_upg(a,b)) #alla temp.värden
	T_vid_07 = np.interp(0.2, x_led, y_led)
	print("==============================\n\n Uppgift 3.d:\n",t_x, f"\nApprox temp vid x=0.2: T ~ {T_vid_07}°C")
	print("==============================\n\n Uppgift 3.e:\n")
	e_upg_konv_studie(0.7,51200)
	print("==============================\n\n Uppgift 3.f:\n",f_upg(0.7))
	plt.show()

