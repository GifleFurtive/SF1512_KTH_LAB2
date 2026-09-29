"""Uppgift 2 -Labb 2 SF1512: Elektrisk krets – högre ordningens differentialekvation. 
BVP problem"""

import numpy as np
import scipy as sp
import matplotlib.pyplot as plt

I_0=0

def c_upg(tid,f,y0,p):
	"""
	löser systemet av diffekv pâ tidsintervallet tid
	param: 
	tid = tidsintervall (tuple)
	f=diff.ekv. som vektor enligt dy/dt=F(t,y)
	y0=randvbärden vid t=0
	p= tuple med r,l,c
	
	return: 
	losning=(tid, q, i) losningen till systemet av diffekv
	"""
	antal_punkter=np.linspace(*tid,200)
	losning=sp.integrate.solve_ivp(f,tid, y0, args=p,t_eval=antal_punkter) #Runge–Kutta av ordning 5(4), även kallad Dormand–Prince / Solve an initial value problem for a system of ODEs.
	return losning["t"],*losning["y"]

def plotting(i_amper,q_kond,tid_led,figure=0):
	"""plottar"""
	plt.figure(figure)
	plt.plot(tid_led,q_kond,label="q(t)")
	plt.plot(tid_led,i_amper,label="I(t)")
	plt.xlabel("t[s]")
	plt.grid(True)
	plt.legend()
	return

def d_upg(tid,f,y0,p,n):
	"""
	löser ett system av ODE med framât EULER
	param: 
	tid = tidsintervall (tuple)
	f=diff.ekv. som vektor enligt dy/dt=F(t,y)
	y0=randvärden vid t=0 [q,I]
	p= tuple med r,l,c
	n=antal tidssteg (/hur man delar upp tidsintervallet)
	
	return: 
	alla_punkter=x_led (med tid) för att plotta funktion
	q_värden=alla q_värden för att plotta (y_led)
	i_värden= idem för i_värden
	"""
	h=tid/n#steglängd
	t=0 #tiden intitieras
	it=0
	maxiter=200000000 
	i_värden=np.array([y0[1]]) #array med värden pâ I
	q_värden=np.array([y0[0]]) #array som innehâller q
	y=y0 #används i formen för frampât euler soim yj: yj+1 = yj + h * F(tj,yj)
	alla_punkter=np.linspace(0,tid,n+1) #t-ledet när vi kmr plotta (x_led)
	while t<tid and it<maxiter: #framât EULER
		t+=h
		y_ny=y + h * f(t,y,*p) #yj+1
		i_värden=np.append(i_värden,y_ny[1]) #lägg till yj+1
		q_värden=np.append(q_värden,y_ny[0]) #idem
		y=y_ny
		it+=1
	if not it<maxiter:
		raise Exception("Iteration error: it>maxiter")
	return alla_punkter,q_värden,i_värden

def e_upg(F,y_0,p):
	"""
	konvergensstudie för framåt Euler
	
	För fallet med dämpad svängning, utför en konvergensstudie för framåt Euler och
	bestäm noggrannhetsordningen för metoden empiriskt. Beräkna felet komponent-
	vis, se förklaring nedan, vid sluttiden T = 20 och använd lösningen från solve_ivp
	som referenslösning.
	Gör så här: Börja med ett värde på N som leder till en stabil numerisk lösning.
	Dubblera sedan N (halvera tidssteget h) successivt och beräkna felen (ett fel per
	komponent i lösningen) för varje värde på N . Följ stegen i 1.2. c) för att beräkna
	noggrannhetsordningen komponentvis, dvs, för varje komponent separat.
	Blir noggrannhetsordningen som förväntat?
	"""
	r,l,c=p
	losning=(sp.integrate.solve_ivp(F,(0,20), y_0, args=p,rtol=1e-6,atol=1e-6))
	q_80_ivp,i_80_ivp=losning["y"]
	p_q=np.array([])
	p_i=np.array([])
	for i in range(7):
		scrap,q_80,i_80=(d_upg(20,F,y_0,(r,l,c),80*(2**i)))
		diff_q_80=np.abs(q_80[-1]-q_80_ivp[-1])
		diff_i_80=np.abs(i_80[-1]-i_80_ivp[-1])
		scrap,q_2x80,i_2x80=(d_upg(20,F,y_0,(r,l,c),2*(80*(2**i))))
		diff_q_2x80=np.abs(q_2x80[-1]-q_80_ivp[-1])
		diff_i_2x80=np.abs(i_2x80[-1]-i_80_ivp[-1])
		p_q=np.append(p_q,(np.log(diff_q_80/diff_q_2x80)) * (1/np.log(2)))
		p_i=np.append(p_i,(np.log(diff_i_80/diff_i_2x80)) * (1/np.log(2)))
	return p_q,p_i
	
	
if __name__ == "__main__":
	"""
	L * d²q/dt² + R * dq/dt + 1/C * q = 0, q(0)=Q⁰, dq/dt (0) = 0
	ersätt dq/dt med i: dq/dt=i -> y' = F(t, y); y = [q ; i] 
	=> man fâr vektorfunktionen F nedan
	"""
	#3.b
	F=lambda t,y,r,l,c: np.array([y[1],(-(r/l)*y[1] - (1/(l*c))*y[0])]) #upggift a,b
	#print(F(t=0, y=[1, 2], r=0.5, l=2, c=0.5)) OUT:[ 2.  -1.5]
	
	#3.c
	tidsintervall=(0,20)
	q_0=1
	y_0=np.array([q_0,I_0])
	r,l,c=1,2,0.5
	tid,q,i=c_upg(tidsintervall,F,y_0,(r,l,c))
	plotting(i,q,tid)
	r,l,c=0,2,0.5
	tid,q,i=c_upg(tidsintervall,F,y_0,(r,l,c))
	plotting(i,q,tid,figure=1)
	
	#3.d
	r,l,c=1,2,0.5
	tid,q,i=(d_upg(20,F,y_0,(r,l,c),20))
	plotting(i,q,tid,figure=2)
	tid,q,i=(d_upg(20,F,y_0,(r,l,c),40))
	plotting(i,q,tid,figure=3)
	tid,q,i=(d_upg(20,F,y_0,(r,l,c),80))
	plotting(i,q,tid,figure=4)
	tid,q,i=(d_upg(20,F,y_0,(r,l,c),160))
	plotting(i,q,tid,figure=5)
	#n=160 & 80 => lösningen är stabil, n=40 => lösningen ser inte ut att växa eller avta -> instabil ? n=20 instabil.
	
	#3.e
	r,l,c=1,2,0.5
	print(e_upg(F,y_0,(r,l,c)))
	plt.show()
	
	
