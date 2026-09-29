"""Uppgift 1 -Labb 2 SF1512: Skalärt begynnelsevärdesproblem med framåt Euler"""

import numpy as np
import matplotlib.pyplot as plt

#dy/dt=f(t,y) ; f(t,y)=1+t-y
#y(0)=1
#t:[0,T]
#T=1.2
T=1.2
Y_0=1
F=lambda t,y:1+t-y

def a_upg(y0,f,t_max,h=0.1):
	"""
	löser BVP problem med framât EULER
	param-
	f:dy/dt=f(t,y) 
	y0:y(0)
	t_max:övre gräns tidsintervall
	h:steglängd
	
	return-
	t_led=tidsaxel (x_axel)
	y_led=y led, losningar
	"""
	n = int(round(t_max / h))
	t=np.linspace(0,t_max,n+1)
	y = np.zeros(n + 1)
	y[0] = y0
	for i in range(n):
		y[i+1]=y[i]+h*f(t[i],y[i])
	y_led=y
	t_led=t
	return t_led,y_led

def b_upg(y_T,yh_T):
	"""
	Verifiera att din lösning med framåt Euler vid tiden t = T ger felet 0.0188
	param:
	y_T=exakt lösning vid t=T
	yh_T= Euler lösning vid t=T
	
	return:
	diff=felet
	"""
	diff=np.abs(yh_T-y_T)
	return diff

def plotting(figure,x,y,label_text):
	"""plottar"""
	plt.figure(figure)
	plt.plot(x,y,label=label_text)
	plt.plot(x,np.exp(-x) + x,label="e^-t + t")
	plt.xlabel("t[s]")
	plt.grid(True)
	plt.legend()
	return

def konv_studie_12a(n=7):
	"""
	Nu vill vi göra en konvergensstudie för problemet
	param: 
	n=antal gâgner vi halverar h
	
	return: 
	diff=dict, värden pâ felen för y(T) vid steglängd h
	yh_T=dict, värden pâ y(T) enl. framât euler vid steglängd h
	p=dict, noggranhetsordningarna
	"""
	yh_T={}
	diff={}
	exaktlosning=(np.exp(-T) + T)
	for i in range(n):
		h=0.2/(2**i)
		t_led,y_led=a_upg(Y_0,F,T,h)
		yh_T[h]=y_led[-1]
		diff[h]=np.abs(y_led[-1]-exaktlosning)
	alla_steglängder = list(diff.keys())
	p={}
	for h,h1 in zip(alla_steglängder[:-1],alla_steglängder[1:]):
		p[h]=((np.log(diff[h]/diff[h1])) * (1/np.log(2)))
	return diff,yh_T,p
	

if __name__=="__main__":
	t_led,y_led=a_upg(Y_0,F,T)
	plotting(0,t_led,y_led,"Famât euler")
	print("1.1 b\n",b_upg(np.exp(-T) + T,y_led[-1])) #0.0188
	print("-------------------------------------")
	felen,scrap,noggranhetordn=konv_studie_12a(5)
	print("1.2 b&c\n",felen,"\n",noggranhetordn)
	plt.show()
