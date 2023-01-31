import numpy as np

#El siguiente bloque es la función de costo vista en la presentación/reporte implementada. 

def CostFunction(W,b,X,Y):
	nu = W.shape[0]
	ni = Y.shape[1]
	aux = 0
	for j in range(nu):
		for i in range(ni):
			if Y[j][i]!=-1:
				aux += (np.dot(W[j],X[i]) + b[j] - Y[j][i])**2
		for num in W[j]:
			aux += (num)**2
	for j in range(ni):
		for num in X[j]:
			aux += (num)**2
	aux *= .5
	return aux

#Estos tres bloques siguientes de código son usados para actualizar las matrices W,X y el vector b para el algoritmo de Gradiente descendente. Se calcula la derivada parcial en cada entrada con la fórmula de aproximación f'(z) = (f(z+h)-f(z-h))/2h, con h = 0.0001. También se usa el coeficiente \alpha = 0.001. en el algoritmo de Gradiente descendente en la fórmula x_{n+1} = x_n - \alpha* Df(x_n)

def ActualW(f,W,b,X,Y):
	for i in range(W.shape[0]):
		for j in range(W.shape[1]):
			wh = W.copy()
			hw = W.copy()
			wh[i][j] = wh[i][j] + 0.0001
			hw[i][j] = hw[i][j] - 0.0001
			W[i][j]  -= 0.001 * (f(wh,b,X,Y)-f(hw,b,X,Y))/ (0.0002)
	return W
	
def Actualb(f,W,b,X,Y):
	for i in range(len(b)):
		bh = b.copy()
		hb = b.copy()
		bh[i] += 0.0001
		hb[i] -= 0.0001
		b[i] -= 0.001 * (f(W,bh,X,Y)-f(W,hb,X,Y))/ (0.0002)
	return b
	

def ActualX(f,W,b,X,Y):
	for i in range(X.shape[0]):
		for j in range(X.shape[1]):
			xh = X.copy()
			hx = X.copy()
			xh[i][j] += 0.0001
			hx[i][j] -= 0.0001
			X[i][j]  -= 0.001 * (f(W,b,xh,Y)-f(W,b,hx,Y))/ (0.0002)
	return X

#El siguiente es el algoritmo de Gradiente descendente. EL primer for todo el algoritmo. Actualizamos los velores de W0, b0, X0. 65 veces, me pareció suficiente.
	
def GradientDescent(f,Y,W0,b0,X0):
	ite = 60
	for i in range(ite):
		W0 = ActualW(CostFunction,W0,b0,X0,Y)
		b0 = Actualb(CostFunction,W0,b0,X0,Y)
		X0 = ActualX(CostFunction,W0,b0,X0,Y)

#La siguiente parte es para escribir los pesos, los trasdalos y las características en documentos separados para un uso posterior que se les pueda dar. Por ejemplo, si se agregan más usuarios o objetos, podríamos usar estos datos cómo datos iniciales.

	with open("Pesos.txt",'w') as docu:
		for i in range(W0.shape[0]):
			a = " ".join([f"{num} " for  num in W0[i]])
			docu.write(f"{a}\n")
	
	with open("Trasdalos.txt",'w') as docu:
		for i in range(len(b)):
			docu.write(f"{b[i]}\n")
	
	with open("Caracteristicas.txt",'w') as docu:
		for i in range(X0.shape[0]):
			a = " ".join([f"{num} " for  num in X0[i]])
			docu.write(f"{a}\n")
	return [W0,b0,X0]

#Esto es lo parte donde pasa todo lo del programa.

print("Escriba el nombre del documento dónde están los datos. ('Prueba.txt' es el que tengo)")
nombre = input()
datos = open(nombre)
lineas = datos.readlines()
Y = []
for linea in lineas:
	linea = linea.split()
	a = [float(num) for num in linea]
	Y.append(a)
Y = np.array(Y)


b = [1.0 for i in range(Y.shape[0])]
W = np.ones((Y.shape[0],10))
X = np.ones((Y.shape[1],10))

PCT = GradientDescent(CostFunction,Y,W,b,X)

#Esto último son las recomendaciones. Para cada usuario, calcula la proyección para cada elemento que no haya evaluado. Si la proyección es mayor o igual que 3.5 (número elegido por mí, completamente aleatorio), entonces se lo recomienda al usuario.

with open("Recomendaciones.txt",'w') as docu:
	for i in range(Y.shape[0]):
		a=[]
		for j in range(Y.shape[1]):
			if Y[i][j]==-1 and np.dot(PCT[0][i],PCT[2][j]) + PCT[1][i] >= 3.5:
				a.append(j)
		docu.write(f"Al usuario {i} se le recomiendan los objetos {a}\n")

#Es el mismo codigo que el anterior, sólo que aquí lo dejo doy las recomendaciones si la predicción es mayor o igual a 3.
		
with open("Recomendaciones2.txt",'w') as docu:
	for i in range(Y.shape[0]):
		a=[]
		for j in range(Y.shape[1]):
			if Y[i][j]==-1 and np.dot(PCT[0][i],PCT[2][j]) + PCT[1][i] >= 3:
				a.append(j)
		docu.write(f"Al usuario {i} se le recomiendan los objetos {a}\n")
