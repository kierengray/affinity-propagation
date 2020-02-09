import numpy as np
import pandas as pd
from scipy.spatial.distance import pdist, squareform
import csv
import sys

def main(argv):
	df= pd.read_csv(argv[0], header=0)
	names=df[df.columns[0]].tolist()
	df.drop(df.columns[0], axis=1, inplace=True)
	maxits=	int(argv[1])	#Algorithm terminates when maxits reached unless convits reached before.
	convits=int(argv[2])	#Algorithm terminates if exemplars unchanged of convits iterations.
	dampfact=float(argv[3]) #helps prevent oscilattions
	preference=float(argv[4])

	S= similarity(df, preference) #calculates similarities and returns square matrix
	assignment= ap(S, maxits, convits, dampfact) #runs affinity propagation algorithm
	results(assignment, names, df)

def similarity(df, preference):
		array= np.asmatrix(df.as_matrix())
		n=array.shape[0]
		S=np.zeros((n, n))
		S = squareform(pdist(array, 'euclidean'))
		S= -S
		median=np.median(S)
		preference=median*preference
		#similarity matrix of all points.
		for i in range(0, n): #put preference in diagonal
			S[i,i]= preference
		return S

def ap(S, maxits, convits, dampfact):
		n=S.shape[0]

		#Create empty Availability and Responsibility matrix and Exemplars list
		A=np.zeros((n, n))
		R=np.zeros((n, n))
		exemplars=[]
		count=0

		#start iterations
		for m in range(0, maxits):
		      # Compute responsibilities
			Rold = R
			AS = A + S
			Y= AS.max(1)
			I= AS.argmax(1)
			for i in range(n) :
				AS[i,I[i]] = -1000000
			Y2 = AS.max(1)
			I2 = AS.argmax(1)
			temp=np.repeat(Y, n).reshape(n, n)
			R = S - temp
			for i in range(n) :
				R[i,I[i]] = S[i,I[i]]-Y2[i]
			R = (1-dampfact)*R+dampfact*Rold


			# Compute availabilities
			Aold = A
			Rp = np.maximum(R,0)
			for i in range(n) :
				Rp[i,i] = R[i,i]
			temp2=np.ones((n,1))
			temp3=Rp.sum(0)
			A = np.kron(temp2, temp3)
			A= A-Rp
			diag = np.diag(A)
			A = np.minimum(A,0)
			for i in range(n) :
				A[i,i] = diag[i]
			A = (1-dampfact)*A + dampfact*Aold

			tempexemplars= []
			for i in range(0, n):
				if (R[i,i]+A[i,i])>0:
					tempexemplars.append(i)

			if(tempexemplars==exemplars):
				count=count+1
				if(count==convits):
					break
			else:
				count=0
				exemplars=list(tempexemplars)

		#Assigning datapoints to Exemplar
		assignment= np.zeros(n)

		for i in range(0,n):
			closest=0;
			currentbest=-1000000
			for j in range(0, len(exemplars)):
				if S[i,exemplars[j]]>currentbest:
					currentbest=S[i,exemplars[j]]
					closest=exemplars[j]
				if i==exemplars[j]:
					closest=exemplars[j]
					break
			assignment[i]=closest


		return assignment

def results(assignment, names, df):
	results=zip(names, assignment)
	with open("results.csv", "wb") as f:
		writer = csv.writer(f)
		writer.writerows(results)
	return


main(sys.argv[1:])
