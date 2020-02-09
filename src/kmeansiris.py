import pandas as pd
import csv
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn import preprocessing


def main():
	df= pd.read_csv('iris.csv', header=0)
	names=df[df.columns[0]].tolist()
	df.drop(df.columns[0], axis=1, inplace=True)
	mat=df.as_matrix()
	scaler = preprocessing.MaxAbsScaler()
	mat = scaler.fit_transform(mat)
	km = KMeans(n_clusters=3)
	a= km.fit_predict(mat)
	results(a, names, df)
	plot(df, a)

def results(assignment, names, df):
	results=zip(names, assignment)
	with open("kmeansirisresults.csv", "wb") as f:
		writer = csv.writer(f)
		writer.writerows(results)
	return

def plot(df, assignment):
	se=pd.Series(assignment)
	df['Clusters']=se.values
	sns.pairplot(df, vars=['Petal.Width', 'Petal.Length', 'Sepal.Width', 'Sepal.Length'], hue='Clusters',diag_kind="kde").savefig("output.png")
	sns.plt.show()
	return

main()
