Affinity Propagation algorithm:

requirements:
python
numpy
scipy
pandas
matplotlib
seaborn
scikit-learn

To run, open a terminal, navigate to the src folder.
enter: python ap.py foo.csv maxits convits dampfact preference

where:
foo.csv is any dataset with the class label/object name as the first variable
maxits is an integer
convits is an integer smaller that maxits
dampfact is any number between 0.5 and 1
preference is any number

This will generate a results.csv file which includes every datapoint label with assigned exemplar.



To run the parameter testing shown in the comparison chapter:
enter python parameter.py
This will output 2 plots which measure a parameter against silhouette coefficient



To run the affinity propagation on iris example:
enter: python apiris.py
This will generate a a apirisresults.csv file which contains the class labels and assignment
It will also output a plot which shows the scatterplot matrix of the clusters with the
kernel density estimate of each variable in the diagonal.



To run the k-means on iris example:
enter: python kmeansiris.py

This will generate a a kmeansirisresults.csv file which contains the class labels and assignment
It will also output a plot which shows the scatterplot matrix of the clusters with the
kernel density estimate of each variable in the diagonal.


For the example used in the cne chapter:
enter: python apcne.py

This will generate a apcneresults.csv file which contains the class labels and assignment.
