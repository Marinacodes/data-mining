from sklearn import tree
from sklearn.tree import DecisionTreeClassifier
from util2 import Arff2Skl
from sklearn.externals.six import StringIO  
import pydotplus as pydot

import os     

os.environ['PATH'] += os.pathsep + 'C:/Program Files (x86)/Graphviz2.38/bin/'

cvt = Arff2Skl('contact-lenses.arff')
label = cvt.meta.names()[-1]
X, y = cvt.transform(label)

clf = DecisionTreeClassifier(criterion='entropy')
clf.fit(X, y)

print("X[7] is ", X[7])
print("X[8] is ", X[8])

tree.export_graphviz(clf, out_file='tree.dot')
dot_data = StringIO() 
tree.export_graphviz(clf, out_file=dot_data) 
graph = pydot.graph_from_dot_data(dot_data.getvalue()) 
graph.write_pdf('tree.pdf')