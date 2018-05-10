#SENG474 - Assignment 1
#Devroop Banerjee
#V00837868

#Q1.b)

from util2 import Arff2Skl
from sklearn import tree
import graphviz

cvt = Arff2Skl('contact-lenses.arff')
label = cvt.meta.names()[-1]
X, y = cvt.transform(label)

decisionTree = tree.DecisionTreeClassifier(criterion ='entropy')
decisionTree.fit(X, y)

dotDot = tree.export_graphviz(decisionTree, out_file = None)

graph = graphviz.Source(dotDot)
graph.render("tree")

#Graph generated using http://www.webgraphviz.com/ because render didn't work