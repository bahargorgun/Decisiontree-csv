import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import plot_tree
from sklearn.metrics import accuracy_score


df = pd.read_csv("heart.csv")


df.columns
y = df['output'].values 
X = df.drop('output', axis =1).values

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size = 0.25, random_state = 0)

tree = DecisionTreeClassifier(criterion='entropy',max_depth= 3,random_state = 0)
tree.fit(X_train, y_train)
plot_tree(tree, feature_names = df.columns ,fontsize = 8)
y_pred_test = tree.predict(X_test)
print("Accuracy of Decision Tree: ", accuracy_score(y_test, y_pred_test))

#gini
tree_gin = DecisionTreeClassifier(criterion = 'gini',max_depth =  4, random_state =  0 )
tree_gin.fit(X_train, y_train)
y_pred_test_gin  = tree_gin.predict(X_test)
gini_score = round(accuracy_score(y_test, y_pred_test_gin) * 100 ,2)

gini_score
print("Gini Accuracy  : ", accuracy_score(y_test, y_pred_test_gin) * 100, "%")

#entropy
tree_ent = DecisionTreeClassifier(criterion = 'entropy', max_depth =4,random_state =  0 )
tree_ent.fit(X_train, y_train)
y_pred_test_ent  = tree_ent.predict(X_test)
entropy_score = round(accuracy_score(y_test, y_pred_test_ent) * 100,2)


print("Entropy Accuracy  : ", accuracy_score(y_test, y_pred_test_ent) * 100, "%\n")

#Random Forest
forest = RandomForestClassifier()
forest.fit(X_train, y_train)
forest_predictions = forest.predict(X_test)
forest_predictions[0:10]
y_test[0:10]
print("Accuracy of Random Forest: {}%".format(forest.score(X_test, y_test) * 100 ), "%\n")
randomf_score = round(forest.score(X_test, y_test) * 100 ,2)
