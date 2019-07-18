import numpy as np
import pandas as pd
import pydotplus
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import cross_val_score
from IPython.display import Image
from sklearn.externals.six import StringIO
from sklearn.tree import export_graphviz
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier





data = pd.read_csv("/home/yatin/PycharmProjects/practicing/DecisionTrees/Data/titanic.csv")
# print(data.head(n=5))
# print(data.info())


columns_to_drop = ["PassengerId", "Name", "Ticket", "Cabin", "Embarked"]
data_clean = data.drop(columns_to_drop, axis=1)
# print(data_clean.head(n=5))


le = LabelEncoder()
data_clean["Sex"] = le.fit_transform(data_clean["Sex"])
# print(data_clean.head(n=5))


data_clean = data_clean.fillna(data_clean["Age"].mean())
# print(data_clean.head(n=5))
# print(data_clean.loc[1])

input_cols = ['Pclass', "Sex", "Age", "SibSp", "Parch", "Fare"]
output_cols = ["Survived"]

X = data_clean[input_cols]
Y = data_clean[output_cols]

# print(X.shape, Y.shape)
# print(type(X))
# print(X.head(n=5))
# print(Y.head(n=5))



# Define Entropy and Information Gain
def entropy(col):
    counts = np.unique(col, return_counts=True)
    N = float(col.shape[0])

    ent = 0.0

    for ix in counts[1]:
        p = ix / N
        ent += (((-1.0) * p) * np.log2(p))

    return ent



def divide_data(x_data, fkey, fval):
    # Work with Pandas Data Frames
    x_right = pd.DataFrame([], columns=x_data.columns)
    x_left = pd.DataFrame([], columns=x_data.columns)

    for ix in range(x_data.shape[0]):
        val = x_data[fkey].loc[ix]

        if val > fval:
            x_right = x_right.append(x_data.loc[ix])
        else:
            x_left = x_left.append(x_data.loc[ix])

    return x_left, x_right


# x_left,x_right = divide_data(data_clean[:10],'Sex',0.5)
# print(x_left)
# print(x_right)



def information_gain(x_data, fkey, fval):
    left, right = divide_data(x_data, fkey, fval)

    # % of total samples are on left and right
    l = float(left.shape[0]) / x_data.shape[0]
    r = float(right.shape[0]) / x_data.shape[0]

    # All examples come to one side!
    if left.shape[0] == 0 or right.shape[0] == 0:
        return -1000000  # Min Information Gain

    i_gain = entropy(x_data.Survived) - (l * entropy(left.Survived) + r * entropy(right.Survived))
    return i_gain


# Test our function
# for fx in X.columns:
#     print(fx)
#     print(information_gain(data_clean, fx, data_clean[fx].mean()))



class DecisionTree:

    # Constructor
    def __init__(self, depth=0, max_depth=5):
        self.left = None
        self.right = None
        self.fkey = None
        self.fval = None
        self.max_depth = max_depth
        self.depth = depth
        self.target = None

    def train(self, X_train):

        features = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare']
        info_gains = []

        for ix in features:
            i_gain = information_gain(X_train, ix, X_train[ix].mean())
            info_gains.append(i_gain)

        self.fkey = features[np.argmax(info_gains)]
        self.fval = X_train[self.fkey].mean()
        # print("Making Tree Features is", self.fkey)

        # Split Data
        data_left, data_right = divide_data(X_train, self.fkey, self.fval)
        data_left = data_left.reset_index(drop=True)
        data_right = data_right.reset_index(drop=True)

        # Truly a left node
        if data_left.shape[0] == 0 or data_right.shape[0] == 0:
            if X_train.Survived.mean() >= 0.5:
                self.target = "Survive"
            else:
                self.target = "Dead"
            return
        # Stop earyly when depth >=max depth
        if (self.depth >= self.max_depth):
            if X_train.Survived.mean() >= 0.5:
                self.target = "Survive"
            else:
                self.target = "Dead"
            return

        # Recursive Case
        self.left = DecisionTree(depth=self.depth + 1, max_depth=self.max_depth)
        self.left.train(data_left)

        self.right = DecisionTree(depth=self.depth + 1, max_depth=self.max_depth)
        self.right.train(data_right)

        # You can set the target at every node
        if X_train.Survived.mean() >= 0.5:
            self.target = "Survive"
        else:
            self.target = "Dead"
        return

    def predict(self, test):
        if test[self.fkey] > self.fval:
            # go to right
            if self.right is None:
                return self.target
            return self.right.predict(test)
        else:
            if self.left is None:
                return self.target
            return self.left.predict(test)


# Train-Validation-Test Set Split
split = int(0.7*data_clean.shape[0])
train_data = data_clean[:split]
test_data = data_clean[split:]
# print(test_data.head(n=5))
test_data = test_data.reset_index(drop=True)
# print(test_data.head(n=5))
# print(train_data.shape,test_data.shape)


dt = DecisionTree()
dt.train(train_data)
# print(dt.fkey)
# print(dt.fval)
# print(dt.left.fkey)
# print(dt.right.fkey)


y_pred = []
for ix in range(test_data.shape[0]):
    y_pred.append(dt.predict(test_data.loc[ix]))

# print(y_pred)


y_actual = test_data[output_cols]
le = LabelEncoder()
y_pred = le.fit_transform(y_pred)
# print(y_pred)
y_pred = np.array(y_pred).reshape((-1, 1))
# print(y_pred.shape)

acc = np.sum(y_pred == y_actual) / y_pred.shape[0]
print(acc)
# acc = np.sum(np.array(y_pred) == np.array(y_actual)) / y_pred.shape[0]
# print(acc)



# Decision Tree using Sklearn
sk_tree = DecisionTreeClassifier(criterion='gini',max_depth=5)
sk_tree.fit(train_data[input_cols],train_data[output_cols])
sk_tree.predict(test_data[input_cols])
print("Decision Tree using Sklearn 'Gini' ",sk_tree.score(test_data[input_cols], test_data[output_cols]))


X_train = train_data[input_cols]
Y_train = np.array(train_data[output_cols]).reshape((-1,))
X_test = test_data[input_cols]
Y_test = np.array(test_data[output_cols]).reshape((-1,))

sk_tree = DecisionTreeClassifier(criterion='entropy',max_depth=5)
sk_tree.fit(X_train,Y_train)
print("Decision Tree using Sklearn criterion='Entropy' ",sk_tree.score(X_train,Y_train))
print("Decision Tree using Sklearn criterion='Entropy' ",sk_tree.score(X_test,Y_test))


# Visualise a Decison Tree
dot_data = StringIO()
export_graphviz(sk_tree,out_file=dot_data,filled=True,rounded=True)

graph = pydotplus.graph_from_dot_data(dot_data.getvalue())
Image(graph.create_png())



# Random Forests
rf = RandomForestClassifier(n_estimators=10,criterion='entropy',max_depth=5)
rf.fit(X_train,Y_train)
print("Random Forests n_estimators=10 ",rf.score(X_train,Y_train))
print("Random Forests n_estimators=10 ",rf.score(X_test,Y_test))

rf = RandomForestClassifier(n_estimators=22,criterion='entropy',max_depth=5,)
rf.fit(X_train,Y_train)
print("Random Forests n_estimators=22 ",rf.score(X_train,Y_train))
print("Random Forests n_estimators=22 ",rf.score(X_test,Y_test))




acc = cross_val_score(RandomForestClassifier(n_estimators=40,criterion='entropy',max_depth=5,),X_train,Y_train,cv=5).mean()
# print(acc)
acc_list = []
for i in range(1,50):
    acc = cross_val_score(RandomForestClassifier(n_estimators=i,max_depth=5),X_train,Y_train,cv=5).mean()
    acc_list.append(acc)

# print(acc_list)
# print(np.argmax(acc_list))
plt.style.use("seaborn")
plt.plot(acc_list)
plt.show()