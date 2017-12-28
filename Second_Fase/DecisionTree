from DataManipulate import get_manipulated_train
from sklearn import tree
def main():
    (X_train,X_test,Y_train,Y_test)=get_manipulated_train()
    dt = tree.DecisionTreeClassifier(max_depth=5)
    score=0
    for i in range(20):
        dt.fit(X_train,Y_train)
        score+=dt.score(X_test,X_test)
    print(score/20)

if __name__ == "__main__":
    main()
