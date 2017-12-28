import pandas as pd
import numpy as np
from sklearn import cross_validation
from IPython.display import display # Allows the use of display() for DataFrames
import re as re
from DataManipulate import get_manipulated_train
from sklearn import tree


def get_manipulated_train():
	df = pd.read_csv('../data/train.csv',header=0)
	columns = ['Ticket','Cabin']
	df = df.drop(columns,axis=1)
	redundant = []
	columns = ['Pclass','Sex','Embarked']
	for col in columns:
		redundant.append(pd.get_dummies(df[col]))
	titanic_redundant = pd.concat(redundant, axis=1)
	df = pd.concat((df,titanic_redundant),axis=1)
	df = df.drop(['Pclass','Sex','Embarked'],axis=1)
	#old method for age
	#df['Age'] = df['Age'].interpolate()
	#new method but not efficient for age
	#age_avg = df['Age'].mean()
	#age_std = df['Age'].std()
	#age_null_count = df['Age'].isnull().sum()
	#age_null_random_list = np.random.randint(age_avg - age_std, age_avg + age_std, size=age_null_count)
	#df['Age'][np.isnan(df['Age'])] = age_null_random_list
	#df['Age'] = df['Age'].astype(int)
	#make new column for titels with passed function
	df['Title'] = df['Name'].apply(get_title)
	#new and better than before for fill empty ages
	#iterates through df and fill empty ages according to relation between sex, pclass, title shown in pic. no. 5
	#row 1 means existence of passenger in Pclass 1

	for index, row in df.iterrows():
		if np.isnan(row['Age']):
			if row['male'] == 0 and row[1] == 1:
				if row['Title'] == 'Miss':
					df.set_value(index, 'Age', '30')
				elif row['Title'] == 'Mrs':
					df.set_value(index, 'Age', '45')
				elif row['Title'] == 'Officer':
					rdf.set_value(index, 'Age', '49')
				elif row['Title'] == 'Royalty':
					df.set_value(index, 'Age', '39')
			
			elif row['male'] == 0 and row[2] == 1:
				if row['Title'] == 'Miss':
					df.set_value(index, 'Age', '20')
				elif row['Title'] == 'Mrs':
					rdf.set_value(index, 'Age', '30')

			elif row['male'] == 0 and row[3] == 1:
				if row['Title'] == 'Miss':
					df.set_value(index, 'Age', '18')
				elif row['Title'] == 'Mrs':
					df.set_value(index, 'Age', '31')

			elif row['male'] == 1 and row[1] == 1:
				if row['Title'] == 'Master':
					df.set_value(index, 'Age', '6')
				elif row['Title'] == 'Mr':
					df.set_value(index, 'Age', '41.5')
				elif row['Title'] == 'Officer':
					df.set_value(index, 'Age', '52')
				elif row['Title'] == 'Royalty':
					df.set_value(index, 'Age', '40')

			elif row['male'] == 1 and row[2] == 1:
				if row['Title'] == 'Master':
					df.set_value(index, 'Age', '2')
				elif row['Title'] == 'Mr':
					df.set_value(index, 'Age', '30')
				elif row['Title'] == 'Officer':
					df.set_value(index, 'Age', '41.5')

			elif row['male'] == 1 and row[3] == 1:
				if row['Title'] == 'Master':
					df.set_value(index, 'Age', '6')
				elif row['Title'] == 'Mr':
					df.set_value(index, 'Age', '26')


	df.loc[ df['Age'] <= 16, 'Age'] = 0
	df.loc[(df['Age'] > 16) & (df['Age'] <= 32), 'Age'] = 1
	df.loc[(df['Age'] > 32) & (df['Age'] <= 48), 'Age'] = 2
	df.loc[(df['Age'] > 48) & (df['Age'] <= 64), 'Age'] = 3
	df.loc[ df['Age'] > 64, 'Age'] = 4

	df.loc[ df['Fare'] <= 7.91, 'Fare'] = 0
	df.loc[(df['Fare'] > 7.91) & (df['Fare'] <= 14.454), 'Fare'] = 1
	df.loc[(df['Fare'] > 14.454) & (df['Fare'] <= 31), 'Fare']   = 2
	df.loc[ df['Fare'] > 31, 'Fare'] = 3
	df['Fare'] = df['Fare'].astype(int)
	#Calculate familiSize based on SibSo and Parch and add new column as IsAlone
	df['FamilySize'] = df['SibSp'] + df['Parch'] + 1
	df['IsAlone'] = 0
	df.loc[df['FamilySize'] == 1, 'IsAlone'] = 1
	df = df.drop(['FamilySize'],axis=1)
	#Replace less-used titels with more used titels and map number for each

	df['Title'] = df['Title'].replace(['Lady', 'Countess','Capt', 'Col','Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare')
	df['Title'] = df['Title'].replace('Ms', 'Miss')
	df['Title'] = df['Title'].replace('Mlle', 'Miss')
	df['Title'] = df['Title'].replace('Mme', 'Mrs')
	title_mapping = {"Mr": 1, "Miss": 2, "Mrs": 3, "Master": 4, "Rare": 5}
	df['Title'] = df['Title'].map(title_mapping)
	df['Title'] = df['Title'].fillna(0)
	df = df.drop(['Name', 'SibSp', 'Parch'],axis=1)

	print(df.head())

	Y = df['Survived'].values
	X = df.values
	X = np.delete(X,1,axis=1)

	return  cross_validation.train_test_split(X,Y,test_size=0.3,random_state=0)



def get_title(name):
	title_search = re.search(' ([A-Za-z]+)\.', name)
	# If the title exists, extract and return it.
	if title_search:
		return title_search.group(1)
	return ""


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
