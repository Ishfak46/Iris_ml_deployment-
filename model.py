import joblib
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier

iris=load_iris()

x=iris.data
y=iris.target

rf= RandomForestClassifier().fit(x,y)

joblib.dump(rf,'./D:\IRIS ML API DEPLOYMENT\Iris_ml_deployment-\api/mlmodel.joblib')
