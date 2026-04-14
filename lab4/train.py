from sklearn.model_selection import train_test_split
from data.data_processing import prepare_data
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler

#импорт моделей
from models.decision_tree_model import get_model as get_tree_model
from models.linear_regression_model import  get_model as get_linear_model
from models.svm_kernel_model import get_model as get_svm_kernel_model

X,y = prepare_data()
print(X.shape,y.shape)

X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                    test_size=0.2,
                                                    random_state=42,
                                                    )

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

#дерево
print("-" * 30)
print('decision_tree_model')
tree_model = get_tree_model(X_train_scaled, y_train)
y_pred_tree = tree_model.predict(X_test_scaled)
mae_tree = mean_squared_error(y_test, y_pred_tree)
r2_tree = r2_score(y_test, y_pred_tree)
print("Абсолтная ошибка и R2:",mae_tree,r2_tree)

#линейная регрессия
print("-" * 30)
print('linear_regression_model')
linear_model = get_linear_model(X_train_scaled, y_train)
y_pred_linear = linear_model.predict(X_test_scaled)
mae_linear = mean_squared_error(y_test, y_pred_linear)
r2_linear = r2_score(y_test, y_pred_linear)
print("Абсолтная ошибка и R2:",mae_linear,r2_linear)

#svm
print("-" * 30)
print('svm_kernel_model')
svm_kernel_model = get_svm_kernel_model(X_train_scaled, y_train)
y_pred_svm = svm_kernel_model.predict(X_test_scaled)
mae_svm = mean_squared_error(y_test, y_pred_svm)
r2_svm = r2_score(y_test, y_pred_svm)
print("Абсолтная ошибка и R2:",mae_svm,r2_svm)
