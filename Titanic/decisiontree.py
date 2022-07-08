from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix, accuracy_score, precision_recall_curve
import dataprocessing

x_train, y_train, x_val, y_val, test = dataprocessing.getDataset()

model = DecisionTreeClassifier(random_state=2)
model.fit(x_train, y_train)

val_pred = model.predict(x_val)
accuracy = accuracy_score(y_val, val_pred)
print("Accuracy: ", accuracy)