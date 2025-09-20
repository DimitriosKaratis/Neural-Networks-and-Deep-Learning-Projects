import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import numpy as np

class Visualization:
    def generate_conf_matrix(label_test, label_pred, title):
        cm = confusion_matrix(label_test, label_pred, labels=np.unique(label_test))
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=np.unique(label_test))
        disp.plot(cmap=plt.cm.Blues)
        plt.title(title)
        plt.show()
