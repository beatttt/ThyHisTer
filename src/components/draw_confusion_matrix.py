import numpy as np
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import itertools


class ClassifyMetric:
    # 这里的 labels 可以使用 None，这样就会让sklearn自己决定标签
    def __init__(self, numClass, labels=None):
        self.labels = labels
        self.numClass = numClass
        self.confusionMatrix = np.zeros((self.numClass,)*2)

    def genConfusionMatrix(self, y_true, y_pred):
        return confusion_matrix(y_true, y_pred, labels=self.labels)

    def addBatch(self, y_true, y_pred):
        assert np.array(y_true).shape == np.array(y_pred).shape
        self.confusionMatrix += self.genConfusionMatrix(y_true, y_pred)

    def reset(self):
        self.confusionMatrix = np.zeros((self.numClass, self.numClass))

    def accuracy(self):
        accuracy = np.diag(self.confusionMatrix).sum() / \
            self.confusionMatrix.sum()
        return accuracy

    def precision(self):
        precision = np.diag(self.confusionMatrix) / \
            self.confusionMatrix.sum(axis=0)
        return np.nan_to_num(precision)

    def recall(self):
        recall = np.diag(self.confusionMatrix) / \
            self.confusionMatrix.sum(axis=1)
        return recall

    def f1_score(self):
        precision = self.precision()
        recall = self.recall()
        f1_score = 2 * (precision*recall) / (precision+recall)
        return np.nan_to_num(f1_score)


#  绘制混淆矩阵
def plot_confusion_matrix(cm, classes, normalize=True, title='Confusion matrix', cmap=plt.cm.Blues):
    """
    - cm : 计算出的混淆矩阵的值
    - classes : 混淆矩阵中每一行每一列对应的列
    - normalize : True:显示百分比, False:显示个数
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("显示百分比：")
        np.set_printoptions(formatter={'float': '{: 0.2f}'.format})
        print(cm)
    else:
        print('显示具体数字：')
        print(cm)
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)
    #  matplotlib版本问题，如果不加下面这行代码，则绘制的混淆矩阵上下只能显示一半，有的版本的matplotlib不需要下面的代码，分别试一下即可
    plt.ylim(len(classes) - 0.5, -0.5)
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.show()


#  def plot_confusion_matrix(cm, classes,
                          #  title='Confusion matrix',
                          #  cmap=plt.cm.Blues):
    #  """
    #  This function prints and plots the confusion matrix.
    #  cm:混淆矩阵值
    #  classes:分类标签
    #  """
    #  plt.imshow(cm, interpolation='nearest', cmap=cmap)
    #  # plt.title(title, y=-0.2)
    #  plt.title(title)
    #  plt.colorbar()
    #  tick_marks = np.arange(len(classes))
    #  plt.xticks(tick_marks, classes, rotation=0)
    #  plt.yticks(tick_marks, classes)

    #  thresh = cm.max() / 2.
    #  for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        #  plt.text(j, i, cm[i, j],
                 #  horizontalalignment="center",
                 #  color="white" if cm[i, j] > thresh else "black")

    #  plt.tight_layout()
    #  plt.ylabel('True label')
    #  plt.xlabel('Predicted label')

if __name__ == '__main__':
    labels = ['LP', 'LP', 'LP', 'LP', 'LP', 'LP', 'LP', 'LP', 'LP', 'LX', 'LX', 'LX', 'LX', 'LX', 'LX', 'RT', 'RT', 'RT', 'RT', 'RT', 'RT', 'RT', 'RT', 'RT', 'RT', 'RT', 'RT', 'RT', 'RT', 'RT', 'RT', 'RT', 'RT', 'RT']
    prediction = ['LP', 'LP', 'LP', 'LP', 'LP', 'LP', 'LP', 'LP', 'LP', 'LX', 'LP', 'LX', 'LP', 'LP', 'LP', 'LP', 'LX', 'LX', 'LP', 'LX', 'LP', 'RT', 'LP', 'RT', 'LP', 'LX', 'LX', 'LP', 'RT', 'LX', 'RT', 'RT', 'RT', 'RT']
    title = 'max'


    metric = ClassifyMetric(3, ['LP', 'LX', 'RT'])
    matrix = metric.genConfusionMatrix(labels, prediction)
    print(type(matrix))
    print(matrix)
    plot_confusion_matrix(matrix, ['LP', 'LX', 'RT'], title=title)
    plt.show()
