import torch.nn as nn
import torch
from torch.utils.data import DataLoader
from components.dataset import slide_eval_dataset
from torchvision import transforms
import numpy as np
import os
from itertools import cycle
from sklearn.metrics import RocCurveDisplay
import sys
import logging
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import LabelBinarizer
import matplotlib.pyplot as plt
import itertools
from components.builder import build
from tqdm import tqdm
from components.se_sep_transformer import FuSepViT
from torchvision import models
from components.fusion import fusion


lg = logging.getLogger()
lg.setLevel("INFO") # 不想打印log时，把INFO替换成ERROR


def test(model, data_loader):
    total_confidence = None
    total_pred = None
    total_target = None
    for data, target in data_loader:
        data, target = data.to(device), target.to(device)
        s_output = model(data).cpu().detach().numpy()
        local_target = target.cpu().detach().numpy()
        pred = np.argmax(s_output, 1)
        if total_confidence is None:
            total_confidence = s_output
            total_pred = pred
            total_target = local_target
            lg.info("=" * 80)
            lg.info("data.shape:")
            lg.info(data.shape)
            lg.info("target.shape:")
            lg.info(target.shape)
            lg.info("s_output.shape:")
            lg.info(s_output.shape)
            lg.info("pred.shape:")
            lg.info(pred.shape)
            continue
        total_pred = np.concatenate((total_pred, pred))
        total_confidence = np.concatenate((total_confidence, s_output))
        total_target = np.concatenate((total_target, local_target))
    lg.info("=" * 80)
    lg.info("total_confidence.shape:")
    lg.info(total_confidence.shape)
    lg.info("total_pred.shape:")
    lg.info(total_pred.shape)
    return total_pred, total_confidence, total_target


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
def draw_confusion_matrix(cm, classes, normalize=True, title='Confusion matrix', cmap=plt.cm.Blues):
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
    plt.savefig(os.path.join(args.save_path, args.model_name + "_confusion_matrix.png"))


# 绘制ROC曲线
def draw_ROC_curve(target, confidence):
    one_hot_target = LabelBinarizer().fit_transform(target)
    print("total_target.shape:")
    print(total_target.shape)
    print(total_target[:3])
    print("total_confidence.shape:")
    print(total_confidence.shape)
    print(total_confidence[:3])
    print("one_hot_pred.shape:")
    print(one_hot_target.shape)
    print(one_hot_target[:3])
    fig, ax = plt.subplots(figsize=(6, 6))
    colors = cycle(["aqua", "darkorange", "cornflowerblue"])

    for class_id, color in zip(range(3), colors):
        RocCurveDisplay.from_predictions(
            one_hot_target[:, class_id],
            confidence[:, class_id],
            name=f"ROC curve for {idx_to_class[class_id]}",
            color=color,
            ax=ax,
        )

    plt.plot([0, 1], [0, 1], "k--", label="ROC curve for SeSepViT")
    # plt.plot([0, 1], [0, 1], "k--", label="ROC curve for " + args.model_name)
    plt.axis("square")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Extension of Receiver Operating Characteristic\nto One-vs-Rest multiclass")
    plt.legend()
    plt.savefig(os.path.join(args.save_path, args.model_name + "_ROC.png"))

# python .\slide_test.py --model-name resnet18 --dataset E:\patch\validation\ --batch-size 2048 --data-parallel
if __name__ == "__main__":
    model, args = build()

    if args.model_name == "fusion":
        model1 = FuSepViT(                                                               
            num_classes = 1000,
            dim = 32,           
            dim_head = 32,          
            heads = (1, 2, 4, 8),  
            depth = (1, 2, 6, 2),  
            window_size = 7,       
            dropout = 0.1          
        )

        model2 = models.resnet18(weights=None)
        layers = list(model2.children())[:-2]
        model2 = torch.nn.Sequential(*layers)

        model = fusion(model1, model2, 768, 3)


    if args.weights is not None:
        print("Loading weights...")
        checkpoint = torch.load(args.weights, map_location="cpu")

        if list(checkpoint.keys())[0].split(".")[0] == "module":
            for k in list(checkpoint):
                tmp = "."
                checkpoint[tmp.join(k.split(".")[1:])] = checkpoint.pop(k)

        model.load_state_dict(checkpoint)

    if args.data_parallel:
        model = nn.DataParallel(model)
        

    output_file = open(os.path.join(args.save_path, args.model_name + ".txt"), mode="w")
    stdout = sys.stdout
    sys.stdout = output_file

    idx_to_class = {0: 'FTC', 1: 'BN', 2: 'PTC'}
    metrics = ClassifyMetric(3, [0, 1, 2])
    total_confidence = None
    total_pred = None
    total_target = None

    device = torch.device(args.device)
    model = model.to(device)
    model.eval()
    print("Evaluating : " + args.model_name + "\n")
    print(model)
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.CenterCrop(224)
    ])
    labels = []
    prediction = []
    ave = []
    count = []
    patients = os.listdir(args.dataset)

    result_folder_path = None
    confidence_path = None
    pred_path = None
    total_confidence = None
    total_pred = None
    confidence = None
    pred = None
    target = None

    if args.confidence is not None:
        result_folder_path = os.path.join(args.confidence, args.model_name)
    if not os.path.exists(result_folder_path):
        os.mkdir(result_folder_path)

    for patient in tqdm(patients):
        if args.confidence is not None:
            confidence_path = os.path.join(result_folder_path, patient + "_confidence.npy")
            pred_path = os.path.join(result_folder_path, patient + "_pred.npy")
            target_path = os.path.join(result_folder_path, patient + "_target.npy")

        if args.confidence is not None and os.path.exists(confidence_path):
            confidence = np.load(confidence_path)
            pred = np.load(pred_path)
            target = np.load(target_path)

        else:
            cur_path = os.path.join(args.dataset, patient)
            dataset = slide_eval_dataset(cur_path, transform=transform)
            dataloader = DataLoader(dataset, batch_size=int(args.batch_size), shuffle=True, num_workers=20)
            pred, confidence, target = test(model, dataloader)
            if args.confidence is not None:
                np.save(confidence_path, confidence)
                np.save(pred_path, pred)
                np.save(target_path, target)

        labels.append(patient.split('_')[0])
        # res = confidence.max(axis=0) # max aggregation

        idxs = np.argmax(confidence, axis=1)
        sums = [(0, 0), (0, 0), (0, 0)]
        for idx, conf in zip(idxs, confidence):
            sums[idx] = (sums[idx][0] + 1, sums[idx][1] + conf[idx])

        res = np.array([sums[i][1]/sums[i][0] if sums[i][0] > 0 else 0 for i in [0, 1, 2]])

        res = np.argmax(res, axis=0)
        prediction.append(idx_to_class[res])
        metrics.addBatch(target, pred)

        if total_confidence is None:
            total_confidence = confidence
            total_target = target
        else:
            total_target = np.concatenate((total_target, target))
            total_confidence = np.concatenate((total_confidence, confidence))


    draw_ROC_curve(total_target, total_confidence)

    accuracy = metrics.accuracy()
    precision = metrics.precision()
    recall = metrics.recall()
    f1_score = metrics.f1_score()
    print(40 * '*')
    print("Patch level results:")
    print(40 * '*')
    print('labels:')
    print(labels)
    print('prediction:')
    print(prediction)
    print(40 * '*')
    print('accuracy:')
    print(accuracy * 100)
    print('precision:')
    print(precision, end=", Average: ")
    print(np.mean(precision) * 100)
    print('recall:')
    print(recall, end=", Average: ")
    print(np.mean(recall) * 100)
    print('f1_score:')
    print(f1_score, end=", Average: ")
    print(np.mean(f1_score) * 100)
    output_file.flush()

    metrics = ClassifyMetric(3, ["FTC", "BN", "PTC"])
    metrics.addBatch(labels, prediction)
    matrix = metrics.genConfusionMatrix(labels, prediction)
    # draw_confusion_matrix(matrix, ["FTC", "BN", "PTC"], title=args.model_name)

    accuracy = metrics.accuracy() * 100
    precision = metrics.precision() * 100
    recall = metrics.recall() * 100
    f1_score = metrics.f1_score() * 100

    print(40 * '*')
    print("Slide level results:")
    print(40 * '*')
    print(matrix)
    print('accuracy:')
    print(accuracy)
    print('precision:')
    print(precision, end=", Average: ")
    print(np.mean(precision))
    print('recall:')
    print(recall, end=", Average: ")
    print(np.mean(recall))
    print('f1_score:')
    print(f1_score, end=", Average: ")
    print(np.mean(f1_score))

    sys.stdout = stdout 
    output_file.close()
