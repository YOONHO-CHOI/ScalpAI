#%% Import modules
import os
import numpy as np
import pandas as pd
import seaborn as sns
import tensorflow as tf
import dataframe_image as dfi
import matplotlib.pyplot as plt

from tqdm import tqdm
from tensorflow.keras import layers
from tensorflow.keras.models import Model
from tensorflow.keras import backend as K
from tensorflow.keras.utils import plot_model, Sequence
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import Input, GlobalAveragePooling2D, Dense
from tensorflow.keras.optimizers import Adam, SGD, RMSprop, Adamax, Nadam
from tensorflow.keras.applications import VGG16, ResNet50, InceptionResNetV2, MobileNet, MobileNetV2, InceptionV3
from sklearn.metrics import confusion_matrix, multilabel_confusion_matrix, ConfusionMatrixDisplay,\
    classification_report, roc_curve, auc, precision_recall_curve, f1_score, accuracy_score, roc_auc_score


#%% Define functions
class DataGenerator(tf.keras.utils.Sequence):
    def __init__(self, df, dir, x_col, y_col, batch_size=32, target_size=(480, 640), shuffle=True, rescale=1./255,
                 zoom_range=0, rotation_range=0, brightness_range=[1,1], horiziontal_flip=False, vertical_flip=False,
                 multi_class=False, parsing=False):
        self.multi_class = multi_class
        self.parsing = parsing
        self.len_df = len(df)
        self.batch_size = batch_size
        self.target_size = target_size
        self.shuffle = shuffle
        self.y_col = y_col
        self.generator = ImageDataGenerator(rescale=rescale,
                                            zoom_range=zoom_range,
                                            rotation_range=rotation_range,
                                            brightness_range=brightness_range,
                                            fill_mode='nearest',
                                            horizontal_flip=horiziontal_flip,
                                            vertical_flip=vertical_flip)

        self.df_generator = self.generator.flow_from_dataframe(dataframe=df,
                                                               directory=dir,
                                                               x_col=x_col,
                                                               y_col=y_col,
                                                               batch_size=batch_size,
                                                               seed=42,
                                                               shuffle=shuffle,
                                                               class_mode="raw",
                                                               target_size=target_size)
        self.on_epoch_end()

    def __len__(self):
        return int(np.floor(self.len_df) / self.batch_size)

    def on_epoch_end(self):
        self.indexes = np.arange(self.len_df)
        if self.shuffle:
            np.random.shuffle(self.indexes)

    def __getitem__(self, index):
        images, labels = self.df_generator.__getitem__(index)
        if self.parsing:
            labels =self.__label_parsing(labels)
        if not self.multi_class:
            labels = (labels>0).astype(float)
        # return input and multi-output
        return images, labels

    def __label_parsing(self, labels):
        parsed_labels = np.split(labels, len(self.y_col), axis=-1)
        return parsed_labels

def calculating_class_weights(lbl_arr):
    from sklearn.utils.class_weight import compute_class_weight
    number_dim = np.shape(lbl_arr)[1]
    weights = np.empty([number_dim, 2])
    for i in range(number_dim):
        weights[i] = compute_class_weight(class_weight = 'balanced', classes =np.unique(lbl_arr), y=lbl_arr[:, i])
    return weights

def get_weighted_loss(weights):
    def loss(y_true, y_pred):
        return K.mean((weights[:,0]**(1-y_true))*(weights[:,1]**(y_true))*K.binary_crossentropy(y_true, y_pred), axis=-1)
    return loss

def makedirs(path):
    # Intended behavior: try to create the directory,
    # pass if the directory exists already, fails otherwise.
    try:
        os.makedirs(path)
    except OSError:
        if not os.path.isdir(path):
            raise

def plot(hist, epochs, savepath, name):
    train_loss = hist['loss']
    val_loss = hist['val_loss']
    # acc = hist['acc']
    # val_acc = hist['val_acc']

    plt.figure()
    epochs_len = np.arange(1, len(train_loss) + 1, 1)
    plt.plot(epochs_len, train_loss, 'b', label='Training Loss')
    plt.plot(epochs_len, val_loss, 'r', label='Validation Loss')
    plt.grid(color='gray', linestyle='--')
    plt.legend()
    plt.title('Loss, Model={}, Epochs={}'.format(name, epochs))
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    # plt.show() # <- When we run this before 'savefig', blank image files are generated.
    plt.savefig('{}/{}_Loss.png'.format(savepath, name), format='png')

    # plt.figure()
    # plt.plot(epochs_len, acc, 'b', label='Training accuracy')
    # plt.plot(epochs_len, val_acc, 'r', label='Validation accuracy')
    # plt.grid(color='gray', linestyle='--')
    # plt.legend()
    # plt.title('Accuracy, Model={}, Epochs={}, Batch={}'.format(fullname, epochnum, batchnum))
    # plt.xlabel('Epochs')
    # plt.ylabel('Accuracy')
    # # plt.show()
    # plt.savefig('{}/{}_Accuracy.png'.format(savepath, fullname), format='png')

def print_auroc(fpr_dict, tpr_dict, auroc_dict, columns_readable, dir=None, fontsize=14):
    fig, ax = plt.subplots(3, 2, figsize=(10, 12))
    for axes, c in zip(ax.flatten(), columns_readable):
        # Plot of a ROC curve for a specific class
        axes.plot(fpr_dict[c], tpr_dict[c], label='Ours: %0.4f' % auroc_dict[c])
        axes.plot([0, 1], [0, 1], 'k--', label='Baseline: %0.4f' % 0.5)
        axes.set_xlim([0.0, 1.05])
        axes.set_ylim([0.0, 1.05])
        axes.set_xlabel('False Positive Rate', fontsize=fontsize)
        axes.set_ylabel('True Positive Rate', fontsize=fontsize)
        axes.set_title('AUROC - {}'.format(c), fontsize=fontsize + 2, fontweight='bold')
        axes.legend(loc="lower right", fontsize=fontsize)
    fig.tight_layout()
    plt.subplots_adjust(wspace=0.2)
    if dir:
        plt.savefig(dir)
    plt.show()

def print_auprc(p_dict, r_dict, auprc_dict, positives, columns_readable, dir=None, fontsize=14):
    fig, ax = plt.subplots(3, 2, figsize=(10, 12))
    for i, (axes, c) in enumerate(zip(ax.flatten(), columns_readable)):
        # Plot of a ROC curve for a specific class
        axes.plot(r_dict[c], p_dict[c], label='Ours: %0.4f' % auprc_dict[c])
        no_skill = positives[i]
        axes.plot([0, 1], [no_skill, no_skill], 'k--', label='Baseline: %0.4f' % no_skill)
        axes.set_xlim([0.0, 1.05])
        axes.set_ylim([0.0, 1.05])
        axes.set_xlabel('Recall', fontsize=fontsize)
        axes.set_ylabel('Precision', fontsize=fontsize)
        axes.set_title('AUPRC - {}'.format(c), fontsize=fontsize + 2, fontweight='bold')
        axes.legend(loc="upper right", fontsize=fontsize)
    fig.tight_layout()
    plt.subplots_adjust(wspace=0.2)
    if dir:
        plt.savefig(dir)
    plt.show()

def print_confusion_matrix(cm, class_label, dir=None, class_names=["N", "Y"], fontsize=14, cmap='PuBu'):
    fig, ax = plt.subplots(3, 2, figsize=(12, 7))
    for axes, cfs_matrix, label in zip(ax.flatten(), cm, class_label):
        sns.set(font_scale=1)
        df_cm = pd.DataFrame(cfs_matrix, index=class_names, columns=class_names)
        try:
            heatmap = sns.heatmap(df_cm, annot=True, fmt="d", cbar=False, ax=axes, cmap=cmap)
        except ValueError:
            raise ValueError("Confusion matrix values must be integers.")
        heatmap.yaxis.set_ticklabels(heatmap.yaxis.get_ticklabels(), rotation=0, ha='right', fontsize=fontsize)
        heatmap.xaxis.set_ticklabels(heatmap.xaxis.get_ticklabels(), rotation=0, ha='right', fontsize=fontsize)
        axes.set_ylabel('True label', fontsize=fontsize)
        axes.set_xlabel('Predicted label', fontsize=fontsize)
        axes.set_title(label, fontsize=fontsize+2, fontweight='bold')
    fig.tight_layout()
    plt.subplots_adjust(wspace=0.2)
    if dir:
        plt.savefig(dir)
    plt.show()


#%% Directories
root = os.getcwd()
data_root = os.path.join(root, 'Data', 'Unified')
train_dir = os.path.join(data_root, 'Training', 'images')
test_dir  = os.path.join(data_root, 'Validation', 'images')
train_lbl = pd.read_csv(os.path.join(data_root, 'Training', 'labels', 'train_labels.csv'))
test_lbl  = pd.read_csv(os.path.join(data_root, 'Validation', 'labels', 'valid_labels.csv'))
save_root = os.path.join(root, 'Result')


#%% Settings
version = 'allinone BCE'
monitor = 'loss'
mode = 'min'
epochs = 20
loss = 'binary_crossentropy'
weight_dir   = os.path.join(save_root, 'weights')
plot_dir  = os.path.join(save_root, 'plots', '_'.join([version, monitor]))
makedirs(weight_dir)
makedirs(plot_dir)

# lbl_arr = (train_lbl.to_numpy()[:,2:] > 0).astype(int)
# class_weight = calculating_class_weights(lbl_arr)
# loss = get_weighted_loss(class_weight)


columns = train_lbl.keys()[2:]
columns_readable = ['Microkeratin', 'Sebaceous', 'Erythema', 'Erythema pustules', 'Dandruff', 'Hair loss...']
# train generator
custom_train_generator = DataGenerator(train_lbl,
                                       train_dir,
                                       "image_file_name",
                                       columns,
                                       batch_size=2,
                                       rescale=1./255,
                                       zoom_range=0.2,
                                       rotation_range=5,
                                       target_size=(120, 160),
                                       brightness_range=[0.8,1.2],
                                       shuffle=True,
                                       vertical_flip=True,
                                       horiziontal_flip=True)
# validation generator
custom_validation_generator = DataGenerator(test_lbl[:1000],
                                            test_dir,
                                            "image_file_name",
                                            columns,
                                            batch_size=1,
                                            rescale = 1./255,
                                            target_size=(120, 160),
                                            shuffle=True,
                                            multi_class=False,
                                            parsing=False)

custom_test_generator = DataGenerator(test_lbl,
                                      test_dir,
                                      "image_file_name",
                                      columns,
                                      batch_size=1,
                                      rescale = 1./255,
                                      target_size=(120, 160),
                                      shuffle=False,
                                      multi_class=False,
                                      parsing=False)

#%% Model construction
input_shape = (120, 160, 3)
inputs = tf.keras.Input(shape=input_shape)
base_model = ResNet50(input_shape=input_shape, weights='imagenet', include_top=False)
GAP = GlobalAveragePooling2D(name='gap')(base_model.output)
outputs = Dense(6, activation='sigmoid')(GAP)
model = Model(inputs=base_model.input, outputs=outputs)
model.summary()


#%% Train model
model_path = os.path.join(weight_dir, version + '_'+monitor+'.hdf5')
checkpoint = tf.keras.callbacks.ModelCheckpoint(model_path, monitor='val_'+monitor, verbose=1,
                                                save_best_only=True, save_weights_only=True, mode=mode, period=1)
model.compile(optimizer=Adam(lr=5e-3), loss=loss, metrics=[tf.keras.metrics.PrecisionAtRecall(recall=0.8)])
history = model.fit(custom_train_generator,
                    validation_data = custom_validation_generator,
                    steps_per_epoch=custom_train_generator.len_df//2,
                    epochs=epochs,
                    callbacks = [checkpoint])
hist = history.history
plot(hist, epochs, plot_dir, version)


model.load_weights(model_path)


#%% Test
true_ = []
prob_ = []
for i in tqdm(range(custom_test_generator.len_df)):
    image, label = custom_test_generator.__getitem__(i)
    prob_.append(model.predict(image))
    true_.append(label)
true = np.squeeze(np.concatenate([true_]))
prob = np.squeeze(np.concatenate([prob_]))
pred = np.round(prob)


#%% Compute result
result = classification_report(true, pred, zero_division=True, output_dict=True, target_names=columns_readable)
result_df = pd.DataFrame(result).transpose()
result_df = result_df.round(decimals=3)
result_df.style
result_df_styled = result_df.style.background_gradient()
dfi.export(result_df_styled,os.path.join(plot_dir, version+"_report.png"))


#%% Confusion matrix
cm = multilabel_confusion_matrix(true, pred)
print_confusion_matrix(cm, columns_readable, os.path.join(plot_dir,version+'_cm.png'),class_names=["N", "Y"], fontsize=14, cmap='PuBu')


#%% AUROC
fpr_dict, tpr_dict, auroc_dict = dict(), dict(), dict()
for i, name in enumerate(columns_readable):
    fpr_dict[name], tpr_dict[name], _ = roc_curve(true[:, i], prob[:, i])
    auroc_dict[name] = auc(fpr_dict[name], tpr_dict[name])
print_auroc(fpr_dict, tpr_dict, auroc_dict, columns_readable, os.path.join(plot_dir,version+'_auroc.png'), fontsize=14)


#%% AUPRC
p_dict, r_dict, auprc_dict, positives = dict(), dict(), dict(), []
for i, name in enumerate(columns_readable):
    positives.append((true[:,i]).sum()/len(true))
    p_dict[name], r_dict[name], _ = precision_recall_curve(true[:, i], prob[:, i])
    auprc_dict[name] = auc(r_dict[name], p_dict[name])
print_auprc(p_dict, r_dict, auprc_dict, positives, columns_readable, os.path.join(plot_dir,version+'_auprc.png'), fontsize=14)
