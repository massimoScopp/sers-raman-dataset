import csv
import glob
import os
import pickle
from builtins import print
from pathlib import Path

import numpy as np
import seglearn
import sklearn
from sklearn import metrics
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, MinMaxScaler, StandardScaler
from generate_window import generate

import pandas as pd


def main():
    pi_dict = dict()

    pickle_dir = "./generated_pickle_files_cytoplasm"
    pickle_files = [pickle_dir]
    size = str(10)
    overlap =str(0.75)
    prima = True
    #legge i dati per tutte chiavi dello specifico pickle
    files = []
    for dataset in pickle_files:
        dataset_files = glob.glob(dataset + '/dataset_raman_segmented-window-size-'+size+'_'+'overlap-'+overlap+'.pickle')
        for x in dataset_files:
            files.append(x)
    # sorted genera una lista ordinata di item partendo dall'oggetto
    files = sorted(files)
    print(files)
    for file in files:
        infile = open(file, 'rb')
        pi_dict = pickle.load(infile)
        for k in pi_dict.keys():
            if prima :
                prima = False
                x_tot =pi_dict[k]['X']
                y_tot = pi_dict[k]['y']
                #x_test = pi_dict[k]['X_test']
                #y_test = pi_dict[k]['y_test']
            else :
                xx_tot = pi_dict[k]['X']
                x_tot = np.vstack([x_tot, xx_tot])
                yy_tot = pi_dict[k]['y']
                y_tot = np.concatenate([y_tot, yy_tot], axis =None)
                #xx_test = pi_dict[k]['X_test']
                #x_test = np.vstack([x_test, xx_test])
                #yy_test = pi_dict[k]['y_test']
                #y_test = np.concatenate([y_test, yy_test], axis = None)

        infile.close()
        x_train, x_test, y_train, y_test = train_test_split(x_tot, y_tot,
                                                            test_size=0.3, shuffle=True, random_state= 36)


    # Create Decision Tree classifer object
    #DecisionTreeClassifier =sklearn.tree
    clf = DecisionTreeClassifier(criterion="entropy", max_depth=100)

    # Train Decision Tree Classifer
    clf = clf.fit(x_train, y_train)

    # Predict the response for test dataset
    y_pred = clf.predict(x_test)

    # Metriche per J48
    print("Accuracy per J48:", metrics.accuracy_score(y_test, y_pred))
    ###Metriche###
    # Training predictions (to demonstrate overfitting)
    train_rf_predictions = clf.predict(x_train)
    train_rf_probs = clf.predict_proba(x_train)[:, 1]

    # Testing predictions (to determine performance)
    rf_predictions = clf.predict(x_test)
    rf_probs = clf.predict_proba(x_test)[:, 1]

    from sklearn.metrics import precision_score, recall_score, roc_auc_score, roc_curve
    import matplotlib.pyplot as plt

    # Plot formatting
    plt.style.use('fivethirtyeight')
    plt.rcParams['font.size'] = 18

    def evaluate_model(predictions, probs, train_predictions, train_probs):
        """Compare machine learning model to baseline performance.
        Computes statistics and shows ROC curve."""

        baseline = {}

        baseline['recall'] = recall_score(y_test,
                                          [1 for _ in range(len(y_test))])
        baseline['precision'] = precision_score(y_test,
                                                [1 for _ in range(len(y_test))])
        baseline['roc'] = 0.5

        results = {}

        results['recall'] = recall_score(y_test, predictions)
        results['precision'] = precision_score(y_test, predictions)
        results['roc'] = roc_auc_score(y_test, probs)

        train_results = {}
        train_results['recall'] = recall_score(y_train, train_predictions)
        train_results['precision'] = precision_score(y_train, train_predictions)
        train_results['roc'] = roc_auc_score(y_train, train_probs)

        for metric in ['recall', 'precision', 'roc']:
            print(
                f'{metric.capitalize()} Baseline: {round(baseline[metric], 2)} Test: {round(results[metric], 2)} Train: {round(train_results[metric], 2)}')

        # Calculate false positive rates and true positive rates
        base_fpr, base_tpr, _ = roc_curve(y_test, [1 for _ in range(len(y_test))])
        model_fpr, model_tpr, _ = roc_curve(y_test, probs)

        plt.figure(figsize=(8, 6))
        plt.rcParams['font.size'] = 16

        # Plot both curves
        plt.plot(base_fpr, base_tpr, 'b', label='baseline')
        plt.plot(model_fpr, model_tpr, 'r', label='model')
        plt.legend()
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curves')
        plt.show()

    evaluate_model(rf_predictions, rf_probs, train_rf_predictions, train_rf_probs)
    plt.savefig('roc_auc_curve.png')
    plt.style.use('fivethirtyeight')
    plt.rcParams['font.size'] = 18
    from sklearn.metrics import confusion_matrix
    import itertools
    def plot_confusion_matrix(cm, classes,
                              normalize=False,
                              title='Confusion matrix',
                              cmap=plt.cm.Oranges):
        """
        This function prints and plots the confusion matrix.
        Normalization can be applied by setting `normalize=True`.
        Source: http://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html
        """
        if normalize:
            cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
            print("Normalized confusion matrix")
        else:
            print('Confusion matrix, without normalization')

        print(cm)

        # Plot the confusion matrix
        plt.figure(figsize=(10, 10))
        plt.imshow(cm, interpolation='nearest', cmap=cmap)
        plt.title(title, size=24)
        plt.colorbar(aspect=4)
        tick_marks = np.arange(len(classes))
        plt.xticks(tick_marks, classes, rotation=45, size=14)
        plt.yticks(tick_marks, classes, size=14)

        fmt = '.2f' if normalize else 'd'
        thresh = cm.max() / 2.

        # Labeling the plot
        for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
            plt.text(j, i, format(cm[i, j], fmt), fontsize=20,
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")

        plt.grid(None)
        plt.tight_layout()
        plt.ylabel('True label', size=18)
        plt.xlabel('Predicted label', size=18)

    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    plot_confusion_matrix(cm, classes=['Poor Health', 'Good Health'],
                          title='Health Confusion Matrix')
    plt.savefig('cm.png')

    #modello random forest feature  = x labels =y
    # Import the model we are using
    from sklearn.ensemble import RandomForestClassifier  # Instantiate model with 1000 decision trees
    rf = RandomForestClassifier( random_state=42)  # Train the model on training data
    rf.fit(x_train, y_train)
    # Use the forest's predict method on the test data
    predictions = rf.predict(x_test)  # Calculate the absolute errors
    # Calculate and display accuracy
    # Calculate mean absolute percentage error (MAPE)
    # Calculate the absolute errors
    print("Accuracy per RandomForest:", metrics.accuracy_score(y_test, predictions))


if __name__ == '__main__':
    main()
