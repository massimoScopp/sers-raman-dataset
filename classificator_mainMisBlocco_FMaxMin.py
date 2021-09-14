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
from  sklearn.linear_model import  LogisticRegression
from sklearn.preprocessing import LabelEncoder, MinMaxScaler, StandardScaler
from generate_window import generate

import pandas as pd


def main():
    pi_dict = dict()
    tipoCell ='nucleus'
    pickle_dir = "./generated_pickle_files_"+tipoCell
    pickle_files = [pickle_dir]
    size = [2, 10, 20, 30]
    overlap = [0.5, 0.75, 0, 1]

    #legge i dati per tutte chiavi dello specifico pickle
    import csv
    with open('misurazioni_'+tipoCell+'.csv', mode='w', newline='') as csv_file:
        #colonne = ['Nome', 'tipo ML', 'accuracy']
        writer = csv.writer(csv_file, quoting=csv.QUOTE_ALL,delimiter=';')

        writer.writerow(['Nome', 'tipo ML', 'accuracy'])

    for s in size:
        for ov in overlap:
            files = []
            prima = True
            for dataset in pickle_files:
                dataset_files = glob.glob(dataset + '/dataset_raman_segmented-window-size-'+str(s)+'_'+'overlap-'+str(ov)+'.pickle')
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
                        ### frequenze aggiunte ad ogni record ([:, 0] permette di prendere solo la prima colonna)
                        f_totA = pi_dict[k]['freq'][:, 0]
                        ### commentato permette di prendere anche l'ultima frequenza del periodo
                        f_totB = pi_dict[k]['freq'][:, s-1]
                        #x_tot = np.append(f_totA, x_tot, axis=0)
                        x_tot =np.insert(x_tot, 0, f_totA, axis=1) #viene inserita la frequenza come prima colonna delle x_tot
                        x_tot = np.insert(x_tot, 1, f_totB, axis=1)  # viene inserita la frequenza come prima colonna delle x_tot

                    else :
                        xx_tot = pi_dict[k]['X']
                        #vengono inserite le frequenze di base all'interno di x_tot
                        f_totA = pi_dict[k]['freq'][:, 0]
                        f_totB = pi_dict[k]['freq'][:, s - 1]
                        #xx_tot = np.append(xx_tot, f_totA, axis=0)
                        xx_tot = np.insert(xx_tot, 0, f_totA, axis=1)
                        xx_tot = np.insert(xx_tot, 1, f_totB, axis=1) #serve per freq Max e min
                        x_tot = np.vstack([x_tot, xx_tot])
                        yy_tot = pi_dict[k]['y']
                        y_tot = np.concatenate([y_tot, yy_tot], axis =None)



                infile.close()
                x_train, x_test, y_train, y_test = train_test_split(x_tot, y_tot,
                                                                    test_size=0.3, shuffle=True, random_state= 36)


            # Spot Check Algorithms
            from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
            from sklearn.neighbors import KNeighborsClassifier
            from sklearn.naive_bayes import GaussianNB
            from sklearn.svm import SVC
            from sklearn.model_selection import StratifiedKFold
            from sklearn.model_selection import cross_val_score
            models = []
            models.append(('LR', LogisticRegression(solver='liblinear', multi_class='ovr')))
            models.append(('LDA', LinearDiscriminantAnalysis()))
            #models.append(('KNN', KNeighborsClassifier()))#prima non funzionava
            models.append(('CART', DecisionTreeClassifier()))
            models.append(('NB', GaussianNB()))
            #models.append(('SVM', SVC(gamma='auto'))) #prima non funzionava
            # evaluate each model in turn
            results = []
            names = []
            for name, model in models:
                kfold = StratifiedKFold(n_splits=10, random_state=1, shuffle=True)
                cv_results = cross_val_score(model, x_train, y_train, cv=kfold, scoring='accuracy')
                results.append(cv_results)
                names.append(name)
                print('%s: %f (%f)' % (name, cv_results.mean(), cv_results.std()))
            #fine prova
            # Create Decision Tree classifer object
            #DecisionTreeClassifier =sklearn.tree
            clf = DecisionTreeClassifier(criterion="entropy", max_depth=5000)

            # Train Decision Tree Classifer
            clf = clf.fit(x_train, y_train)

            # Predict the response for test dataset
            y_pred = clf.predict(x_test)
            score = clf.score(x_test, y_pred)
            # Metriche per J48
            j48score = metrics.accuracy_score(y_test, y_pred)
            print("Accuracy per J$48:", j48score )



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

                #plt.figure(figsize=(8, 6))
                #plt.rcParams['font.size'] = 16

                # Plot both curves
                #plt.plot(base_fpr, base_tpr, 'b', label='baseline')
                #plt.plot(model_fpr, model_tpr, 'r', label='model')
                #plt.legend()
                #plt.xlabel('False Positive Rate')
                #plt.ylabel('True Positive Rate')
                #plt.title('ROC Curves')
                #plt.show()

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
            plot_confusion_matrix(cm, classes=['Tumorale (1)', 'Non Tumorale(0)'],
                                  title='Health Confusion Matrix')
            plt.savefig('cm.png')

            #modello random forest feature  = x labels =y
            # Import the model we are using
            from sklearn.ensemble import RandomForestClassifier  # Instantiate model with 1000 decision trees
            rf = RandomForestClassifier(max_depth=1000 ,n_estimators= 1000)  # Train the model on training data
            rf.fit(x_train, y_train)
            # Use the forest's predict method on the test data
            predictions = rf.predict(x_test)  # Calculate the absolute errors
            # Calculate and display accuracy
            # Calculate mean absolute percentage error (MAPE)
            # Calculate the absolute errors
            rforestscore = metrics.accuracy_score(y_test, predictions)
            print("Accuracy per RandomForest:", rforestscore)
            with open('misurazioni_' + tipoCell + '.csv', 'a', newline='') as csv_file:
                writer = csv.writer(csv_file, delimiter=';')
                nome = tipoCell + '_size_' + str(s) + '_overlap_' + str(ov)
                writer.writerow([nome, 'J48', j48score])
                writer.writerow([nome, 'RandomForest', rforestscore])

if __name__ == '__main__':
    main()
