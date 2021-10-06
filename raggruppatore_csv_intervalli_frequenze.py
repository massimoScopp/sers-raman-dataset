import csv
import glob
import os
import pickle
import statistics
from pathlib import Path

import numpy
import seglearn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, MinMaxScaler, StandardScaler
from generate_window import generate

import pandas as pd

# un unico main che si occupa della segmentazione dei dataset in finestre .pickle
# -global parameters---------------------------------------------
#strnome specifica il nome del dataset e del .pickle
strnome ='nucleus'
path_dataset = './dataset_raman_original_csv_'+strnome
csv_groupF_dir = "./dataset_raman_interval_freq_"+strnome





# --------------------------------------------------------------

def create_unsegmented_dataset():
    # modificato per distinzione con citoplasma

    if not os.path.exists(csv_groupF_dir+'.csv'):
        dataset_root_path = 'datasets'
        datasets_csv = [path_dataset]

        files = []
        for dataset in datasets_csv:
            dataset_files = glob.glob(dataset + '/*.csv')
            for x in dataset_files:
                files.append(x)
        # sorted genera una lista ordinata di item partendo dall'oggetto
        files = sorted(files)
        print(files)



        for trip in files:
            print("Processing spectrum: ", trip)
            print("Processing x axis..")
            reader = csv.reader(open(trip, 'r'))
            raman_shift = None
            for line in reader:
                if len(line) > 0 and line[0].startswith('\t--\t'):
                    headerLine = line[0].split("\t")
                    raman_shift = numpy.array([float(x) for x in headerLine[2:]])
                    #modificare per ottenere intervalli di frequenza
                    #newraman_shift = ['', '--']
                    newraman_shift=[]
                    j = 0
                    fnew =603.138
                    step =6.56
                    for j in range(362):
                        if(j % 2 == 0) and j == 0 :
                            newraman_shift.append(str(fnew))
                        if(j % 2 ==0):
                            fnew =fnew + step
                            newraman_shift.append(str(fnew))
                print("intestazione frequenze")
                # legge i dati per tutte chiavi dello specifico pickle
                #import csv
                ins= dict()
                with open(csv_groupF_dir + os.sep + 'dataset_csv_raggruppato_max_' + strnome + '.csv', mode='w',
                          newline='') as csv_file:
                    # colonne = ['Nome', 'tipo ML', 'accuracy']
                    writer = csv.writer(csv_file, quoting=csv.QUOTE_NONE, escapechar=' ')
                    s1 = '\t--\t'
                    #separator =''
                    #s1 =separator.join(newraman_shift)
                    for s in newraman_shift:
                        s1 += str(s)+"\t"
                    riga = [s1[:-1]]

                    writer.writerow(riga)



        dataset_raman = dict()
        nr = -1
        for trip in files:
            print(50 * "_")
            reader = csv.reader(open(trip, 'r'))
            # crea array numpy 2D
            trip_data = numpy.ndarray((0, len(headerLine)))
            dataLine = None
            for line in reader:
                if len(line) > 0 and line[0].startswith('\t--\t'):
                    pass
                elif len(line) > 0:
                    dataLine = line[0].split("\t")
                    cell = dataLine[0]
                    # ogni volta che viene cambiata la cellula viene incrementato nr che corrisponde ad un id unico della prova
                    if cell == '1.1':
                        nr += 1
                    kind = dataLine[1]

                    print("cellula: ", cell)
                    samples = numpy.array([float(x) for x in dataLine[2:]])
                    #modificare per ottenere frequenza max o media per frequenza
                    #import array as newsamples
                    newsamples =[]
                    limite_superiore =1

                    i = 2
                    j = 0
                    arr_appoggio = []
                    while i<=len(samples.data):

                        arr_appoggio = samples[i-2:i]

                        # #per massimo
                        # valore_da_inserire = max(arr_appoggio)
                        #per media
                        valore_da_inserire = statistics.mean(arr_appoggio)
                        newsamples.append(valore_da_inserire)


                        i = i+2

                    print("spectrum:")
                    with open(csv_groupF_dir + os.sep +'dataset_csv_raggruppato_max_'+strnome+ '.csv', mode='a', newline='') as csv_file:
                        # colonne = ['Nome', 'tipo ML', 'accuracy']
                        writer = csv.writer(csv_file, quoting=csv.QUOTE_NONE, delimiter = ' ', escapechar = ' ')
                        s1=''
                        for s in newsamples:
                            s1+=str(s)+'\t'

                        s2 =cell+'\t'+kind+'\t'+s1[:-1]
                        riga =[s2]

                        writer.writerow(riga)






def main():
    if not os.path.exists(csv_groupF_dir):
        Path(csv_groupF_dir).mkdir(parents=False, exist_ok=True)

    dataset_raman = create_unsegmented_dataset()




    

if __name__ == '__main__':
    main()
