import csv
import glob
import os
import pickle
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
strnome ='nucleusB'
path_dataset = './dataset_raman_csv_'+strnome
#path_dataset = './dataset_raman_interval_freqMax_'+strnome
pickle_dir = "./generated_pickle_files_"+strnome

dump_segmented_pickles = True
scale_data = False
testset_ratio = 0.3

segmenters = [
    ('window-size-2_overlap-0', generate(2)),
    ('window-size-10_overlap-0', generate(10)),
    ('window-size-20_overlap-0', generate(20)),
    ('window-size-30_overlap-0', generate(30)),
    ('window-size-2_overlap-0.5', generate(2, .5)),
    ('window-size-10_overlap-0.5', generate(10, .5)),
    ('window-size-20_overlap-0.5', generate(20, .5)),
    ('window-size-30_overlap-0.5', generate(30, .5)),
    ('window-size-2_overlap-0.75', generate(2, .75)),
    ('window-size-10_overlap-0.75', generate(10, .75)),
    ('window-size-20_overlap-0.75', generate(20, .75)),
    ('window-size-30_overlap-0.75', generate(30, .75)),
    ('window-size-2_overlap-1', generate(2, 1)),
    ('window-size-10_overlap-1', generate(10, 1)),
    ('window-size-20_overlap-1', generate(20, 1)),
    ('window-size-30_overlap-1', generate(30, 1)),
]


# --------------------------------------------------------------

def create_unsegmented_dataset():
    # modificato per distinzione con citoplasma

    if not os.path.exists(pickle_dir+'.pickle'):
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

        def hnorm(param):
            return param

        def hnorm_old(param):
            lowering = str(param).lower()
            breaking = lowering.split("(")[0]
            breaking = breaking.strip()
            subst = breaking.replace(" ", "_").replace("%", "percent").replace("/", "").replace("0-100", "0_to_100_")
            finalstrip = subst.strip()
            return finalstrip

        for trip in files:
            print("Processing spectrum: ", trip)
            print("Processing x axis..")
            reader = csv.reader(open(trip, 'r'))
            raman_shift = None
            for line in reader:
                if len(line) > 0 and line[0].startswith('\t--\t'):
                    headerLine = line[0].split("\t")
                    raman_shift = numpy.array([float(x) for x in headerLine[2:]])

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
                    cell = cell + '.' + str(nr)
                    print("cellula: ", cell)
                    samples = numpy.array([float(x) for x in dataLine[2:]])
                    print("spectrum:", trip, cell, kind, " shape=", samples.shape)
                    dataset_raman.update({
                        cell: {
                            'tumoral': kind,
                            'data': samples,
                            'freq': raman_shift
                        }})

            # inserire codice per categorizzazione, in questo caso categorizza su tumorale mettendo 1 =tumorale 0 altrimenti
            # in item viene creata una copia del dict dataset_raman per items
            item = dataset_raman.items()
            # in item2 viene creata una copia di dataset_raman prendendo i suoi valori, utile a costruire l'attributo categorico tumoral
            item2 = dataset_raman.values()
            # viene creato il dataframe df che servirà per attribuire i valori per ogni indice  0 non tumorale 1 tumorale
            df = pd.DataFrame(item)
            df2 = pd.DataFrame(item2)
            df1 = pd.get_dummies(df2['tumoral'], drop_first=True)
            df = pd.concat([df1, df], axis=1)
            df = df.rename(columns={0: 'index'})
            df.set_index(['index'], inplace=True)
            # viene cambiato il nome della colonna Tum_Cyto per poterla gestire in maniera indipendente
            df.columns.values[0] = 'tumoral'
            # da migliorare
            # for d in dataset_raman.keys():
            # print("valori d: ",d)
            for d, row in df.iterrows():
                # print("riga df", row['tumoral'], d)
                # aggiornamento dataset raman, scorre gli elementi key values quando la chiave è uguale all'indice del dataframe creato sostituisce il valore della colonna tumoral
                for key, value in item:
                    if key == d:
                        dataset_raman[key]['tumoral'] = row['tumoral']
        pickle.dump(dataset_raman, open(pickle_dir + os.sep + "dataset_raman.pickle", "wb"))
    else:
        dataset_raman = pickle.load(open(pickle_dir + os.sep + "dataset_raman.pickle", "rb"))
    return dataset_raman


def main():
    if not os.path.exists(pickle_dir):
        Path(pickle_dir).mkdir(parents=False, exist_ok=True)

    dataset_raman = create_unsegmented_dataset()

    if dataset_raman is not None:
        # riduzione della dimensionalità in un range tra 0-10
        scaler = MinMaxScaler(feature_range=[0, 10])
        for k in dataset_raman.keys():
            item = dataset_raman.get(k)
            data = item.get('data')
            if scale_data:
                scaler.fit(data)
                dataset_raman.get(k).update({'data': scaler.transform(data)})


    segmented_dataset = dict()
    for (label_data, segmenter) in segmenters:
        #spectrum = dataset_raman.get(k).get('data')
        for k in dataset_raman.keys():
            spectrum = dataset_raman.get(k).get('data')#originariamente stava fuori
            spectrum = spectrum.reshape(1, -1)
            labels = numpy.array(len(spectrum) * [dataset_raman.get(k).get('tumoral')])
            labels = labels.reshape(1, -1)
            #prova inserimento frequenze
            freq = numpy.array(len(spectrum) * [dataset_raman.get(k).get('freq')])
            #freq = freq.reshape(1, -1)
            segmenter.fit(freq, spectrum)
            segments_and_freq =segmenter.transform(freq, spectrum)
            #fine
            segmenter.fit(spectrum, labels)
            segments_and_lables = segmenter.transform(spectrum, labels)
            print("Windows layout shape:{0},{1},{2}".format(segments_and_lables[0].shape, segments_and_lables[1].shape,segments_and_freq[1].shape))
            #X_train, X_test, y_train, y_test = train_test_split(segments_and_lables[0], segments_and_lables[1], test_size=testset_ratio, shuffle=True)

            segmented_dataset.update(
                {k: {'X': segments_and_lables[0], 'y': segments_and_lables[1], 'freq': segments_and_freq[0],
                     'segmenter_params': label_data,
                     'original_data': dataset_raman,
                     'segments_data': segments_and_lables[0], 'segments_labels': segments_and_lables[1]
                     }})

        picklefile = "dataset_raman_segmented-" + label_data + ".pickle"
        if dump_segmented_pickles:
            print("Dumping ", picklefile)
            pickle.dump(segmented_dataset, open(pickle_dir + os.sep + picklefile, "wb"))
        else:
            print("Skipping dump of ", picklefile)


if __name__ == '__main__':
    main()
