import os

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns

import glob


def main():
    # todo: load the "results.csv" file from the mia-results directory
    # todo: read the data into a list
    # todo: plot the Dice coefficients per label
    #  (i.e. white matter, gray matter, hippocampus, amygdala, thalamus) in a boxplot

    #Multiboxplot atlas-ml combined features
    data_1 = pd.read_csv('C:/Users/Admin/PycharmProjects/MyMIALab/bin/mia-result/All features, 5 trees, 10 depth/results.csv', ";")
    data_2 = pd.read_csv('C:/Users/Admin/PycharmProjects/MyMIALab/bin/mia-result/Coordinate, intensity, gradient features/results.csv', ";")
    data_3 = pd.read_csv('C:/Users/Admin/PycharmProjects/MyMIALab/bin/mia-result/Coordinate, intensity, gradient, probability map features/results.csv', ";")#trial 2
    data_4 = pd.read_csv('C:/Users/Admin/PycharmProjects/MyMIALab/bin/mia-result/Probability map features/results.csv', ";")#trial 1
    data_5 = pd.read_csv('C:/Users/Admin/PycharmProjects/MyMIALab/bin/mia-result/Gradient_probability_map_features/results.csv', ";")#trial 1

    #Atlas results
    data_atlas =  pd.read_csv('C:/Users/Admin/PycharmProjects/MyMIALab/bin/DiceTestResult/combinedHausdorff.csv', ";")

    cdf = pd.concat([data_1, data_2, data_3, data_4, data_5])
    cdf2 = pd.concat([data_atlas, data_2, data_3])

    #ax = sns.boxplot(x = "LABEL", y = "DICE", hue="TRIAL", data=cdf)
    ax = sns.boxplot(x="LABEL", y="HDRFDST", hue="TRIAL", data=cdf2)
    plt.show()
    plt.clf()
    plt.close()


    '''data_1.boxplot(by='LABEL', column=['DICE'], grid=False)
    data_2.boxplot(by='LABEL', column=['DICE'], grid=False)
    data_3.boxplot(by='LABEL', column=['DICE'], grid=False)
    data_4.boxplot(by='LABEL', column=['DICE'], grid=False)
    plt.title('Boxplot : ML combined with probability map')
    plt.suptitle("")
    plt.show()'''





    '''wdpath_atlas_results = '../bin/DiceTestResult'
    wdpath_ml_results = os.getcwd()

    # Atlas-based results
    os.chdir(wdpath_atlas_results)
    results_file_Atlas = [i for i in glob.glob('*.{}'.format('csv'))]
    combined_csv = pd.concat([pd.read_csv(f) for f in results_file_Atlas])
    combined_csv.to_csv("combinedDice.csv", index=False, encoding='utf-8-sig')

    data = pd.read_csv('combinedDice.csv', ";")
    data.boxplot(by='LABEL', column=['DICE'], grid=False)
    plt.title('Boxplot : Atlas-based')
    plt.suptitle("")
    plt.show()


    # ML-based results
    results_file_RF = []
    for dirpath, subdirs, files in os.walk(wdpath_ml_results):
        for x in files:
            if x.endswith("results.csv"):
                print(os.path.join(dirpath, x))
                results_file_RF.append(os.path.join(dirpath, x))

    for i in range(0, len(results_file_RF)):
        data = pd.read_csv(results_file_RF[i], ";")
        bp_name = str(os.path.basename(os.path.dirname(results_file_RF[i])))
        data.boxplot(by='LABEL', column=['DICE'], grid=False)
        plt.title('Boxplot : %s' % bp_name)
        plt.suptitle("")
        plt.show()'''

    # alternative: instead of manually loading/reading the csv file you could also use the pandas package
    # but you will need to install it first ('pip install pandas') and import it to this file ('import pandas as pd')
    pass  # pass is just a placeholder if there is no other code


if __name__ == '__main__':
    main()
