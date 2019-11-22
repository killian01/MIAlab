import os

import matplotlib.pyplot as plt
import pandas as pd
import glob


def main():
    # todo: load the "results.csv" file from the mia-results directory
    # todo: read the data into a list
    # todo: plot the Dice coefficients per label
    #  (i.e. white matter, gray matter, hippocampus, amygdala, thalamus) in a boxplot

    wdpath_atlas_results = '../bin/DiceTestResult'
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
        plt.show()
    # alternative: instead of manually loading/reading the csv file you could also use the pandas package
    # but you will need to install it first ('pip install pandas') and import it to this file ('import pandas as pd')
    pass  # pass is just a placeholder if there is no other code


if __name__ == '__main__':
    main()
