import os

import matplotlib.pyplot as plt
import pandas as pd


def main():
    # todo: load the "results.csv" file from the mia-results directory
    # todo: read the data into a list
    # todo: plot the Dice coefficients per label
    #  (i.e. white matter, gray matter, hippocampus, amygdala, thalamus) in a boxplot
    # TEST GIT PUSH+COMMIT
    wdpath= os.getcwd()
    print(wdpath)
    results_file = []
    for dirpath, subdirs, files in os.walk(wdpath):
        for x in files:
            if x.endswith("results.csv"):
                results_file.append(os.path.join(dirpath, x))
    for i in range(0, len(results_file)):
        data = pd.read_csv(results_file[i], ";")
        data.boxplot(by='LABEL', column=['DICE'], grid=False)
        plt.title('Boxplot :%i' % i)
        plt.show()
    # alternative: instead of manually loading/reading the csv file you could also use the pandas package
    # but you will need to install it first ('pip install pandas') and import it to this file ('import pandas as pd')
    pass  # pass is just a placeholder if there is no other code


if __name__ == '__main__':
    main()
