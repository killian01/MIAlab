"""A medical image analysis pipeline.

The pipeline is used for brain tissue segmentation using a decision forest classifier.
"""
import argparse
import datetime
import os
import random
import sys
import timeit
import warnings
from matplotlib import pyplot as plt
import pymia.evaluation.evaluator as eval_
import pymia.evaluation.metric as metric

import SimpleITK as sitk
import sklearn.ensemble as sk_ensemble
import numpy as np
import pymia.data.conversion as conversion
import pymia.data.loading as load


sys.path.insert(0, os.path.join(os.path.dirname(sys.argv[0]), '..'))  # append the MIALab root directory to Python path
# fixes the ModuleNotFoundError when executing main.py in the console after code changes (e.g. git pull)
# somehow pip install does not keep track of packages

import mialab.data.structure as structure
import mialab.utilities.file_access_utilities as futil
import mialab.utilities.pipeline_utilities as putil

LOADING_KEYS = [structure.BrainImageTypes.T1w,
                structure.BrainImageTypes.T2w,
                structure.BrainImageTypes.GroundTruth,
                structure.BrainImageTypes.BrainMask,
                structure.BrainImageTypes.RegistrationTransform]  # the list of data we will load

def atlas_creation():
    #Load the train labels_native with their transform
    wdpath = 'C:/Users/Admin/PycharmProjects/MyMIALab/data/train'
    results_labels_nii = []
    results_affine = []
    resample_labels = []

    for dirpath, subdirs, files in os.walk(wdpath):
        for x in files:
            if x.endswith("labels_native.nii.gz"):
                results_labels_nii.append(os.path.join(dirpath, x))
            if x.endswith("affine.txt"):
                results_affine.append(os.path.join(dirpath, x))

    #Resample the train labels_native with the transform
    for i in range(0, len(results_affine)):
        transform = sitk.ReadTransform(results_affine[i])
        labels_image = sitk.ReadImage(results_labels_nii[i])
        resample_image = sitk.Resample(labels_image, transform, sitk.sitkNearestNeighbor, 0, labels_image.GetPixelIDValue())
        resample_labels.append(resample_image)
        #without resample
        #resample_labels.append(labels_image)


    # Threshold the images to sort them in 5 categories
    white_matter_list = []
    grey_matter_list = []
    hippocampus_list =[]
    amygdala_list = []
    thalamus_list = []
    for i in range(0, len(resample_labels)):
        white_matter_list.append(sitk.Threshold(resample_labels[i], 1, 1, 0))
        grey_matter_list.append(sitk.Threshold(resample_labels[i], 2, 2, 0))
        hippocampus_list.append(sitk.Threshold(resample_labels[i], 3, 3, 0))
        amygdala_list.append(sitk.Threshold(resample_labels[i], 4, 4, 0))
        thalamus_list.append(sitk.Threshold(resample_labels[i], 5, 5, 0))


    #sum them up and divide by their number of images to make a probability map
    white_matter_map = 0
    grey_matter_map = 0
    hippocampus_map = 0
    amygdala_map = 0
    thalamus_map = 0

    for i in range(1, len(resample_labels)):
        white_matter_map = sitk.Add(white_matter_map, white_matter_list[i])
        grey_matter_map = sitk.Add(grey_matter_map, grey_matter_list[i])
        hippocampus_map = sitk.Add(hippocampus_map, hippocampus_list[i])
        amygdala_map = sitk.Add(amygdala_map, amygdala_list[i])
        thalamus_map = sitk.Add(thalamus_map, thalamus_list[i])

    white_matter_map = sitk.Divide(white_matter_map, len(white_matter_list))
    grey_matter_map = sitk.Divide(grey_matter_map, len(grey_matter_list))
    hippocampus_map = sitk.Divide(hippocampus_map, len(hippocampus_list))
    amygdala_map = sitk.Divide(amygdala_map, len(amygdala_list))
    thalamus_map = sitk.Divide(thalamus_map, len(thalamus_list))
    #atlas = sitk.Divide(sum_images, len(test_resample))
    #slice = sitk.GetArrayFromImage(atlas)[90,:,:]
    #plt.imshow(slice)

    sitk.WriteImage(hippocampus_map, 'C:/Users/Admin/PycharmProjects/MyMIALab/bin/mia-result/hippocampus_map_no_threshold.nii', False)
    sitk.WriteImage(white_matter_map, 'C:/Users/Admin/PycharmProjects/MyMIALab/bin/mia-result/white_matter_map_no_threshold.nii', False)
    sitk.WriteImage(grey_matter_map, 'C:/Users/Admin/PycharmProjects/MyMIALab/bin/mia-result/grey_matter_map_no_threshold.nii', False)
    sitk.WriteImage(amygdala_map, 'C:/Users/Admin/PycharmProjects/MyMIALab/bin/mia-result/amygdala_map_no_threshold.nii', False)
    sitk.WriteImage(thalamus_map, 'C:/Users/Admin/PycharmProjects/MyMIALab/bin/mia-result/thalamus_map_no_threshold.nii', False)

    #Threhold the 5 different maps to get a binary map
    white_matter_map = sitk.BinaryThreshold(white_matter_map, 0, 1, 1, 0)
    grey_matter_map = sitk.BinaryThreshold(grey_matter_map, 0, 2, 2, 0)
    hippocampus_map = sitk.BinaryThreshold(hippocampus_map, 0, 3, 3, 0)
    amygdala_map = sitk.BinaryThreshold(amygdala_map, 0, 4, 4, 0)
    thalamus_map = sitk.BinaryThreshold(thalamus_map, 0, 5, 5, 0)


    #Save the images
    sitk.WriteImage(grey_matter_map, 'C:/Users/Admin/PycharmProjects/MyMIALab/bin/mia-result/grey_matter_map.nii', False)
    sitk.WriteImage(white_matter_map, 'C:/Users/Admin/PycharmProjects/MyMIALab/bin/mia-result/white_matter_map.nii', False)
    sitk.WriteImage(hippocampus_map, 'C:/Users/Admin/PycharmProjects/MyMIALab/bin/mia-result/hippocampus_map.nii', False)
    sitk.WriteImage(amygdala_map, 'C:/Users/Admin/PycharmProjects/MyMIALab/bin/mia-result/amygdala_map.nii', False)
    sitk.WriteImage(thalamus_map, 'C:/Users/Admin/PycharmProjects/MyMIALab/bin/mia-result/thalamus_map.nii', False)


    #Load the test labels_native and their transform
    wdpath_test = 'C:/Users/Admin/PycharmProjects/MyMIALab/data/test'
    test_results_nii = []
    test_results_affine = []
    test_resample = []
    for dirpath, subdirs, files in os.walk(wdpath_test):
        for x in files:
            if x.endswith("labels_native.nii.gz"):
                test_results_nii.append(os.path.join(dirpath, x))
            if x.endswith("affine.txt"):
                test_results_affine.append(os.path.join(dirpath, x))

    #Resample the labels_native with the transform
    for i in range(0, len(test_results_affine)):
        test_transform = sitk.ReadTransform(test_results_affine[i])
        test_image = sitk.ReadImage(test_results_nii[i])
        test_resample_image = sitk.Resample(test_image, test_transform, sitk.sitkNearestNeighbor)
        test_resample.append(test_resample_image)
        #Without resample
        #test_resample.append(test_image)


    #Save the first test patient labels
    sitk.WriteImage(test_resample[0], 'C:/Users/Admin/PycharmProjects/MyMIALab/bin/mia-result/test.nii', False)


    #Compute the dice coeefficent (and the Hausdorff distance)
    label_list = ['White Matter', 'Grey Matter', 'Hippocampus', 'Amygdala', 'Thalamus']
    map_list = [white_matter_map, grey_matter_map, hippocampus_map, amygdala_map, thalamus_map]
    dice_list = []
    for i in range(0, 5):
        evaluator = eval_.Evaluator(eval_.ConsoleEvaluatorWriter(5))
        evaluator.metrics = [metric.DiceCoefficient(), metric.Sensitivity(), metric.Precision(), metric.Fallout()]
        evaluator.add_writer(eval_.CSVEvaluatorWriter(os.path.join('C:/Users/Admin/PycharmProjects/MyMIALab/bin/mia-result', 'Results_' + label_list[i] + '.csv')))
        evaluator.add_label(i+1, label_list[i])
        for j in range(0, len(test_resample)):
            evaluator.evaluate(test_resample[j], map_list[i], 'Patient ' + str(j))


def main(result_dir: str, data_atlas_dir: str, data_train_dir: str, data_test_dir: str):
    """Brain tissue segmentation using decision forests.

    The main routine executes the medical image analysis pipeline:

        - Image loading
        - Registration
        - Pre-processing
        - Feature extraction
        - Decision forest classifier model building
        - Segmentation using the decision forest classifier model on unseen images
        - Post-processing of the segmentation
        - Evaluation of the segmentation
    """
    seed = 42
    random.seed(seed)
    np.random.seed(seed)

    # load atlas images
    putil.load_atlas_images(data_atlas_dir)
    #atlas_creation()
    #putil.load_atlas_custom_images(data_train_dir)

    print('-' * 5, 'Training...')

    # crawl the training image directories
    crawler = load.FileSystemDataCrawler(data_train_dir,
                                         LOADING_KEYS,
                                         futil.BrainImageFilePathGenerator(),
                                         futil.DataDirectoryFilter())
    pre_process_params = {'skullstrip_pre': True,
                          'normalization_pre': True,
                          'registration_pre': True,
                          'coordinates_feature': True,
                          'intensity_feature': True,
                          'gradient_intensity_feature': True}

    # load images for training and pre-process
    images = putil.pre_process_batch(crawler.data, pre_process_params, multi_process=False)

    # generate feature matrix and label vector
    data_train = np.concatenate([img.feature_matrix[0] for img in images])
    labels_train = np.concatenate([img.feature_matrix[1] for img in images]).squeeze()


    # warnings.warn('Random forest parameters not properly set.')
    # we modified the number of decision trees in the forest to be 20 and the maximum tree depth to be 25
    # note, however, that these settings might not be the optimal ones...
    forest = sk_ensemble.RandomForestClassifier(max_features=images[0].feature_matrix[0].shape[1],
                                                n_estimators=5,
                                                max_depth=10)

    start_time = timeit.default_timer()
    forest.fit(data_train, labels_train)
    print(' Time elapsed:', timeit.default_timer() - start_time, 's')

    # create a result directory with timestamp
    t = datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
    result_dir = os.path.join(result_dir, t)
    os.makedirs(result_dir, exist_ok=True)

    print('-' * 5, 'Testing...')

    # initialize evaluator
    evaluator = putil.init_evaluator(result_dir)

    # crawl the training image directories
    crawler = load.FileSystemDataCrawler(data_test_dir,
                                         LOADING_KEYS,
                                         futil.BrainImageFilePathGenerator(),
                                         futil.DataDirectoryFilter())

    # load images for testing and pre-process
    pre_process_params['training'] = False
    images_test = putil.pre_process_batch(crawler.data, pre_process_params, multi_process=False)

    images_prediction = []
    images_probabilities = []

    for img in images_test:
        print('-' * 10, 'Testing', img.id_)

        start_time = timeit.default_timer()
        predictions = forest.predict(img.feature_matrix[0])
        probabilities = forest.predict_proba(img.feature_matrix[0])
        print(' Time elapsed:', timeit.default_timer() - start_time, 's')

        # convert prediction and probabilities back to SimpleITK images
        image_prediction = conversion.NumpySimpleITKImageBridge.convert(predictions.astype(np.uint8),
                                                                        img.image_properties)
        image_probabilities = conversion.NumpySimpleITKImageBridge.convert(probabilities, img.image_properties)

        # evaluate segmentation without post-processing
        evaluator.evaluate(image_prediction, img.images[structure.BrainImageTypes.GroundTruth], img.id_)

        images_prediction.append(image_prediction)
        images_probabilities.append(image_probabilities)

    # post-process segmentation and evaluate with post-processing
    post_process_params = {'simple_post': True}
    images_post_processed = putil.post_process_batch(images_test, images_prediction, images_probabilities,
                                                     post_process_params, multi_process=True)

    for i, img in enumerate(images_test):
        evaluator.evaluate(images_post_processed[i], img.images[structure.BrainImageTypes.GroundTruth],
                           img.id_ + '-PP')

        # save results
        sitk.WriteImage(images_prediction[i], os.path.join(result_dir, images_test[i].id_ + '_SEG.mha'), True)
        sitk.WriteImage(images_post_processed[i], os.path.join(result_dir, images_test[i].id_ + '_SEG-PP.mha'), True)


if __name__ == "__main__":
    """The program's entry point."""

    script_dir = os.path.dirname(sys.argv[0])

    parser = argparse.ArgumentParser(description='Medical image analysis pipeline for brain tissue segmentation')

    parser.add_argument(
        '--result_dir',
        type=str,
        default=os.path.normpath(os.path.join(script_dir, './mia-result')),
        help='Directory for results.'
    )

    parser.add_argument(
        '--data_atlas_dir',
        type=str,
        default=os.path.normpath(os.path.join(script_dir, '../data/atlas')),
        help='Directory with atlas data.'
    )

    parser.add_argument(
        '--data_train_dir',
        type=str,
        default=os.path.normpath(os.path.join(script_dir, '../data/train/')),
        help='Directory with training data.'
    )

    parser.add_argument(
        '--data_test_dir',
        type=str,
        default=os.path.normpath(os.path.join(script_dir, '../data/test/')),
        help='Directory with testing data.'
    )

    args = parser.parse_args()
    main(args.result_dir, args.data_atlas_dir, args.data_train_dir, args.data_test_dir)
