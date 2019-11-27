"""This module contains utility classes and functions."""
import enum
import os
import typing as t
import warnings
from builtins import print

import numpy as np
import pymia.data.conversion as conversion
import pymia.filtering.filter as fltr
import pymia.evaluation.evaluator as eval_
import pymia.evaluation.metric as metric
import SimpleITK as sitk
from numpy.distutils.system_info import gtkp_2_info
from numpy.testing._private.parameterized import param

import mialab.data.structure as structure
import mialab.filtering.feature_extraction as fltr_feat
import mialab.filtering.postprocessing as fltr_postp
import mialab.filtering.preprocessing as fltr_prep
import mialab.utilities.multi_processor as mproc

atlas_t1 = sitk.Image()
atlas_t2 = sitk.Image()


def load_atlas_images(directory: str):
    """Loads the T1 and T2 atlas images.

    Args:
        directory (str): The atlas data directory.
    """

    global atlas_t1
    global atlas_t2
    atlas_t1 = sitk.ReadImage(os.path.join(directory, 'mni_icbm152_t1_tal_nlin_sym_09a_mask.nii.gz'))
    atlas_t2 = sitk.ReadImage(os.path.join(directory, 'mni_icbm152_t2_tal_nlin_sym_09a.nii.gz'))
    if not conversion.ImageProperties(atlas_t1) == conversion.ImageProperties(atlas_t2):
        raise ValueError('T1w and T2w atlas images have not the same image properties')


def load_atlas_custom_images(wdpath):
    # params_list = list(data_batch.items())
    # print(params_list[0] )
    t1w_list = []
    t2w_list = []
    gt_label_list = []
    brain_mask_list = []
    transform_list = []

    #Load the train labels_native with their transform
    for dirpath, subdirs, files in os.walk(wdpath):
        # print("dirpath", dirpath)
        # print("subdirs", subdirs)
        # print("files", files)
        for x in files:
            if x.endswith("T1native.nii.gz"):
                t1w_list.append(sitk.ReadImage(os.path.join(dirpath, x)))
            elif x.endswith("T2native.nii.gz"):
                t2w_list.append(sitk.ReadImage(os.path.join(dirpath, x)))
            elif x.endswith("labels_native.nii.gz"):
                gt_label_list.append(sitk.ReadImage(os.path.join(dirpath, x)))
            elif x.endswith("Brainmasknative.nii.gz"):
                brain_mask_list.append(sitk.ReadImage(os.path.join(dirpath, x)))
            elif x.endswith("affine.txt"):
                transform_list.append(sitk.ReadTransform(os.path.join(dirpath, x)))
            # else:
            #     print("Problem in CustomAtlas in folder", dirpath)


    #Resample and thershold to get the label
    white_matter_list = []
    grey_matter_list = []
    hippocampus_list = []
    amygdala_list = []
    thalamus_list = []
    for i in range(0, len(gt_label_list)):
        resample_img = sitk.Resample(gt_label_list[i],
                                     transform_list[i],
                                     sitk.sitkNearestNeighbor, 0, gt_label_list[i].GetPixelIDValue())
        white_matter_list.append(sitk.Threshold(resample_img, 1, 1, 0))
        grey_matter_list.append(sitk.Threshold(resample_img, 2, 2, 0))
        hippocampus_list.append(sitk.Threshold(resample_img, 3, 3, 0))
        amygdala_list.append(sitk.Threshold(resample_img, 4, 4, 0))
        thalamus_list.append(sitk.Threshold(resample_img, 5, 5, 0))

    # sum them up and divide by their number of images to make a probability map
    white_matter_map = 0
    grey_matter_map = 0
    hippocampus_map = 0
    amygdala_map = 0
    thalamus_map = 0
    for i in range(1, len(gt_label_list)):
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

    #Threhold the 5 different maps to get a binary map
    white_matter_map = sitk.BinaryThreshold(white_matter_map, 0.5, 5, 1, 0)
    grey_matter_map = sitk.BinaryThreshold(grey_matter_map, 1, 5, 2, 0)
    hippocampus_map = sitk.BinaryThreshold(hippocampus_map, 1, 5, 3, 0)
    amygdala_map = sitk.BinaryThreshold(amygdala_map, 1, 5, 4, 0)
    thalamus_map = sitk.BinaryThreshold(thalamus_map, 1, 5, 5, 0)

    #Save the images
    path_to_save = '../bin/custom_atlas_result/'
    if not os.path.exists(path_to_save):
        os.makedirs(path_to_save)
    sitk.WriteImage(grey_matter_map, os.path.join(path_to_save, 'grey_matter_map.nii'), False)
    sitk.WriteImage(white_matter_map,  os.path.join(path_to_save, 'white_matter_map.nii'), False)
    sitk.WriteImage(hippocampus_map,  os.path.join(path_to_save, 'hippocampus_map.nii'), False)
    sitk.WriteImage(amygdala_map,  os.path.join(path_to_save, 'amygdala_map.nii'), False)
    sitk.WriteImage(thalamus_map,  os.path.join(path_to_save, 'thalamus_map.nii'), False)

    # Load the test labels_native and their transform
    path_to_test = '../data/test'
    test_gt_label_list = []
    test_transform_list = []

    for dirpath, subdirs, files in os.walk(path_to_test):
        for x in files:
            if x.endswith("labels_native.nii.gz"):
                test_gt_label_list.append(sitk.ReadImage(os.path.join(dirpath, x)))
            if x.endswith("affine.txt"):
                test_transform_list.append(sitk.ReadTransform(os.path.join(dirpath, x)))

    #Resample the labels_native with the transform
    test_resample_img = []
    for i in range(0, len(test_gt_label_list)):
        resample_img = sitk.Resample(test_gt_label_list[i],
                                     test_transform_list[i],
                                     sitk.sitkNearestNeighbor)
        test_resample_img.append(resample_img)

    # Save the first test patient labels
    # path_to_save = '../bin/temp_test_result/'
    # if not os.path.exists(path_to_save):
    #     os.makedirs(path_to_save)
    # sitk.WriteImage(test_resample_img[0], os.path.join(path_to_save, 'FirstPatienFromTestList.nii'), False)

    #Compute the dice coeefficent (and the Hausdorff distance)
    label_list = ['White Matter', 'Grey Matter', 'Hippocampus', 'Amygdala', 'Thalamus']
    map_list = [white_matter_map, grey_matter_map, hippocampus_map, amygdala_map, thalamus_map]
    dice_list = []

    path_to_save = '../bin/DiceTestResult/'
    if not os.path.exists(path_to_save):
        os.makedirs(path_to_save)
    for i in range(0, 5):
        evaluator = eval_.Evaluator(eval_.ConsoleEvaluatorWriter(5))
        evaluator.metrics = [metric.DiceCoefficient()]
        evaluator.add_writer(eval_.CSVEvaluatorWriter(os.path.join(path_to_save,
                                                                   'DiceResults_' + label_list[i] + '.csv')))
        evaluator.add_label(i+1, label_list[i])
        for j in range(0, len(test_resample_img)):
            evaluator.evaluate(test_resample_img[j], map_list[i], 'Patient ' + str(j))





    print("END Custom loadAtlas")


class FeatureImageTypes(enum.Enum):
    """Represents the feature image types."""

    ATLAS_COORD = 1
    T1w_INTENSITY = 2
    T1w_GRADIENT_INTENSITY = 3
    T2w_INTENSITY = 4
    T2w_GRADIENT_INTENSITY = 5
    Atlas_Grey_matter = 6


class FeatureExtractor:
    """Represents a feature extractor."""

    def __init__(self, img: structure.BrainImage, **kwargs):
        """Initializes a new instance of the FeatureExtractor class.

        Args:
            img (structure.BrainImage): The image to extract features from.
        """
        self.img = img
        self.training = kwargs.get('training', True)
        self.coordinates_feature = kwargs.get('coordinates_feature', False)
        self.intensity_feature = kwargs.get('intensity_feature', False)
        self.gradient_intensity_feature = kwargs.get('gradient_intensity_feature', False)
        #self.atlas_feature_grey_matter = kwarg.get('atlas_feature_grey_matter', true)
        #self.atlas_feature_white_matter = kwarg.get('atlas_feature_white_matter', true)
        #self.atlas_feature_thalamus = kwarg.get('atlas_feature_thalamus', true)
        #self.atlas_feature_amygdala = kwarg.get('atlas_feature_amygdala', true)
        #self.atlas_feature_hippocampus = kwarg.get('atlas_feature_hippocampus', true)

    def execute(self) -> structure.BrainImage:
        """Extracts features from an image.

        Returns:
            structure.BrainImage: The image with extracted features.
        """
        # todo: add T2w features (Add T2w to the "self" below)
        # warnings.warn('No features from T2-weighted image extracted.')

        if self.coordinates_feature:
            atlas_coordinates = fltr_feat.AtlasCoordinates()
            self.img.feature_images[FeatureImageTypes.ATLAS_COORD] = \
                atlas_coordinates.execute(self.img.images[structure.BrainImageTypes.T1w])
            # Don't need for Atlas (t1 and t2 already aligned)
            self.img.feature_images[FeatureImageTypes.ATLAS_COORD] = \
                atlas_coordinates.execute(self.img.images[structure.BrainImageTypes.T2w])

        if self.intensity_feature:
            self.img.feature_images[FeatureImageTypes.T1w_INTENSITY] = self.img.images[structure.BrainImageTypes.T1w]
            self.img.feature_images[FeatureImageTypes.T2w_INTENSITY] = self.img.images[structure.BrainImageTypes.T2w]

        if self.gradient_intensity_feature:
            # compute gradient magnitude images
            self.img.feature_images[FeatureImageTypes.T1w_GRADIENT_INTENSITY] = \
                sitk.GradientMagnitude(self.img.images[structure.BrainImageTypes.T1w])
            self.img.feature_images[FeatureImageTypes.T2w_GRADIENT_INTENSITY] = \
                sitk.GradientMagnitude(self.img.images[structure.BrainImageTypes.T2w])


        self._generate_feature_matrix()

        return self.img

    def _generate_feature_matrix(self):
        """Generates a feature matrix."""

        mask = None
        if self.training:
            # generate a randomized mask where 1 represents voxels used for training
            # the mask needs to be binary, where the value 1 is considered as a voxel which is to be loaded
            # we have following labels:
            # - 0 (background)
            # - 1 (white matter)
            # - 2 (grey matter)
            # - 3 (Hippocampus)
            # - 4 (Amygdala)
            # - 5 (Thalamus)

            # you can exclude background voxels from the training mask generation
            # mask_background = self.img.images[structure.BrainImageTypes.BrainMask]
            # and use background_mask=mask_background in get_mask()

            mask = fltr_feat.RandomizedTrainingMaskGenerator.get_mask(
                self.img.images[structure.BrainImageTypes.GroundTruth],
                [0, 1, 2, 3, 4, 5],
                [0.0003, 0.004, 0.003, 0.04, 0.04, 0.02])

            # convert the mask to a logical array where value 1 is False and value 0 is True
            mask = sitk.GetArrayFromImage(mask)
            mask = np.logical_not(mask)

        # generate features
        data = np.concatenate(
            [self._image_as_numpy_array(image, mask) for id_, image in self.img.feature_images.items()],
            axis=1)

        # generate labels (note that we assume to have a ground truth even for testing)
        labels = self._image_as_numpy_array(self.img.images[structure.BrainImageTypes.GroundTruth], mask)

        self.img.feature_matrix = (data.astype(np.float32), labels.astype(np.int16))

    @staticmethod
    def _image_as_numpy_array(image: sitk.Image, mask: np.ndarray = None):
        """Gets an image as numpy array where each row is a voxel and each column is a feature.

        Args:
            image (sitk.Image): The image.
            mask (np.ndarray): A mask defining which voxels to return. True is background, False is a masked voxel.

        Returns:
            np.ndarray: An array where each row is a voxel and each column is a feature.
        """

        number_of_components = image.GetNumberOfComponentsPerPixel()  # the number of features for this image
        no_voxels = np.prod(image.GetSize())
        image = sitk.GetArrayFromImage(image)

        if mask is not None:
            no_voxels = np.size(mask) - np.count_nonzero(mask)

            if number_of_components == 1:
                masked_image = np.ma.masked_array(image, mask=mask)
            else:
                # image is a vector image, make a vector mask
                vector_mask = np.expand_dims(mask, axis=3)  # shape is now (z, x, y, 1)
                vector_mask = np.repeat(vector_mask, number_of_components,
                                        axis=3)  # shape is now (z, x, y, number_of_components)
                masked_image = np.ma.masked_array(image, mask=vector_mask)

            image = masked_image[~masked_image.mask]

        return image.reshape((no_voxels, number_of_components))


def pre_process(id_: str, paths: dict, **kwargs) -> structure.BrainImage:
    """Loads and processes an image.

    The processing includes:

    - Registration
    - Pre-processing
    - Feature extraction

    Args:
        id_ (str): An image identifier.
        paths (dict): A dict, where the keys are an image identifier of type structure.BrainImageTypes
            and the values are paths to the images.

    Returns:
        (structure.BrainImage):
    """

    print('-' * 10, 'Processing', id_)

    # load image
    path = paths.pop(id_, '')  # the value with key id_ is the root directory of the image
    path_to_transform = paths.pop(structure.BrainImageTypes.RegistrationTransform, '')
    img = {img_key: sitk.ReadImage(path) for img_key, path in paths.items()}
    transform = sitk.ReadTransform(path_to_transform)
    img = structure.BrainImage(id_, path, img, transform)

    # construct pipeline for brain mask registration
    # we need to perform this before the T1w and T2w pipeline because the registered mask is used for skull-stripping
    pipeline_brain_mask = fltr.FilterPipeline()
    if kwargs.get('registration_pre', False):
        pipeline_brain_mask.add_filter(fltr_prep.ImageRegistration())
        pipeline_brain_mask.set_param(fltr_prep.ImageRegistrationParameters(atlas_t1, img.transformation, True),
                                      len(pipeline_brain_mask.filters) - 1)

    # execute pipeline on the brain mask image
    img.images[structure.BrainImageTypes.BrainMask] = pipeline_brain_mask.execute(
        img.images[structure.BrainImageTypes.BrainMask])

    # construct pipeline for T1w image pre-processing
    pipeline_t1 = fltr.FilterPipeline()
    if kwargs.get('registration_pre', False):
        pipeline_t1.add_filter(fltr_prep.ImageRegistration())
        pipeline_t1.set_param(fltr_prep.ImageRegistrationParameters(atlas_t1, img.transformation),
                              len(pipeline_t1.filters) - 1)
    if kwargs.get('skullstrip_pre', False):
        pipeline_t1.add_filter(fltr_prep.SkullStripping())
        pipeline_t1.set_param(fltr_prep.SkullStrippingParameters(img.images[structure.BrainImageTypes.BrainMask]),
                              len(pipeline_t1.filters) - 1)
    if kwargs.get('normalization_pre', False):
        pipeline_t1.add_filter(fltr_prep.ImageNormalization())

    # execute pipeline on the T1w image
    img.images[structure.BrainImageTypes.T1w] = pipeline_t1.execute(img.images[structure.BrainImageTypes.T1w])

    # construct pipeline for T2w image pre-processing
    pipeline_t2 = fltr.FilterPipeline()
    if kwargs.get('registration_pre', False):
        pipeline_t2.add_filter(fltr_prep.ImageRegistration())
        pipeline_t2.set_param(fltr_prep.ImageRegistrationParameters(atlas_t2, img.transformation),
                              len(pipeline_t2.filters) - 1)
    if kwargs.get('skullstrip_pre', False):
        pipeline_t2.add_filter(fltr_prep.SkullStripping())
        pipeline_t2.set_param(fltr_prep.SkullStrippingParameters(img.images[structure.BrainImageTypes.BrainMask]),
                              len(pipeline_t2.filters) - 1)
    if kwargs.get('normalization_pre', False):
        pipeline_t2.add_filter(fltr_prep.ImageNormalization())

    # execute pipeline on the T2w image
    img.images[structure.BrainImageTypes.T2w] = pipeline_t2.execute(img.images[structure.BrainImageTypes.T2w])

    # construct pipeline for ground truth image pre-processing
    pipeline_gt = fltr.FilterPipeline()
    if kwargs.get('registration_pre', False):
        pipeline_gt.add_filter(fltr_prep.ImageRegistration())
        pipeline_gt.set_param(fltr_prep.ImageRegistrationParameters(atlas_t1, img.transformation, True),
                              len(pipeline_gt.filters) - 1)

    # execute pipeline on the ground truth image
    img.images[structure.BrainImageTypes.GroundTruth] = pipeline_gt.execute(
        img.images[structure.BrainImageTypes.GroundTruth])

    # update image properties to atlas image properties after registration
    img.image_properties = conversion.ImageProperties(img.images[structure.BrainImageTypes.T1w])

    # extract the features
    feature_extractor = FeatureExtractor(img, **kwargs)
    img = feature_extractor.execute()

    img.feature_images = {}  # we free up memory because we only need the img.feature_matrix
    # for training of the classifier

    return img


def post_process(img: structure.BrainImage, segmentation: sitk.Image, probability: sitk.Image,
                 **kwargs) -> sitk.Image:
    """Post-processes a segmentation.

    Args:
        img (structure.BrainImage): The image.
        segmentation (sitk.Image): The segmentation (label image).
        probability (sitk.Image): The probabilities images (a vector image).

    Returns:
        sitk.Image: The post-processed image.
    """

    print('-' * 10, 'Post-processing', img.id_)

    # construct pipeline
    pipeline = fltr.FilterPipeline()
    if kwargs.get('simple_post', False):
        pipeline.add_filter(fltr_postp.ImagePostProcessing())
    if kwargs.get('crf_post', False):
        pipeline.add_filter(fltr_postp.DenseCRF())
        pipeline.set_param(fltr_postp.DenseCRFParams(img.images[structure.BrainImageTypes.T1w],
                                                     img.images[structure.BrainImageTypes.T2w],
                                                     probability), len(pipeline.filters) - 1)

    return pipeline.execute(segmentation)


def init_evaluator(directory: str, result_file_name: str = 'results.csv') -> eval_.Evaluator:
    """Initializes an evaluator.

    Args:
        directory (str): The directory for the results file.
        result_file_name (str): The result file name (CSV file).

    Returns:
        eval.Evaluator: An evaluator.
    """
    os.makedirs(directory, exist_ok=True)  # generate result directory, if it does not exists

    evaluator = eval_.Evaluator(eval_.ConsoleEvaluatorWriter(5))
    evaluator.add_writer(eval_.CSVEvaluatorWriter(os.path.join(directory, result_file_name)))
    evaluator.add_label(1, 'WhiteMatter')
    evaluator.add_label(2, 'GreyMatter')
    evaluator.add_label(3, 'Hippocampus')
    evaluator.add_label(4, 'Amygdala')
    evaluator.add_label(5, 'Thalamus')
    evaluator.metrics = [metric.DiceCoefficient(), metric.HausdorffDistance(95)]  # Solutions
    # todo: add hausdorff distance, 95th percentile (see metric.HausdorffDistance)
    # evaluator.add_metric(metric.HausdorffDistance(95))
    # warnings.warn('Initialized evaluation with the Dice coefficient. Do you know other suitable metrics?')
    return evaluator


def pre_process_batch(data_batch: t.Dict[structure.BrainImageTypes, structure.BrainImage],
                      pre_process_params: dict = None, multi_process=True) -> t.List[structure.BrainImage]:
    """Loads and pre-processes a batch of images.

    The pre-processing includes:

    - Registration
    - Pre-processing
    - Feature extraction

    Args:
        data_batch (Dict[structure.BrainImageTypes, structure.BrainImage]): Batch of images to be processed.
        pre_process_params (dict): Pre-processing parameters.
        multi_process (bool): Whether to use the parallel processing on multiple cores or to run sequentially.

    Returns:
        List[structure.BrainImage]: A list of images.
    """
    if pre_process_params is None:
        pre_process_params = {}

    params_list = list(data_batch.items())

    if multi_process:
        images = mproc.MultiProcessor.run(pre_process, params_list, pre_process_params, mproc.PreProcessingPickleHelper)
    else:
        images = [pre_process(id_, path, **pre_process_params) for id_, path in params_list]
    return images


def post_process_batch(brain_images: t.List[structure.BrainImage], segmentations: t.List[sitk.Image],
                       probabilities: t.List[sitk.Image], post_process_params: dict = None,
                       multi_process=True) -> t.List[sitk.Image]:
    """ Post-processes a batch of images.

    Args:
        brain_images (List[structure.BrainImageTypes]): Original images that were used for the prediction.
        segmentations (List[sitk.Image]): The predicted segmentation.
        probabilities (List[sitk.Image]): The prediction probabilities.
        post_process_params (dict): Post-processing parameters.
        multi_process (bool): Whether to use the parallel processing on multiple cores or to run sequentially.

    Returns:
        List[sitk.Image]: List of post-processed images
    """
    if post_process_params is None:
        post_process_params = {}

    param_list = zip(brain_images, segmentations, probabilities)
    if multi_process:
        pp_images = mproc.MultiProcessor.run(post_process, param_list, post_process_params,
                                             mproc.PostProcessingPickleHelper)
    else:
        pp_images = [post_process(img, seg, prob, **post_process_params) for img, seg, prob in param_list]
    return pp_images
