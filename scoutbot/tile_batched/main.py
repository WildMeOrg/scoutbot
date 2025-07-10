import logging
import os
import time
from typing import List, Optional, Union, Dict

import numpy as np

from sahi.prediction import ObjectPrediction
from sahi.postprocess.combine import (
    GreedyNMMPostprocess,
    LSNMSPostprocess,
    NMMPostprocess,
    NMSPostprocess,
    PostprocessPredictions,
)
from sahi.models.yolov8 import Yolov8DetectionModel as Yolov8DetectionModelBase
from sahi.models.base import DetectionModel
from sahi.utils.cv import read_image_as_pil
from .herdnet_model import HerdNet # NOQA 
import torch
import torchvision.transforms as T
from tqdm import tqdm

POSTPROCESS_NAME_TO_CLASS = {
    "GREEDYNMM": GreedyNMMPostprocess,
    "NMM": NMMPostprocess,
    "NMS": NMSPostprocess,
    "LSNMS": LSNMSPostprocess,
}

DETECTOR_BATCH_SIZE = int(os.getenv('DETECTOR_BATCH_SIZE', 128))


class Yolov8DetectionModel(Yolov8DetectionModelBase):

    def __init__(self, *args, batch_size: Optional[int] = None, **kwargs):
        """
        Initializes the Yolov8DetectionModel with an optional batch size and additional arguments.

        Args:
            *args: Positional arguments to be passed to the base class.
            batch_size (Optional[int]): Batch size for inference.
            **kwargs: Additional keyword arguments to be passed to the base class.
        """
        self.batch_size = batch_size or DETECTOR_BATCH_SIZE
        super().__init__(*args, **kwargs)

    def perform_inference(self, images: List[np.ndarray], batch_size: Optional[int] = None):
        """
        Perform inference using the model and store predictions.

        Args:
            images (List[np.ndarray]): List of images as numpy arrays for prediction.
            batch_size (Optional[int]): Batch size for inference, overrides class-level batch size if provided.
        """
        batch_size = batch_size or self.batch_size

        # Confirm model is loaded
        if self.model is None:
            raise ValueError("Model is not loaded, load it by calling .load_model()")

        all_preds = []
        for i in range(0, len(images), batch_size):
            batch_images = images[i:i + batch_size]
            preds = self.model.predict(
                source=batch_images,
                verbose=False,
                device=self.device
            )
            all_preds.extend(preds)

        prediction_result = [
            result.boxes.data[result.boxes.data[:, 4] >= self.confidence_threshold]
            for result in all_preds
        ]

        self._original_predictions = prediction_result


class HerdNetDetectionModel(DetectionModel):
    
    def __init__(  
        self,  
        model_path: Optional[str] = None,  
        device: Optional[str] = None,  
        confidence_threshold: float = 0.2,  
        category_mapping: Optional[Dict] = None,  
        category_remapping: Optional[Dict] = None,  
        load_at_init: bool = True,  
        image_size: int = None,
        batch_size: Optional[int] = None,  
        dataset: str = 'general',  
    ):
        self.dataset = dataset
        self.batch_size = batch_size or DETECTOR_BATCH_SIZE
        super().__init__(  
            model_path=model_path,  
            device=device,  
            confidence_threshold=confidence_threshold,  
            category_mapping=category_mapping,  
            category_remapping=category_remapping,  
            load_at_init=load_at_init,  
            image_size=image_size,  
        )  
          
    def load_model(self):
        if self.model_path:
            weights = os.path.join(torch.hub.get_dir(), "checkpoints", self.model_path)
            checkpoint = torch.load(weights, map_location=torch.device(self.device))        
            self.CLASS_NAMES = checkpoint["classes"]
            self.num_classes = len(self.CLASS_NAMES) + 1
            self.img_mean = checkpoint['mean']
            self.img_std = checkpoint['std']
            self.transforms = T.Compose([
                T.ToTensor(),
                T.Normalize(mean=self.img_mean, std=self.img_std)
            ])
            self.model = HerdNet(num_classes=self.num_classes, pretrained=False)
            state_dict = checkpoint['model_state_dict']  
            new_state_dict = {k.replace('model.', ''): v for k, v in state_dict.items() if k.startswith('model.')}
            self.model.load_state_dict(new_state_dict, strict=True)
        
        else:
            self.model = HerdNet(pretrained=False)
            self.transforms = T.Compose([
                T.ToTensor()            
            ])
  
    def perform_inference(self, images: List[np.ndarray], batch_size: Optional[int] = None):
        """
        Perform inference using the model and store predictions.

        Args:
            images (List[np.ndarray]): List of images as numpy arrays for prediction.
            batch_size (Optional[int]): Batch size for inference (kept for interface compatibility, 
                                      but HerdNet always uses batch_size=1).
        
        Note:
            HerdNet always processes images with batch_size=1 regardless of the parameter value.
        """
        # HerdNet always uses batch_size=1, ignoring the parameter for consistency with the model's requirements
        all_preds = []
        counts, locs, labels, scores, dscores = self.model.batch_image_detection(images, 
                                                    self.transforms, 
                                                    batch_size=1, 
                                                    device = self.device)
        all_preds = []
        for i in range(len(counts)):
            if sum(counts[i][0]) == 0:
                all_preds.append([]) # add empty array to all_preds because there are no detections for this image
                continue
            preds_array_i = self.process_lmds_results(counts[i], locs[i], labels[i], scores[i], dscores[i], det_conf_thres=0.2, clf_conf_thres=0.2)
            all_preds.append(preds_array_i)
        self._original_predictions = all_preds

    def process_lmds_results(self, counts, locs, labels, scores, dscores, det_conf_thres=0.2, clf_conf_thres=0.2):
        """
        Process the results from the Local Maxima Detection Strategy.

        Args:
            counts (list): 
                Number of detections for each species.
            locs (list): 
                Locations of the detections.
            labels (list): 
                Labels of the detections.
            scores (list): 
                Scores of the detections.
            dscores (list): 
                Detection scores.
            det_conf_thres (float, optional):
                Confidence threshold for detections. Defaults to 0.2.
            clf_conf_thres (float, optional):
                Confidence threshold for classification. Defaults to 0.2.

        Returns:
            numpy.ndarray: Processed detection results.
        """
        # Flatten the lists since its a single image 
        counts = counts[0]  
        locs = locs[0]  
        labels = labels[0]  
        scores = scores[0]
        dscores = dscores[0]  

        total_detections = sum(counts)  
        preds_array = np.empty((total_detections, 6)) #xyxy, confidence, class_id
        detection_idx = 0
        valid_detections_idx = 0
        # Loop through each species  
        for specie_idx in range(len(counts)):  
            count = counts[specie_idx]  
            if count == 0:  
                continue  
            
            # Get the detections for this species  
            species_locs = np.array(locs[detection_idx : detection_idx + count])
            species_locs[:, [0, 1]] = species_locs[:, [1, 0]] # Swap x and y in species_locs (herdnet uses y, x format)
            species_scores = np.array(scores[detection_idx : detection_idx + count])
            species_dscores = np.array(dscores[detection_idx : detection_idx + count])
            species_labels = np.array(labels[detection_idx : detection_idx + count])

            # Apply the confidence threshold
            valid_detections_by_clf_score = species_scores > clf_conf_thres
            valid_detections_by_det_score = species_dscores > det_conf_thres
            valid_detections = np.logical_and(valid_detections_by_clf_score, valid_detections_by_det_score)
            valid_detections_count = np.sum(valid_detections)
            valid_detections_idx += valid_detections_count
            # Fill the preds_array with the valid detections
            if valid_detections_count > 0:
                preds_array[valid_detections_idx - valid_detections_count : valid_detections_idx, :2] = species_locs[valid_detections] - 2
                preds_array[valid_detections_idx - valid_detections_count : valid_detections_idx, 2:4] = species_locs[valid_detections] + 2
                preds_array[valid_detections_idx - valid_detections_count : valid_detections_idx, 4] = species_scores[valid_detections]
                preds_array[valid_detections_idx - valid_detections_count : valid_detections_idx, 5] = species_labels[valid_detections]
            
            detection_idx += count # Move to the next species 
        
        preds_array = preds_array[:valid_detections_idx] # Remove the empty rows
        
        return preds_array        

    def _create_object_prediction_list_from_original_predictions(  
        self,  
        shift_amount_list: Optional[List[List[int]]] = [[0, 0]],  
        full_shape_list: Optional[List[List[int]]] = None,  
    ):  
        # Convert the predictions from HerdNet into a list of ObjectPrediction  
        original_predictions = self._original_predictions
        object_prediction_list_per_image = []
        for image_ind, image_predictions in enumerate(original_predictions):
            shift_amount = shift_amount_list[image_ind]
            full_shape = None if full_shape_list is None else full_shape_list[image_ind]
            object_prediction_list = []
            for prediction in image_predictions:
                x1 = prediction[0]
                y1 = prediction[1]
                x2 = prediction[2]
                y2 = prediction[3]
                bbox = [x1, y1, x2, y2]
                score = prediction[4]
                category_id = int(prediction[5])
                #category_name = self.category_mapping[str(category_id)]
                category_name = None # TODO: Get category name from category_mapping

                # fix negative box coords
                bbox[0] = max(0, bbox[0])
                bbox[1] = max(0, bbox[1])
                bbox[2] = max(0, bbox[2])
                bbox[3] = max(0, bbox[3])

                # fix out of image box coords
                if full_shape is not None:
                    bbox[0] = min(full_shape[1], bbox[0])
                    bbox[1] = min(full_shape[0], bbox[1])
                    bbox[2] = min(full_shape[1], bbox[2])
                    bbox[3] = min(full_shape[0], bbox[3])

                if not (bbox[0] < bbox[2]) or not (bbox[1] < bbox[3]):
                    print(f"ignoring invalid prediction with bbox: {bbox}")
                    continue

                object_prediction = ObjectPrediction(
                    bbox=bbox,
                    category_id=category_id,
                    score=score,
                    segmentation=None,
                    category_name=category_name,
                    shift_amount=shift_amount,
                    full_shape=full_shape,
                )
                object_prediction_list.append(object_prediction)
            object_prediction_list_per_image.append(object_prediction_list)

        self._object_prediction_list_per_image = object_prediction_list_per_image


class PredictionResult:
    def __init__(
        self,
        object_prediction_list: List[ObjectPrediction],
        durations_in_seconds: Dict[str, float],
    ):
        self.object_prediction_list = object_prediction_list
        self.durations_in_seconds = durations_in_seconds


class PseudoPIL:
    def __init__(self, ndarray):
        self.ndarray = ndarray

    @property
    def size(self):
        # Assuming the ndarray is in HWC format (Height x Width x Channels)
        height, width = self.ndarray.shape[:2]
        return (width, height)

    def __array__(self, dtype=None):
        # This method is called by np.asarray. It returns the ndarray.
        if dtype:
            return self.ndarray.astype(dtype)
        return self.ndarray


def read_image_as_pil(image: Union[str, np.ndarray], exif_fix: bool = False):
    return PseudoPIL(image)


logger = logging.getLogger(__name__)
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=os.environ.get("LOGLEVEL", "INFO").upper(),
)


def get_slice_bboxes(
    image_height: int,
    image_width: int,
    slice_height: Optional[int] = None,
    slice_width: Optional[int] = None,
    overlap_height_ratio: float = 0.2,
    overlap_width_ratio: float = 0.2,
) -> List[List[int]]:
    """Slices `image_pil` in crops.
    Corner values of each slice will be generated using the `slice_height`,
    `slice_width`, `overlap_height_ratio` and `overlap_width_ratio` arguments.

    Args:
        image_height (int): Height of the original image.
        image_width (int): Width of the original image.
        slice_height (int, optional): Height of each slice. Default None.
        slice_width (int, optional): Width of each slice. Default None.
        overlap_height_ratio(float): Fractional overlap in height of each
            slice (e.g. an overlap of 0.2 for a slice of size 100 yields an
            overlap of 20 pixels). Default 0.2.
        overlap_width_ratio(float): Fractional overlap in width of each
            slice (e.g. an overlap of 0.2 for a slice of size 100 yields an
            overlap of 20 pixels). Default 0.2.

    Returns:
        List[List[int]]: List of 4 corner coordinates for each N slices.
            [
                [slice_0_left, slice_0_top, slice_0_right, slice_0_bottom],
                ...
                [slice_N_left, slice_N_top, slice_N_right, slice_N_bottom]
            ]
    """
    slice_bboxes = []
    y_max = y_min = 0

    if slice_height and slice_width:
        y_overlap = int(overlap_height_ratio * slice_height)
        x_overlap = int(overlap_width_ratio * slice_width)
    else:
        raise ValueError("Compute type is not auto and slice width and height are not provided.")

    while y_max < image_height:
        x_min = x_max = 0
        y_max = y_min + slice_height
        while x_max < image_width:
            x_max = x_min + slice_width
            if y_max > image_height or x_max > image_width:
                xmax = min(image_width, x_max)
                ymax = min(image_height, y_max)
                xmin = max(0, xmax - slice_width)
                ymin = max(0, ymax - slice_height)
                slice_bboxes.append([xmin, ymin, xmax, ymax])
            else:
                slice_bboxes.append([x_min, y_min, x_max, y_max])
            x_min = x_max - x_overlap
        y_min = y_max - y_overlap
    return slice_bboxes


class SlicedImage:
    def __init__(self, image, starting_pixel):
        """
        image: np.array
            Sliced image.
        starting_pixel: list of list of int
            Starting pixel coordinates of the sliced image.
        """
        self.image = image
        self.starting_pixel = starting_pixel


class SliceImageResult:
    def __init__(self, original_image_size: List[int]):
        """
        original_image_size: list of int
            Size of the unsliced original image in [height, width]
        """
        self.original_image_height = original_image_size[0]
        self.original_image_width = original_image_size[1]

        self._sliced_image_list: List[SlicedImage] = []

    def add_sliced_image(self, sliced_image: SlicedImage):
        if not isinstance(sliced_image, SlicedImage):
            raise TypeError("sliced_image must be a SlicedImage instance")

        self._sliced_image_list.append(sliced_image)

    @property
    def sliced_image_list(self):
        return self._sliced_image_list

    @property
    def images(self):
        """Returns sliced images.

        Returns:
            images: a list of np.array
        """
        images = []
        for sliced_image in self._sliced_image_list:
            images.append(sliced_image.image)
        return images

    @property
    def starting_pixels(self) -> List[int]:
        """Returns a list of starting pixels for each slice.

        Returns:
            starting_pixels: a list of starting pixel coords [x,y]
        """
        starting_pixels = []
        for sliced_image in self._sliced_image_list:
            starting_pixels.append(sliced_image.starting_pixel)
        return starting_pixels

    @property
    def filenames(self) -> List[int]:
        """Returns a list of filenames for each slice.

        Returns:
            filenames: a list of filenames as str
        """
        filenames = []
        for sliced_image in self._sliced_image_list:
            filenames.append(sliced_image.coco_image.file_name)
        return filenames

    def __getitem__(self, i):
        def _prepare_ith_dict(i):
            return {
                "image": self.images[i],
                "coco_image": self.coco_images[i],
                "starting_pixel": self.starting_pixels[i],
                "filename": self.filenames[i],
            }

        if isinstance(i, np.ndarray):
            i = i.tolist()

        if isinstance(i, int):
            return _prepare_ith_dict(i)
        elif isinstance(i, slice):
            start, stop, step = i.indices(len(self))
            return [_prepare_ith_dict(i) for i in range(start, stop, step)]
        elif isinstance(i, (tuple, list)):
            accessed_mapping = map(_prepare_ith_dict, i)
            return list(accessed_mapping)
        else:
            raise NotImplementedError(f"{type(i)}")

    def __len__(self):
        return len(self._sliced_image_list)


def slice_image(
    image: Union[str, np.ndarray],
    slice_height: Optional[int] = None,
    slice_width: Optional[int] = None,
    overlap_height_ratio: float = 0.2,
    overlap_width_ratio: float = 0.2,
    min_area_ratio: float = 0.1,
    verbose: bool = False,
) -> SliceImageResult:
    """Slice a large image into smaller windows.

    Args:
        image (str or PIL.Image): File path of image or Pillow Image to be sliced.
        slice_height (int, optional): Height of each slice. Default None.
        slice_width (int, optional): Width of each slice. Default None.
        overlap_height_ratio (float): Fractional overlap in height of each
            slice (e.g. an overlap of 0.2 for a slice of size 100 yields an
            overlap of 20 pixels). Default 0.2.
        overlap_width_ratio (float): Fractional overlap in width of each
            slice (e.g. an overlap of 0.2 for a slice of size 100 yields an
            overlap of 20 pixels). Default 0.2.
        min_area_ratio (float): If the cropped annotation area to original annotation
            ratio is smaller than this value, the annotation is filtered out. Default 0.1.
        verbose (bool, optional): Switch to print relevant values to screen.
            Default 'False'.

    Returns:
        sliced_image_result: SliceImageResult:
                                sliced_image_list: list of SlicedImage
                                original_image_size: list of int
                                    Size of the unsliced original image in [height, width]
    """

    # define verboseprint
    verboselog = logger.info if verbose else lambda *a, **k: None

    # read image
    image_pil = read_image_as_pil(image)
    verboselog("image.shape: " + str(image_pil.size))

    image_width, image_height = image_pil.size
    if not (image_width != 0 and image_height != 0):
        raise RuntimeError(f"invalid image size: {image_pil.size} for 'slice_image'.")

    slice_bboxes = get_slice_bboxes(
        image_height=image_height,
        image_width=image_width,
        slice_height=slice_height,
        slice_width=slice_width,
        overlap_height_ratio=overlap_height_ratio,
        overlap_width_ratio=overlap_width_ratio,
    )

    n_ims = 0

    # init images and annotations lists
    sliced_image_result = SliceImageResult(original_image_size=[image_height, image_width])

    image_pil_arr = np.asarray(image_pil)
    # iterate over slices
    for slice_bbox in slice_bboxes:
        n_ims += 1

        # extract image
        tlx = slice_bbox[0]
        tly = slice_bbox[1]
        brx = slice_bbox[2]
        bry = slice_bbox[3]
        image_pil_slice = image_pil_arr[tly:bry, tlx:brx]

        slice_width = slice_bbox[2] - slice_bbox[0]
        slice_height = slice_bbox[3] - slice_bbox[1]

        # create sliced image and append to sliced_image_result
        sliced_image = SlicedImage(
            image=image_pil_slice, starting_pixel=[tlx, tly]
        )
        sliced_image_result.add_sliced_image(sliced_image)

    verboselog(
        "Num slices: " + str(n_ims) + " slice_height: " + str(slice_height) + " slice_width: " + str(slice_width)
    )

    return sliced_image_result


def get_prediction_batched(
    image,
    detection_model,
    shift_amount_list: list = [[0, 0]],
    full_shape=None,
    postprocess: Optional[PostprocessPredictions] = None,
    verbose: int = 0,
) -> list:
    """
    Function for performing prediction for given image using given detection_model.

    Arguments:
        image: str or np.ndarray
            Location of image or numpy image matrix to slice
        detection_model: model.DetectionMode
        shift_amount: List
            To shift the box and mask predictions from sliced image to full
            sized image, should be in the form of [shift_x, shift_y]
        full_shape: List
            Size of the full image, should be in the form of [height, width]
        postprocess: sahi.postprocess.combine.PostprocessPredictions
        verbose: int
            0: no print (default)
            1: print prediction duration

    Returns:
        A dict with fields:
            object_prediction_list: a list of ObjectPrediction
            durations_in_seconds: a dict containing elapsed times for profiling
    """
    durations_in_seconds = dict()

    # get prediction
    time_start = time.time()
    detection_model.perform_inference(image)
    time_end = time.time() - time_start
    durations_in_seconds["prediction"] = time_end

    # process prediction
    time_start = time.time()

    full_shape_list = [full_shape] * len(shift_amount_list)
    detection_model.convert_original_predictions(
        shift_amount=shift_amount_list,
        full_shape=full_shape_list,
    )

    object_prediction_list_per_image: List[List[ObjectPrediction]] = detection_model.object_prediction_list_per_image

    if verbose == 1:
        print(
            "Prediction performed in",
            durations_in_seconds["prediction"],
            "seconds.",
        )

    # flatten object_prediction_list_per_image
    object_prediction_list: List[ObjectPrediction] = [pred for preds in object_prediction_list_per_image for pred in preds]

    return PredictionResult(
        object_prediction_list=object_prediction_list, durations_in_seconds=durations_in_seconds
    )


def get_sliced_prediction_batched(
    image,
    detection_model=None,
    slice_height: int = None,
    slice_width: int = None,
    overlap_height_ratio: float = 0.2,
    overlap_width_ratio: float = 0.2,
    postprocess_type: str = "GREEDYNMM",
    postprocess_match_metric: str = "IOS",
    postprocess_match_threshold: float = 0.5,
    postprocess_class_agnostic: bool = False,
    perform_standard_pred: bool = False,
    verbose: int = 1,
) -> list:
    """
    Function for slice image + get predicion for each slice + combine predictions in full image.

    Args:
        image: str or np.ndarray
            Location of image or numpy image matrix to slice
        detection_model: model.DetectionModel
        slice_height: int
            Height of each slice.  Defaults to ``None``.
        slice_width: int
            Width of each slice.  Defaults to ``None``.
        overlap_height_ratio: float
            Fractional overlap in height of each window (e.g. an overlap of 0.2 for a window
            of size 512 yields an overlap of 102 pixels).
            Default to ``0.2``.
        overlap_width_ratio: float
            Fractional overlap in width of each window (e.g. an overlap of 0.2 for a window
            of size 512 yields an overlap of 102 pixels).
            Default to ``0.2``.
        perform_standard_pred: bool
            Perform a standard prediction on top of sliced predictions to increase large object
            detection accuracy. Default: True.
        postprocess_type: str
            Type of the postprocess to be used after sliced inference while merging/eliminating predictions.
            Options are 'NMM', 'GRREDYNMM' or 'NMS'. Default is 'GRREDYNMM'.
        postprocess_match_metric: str
            Metric to be used during object prediction matching after sliced prediction.
            'IOU' for intersection over union, 'IOS' for intersection over smaller area.
        postprocess_match_threshold: float
            Sliced predictions having higher iou than postprocess_match_threshold will be
            postprocessed after sliced prediction.
        postprocess_class_agnostic: bool
            If True, postprocess will ignore category ids.
        verbose: int
            0: no print
            1: print number of slices (default)
            2: print number of slices and slice/prediction durations
    """

    # for profiling
    durations_in_seconds = dict()

    # create slices from full image
    time_start = time.time()
    slice_image_result = slice_image(
        image=image,
        slice_height=slice_height,
        slice_width=slice_width,
        overlap_height_ratio=overlap_height_ratio,
        overlap_width_ratio=overlap_width_ratio,
    )

    num_slices = len(slice_image_result)
    time_end = time.time() - time_start
    durations_in_seconds["slice"] = time_end

    # init match postprocess instance
    if postprocess_type not in POSTPROCESS_NAME_TO_CLASS.keys():
        raise ValueError(
            f"postprocess_type should be one of {list(POSTPROCESS_NAME_TO_CLASS.keys())} but given as {postprocess_type}"
        )
    elif postprocess_type == "UNIONMERGE":
        # deprecated in v0.9.3
        raise ValueError("'UNIONMERGE' postprocess_type is deprecated, use 'GREEDYNMM' instead.")
    postprocess_constructor = POSTPROCESS_NAME_TO_CLASS[postprocess_type]
    postprocess = postprocess_constructor(
        match_threshold=postprocess_match_threshold,
        match_metric=postprocess_match_metric,
        class_agnostic=postprocess_class_agnostic,
    )

    # create prediction input
    if verbose == 1 or verbose == 2:
        print(f"Performing prediction on {num_slices} number of slices.")

    final_object_prediction_list: List[ObjectPrediction] = []

    # perform batch prediction
    prediction_result = get_prediction_batched(
        image=slice_image_result.images,
        detection_model=detection_model,
        shift_amount_list=slice_image_result.starting_pixels,
        full_shape=[
            slice_image_result.original_image_height,
            slice_image_result.original_image_width,
        ],
    )

    # convert sliced predictions to full predictions
    for object_prediction in prediction_result.object_prediction_list:
        if object_prediction:  # if not empty
            final_object_prediction_list.append(object_prediction.get_shifted_object_prediction())

    # merge matching predictions
    if len(final_object_prediction_list) > 1:
        final_object_prediction_list = postprocess(final_object_prediction_list)

    time_end = time.time() - time_start
    durations_in_seconds["prediction"] = time_end

    if verbose == 2:
        print(
            "Slicing performed in",
            durations_in_seconds["slice"],
            "seconds.",
        )
        print(
            "Prediction performed in",
            durations_in_seconds["prediction"],
            "seconds.",
        )

    return PredictionResult(
        object_prediction_list=final_object_prediction_list, durations_in_seconds=durations_in_seconds
    )
