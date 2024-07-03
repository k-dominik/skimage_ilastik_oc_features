from typing import Dict, List

from typing_extensions import NotRequired, TypedDict

import numpy
import numpy.typing as npt
from skimage.measure import regionprops
import vigra
from ilastik.plugins.types import ObjectFeaturesPlugin


class FeatureDescription(TypedDict):
    displaytext: str
    detailtext: str
    tooltip: str
    advanced: bool
    group: str
    margin: NotRequired[int]
    # features are assumed to be able to do 2D and 3D. If your feature
    # cannot do one of them, you can mark those accordingly by setting
    # one of those keys in the feature description
    no_3D: NotRequired[bool]
    no_2D: NotRequired[bool]
    # if raw data is accessed for your feature, the most of the
    channel_aware: NotRequired[bool]


class Skimage4ilastik(ObjectFeaturesPlugin):
    """Plugins of this class calculate object features.

    """

    name = "skimage_ilastik_oc_features"

    _feature_dict: Dict[str, FeatureDescription] = {
        "area": {
            "displaytext": "Area of the object",
            "detailtext": "Area of the region i.e. number of pixels of the region.",
            "tooltip": "area",
            "advanced": False,
            "group": "Area",
        },
        "area_bbox": {
            "displaytext": "Bounding box area",
            "detailtext": "Area of the bounding box i.e. number of pixels of bounding box.",
            "tooltip": "area_bbox",
            "advanced": False,
            "group": "Area",
        },
        "area_convex": {
            "displaytext": "Convex hull area",
            "detailtext": "Area of the convex hull image, which is the smallest convex polygon that encloses the region.",
            "tooltip": "area_convex",
            "advanced": False,
            "group": "Area",
        },
        "area_filled": {
            "displaytext": "Convex hull area",
            "detailtext": "Area of the region with all the holes filled in.",
            "tooltip": "area_filled",
            "advanced": False,
            "group": "Area",
        },
        "axis_major_length": {
            "displaytext": "Ellipse major axis length",
            "detailtext": "The length of the major axis of the ellipse that has the same normalized second central moments as the region.",
            "tooltip": "axis_major_length",
            "advanced": False,
            "group": "Shape",
        },
        "axis_minor_length": {
            "displaytext": "Ellipse minor axis length",
            "detailtext": "The length of the minor axis of the ellipse that has the same normalized second central moments as the region.",
            "tooltip":
            "axis_minor_length",
            "advanced": False,
            "group": "Shape"
        },
        "bbox": {
            "displaytext": "Bounding box coordinates",
            "detailtext": "Bounding box (min_row, min_col, max_row, max_col). Pixels belonging to the bounding box are in the half-open interval [min_row; max_row) and [min_col; max_col).",
            "tooltip": "bbox",
            "advanced": False,
            "group": "Location",
        },
        "centroid": {
            "displaytext": "Centroid coordinates",
            "detailtext": "Centroid coordinate tuple (row, col).",
            "tooltip": "centroid",
            "advanced": False,
            "group": "Location",
        },
        "centroid_local": {
            "displaytext": "Local centroid coordinates",
            "detailtext": "Centroid coordinate tuple (row, col), relative to region bounding box.",
            "tooltip": "centroid_local",
            "advanced": False,
            "group": "Location",
        },
        "centroid_weighted": {
            "displaytext": "Centroid coordinates weighted",
            "detailtext": "Centroid coordinate tuple (row, col) weighted with intensity image.",
            "tooltip": "centroid_weighted",
            "advanced": False,
            "group": "Location",
            "channel_aware": True,
        },
        "centroid_weighted_local": {
            "displaytext": "Local centroid coordinates weighted",
            "detailtext": "Centroid coordinate tuple (row, col), relative to region bounding box, weighted with intensity image.",
            "tooltip": "centroid_weighted_local",
            "advanced": False,
            "group": "Location",
            "channel_aware": True,
        },
        "eccentricity": {
            "displaytext": "Eccentricity",
            "detailtext": "Eccentricity of the ellipse that has the same second-moments as the region. The eccentricity is the ratio of the focal distance (distance between focal points) over the major axis length. The value is in the interval [0, 1). When it is 0, the ellipse becomes a circle.",
            "tooltip": "eccentricity",
            "advanced": False,
            "group": "Shape",
            "no_3D": True,
        },
        "equivalent_diameter_area": {
            "displaytext": "Equivalent diameter area",
            "detailtext": "The diameter of a circle with the same area as the region.",
            "tooltip": "equivalent_diameter_area",
            "advanced": False,
            "group": "Shape",
        },
        "extent": {
            "displaytext": "Extent",
            "detailtext": "Ratio of pixels in the region to pixels in the total bounding box. Computed as area / (rows * cols).",
            "tooltip": "extent",
            "advanced": False,
            "group": "Shape",
        },
        "feret_diameter_max": {
            "displaytext": "Feret diameter max",
            "detailtext": "Maximum Feret’s diameter computed as the longest distance between points around a region’s convex hull contour as determined by find_contours.",
            "tooltip": "feret_diameter_max",
            "advanced": False,
            "group": "Shape",
        },
        "inertia_tensor": {
            "displaytext": "Inertia tensor",
            "detailtext": "Inertia tensor of the region for the rotation around its mass.",
            "tooltip": "inertia_tensor",
            "advanced": False,
            "group": "Shape",
        },
        "inertia_tensor_eigvals": {
            "displaytext": "Inertia tensor eigvals",
            "detailtext": "The eigenvalues of the inertia tensor in decreasing order.",
            "tooltip": "inertia_tensor_eigvals",
            "advanced": False,
            "group": "Shape",
        },
        "intensity_max": {
            "displaytext": "Intensity max",
            "detailtext": "Value with the greatest intensity in the region.",
            "tooltip": "intensity_max",
            "advanced": False,
            "group": "Intensity",
            "channel_aware": True,
        },
        "intensity_mean": {
            "displaytext": "Intensity mean",
            "detailtext": "Value with the mean intensity in the region.",
            "tooltip": "intensity_mean",
            "advanced": False,
            "group": "Shape",
        },
        "intensity_min": {
            "displaytext": "Intensity min",
            "detailtext": "Value with the least intensity in the region.",
            "tooltip": "intensity_min",
            "advanced": False,
            "group": "Intensity",
            "channel_aware": True,
        },
        "intensity_std": {
             "displaytext": "Intensity std",
             "detailtext": "Standard deviation of the intensity in the region.",
             "tooltip": "intensity_std",
             "advanced": False,
             "group": "Shape",
             "channel_aware": True,
         },
        "label": {
            "displaytext": "Label",
            "detailtext": "The label in the labeled input image.",
            "tooltip": "label",
            "advanced": False,
            "group": "Other",
        },
        "moments": {
            "displaytext": "Moments",
            "detailtext": "Spatial moments up to 3rd order: m_ij = sum{ array(row, col) * row^i * col^j } where the sum is over the row, col coordinates of the region.",
            "tooltip": "moments",
            "advanced": False,
            "group": "Shape",
        },
        "moments_central": {
            "displaytext": "Moments central",
            "detailtext": "Central moments (translation invariant) up to 3rd order: mu_ij = sum{ array(row, col) * (row - row_c)^i * (col - col_c)^j } where the sum is over the row, col coordinates of the region, and row_c and col_c are the coordinates of the region’s centroid.",
            "tooltip": "moments_central",
            "advanced": False,
            "group": "Shape",
        },
        "moments_hu": {
            "displaytext": "Moments hu",
            "detailtext": "Hu moments (translation, scale and rotation invariant).",
            "tooltip": "moments_hu",
            "advanced": False,
            "group": "Shape",
            "no_3D": True,
        },
        "moments_normalized": {
            "displaytext": "Moments normalized",
            "detailtext": "Normalized moments (translation and scale invariant) up to 3rd order: nu_ij = mu_ij / m_00^[(i+j)/2 + 1] where m_00 is the zeroth spatial moment.",
            "tooltip": "moments_normalized",
            "advanced": False,
            "group": "Shape",
        },
        "moments_weighted": {
            "displaytext": "Moments weighted",
            "detailtext": "Spatial moments of intensity image up to 3rd order: wm_ij = sum{ array(row, col) * row^i * col^j } where the sum is over the row, col coordinates of the region.",
            "tooltip": "moments_weighted",
            "advanced": False,
            "group": "Shape",
            "channel_aware": True,
        },
        "moments_weighted_central": {
            "displaytext": "Moments weighted central",
            "detailtext": "Central moments (translation invariant) of intensity image up to 3rd order: wmu_ij = sum{ array(row, col) * (row - row_c)^i * (col - col_c)^j } where the sum is over the row, col coordinates of the region, and row_c and col_c are the coordinates of the region’s weighted centroid.",
            "tooltip": "moments_weighted_central",
            "advanced": False,
            "group": "Shape",
            "channel_aware": True,
        },
        "moments_weighted_hu": {
            "displaytext": "Moments weighted hu",
            "detailtext": "Hu moments (translation, scale and rotation invariant) of intensity image.",
            "tooltip": "moments_weighted_hu",
            "advanced": False,
            "group": "Shape",
            "no_3D": True,
            "channel_aware": True,
        },
        "moments_weighted_normalized": {
            "displaytext": "Moments weighted normalized",
            "detailtext": "Normalized moments (translation and scale invariant) of intensity image up to 3rd order: wnu_ij = wmu_ij / wm_00^[(i+j)/2 + 1] where wm_00 is the zeroth spatial moment (intensity-weighted area).",
            "tooltip": "moments_weighted_normalized",
            "advanced": False,
            "group": "Shape",
            "channel_aware": True,
        },
        "num_pixels": {
            "displaytext": "Number of pixels",
            "detailtext": "Number of foreground pixels.",
            "tooltip": "num_pixels",
            "advanced": False,
            "group": "Shape",
        },
        "orientation": {
            "displaytext": "Orientation",
            "detailtext": "Angle between the 0th axis (rows) and the major axis of the ellipse that has the same second moments as the region, ranging from -pi/2 to pi/2 counter-clockwise.",
            "tooltip": "orientation",
            "advanced": False,
            "group": "Shape",
            "no_3D": True,
        },
        "perimeter": {
            "displaytext": "Perimeter",
            "detailtext": "Perimeter of object which approximates the contour as a line through the centers of border pixels using a 4-connectivity.",
            "tooltip": "perimeter",
            "advanced": False,
            "group": "Shape",
            "no_3D": True,
        },
        "perimeter_crofton": {
            "displaytext": "Perimeter crofton",
            "detailtext": "Perimeter of object approximated by the Crofton formula in 4 directions.",
            "tooltip": "perimeter_crofton",
            "advanced": False,
            "group": "Shape",
            "no_3D": False,
        },
        "solidity": {
            "displaytext": "Solidity",
            "detailtext": "Ratio of pixels in the region to pixels of the convex hull image.",
            "tooltip": "solidity",
            "advanced": False,
            "group": "Shape",
        },
    }

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._selectedFeatures = []

    def availableFeatures(self, image: vigra.VigraArray, labels: vigra.VigraArray):
        """Reports which features this plugin can compute on a
        particular image and label image.

        often plugins may return a slightly different feature set depending
        on the image being 2D or 3D.

        Args:
            image: tagged vigra array
            labels: tagged vigra array

        Returns:
            a nested dictionary, where dict[feature_name] is a
            dictionary of parameters.

        """
        is_3d = all(ax > 1 for ax in labels.withAxes("zyx").shape)

        def is_compatible(fdict):
            if not is_3d:
                return not fdict.get("no_2D", False)

            return not fdict.get("no_3D", False)

        return {k: v for k, v in self._feature_dict.items() if is_compatible(v)}

    def compute_global(
        self, image: vigra.VigraArray, labels: vigra.VigraArray, features: Dict[str, Dict], axes
    ) -> Dict[str, npt.ArrayLike]:
        """calculate the requested features.

        Object id 0 should be excluded from feature computation here.
        (ilastik expects it that way.)

        Args;
            image: VigraArray of the image with axistags, always xyzc, full image
            labels: VigraArray of the labels with axistags, always xyz, full label
                image with all objects having unique pixel values.
                ObjectID 0 is considered background, object ids are positive integers
            features: list of feature names; which features to compute
            axes: axis tags, DEPRECATED; use `.axistags` attribute of image/labels

        Returns
            dictionary with one entry per feature. dict[feature_name] is a
            numpy.ndarray with shape (n_objs, n_featvals), where featvals is
            the number of elements for a single object for the particular feature
        """
        # ilastik will give all features here, so we need to exclude the local ones
        global_features = {k: v for k, v in features.items() if "margin" not in v}
        # For skimage we need to treat channel aware features differently
        features_channnel_aware = [f for f, d in global_features.items() if d.get("_channel_aware", False)]
        features_no_channel = [f for f, d in global_features.items() if not d.get("_channel_aware", False)]

        def compute_assume_single_channel(
            _image: vigra.VigraArray, _labels: vigra.VigraArray, axes, _features: List[str]
        ):

            regions = regionprops(_labels.squeeze(), intensity_image=_image.squeeze())
            computed_features: Dict[str, npt.ArrayLike] = {}
            for prop in _features:
                computed_features[prop] = numpy.atleast_2d([numpy.array(rr[prop]).flatten() for rr in regions])

            return computed_features

        ret = compute_assume_single_channel(image, labels, axes, features_no_channel)

        ret.update(
            self.do_channels(compute_assume_single_channel, image, labels, axes, _features=features_channnel_aware)
        )

        return ret

    def compute_local(self, image: vigra.VigraArray, binary_bbox: vigra.VigraArray, features: Dict[str, Dict], axes):
        """Calculate features on a single object.

        Args:
            image: VigraArray of the image with axistags, always xyzc, for one object, includes margin around
            binary_bbox: VigraArray of the image with axistags, always xyz, for one object, includes margin around
            features: which features to compute
            axes: axis tags, DEPRECATED; use `.axistags` attribute of image/labels

        Returns:
            a dictionary with one entry per feature.
            dict[feature_name] is a numpy.ndarray with ndim=1

        """
        # call parent class method if local features are not needed
        feature_dict = {}

        local_features = {k: v for k, v in features.items() if "margin" in v}

        assert len(local_features) == 0, "No local features expected for this plugin"

        return feature_dict


    def do_channels(self, fn, image: vigra.VigraArray, labels: vigra.VigraArray, axes, **kwargs):
        """Helper for features that only take one channel.

        :param fn: function that computes features

        """
        results = []
        slc = [slice(None)] * 4
        channel_index = image.channelIndex
        for channel in range(image.shape[channel_index]):
            slc[channel_index] = channel
            # a dictionary for the channel

            result = fn(image[slc], labels, axes, **kwargs)
            results.append(result)

        return self.combine_dicts_with_numpy(results)

    def fill_properties(self, feature_dict):
        """Augment die feature dictionary with additional fields

        For every feature in the feature dictionary, fill in its properties,
        such as 'detailtext', which will be displayed in help, or 'displaytext'
        which will be displayed instead of the feature name

        Only necessary if these keys are not present to begin with.

        Args:
            feature_dict: list of feature names

        Returns:
            same dictionary, with additional fields filled for each feature

        """
        return super().fill_properties(feature_dict)
