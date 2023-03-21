from typing import cast, List, Tuple
import pydicom
import PIL.Image
import numpy as np
from functools import cached_property
from torch.utils.data import Dataset
from enum import Enum
import random

VOI_LUT = Enum('VOI_LUT', ['function', 'sequence'])


def mammogram_to_pil(image, resize=False):
    image_max = image.max()
    image_min = image.min()

    output = (((image - image_min) / (image_max - image_min)) * 255).astype(np.uint8)
    pil_image = PIL.Image.fromarray(output, mode="L")

    if resize is not False:
        shape = np.asarray(image.shape)
        shape = shape / shape.max()
        shape = (shape * resize).astype(int).tolist()
        pil_image = pil_image.resize(shape[::-1])

    return pil_image


def apply_windowing(arr: "np.ndarray", ds: "Dataset", index: int = 0) -> "np.ndarray":
    """Apply a windowing operation to `arr`.
    .. versionadded:: 2.1
    Parameters
    ----------
    arr : numpy.ndarray
        The :class:`~numpy.ndarray` to apply the windowing operation to.
    ds : dataset.Dataset
        A dataset containing a :dcm:`VOI LUT Module<part03/sect_C.11.2.html>`.
        If (0028,1050) *Window Center* and (0028,1051) *Window Width* are
        present then returns an array of ``np.float64``, otherwise `arr` will
        be returned unchanged.
    index : int, optional
        When the VOI LUT Module contains multiple alternative views, this is
        the index of the view to return (default ``0``).
    Returns
    -------
    numpy.ndarray
        An array with applied windowing operation.
    Notes
    -----
    When the dataset requires a modality LUT or rescale operation as part of
    the Modality LUT module then that must be applied before any windowing
    operation.
    See Also
    --------
    :func:`~pydicom.pixel_data_handlers.util.apply_modality_lut`
    :func:`~pydicom.pixel_data_handlers.util.apply_voi`
    References
    ----------
    * DICOM Standard, Part 3, :dcm:`Annex C.11.2
      <part03/sect_C.11.html#sect_C.11.2>`
    * DICOM Standard, Part 3, :dcm:`Annex C.8.11.3.1.5
      <part03/sect_C.8.11.3.html#sect_C.8.11.3.1.5>`
    * DICOM Standard, Part 4, :dcm:`Annex N.2.1.1
      <part04/sect_N.2.html#sect_N.2.1.1>`
    """
    if "WindowWidth" not in ds and "WindowCenter" not in ds:
        return arr

    # May be LINEAR (default), LINEAR_EXACT, SIGMOID or not present, VM 1
    voi_func = cast(str, getattr(ds, "VOILUTFunction", "LINEAR")).upper()
    # VR DS, VM 1-n
    elem = ds["WindowCenter"]
    center = cast(List[float], elem.value)[index] if elem.VM > 1 else elem.value
    center = cast(float, center)
    elem = ds["WindowWidth"]
    width = cast(List[float], elem.value)[index] if elem.VM > 1 else elem.value
    width = cast(float, width)

    # The output range depends on whether or not a modality LUT or rescale
    #   operation has been applied
    ds.BitsStored = cast(int, ds.BitsStored)
    y_min: float
    y_max: float
    if ds.get("ModalityLUTSequence"):
        # Unsigned - see PS3.3 C.11.1.1.1
        y_min = 0
        item = cast(List["Dataset"], ds.ModalityLUTSequence)[0]
        bit_depth = cast(List[int], item.LUTDescriptor)[2]
        y_max = 2 ** bit_depth - 1
    elif ds.PixelRepresentation == 0:
        # Unsigned
        y_min = 0
        y_max = 2 ** ds.BitsStored - 1
    else:
        # Signed
        y_min = -(2 ** (ds.BitsStored - 1))
        y_max = 2 ** (ds.BitsStored - 1) - 1

    slope = ds.get("RescaleSlope", None)
    intercept = ds.get("RescaleIntercept", None)
    if slope is not None and intercept is not None:
        ds.RescaleSlope = cast(float, ds.RescaleSlope)
        ds.RescaleIntercept = cast(float, ds.RescaleIntercept)
        # Otherwise its the actual data range
        y_min = y_min * ds.RescaleSlope + ds.RescaleIntercept
        y_max = y_max * ds.RescaleSlope + ds.RescaleIntercept

    y_range = y_max - y_min
    arr = arr.astype("float64")

    if voi_func in ["LINEAR", "LINEAR_EXACT"]:
        # PS3.3 C.11.2.1.2.1 and C.11.2.1.3.2
        if voi_func == "LINEAR":
            if width < 1:
                raise ValueError(
                    "The (0028,1051) Window Width must be greater than or "
                    "equal to 1 for a 'LINEAR' windowing operation"
                )
            center -= 0.5
            width -= 1
        elif width <= 0:
            raise ValueError(
                "The (0028,1051) Window Width must be greater than 0 "
                "for a 'LINEAR_EXACT' windowing operation"
            )

        below = arr <= (center - width / 2)
        above = arr > (center + width / 2)
        between = np.logical_and(~below, ~above)

        arr[below] = y_min
        arr[above] = y_max
        if between.any():
            arr[between] = ((arr[between] - center) / width + 0.5) * y_range + y_min
    elif voi_func == "SIGMOID":
        # PS3.3 C.11.2.1.3.1
        if width <= 0:
            raise ValueError(
                "The (0028,1051) Window Width must be greater than 0 "
                "for a 'SIGMOID' windowing operation"
            )

        arr = y_range / (1 + np.exp(-4 * (arr - center) / width)) + y_min
    else:
        raise ValueError(f"Unsupported (0028,1056) VOI LUT Function value '{voi_func}'")

    return arr


class Mammogram:
    def __init__(self, filename, flip_if_required=False):
        self._filename = filename
        self._dcm_obj = pydicom.read_file(filename)
        if not self._dcm_obj.Modality == "MG":
            raise RuntimeError(f"Not a mammogram. Modality {self._dcm_obj.Modality}")

        self._voi_lut_function_name = None

        # check
        intercept = getattr(self._dcm_obj, "RescaleIntercept", 0.0)
        slope = getattr(self._dcm_obj, "RescaleSlope", 1.0)
        assert intercept == 0.0 and slope == 1.0

        self.window_center = None
        self.window_width = None
        self.num_windows = 0
        self.num_lut_funcs = 0
        self.num_lut_seqs = 0

        self.__lut_func_index = None
        self.__lut_seq_index = None

        self._flip_if_required = flip_if_required

        self._parse_window_levels()
        self._parse_lookup_tables()

    @property
    def _requires_horizontal_flip(self):
        needs_flip = getattr(self._dcm_obj, "FieldOfViewHorizontalFlip", "NO") == "YES"

        orientation = getattr(self._dcm_obj, "PatientOrientation", None)
        if orientation is not None and len(orientation) > 0:
            if (
                self.laterality == "L"
                and orientation[0] == "P"
                or self.laterality == "R"
                and orientation[0] == "A"
            ):
                needs_flip = True

        return needs_flip

    @property
    def raw_array(self):
        return self._dcm_obj.pixel_array

    @property
    def raw_array_as_pil(self):
        return mammogram_to_pil(self.raw_array)

    @property
    def spacing(self):
        spacing = getattr(self._dcm_obj, "PixelSpacing", None)
        if spacing is None:
            spacing = getattr(self._dcm_obj, "ImagerPixelSpacing", None)

        if not spacing:
            raise ValueError("No pixel spacing found")

        if spacing[0] != spacing[1]:
            raise ValueError("Pixel spacing is not uniform")
        return float(spacing[0])

    @cached_property
    def array(self):
        if not hasattr(self._dcm_obj, "PhotometricInterpretation"):
            raise ValueError("DICOM file has no PhotometricInterpretation")

        if self._dcm_obj.PhotometricInterpretation not in [
            "MONOCHROME1",
            "MONOCHROME2",
        ]:
            raise ValueError(
                "When performing a windowing operation only 'MONOCHROME1' and "
                "'MONOCHROME2' are allowed for (0028,0004) Photometric "
                "Interpretation"
            )

        array = self.raw_array.copy()

        if self.num_lut_funcs > 0 or self.num_lut_seqs > 0:
            if self.__lut_func_index is None and self.__lut_seq_index is None:
                raise RuntimeError("Need to select a VOI or LUT")

        if self.__lut_func_index is not None:  # Takes precedence
            array = pydicom.pixel_data_handlers.apply_voi(
                array, self._dcm_obj, index=self.__lut_func_index
            )

        elif self.__lut_seq_index:
            array = apply_windowing(array, self._dcm_obj, index=self.__lut_seq_index)

        # Photometric
        if self.__lut_seq_index is not None and self._requires_inversion:
            print(
                f"Not sure what is happening here: {self._filename}. Requires inversion. Ignoring."
            )

        if self.__lut_seq_index is None:
            if self._requires_inversion:
                array = array.max() - array

        if self._requires_horizontal_flip and self._flip_if_required:
            array = np.ascontiguousarray(np.fliplr(array))

        return array

    @property
    def as_pil(self):
        if not self.lut_set:
            if self.num_lut_funcs > 0:
                self.set_lut_func(self.num_lut_funcs // 2)
            elif self.num_lut_seqs > 0:
                self.set_lut_seq(self.num_lut_seqs // 2)

        image = self.array
        if not image.ndim == 2:
            raise ValueError("wrong image dimensions")
        image_max = image.max()
        image_min = image.min()

        output = (((image - image_min) / (image_max - image_min + 0.0001)) * 255).astype(
            np.uint8
        )
        pil_image = PIL.Image.fromarray(output, mode="L")

        return pil_image

    @property
    def patient_id(self):
        return self._dcm_obj.PatientID

    @property
    def laterality(self):
        laterality_0 = getattr(self._dcm_obj, "Laterality", None)
        laterality_1 = getattr(self._dcm_obj, "ImageLaterality", None)
        if laterality_0 is None:
            return laterality_1
        return laterality_0

    @property
    def view(self):
        return getattr(self._dcm_obj, "ViewPosition", None)

    @property
    def _requires_inversion(self):
        return getattr(self._dcm_obj, "PhotometricInterpretation", "") == "MONOCHROME1"

    @property
    def lut_set(self):
        return not (self.__lut_func_index is None
                    and self.__lut_seq_index is None)

    def set_lut_func(self, idx):
        self.unset_lut_seq()
        self.__lut_func_index = idx

    def unset_lut_func(self):
        self.__lut_func_index = None

    def set_lut_seq(self, idx):
        self.unset_lut_func()
        self.__lut_seq_index = idx

    def unset_lut_seq(self):
        self.__lut_seq_index = None

    def _parse_window_levels(self):
        # TODO: A check that window_center and width are the same length
        window_center = getattr(self._dcm_obj, "WindowCenter", None)
        window_width = getattr(self._dcm_obj, "WindowWidth", None)

        if isinstance(window_center, pydicom.valuerep.DSfloat):
            window_center = [float(window_center)]
        if isinstance(window_width, pydicom.valuerep.DSfloat):
            window_width = [float(window_width)]

        self.window_center = window_center
        self.window_width = window_width
        self.num_lut_funcs = len(window_center) if window_center is not None else 0

    def _parse_lookup_tables(self):
        voi_lut_sequence = self._dcm_obj.get("VOILUTSequence", None)
        if voi_lut_sequence is None:
            self.num_lut_seqs = 0
        else:
            self.num_lut_seqs = len(voi_lut_sequence)

    @property
    def available_luts(self):
        lut_funcs = []
        for i in range(self.num_lut_funcs):
            lut_funcs.append((VOI_LUT.function, i))

        lut_seqs = []
        for i in range(self.num_lut_seqs):
            lut_seqs.append((VOI_LUT.sequence, i))

        return lut_funcs + lut_seqs

    def set_lut(self, lut: Tuple):
        if lut[0] == VOI_LUT.function:
            self.set_lut_func(lut[1])
        elif lut[0] == VOI_LUT.sequence:
            self.set_lut_seq(lut[1])

    def set_random_lut(self):
        luts = self.available_luts

        rand_lut = random.choice(luts)
        self.set_lut(rand_lut)
