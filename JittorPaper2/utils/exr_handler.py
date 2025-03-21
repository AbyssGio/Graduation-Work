import Imath
import OpenEXR
import numpy as np
import pyexr
import cv2


def png_depth_loader(path_to_img):
    depth = cv2.imread(path_to_img, -1)
    depth = np.array(depth, dtype=np.float32)
    if len(depth.shape) == 3:
        return depth[:, :, 0]
    else:
        return depth


def exr_loader(exr_path, ndim=3):
    """Loads a .exr file as a numpy array
    Args:
        exr_path: path to the exr file
        ndim: number of channels that should be in returned array. Valid values are 1 and 3.
                        if ndim=1, only the 'R' channel is taken from exr file
                        if ndim=3, the 'R', 'G' and 'B' channels are taken from exr file.
                            The exr file must have 3 channels in this case.
    Returns:
        numpy.ndarray (dtype=np.float32): If ndim=1, shape is (height x width)
                                          If ndim=3, shape is (3 x height x width)
    """

    exr_file = OpenEXR.InputFile(exr_path)
    cm_dw = exr_file.header()['dataWindow']
    size = (cm_dw.max.x - cm_dw.min.x + 1, cm_dw.max.y - cm_dw.min.y + 1)

    pt = Imath.PixelType(Imath.PixelType.FLOAT)

    if ndim == 3:
        # read channels individually
        all_channel = []
        for c in ['R', 'G', 'B']:
            # transform data to numpy
            channel = np.frombuffer(exr_file.channel(c, pt), dtype=np.float32)
            channel.shape = (size[1], size[0])
            all_channel.append(channel)

        # create array and transpose dimensions to match tensor style
        exr_arr = np.array(all_channel).transpose((0, 1, 2))
        return exr_arr

    if ndim == 1:
        # transform data to numpy
        channel = np.frombuffer(exr_file.channel('R', pt), dtype=np.float32)
        channel.shape = (size[1], size[0])  # Numpy arrays are (row, col)
        exr_arr = np.array(channel)
        return exr_arr


def exr_writer(img_array, path_to_exr):
    """
    write depth to .exr file
    :param img_array: depth array
    :param path_to_exr: save path
    :return: None
    """
    assert len(img_array.shape) == 2
    exr_channel = ["R"]
    pyexr.write(path_to_exr, img_array, exr_channel, compression=pyexr.NO_COMPRESSION)


def png_writer(img_array, path_to_png):
    assert len(img_array.shape) == 2
    img_array = img_array.astype(np.uint16)
    cv2.imwrite(path_to_png, img_array)
