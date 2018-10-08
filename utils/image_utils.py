""" This module exposes the image processing methods as module level global
variables, that are loaded based on user defined flags.

Since there are multiple different implementations and libraries available for each
of the image tasks we want perform (e.g 'load jpeg' and 'resize'), we give the
user the choice of using different versions of each function without changing
the rest of the code. One example in which one would use a 'safe' flag is if
there are non jpeg images in the dataset, as imageio.imread can read them,
but not jpeg4py.

The basic changes introduced here are the libjpeg-turbo which is a instruction
level parallel (SSE, AVX, etc) implementation of the default C++ jpeg lib
libjpeg. Morover we also switched the default Pillow for a fork called PIL-SIMD,
which is identical in terms of the API but uses vectorized instructions as well
as heavy loop unrolling to optimize some of Pillows functions (particularly the
resize function). Since the PIL-SIMD is a drop-in replacement for Pillow, even
if the SIMD version is not installed the program should work just as fine with
the default Pillow (without the massive speed gains, though).
"""
import numpy as np
from scipy.misc import imresize
import PIL
from PIL import Image
try:
    import jpeg4py as jpeg
except (KeyboardInterrupt, EOFError):
    raise
except Exception as e:
    print('[IMAGE-UTILS] package jpeg4py not available. Continuing...')
    LIBJPEG_TURBO_PRESENT = False
else:
    LIBJPEG_TURBO_PRESENT = True


from utils.exceptions import InvalidOption

VALID_FLAGS = ['fast', 'safe']
PIL_FLAGS = {'bilinear': PIL.Image.BILINEAR, 'bicubic': PIL.Image.BICUBIC,
             'nearest': PIL.Image.NEAREST}


def decode_jpeg_fast(img_path):
    """ Jpeg decoding method implemented by jpeg4py, available in
    https://github.com/ajkxyz/jpeg4py . This library binds the libjpeg-turbo
    C++ library (available in
    https://github.com/libjpeg-turbo/libjpeg-turbo/blob/master/BUILDING.md),
    and can be up to 9 times faster than the non SIMD implementation.
    Requires libjpeg-turbo, built and installed correctly.
    """
    return jpeg.JPEG(img_path).decode()


def get_decode_jpeg_fcn(flag='fast'):
    """ Yields the function demanded by the user based on the flags given and
    the the system responses to imports. If the demanded function is not
    available an exception is raised and the user is informed it should try
    using another flag.
    """
    if flag == 'fast':
        assert LIBJPEG_TURBO_PRESENT, ('[IMAGE-UTILS] Error: It seems that the '
                                       'used image utils flag is not available,'
                                       ' try setting the flag to \'safe\'.')
        decode_jpeg_fcn = decode_jpeg_fast
    elif flag == 'safe':
        from imageio import imread
        decode_jpeg_fcn = imread
    else:
        raise InvalidOption('The informed flag: {}, is not valid. Valid flags '
                            "include: {}".format(flag, VALID_FLAGS['decode_jpeg']))

    return decode_jpeg_fcn


def resize_fast(img, size_tup, interp='bilinear'):
    """ Implements the PIL resize method from a numpy image input, using the
    same interface as the scipy imresize method.
    OBS: if concerned with the resizing of the correlation score map, the
    default behavior of the PIL resize is to align the corners, so we can
    simply use resize(img, (129,129)) without any problems. The center pixel
    is kept in the center (at least for the 4x upscale) and the corner pixels
    aligned.

    Args:
        img: (numpy.ndarray) A numpy RGB image.
        size_tup: (tuple) A 2D tuple containing the height and weight of the
            resized image.
        interp: (str) The flag indicating the interpolation method. Available
            methods include 'bilinear', 'bicubic' and 'nearest'.
    Returns:
        img_res: (numpy.ndarray) The resized image
    """
    # The order of the size tuple is inverted in PIL compared to scipy
    size_tup = (size_tup[1], size_tup[0])
    original_type = img.dtype
    img_res = Image.fromarray(img.astype('uint8', copy=False), 'RGB')
    img_res = img_res.resize(size_tup, PIL_FLAGS[interp])
    img_res = np.asarray(img_res)
    img_res = img_res.astype(original_type)
    return img_res


def get_resize_fcn(flag='fast'):
    """Yields the resize function demanded by the user based on the flags given
    and the the system responses to imports. If the demanded function is not
    available an exception is raised and the user is informed it should try
    using another flag.
    """
    if flag == 'fast':
        return resize_fast
    elif flag == 'safe':
        return imresize
    else:
        raise InvalidOption('The informed flag: {}, is not valid. Valid flags '
                            "include: {}".format(flag, VALID_FLAGS['resize']))
