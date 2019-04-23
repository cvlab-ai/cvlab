import json
import os


names = {
    '2d features framework': 'Features 2D',
    '3d visualizer': '3D visualization',
    'autocalibration': '3D calibration & reconstruction',
    'background segmentation': 'Segmentation',
    'bitwise logical operations': 'Logical',
    'c api': 'API',
    'c api for video i/o': 'API',
    'c structures and operations': 'API',
    'camera calibration and 3d reconstruction': '3D calibration & reconstruction',
    'clustering': 'Clustering',
    'color space processing': 'Color',
    'computational photography': 'Photo',
    'connections with c++': 'API',
    'deep neural network module': 'Deep neural networks',
    'denoising': 'Filters',
    'drawing function of keypoints and matches': 'Drawing',
    'feature detection': 'Features 2D',
    'feature detection and description': 'Features 2D',
    'features2d_hal_interface': 'Features 2D',
    'fisheye camera model': '3D calibration & reconstruction',
    'global motion estimation': 'Motion detection',
    'hdr imaging': 'Photo',
    'high-level gui': 'GUI',
    'highgui_c': 'API',
    'highgui_winrt': 'GUI',
    'histogram calculation': 'Histogram',
    'hough transform': 'Features 2D',
    'image blenders': 'Blending',
    'image file reading and writing': 'Image IO',
    'image filtering': 'Filters',
    'image warping': 'Transforms',
    'images stitching': 'Stitching',
    'imgproc_c': 'API',
    'imgproc_color_conversions': 'Color',
    'imgproc_colormap': 'Color',
    'imgproc_draw': 'Drawing',
    'imgproc_feature': 'Features 2D',
    'imgproc_hist': 'Histogram',
    'imgproc_motion': 'Motion detection',
    'imgproc_object': 'Object detection',
    'imgproc_shape': 'Shape',
    'imgproc_transform': 'Transforms',
    'imgproc_misc': "Miscellaneous",
    'legacy support': 'API',
    'machine learning': 'Machine learning',
    'motion analysis': 'Motion detection',
    'object detection': 'Object detection',
    'object tracking': 'Object tracking',
    'operations on arrays': 'Miscellaneous',
    'optimization algorithms': 'Optimization',
    'photo_c': 'Photo',
    'photo_clone': 'Photo',
    'photo_render': 'Photo',
    'qt new functions': 'GUI',
    'rotation estimation': 'Stitching',
    'shape distance and matching': 'Shape',
    'super resolution': 'Super resolution',
    'utility and system functions and macros': 'Miscellaneous',
    'video encoding/decoding': 'Video IO',
    'video stabilization': 'Video stabilization',
}

base_groups = json.load(open(os.path.dirname(__file__)+"/base_groups.json"))

def get_groups():
    try:
        file = os.path.dirname(__file__) + "/typelist_groups.json"
        return json.load(open(file))
    except Exception:
        return {}


groups = get_groups()


def get_name(name):
    return names.get(name.lower(), name)


def get_group(name):
    name = name.lower()

    if name in base_groups:
        return base_groups[name]

    group = groups.get(name, None)
    if not group: return None
    if group.lower() in names:
        group = names[group.lower()]
    else:
        group = "OpenCV: " + group

    return group

