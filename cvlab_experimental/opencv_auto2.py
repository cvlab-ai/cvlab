# Source generated with cvlab/tools/generate_opencv.py
# See: https://github.com/cvlab-ai/cvlab
   
import cv2
from cvlab.diagram.elements.base import *

### GaussianBlur ###

class OpenCVAuto2_GaussianBlur(NormalElement):
    name = 'GaussianBlur'
    comment = '''GaussianBlur(src, ksize, sigmaX[, dst[, sigmaY[, borderType]]]) -> dst\n@brief Blurs an image using a Gaussian filter.\n\nThe function convolves the source image with the specified Gaussian kernel. In-place filtering is\nsupported.\n\n@param src input image; the image can have any number of channels, which are processed\nindependently, but the depth should be CV_8U, CV_16U, CV_16S, CV_32F or CV_64F.\n@param dst output image of the same size and type as src.\n@param ksize Gaussian kernel size. ksize.width and ksize.height can differ but they both must be\npositive and odd. Or, they can be zero's and then they are computed from sigma.\n@param sigmaX Gaussian kernel standard deviation in X direction.\n@param sigmaY Gaussian kernel standard deviation in Y direction; if sigmaY is zero, it is set to be\nequal to sigmaX, if both sigmas are zeros, they are computed from ksize.width and ksize.height,\nrespectively (see #getGaussianKernel for details); to fully control the result regardless of\npossible future modifications of all this semantics, it is recommended to specify all of ksize,\nsigmaX, and sigmaY.\n@param borderType pixel extrapolation method, see #BorderTypes\n\n@sa  sepFilter2D, filter2D, blur, boxFilter, bilateralFilter, medianBlur'''

    def get_attributes(self):
        return [Input('src', 'src')], \
               [Output('dst', 'dst')], \
               [SizeParameter('ksize', 'ksize'),
                FloatParameter('sigmaX', 'sigmaX'),
                FloatParameter('sigmaY', 'sigmaY'),
                ComboboxParameter('borderType', [('BORDER_CONSTANT',0),('BORDER_DEFAULT',4),('BORDER_ISOLATED',16),('BORDER_REFLECT',2),('BORDER_REFLECT101',4),('BORDER_REFLECT_101',4),('BORDER_REPLICATE',1),('BORDER_TRANSPARENT',5),('BORDER_WRAP',3)])]

    def process_inputs(self, inputs, outputs, parameters):
        src = inputs['src'].value
        ksize = parameters['ksize']
        sigmaX = parameters['sigmaX']
        sigmaY = parameters['sigmaY']
        borderType = parameters['borderType']
        dst = cv2.GaussianBlur(src=src, ksize=ksize, sigmaX=sigmaX, sigmaY=sigmaY, borderType=borderType)
        outputs['dst'] = Data(dst)

### HuMoments ###

class OpenCVAuto2_HuMoments(NormalElement):
    name = 'HuMoments'
    comment = '''HuMoments(m[, hu]) -> hu\n@overload'''

    def get_attributes(self):
        return [Input('m', 'm')], \
               [Output('hu', 'hu')], \
               []

    def process_inputs(self, inputs, outputs, parameters):
        m = inputs['m'].value
        hu = cv2.HuMoments(m=m)
        outputs['hu'] = Data(hu)

### KeyPoint_convert ###

class OpenCVAuto2_KeyPoint_convert(NormalElement):
    name = 'KeyPoint_convert'
    comment = '''KeyPoint_convert(keypoints[, keypointIndexes]) -> points2f\nThis method converts vector of keypoints to vector of points or the reverse, where each keypoint is\nassigned the same size and the same orientation.\n\n@param keypoints Keypoints obtained from any feature detection algorithm like SIFT/SURF/ORB\n@param points2f Array of (x,y) coordinates of each keypoint\n@param keypointIndexes Array of indexes of keypoints to be converted to points. (Acts like a mask to\nconvert only specified keypoints)



KeyPoint_convert(points2f[, size[, response[, octave[, class_id]]]]) -> keypoints\n@overload\n@param points2f Array of (x,y) coordinates of each keypoint\n@param keypoints Keypoints obtained from any feature detection algorithm like SIFT/SURF/ORB\n@param size keypoint diameter\n@param response keypoint detector response on the keypoint (that is, strength of the keypoint)\n@param octave pyramid octave in which the keypoint has been detected\n@param class_id object id'''

    def get_attributes(self):
        return [Input('keypoints', 'keypoints')], \
               [Output('points2f', 'points2f')], \
               []

    def process_inputs(self, inputs, outputs, parameters):
        keypoints = inputs['keypoints'].value
        points2f = cv2.KeyPoint_convert(keypoints=keypoints)
        outputs['points2f'] = Data(points2f)

### Laplacian ###

class OpenCVAuto2_Laplacian(NormalElement):
    name = 'Laplacian'
    comment = '''Laplacian(src, ddepth[, dst[, ksize[, scale[, delta[, borderType]]]]]) -> dst\n@brief Calculates the Laplacian of an image.\n\nThe function calculates the Laplacian of the source image by adding up the second x and y\nderivatives calculated using the Sobel operator:\n\n\f[\texttt{dst} =  \Delta \texttt{src} =  \frac{\partial^2 \texttt{src}}{\partial x^2} +  \frac{\partial^2 \texttt{src}}{\partial y^2}\f]\n\nThis is done when `ksize > 1`. When `ksize == 1`, the Laplacian is computed by filtering the image\nwith the following \f$3 \times 3\f$ aperture:\n\n\f[\vecthreethree {0}{1}{0}{1}{-4}{1}{0}{1}{0}\f]\n\n@param src Source image.\n@param dst Destination image of the same size and the same number of channels as src .\n@param ddepth Desired depth of the destination image.\n@param ksize Aperture size used to compute the second-derivative filters. See #getDerivKernels for\ndetails. The size must be positive and odd.\n@param scale Optional scale factor for the computed Laplacian values. By default, no scaling is\napplied. See #getDerivKernels for details.\n@param delta Optional delta value that is added to the results prior to storing them in dst .\n@param borderType Pixel extrapolation method, see #BorderTypes\n@sa  Sobel, Scharr'''

    def get_attributes(self):
        return [Input('src', 'src')], \
               [Output('dst', 'dst')], \
               [ComboboxParameter('ddepth', [('NONE',0),('CV_8U',0),('CV_8UC1',0),('CV_8UC2',8),('CV_8UC3',16),('CV_8UC4',24),('CV_8S',1),('CV_8SC1',1),('CV_8SC2',9),('CV_8SC3',17),('CV_8SC4',25),('CV_16U',2),('CV_16UC1',2),('CV_16UC2',10),('CV_16UC3',18),('CV_16UC4',26),('CV_16S',3),('CV_16SC1',3),('CV_16SC2',11),('CV_16SC3',19),('CV_16SC4',27),('CV_32S',4),('CV_32SC1',4),('CV_32SC2',12),('CV_32SC3',20),('CV_32SC4',28),('CV_32F',5),('CV_32FC1',5),('CV_32FC2',13),('CV_32FC3',21),('CV_32FC4',29),('CV_64F',6),('CV_64FC1',6),('CV_64FC2',14),('CV_64FC3',22),('CV_64FC4',30)]),
                SizeParameter('ksize', 'ksize'),
                FloatParameter('scale', 'scale'),
                ComboboxParameter('borderType', [('BORDER_CONSTANT',0),('BORDER_DEFAULT',4),('BORDER_ISOLATED',16),('BORDER_REFLECT',2),('BORDER_REFLECT101',4),('BORDER_REFLECT_101',4),('BORDER_REPLICATE',1),('BORDER_TRANSPARENT',5),('BORDER_WRAP',3)])]

    def process_inputs(self, inputs, outputs, parameters):
        src = inputs['src'].value
        ddepth = parameters['ddepth']
        ksize = parameters['ksize']
        scale = parameters['scale']
        borderType = parameters['borderType']
        dst = cv2.Laplacian(src=src, ddepth=ddepth, ksize=ksize, scale=scale, borderType=borderType)
        outputs['dst'] = Data(dst)

### RQDecomp3x3 ###

class OpenCVAuto2_RQDecomp3x3(NormalElement):
    name = 'RQDecomp3x3'
    comment = '''RQDecomp3x3(src[, mtxR[, mtxQ[, Qx[, Qy[, Qz]]]]]) -> retval, mtxR, mtxQ, Qx, Qy, Qz\n@brief Computes an RQ decomposition of 3x3 matrices.\n\n@param src 3x3 input matrix.\n@param mtxR Output 3x3 upper-triangular matrix.\n@param mtxQ Output 3x3 orthogonal matrix.\n@param Qx Optional output 3x3 rotation matrix around x-axis.\n@param Qy Optional output 3x3 rotation matrix around y-axis.\n@param Qz Optional output 3x3 rotation matrix around z-axis.\n\nThe function computes a RQ decomposition using the given rotations. This function is used in\ndecomposeProjectionMatrix to decompose the left 3x3 submatrix of a projection matrix into a camera\nand a rotation matrix.\n\nIt optionally returns three rotation matrices, one for each axis, and the three Euler angles in\ndegrees (as the return value) that could be used in OpenGL. Note, there is always more than one\nsequence of rotations about the three principal axes that results in the same orientation of an\nobject, e.g. see @cite Slabaugh . Returned tree rotation matrices and corresponding three Euler angles\nare only one of the possible solutions.'''

    def get_attributes(self):
        return [Input('src', 'src')], \
               [Output('mtxR', 'mtxR'),
                Output('mtxQ', 'mtxQ'),
                Output('Qx', 'Qx'),
                Output('Qy', 'Qy'),
                Output('Qz', 'Qz')], \
               []

    def process_inputs(self, inputs, outputs, parameters):
        src = inputs['src'].value
        retval, mtxR, mtxQ, Qx, Qy, Qz = cv2.RQDecomp3x3(src=src)
        outputs['mtxR'] = Data(mtxR)
        outputs['mtxQ'] = Data(mtxQ)
        outputs['Qx'] = Data(Qx)
        outputs['Qy'] = Data(Qy)
        outputs['Qz'] = Data(Qz)

### Rodrigues ###

class OpenCVAuto2_Rodrigues(NormalElement):
    name = 'Rodrigues'
    comment = '''Rodrigues(src[, dst[, jacobian]]) -> dst, jacobian\n@brief Converts a rotation matrix to a rotation vector or vice versa.\n\n@param src Input rotation vector (3x1 or 1x3) or rotation matrix (3x3).\n@param dst Output rotation matrix (3x3) or rotation vector (3x1 or 1x3), respectively.\n@param jacobian Optional output Jacobian matrix, 3x9 or 9x3, which is a matrix of partial\nderivatives of the output array components with respect to the input array components.\n\n\f[\begin{array}{l} \theta \leftarrow norm(r) \\ r  \leftarrow r/ \theta \\ R =  \cos{\theta} I + (1- \cos{\theta} ) r r^T +  \sin{\theta} \vecthreethree{0}{-r_z}{r_y}{r_z}{0}{-r_x}{-r_y}{r_x}{0} \end{array}\f]\n\nInverse transformation can be also done easily, since\n\n\f[\sin ( \theta ) \vecthreethree{0}{-r_z}{r_y}{r_z}{0}{-r_x}{-r_y}{r_x}{0} = \frac{R - R^T}{2}\f]\n\nA rotation vector is a convenient and most compact representation of a rotation matrix (since any\nrotation matrix has just 3 degrees of freedom). The representation is used in the global 3D geometry\noptimization procedures like calibrateCamera, stereoCalibrate, or solvePnP .'''

    def get_attributes(self):
        return [Input('src', 'src')], \
               [Output('dst', 'dst'),
                Output('jacobian', 'jacobian')], \
               []

    def process_inputs(self, inputs, outputs, parameters):
        src = inputs['src'].value
        dst, jacobian = cv2.Rodrigues(src=src)
        outputs['dst'] = Data(dst)
        outputs['jacobian'] = Data(jacobian)

### Scharr ###

class OpenCVAuto2_Scharr(NormalElement):
    name = 'Scharr'
    comment = '''Scharr(src, ddepth, dx, dy[, dst[, scale[, delta[, borderType]]]]) -> dst\n@brief Calculates the first x- or y- image derivative using Scharr operator.\n\nThe function computes the first x- or y- spatial image derivative using the Scharr operator. The\ncall\n\n\f[\texttt{Scharr(src, dst, ddepth, dx, dy, scale, delta, borderType)}\f]\n\nis equivalent to\n\n\f[\texttt{Sobel(src, dst, ddepth, dx, dy, CV_SCHARR, scale, delta, borderType)} .\f]\n\n@param src input image.\n@param dst output image of the same size and the same number of channels as src.\n@param ddepth output image depth, see @ref filter_depths "combinations"\n@param dx order of the derivative x.\n@param dy order of the derivative y.\n@param scale optional scale factor for the computed derivative values; by default, no scaling is\napplied (see #getDerivKernels for details).\n@param delta optional delta value that is added to the results prior to storing them in dst.\n@param borderType pixel extrapolation method, see #BorderTypes\n@sa  cartToPolar'''

    def get_attributes(self):
        return [Input('src', 'src')], \
               [Output('dst', 'dst')], \
               [ComboboxParameter('ddepth', [('NONE',0),('CV_8U',0),('CV_8UC1',0),('CV_8UC2',8),('CV_8UC3',16),('CV_8UC4',24),('CV_8S',1),('CV_8SC1',1),('CV_8SC2',9),('CV_8SC3',17),('CV_8SC4',25),('CV_16U',2),('CV_16UC1',2),('CV_16UC2',10),('CV_16UC3',18),('CV_16UC4',26),('CV_16S',3),('CV_16SC1',3),('CV_16SC2',11),('CV_16SC3',19),('CV_16SC4',27),('CV_32S',4),('CV_32SC1',4),('CV_32SC2',12),('CV_32SC3',20),('CV_32SC4',28),('CV_32F',5),('CV_32FC1',5),('CV_32FC2',13),('CV_32FC3',21),('CV_32FC4',29),('CV_64F',6),('CV_64FC1',6),('CV_64FC2',14),('CV_64FC3',22),('CV_64FC4',30)]),
                IntParameter('dx', 'dx'),
                IntParameter('dy', 'dy'),
                FloatParameter('scale', 'scale'),
                ComboboxParameter('borderType', [('BORDER_CONSTANT',0),('BORDER_DEFAULT',4),('BORDER_ISOLATED',16),('BORDER_REFLECT',2),('BORDER_REFLECT101',4),('BORDER_REFLECT_101',4),('BORDER_REPLICATE',1),('BORDER_TRANSPARENT',5),('BORDER_WRAP',3)])]

    def process_inputs(self, inputs, outputs, parameters):
        src = inputs['src'].value
        ddepth = parameters['ddepth']
        dx = parameters['dx']
        dy = parameters['dy']
        scale = parameters['scale']
        borderType = parameters['borderType']
        dst = cv2.Scharr(src=src, ddepth=ddepth, dx=dx, dy=dy, scale=scale, borderType=borderType)
        outputs['dst'] = Data(dst)

### Sobel ###

class OpenCVAuto2_Sobel(NormalElement):
    name = 'Sobel'
    comment = '''Sobel(src, ddepth, dx, dy[, dst[, ksize[, scale[, delta[, borderType]]]]]) -> dst\n@brief Calculates the first, second, third, or mixed image derivatives using an extended Sobel operator.\n\nIn all cases except one, the \f$\texttt{ksize} \times \texttt{ksize}\f$ separable kernel is used to\ncalculate the derivative. When \f$\texttt{ksize = 1}\f$, the \f$3 \times 1\f$ or \f$1 \times 3\f$\nkernel is used (that is, no Gaussian smoothing is done). `ksize = 1` can only be used for the first\nor the second x- or y- derivatives.\n\nThere is also the special value `ksize = #CV_SCHARR (-1)` that corresponds to the \f$3\times3\f$ Scharr\nfilter that may give more accurate results than the \f$3\times3\f$ Sobel. The Scharr aperture is\n\n\f[\vecthreethree{-3}{0}{3}{-10}{0}{10}{-3}{0}{3}\f]\n\nfor the x-derivative, or transposed for the y-derivative.\n\nThe function calculates an image derivative by convolving the image with the appropriate kernel:\n\n\f[\texttt{dst} =  \frac{\partial^{xorder+yorder} \texttt{src}}{\partial x^{xorder} \partial y^{yorder}}\f]\n\nThe Sobel operators combine Gaussian smoothing and differentiation, so the result is more or less\nresistant to the noise. Most often, the function is called with ( xorder = 1, yorder = 0, ksize = 3)\nor ( xorder = 0, yorder = 1, ksize = 3) to calculate the first x- or y- image derivative. The first\ncase corresponds to a kernel of:\n\n\f[\vecthreethree{-1}{0}{1}{-2}{0}{2}{-1}{0}{1}\f]\n\nThe second case corresponds to a kernel of:\n\n\f[\vecthreethree{-1}{-2}{-1}{0}{0}{0}{1}{2}{1}\f]\n\n@param src input image.\n@param dst output image of the same size and the same number of channels as src .\n@param ddepth output image depth, see @ref filter_depths "combinations"; in the case of\n8-bit input images it will result in truncated derivatives.\n@param dx order of the derivative x.\n@param dy order of the derivative y.\n@param ksize size of the extended Sobel kernel; it must be 1, 3, 5, or 7.\n@param scale optional scale factor for the computed derivative values; by default, no scaling is\napplied (see #getDerivKernels for details).\n@param delta optional delta value that is added to the results prior to storing them in dst.\n@param borderType pixel extrapolation method, see #BorderTypes\n@sa  Scharr, Laplacian, sepFilter2D, filter2D, GaussianBlur, cartToPolar'''

    def get_attributes(self):
        return [Input('src', 'src')], \
               [Output('dst', 'dst')], \
               [ComboboxParameter('ddepth', [('NONE',0),('CV_8U',0),('CV_8UC1',0),('CV_8UC2',8),('CV_8UC3',16),('CV_8UC4',24),('CV_8S',1),('CV_8SC1',1),('CV_8SC2',9),('CV_8SC3',17),('CV_8SC4',25),('CV_16U',2),('CV_16UC1',2),('CV_16UC2',10),('CV_16UC3',18),('CV_16UC4',26),('CV_16S',3),('CV_16SC1',3),('CV_16SC2',11),('CV_16SC3',19),('CV_16SC4',27),('CV_32S',4),('CV_32SC1',4),('CV_32SC2',12),('CV_32SC3',20),('CV_32SC4',28),('CV_32F',5),('CV_32FC1',5),('CV_32FC2',13),('CV_32FC3',21),('CV_32FC4',29),('CV_64F',6),('CV_64FC1',6),('CV_64FC2',14),('CV_64FC3',22),('CV_64FC4',30)]),
                IntParameter('dx', 'dx'),
                IntParameter('dy', 'dy'),
                SizeParameter('ksize', 'ksize'),
                FloatParameter('scale', 'scale'),
                ComboboxParameter('borderType', [('BORDER_CONSTANT',0),('BORDER_DEFAULT',4),('BORDER_ISOLATED',16),('BORDER_REFLECT',2),('BORDER_REFLECT101',4),('BORDER_REFLECT_101',4),('BORDER_REPLICATE',1),('BORDER_TRANSPARENT',5),('BORDER_WRAP',3)])]

    def process_inputs(self, inputs, outputs, parameters):
        src = inputs['src'].value
        ddepth = parameters['ddepth']
        dx = parameters['dx']
        dy = parameters['dy']
        ksize = parameters['ksize']
        scale = parameters['scale']
        borderType = parameters['borderType']
        dst = cv2.Sobel(src=src, ddepth=ddepth, dx=dx, dy=dy, ksize=ksize, scale=scale, borderType=borderType)
        outputs['dst'] = Data(dst)

### absdiff ###

class OpenCVAuto2_Absdiff(NormalElement):
    name = 'Absdiff'
    comment = '''absdiff(src1, src2[, dst]) -> dst\n@brief Calculates the per-element absolute difference between two arrays or between an array and a scalar.\n\nThe function cv::absdiff calculates:\n*   Absolute difference between two arrays when they have the same\nsize and type:\n\f[\texttt{dst}(I) =  \texttt{saturate} (| \texttt{src1}(I) -  \texttt{src2}(I)|)\f]\n*   Absolute difference between an array and a scalar when the second\narray is constructed from Scalar or has as many elements as the\nnumber of channels in `src1`:\n\f[\texttt{dst}(I) =  \texttt{saturate} (| \texttt{src1}(I) -  \texttt{src2} |)\f]\n*   Absolute difference between a scalar and an array when the first\narray is constructed from Scalar or has as many elements as the\nnumber of channels in `src2`:\n\f[\texttt{dst}(I) =  \texttt{saturate} (| \texttt{src1} -  \texttt{src2}(I) |)\f]\nwhere I is a multi-dimensional index of array elements. In case of\nmulti-channel arrays, each channel is processed independently.\n@note Saturation is not applied when the arrays have the depth CV_32S.\nYou may even get a negative value in the case of overflow.\n@param src1 first input array or a scalar.\n@param src2 second input array or a scalar.\n@param dst output array that has the same size and type as input arrays.\n@sa cv::abs(const Mat&)'''

    def get_attributes(self):
        return [Input('src1', 'src1'),
                Input('src2', 'src2')], \
               [Output('dst', 'dst')], \
               []

    def process_inputs(self, inputs, outputs, parameters):
        src1 = inputs['src1'].value
        src2 = inputs['src2'].value
        dst = cv2.absdiff(src1=src1, src2=src2)
        outputs['dst'] = Data(dst)

### accumulate ###

class OpenCVAuto2_Accumulate(NormalElement):
    name = 'Accumulate'
    comment = '''accumulate(src, dst[, mask]) -> dst\n@brief Adds an image to the accumulator image.\n\nThe function adds src or some of its elements to dst :\n\n\f[\texttt{dst} (x,y)  \leftarrow \texttt{dst} (x,y) +  \texttt{src} (x,y)  \quad \text{if} \quad \texttt{mask} (x,y)  \ne 0\f]\n\nThe function supports multi-channel images. Each channel is processed independently.\n\nThe function cv::accumulate can be used, for example, to collect statistics of a scene background\nviewed by a still camera and for the further foreground-background segmentation.\n\n@param src Input image of type CV_8UC(n), CV_16UC(n), CV_32FC(n) or CV_64FC(n), where n is a positive integer.\n@param dst %Accumulator image with the same number of channels as input image, and a depth of CV_32F or CV_64F.\n@param mask Optional operation mask.\n\n@sa  accumulateSquare, accumulateProduct, accumulateWeighted'''

    def get_attributes(self):
        return [Input('src', 'src'),
                Input('dst', 'dst'),
                Input('mask', 'mask', optional=True)], \
               [Output('dst', 'dst')], \
               []

    def process_inputs(self, inputs, outputs, parameters):
        src = inputs['src'].value
        dst = inputs['dst'].value
        mask = inputs['mask'].value
        dst = cv2.accumulate(src=src, dst=dst, mask=mask)
        outputs['dst'] = Data(dst)

### accumulateProduct ###

class OpenCVAuto2_AccumulateProduct(NormalElement):
    name = 'AccumulateProduct'
    comment = '''accumulateProduct(src1, src2, dst[, mask]) -> dst\n@brief Adds the per-element product of two input images to the accumulator image.\n\nThe function adds the product of two images or their selected regions to the accumulator dst :\n\n\f[\texttt{dst} (x,y)  \leftarrow \texttt{dst} (x,y) +  \texttt{src1} (x,y)  \cdot \texttt{src2} (x,y)  \quad \text{if} \quad \texttt{mask} (x,y)  \ne 0\f]\n\nThe function supports multi-channel images. Each channel is processed independently.\n\n@param src1 First input image, 1- or 3-channel, 8-bit or 32-bit floating point.\n@param src2 Second input image of the same type and the same size as src1 .\n@param dst %Accumulator image with the same number of channels as input images, 32-bit or 64-bit\nfloating-point.\n@param mask Optional operation mask.\n\n@sa  accumulate, accumulateSquare, accumulateWeighted'''

    def get_attributes(self):
        return [Input('src1', 'src1'),
                Input('src2', 'src2'),
                Input('dst', 'dst'),
                Input('mask', 'mask', optional=True)], \
               [Output('dst', 'dst')], \
               []

    def process_inputs(self, inputs, outputs, parameters):
        src1 = inputs['src1'].value
        src2 = inputs['src2'].value
        dst = inputs['dst'].value
        mask = inputs['mask'].value
        dst = cv2.accumulateProduct(src1=src1, src2=src2, dst=dst, mask=mask)
        outputs['dst'] = Data(dst)

### accumulateSquare ###

class OpenCVAuto2_AccumulateSquare(NormalElement):
    name = 'AccumulateSquare'
    comment = '''accumulateSquare(src, dst[, mask]) -> dst\n@brief Adds the square of a source image to the accumulator image.\n\nThe function adds the input image src or its selected region, raised to a power of 2, to the\naccumulator dst :\n\n\f[\texttt{dst} (x,y)  \leftarrow \texttt{dst} (x,y) +  \texttt{src} (x,y)^2  \quad \text{if} \quad \texttt{mask} (x,y)  \ne 0\f]\n\nThe function supports multi-channel images. Each channel is processed independently.\n\n@param src Input image as 1- or 3-channel, 8-bit or 32-bit floating point.\n@param dst %Accumulator image with the same number of channels as input image, 32-bit or 64-bit\nfloating-point.\n@param mask Optional operation mask.\n\n@sa  accumulateSquare, accumulateProduct, accumulateWeighted'''

    def get_attributes(self):
        return [Input('src', 'src'),
                Input('dst', 'dst'),
                Input('mask', 'mask', optional=True)], \
               [Output('dst', 'dst')], \
               []

    def process_inputs(self, inputs, outputs, parameters):
        src = inputs['src'].value
        dst = inputs['dst'].value
        mask = inputs['mask'].value
        dst = cv2.accumulateSquare(src=src, dst=dst, mask=mask)
        outputs['dst'] = Data(dst)

### accumulateWeighted ###

class OpenCVAuto2_AccumulateWeighted(NormalElement):
    name = 'AccumulateWeighted'
    comment = '''accumulateWeighted(src, dst, alpha[, mask]) -> dst\n@brief Updates a running average.\n\nThe function calculates the weighted sum of the input image src and the accumulator dst so that dst\nbecomes a running average of a frame sequence:\n\n\f[\texttt{dst} (x,y)  \leftarrow (1- \texttt{alpha} )  \cdot \texttt{dst} (x,y) +  \texttt{alpha} \cdot \texttt{src} (x,y)  \quad \text{if} \quad \texttt{mask} (x,y)  \ne 0\f]\n\nThat is, alpha regulates the update speed (how fast the accumulator "forgets" about earlier images).\nThe function supports multi-channel images. Each channel is processed independently.\n\n@param src Input image as 1- or 3-channel, 8-bit or 32-bit floating point.\n@param dst %Accumulator image with the same number of channels as input image, 32-bit or 64-bit\nfloating-point.\n@param alpha Weight of the input image.\n@param mask Optional operation mask.\n\n@sa  accumulate, accumulateSquare, accumulateProduct'''

    def get_attributes(self):
        return [Input('src', 'src'),
                Input('dst', 'dst'),
                Input('mask', 'mask', optional=True)], \
               [Output('dst', 'dst')], \
               [FloatParameter('alpha', 'alpha')]

    def process_inputs(self, inputs, outputs, parameters):
        src = inputs['src'].value
        dst = inputs['dst'].value
        mask = inputs['mask'].value
        alpha = parameters['alpha']
        dst = cv2.accumulateWeighted(src=src, dst=dst, mask=mask, alpha=alpha)
        outputs['dst'] = Data(dst)

### add ###

class OpenCVAuto2_Add(NormalElement):
    name = 'Add'
    comment = '''add(src1, src2[, dst[, mask[, dtype]]]) -> dst\n@brief Calculates the per-element sum of two arrays or an array and a scalar.\n\nThe function add calculates:\n- Sum of two arrays when both input arrays have the same size and the same number of channels:\n\f[\texttt{dst}(I) =  \texttt{saturate} ( \texttt{src1}(I) +  \texttt{src2}(I)) \quad \texttt{if mask}(I) \ne0\f]\n- Sum of an array and a scalar when src2 is constructed from Scalar or has the same number of\nelements as `src1.channels()`:\n\f[\texttt{dst}(I) =  \texttt{saturate} ( \texttt{src1}(I) +  \texttt{src2} ) \quad \texttt{if mask}(I) \ne0\f]\n- Sum of a scalar and an array when src1 is constructed from Scalar or has the same number of\nelements as `src2.channels()`:\n\f[\texttt{dst}(I) =  \texttt{saturate} ( \texttt{src1} +  \texttt{src2}(I) ) \quad \texttt{if mask}(I) \ne0\f]\nwhere `I` is a multi-dimensional index of array elements. In case of multi-channel arrays, each\nchannel is processed independently.\n\nThe first function in the list above can be replaced with matrix expressions:\n@code{.cpp}\ndst = src1 + src2;\ndst += src1; // equivalent to add(dst, src1, dst);\n@endcode\nThe input arrays and the output array can all have the same or different depths. For example, you\ncan add a 16-bit unsigned array to a 8-bit signed array and store the sum as a 32-bit\nfloating-point array. Depth of the output array is determined by the dtype parameter. In the second\nand third cases above, as well as in the first case, when src1.depth() == src2.depth(), dtype can\nbe set to the default -1. In this case, the output array will have the same depth as the input\narray, be it src1, src2 or both.\n@note Saturation is not applied when the output array has the depth CV_32S. You may even get\nresult of an incorrect sign in the case of overflow.\n@param src1 first input array or a scalar.\n@param src2 second input array or a scalar.\n@param dst output array that has the same size and number of channels as the input array(s); the\ndepth is defined by dtype or src1/src2.\n@param mask optional operation mask - 8-bit single channel array, that specifies elements of the\noutput array to be changed.\n@param dtype optional depth of the output array (see the discussion below).\n@sa subtract, addWeighted, scaleAdd, Mat::convertTo'''

    def get_attributes(self):
        return [Input('src1', 'src1'),
                Input('src2', 'src2'),
                Input('mask', 'mask', optional=True)], \
               [Output('dst', 'dst')], \
               [ComboboxParameter('dtype', [('NONE',0),('CV_8U',0),('CV_8UC1',0),('CV_8UC2',8),('CV_8UC3',16),('CV_8UC4',24),('CV_8S',1),('CV_8SC1',1),('CV_8SC2',9),('CV_8SC3',17),('CV_8SC4',25),('CV_16U',2),('CV_16UC1',2),('CV_16UC2',10),('CV_16UC3',18),('CV_16UC4',26),('CV_16S',3),('CV_16SC1',3),('CV_16SC2',11),('CV_16SC3',19),('CV_16SC4',27),('CV_32S',4),('CV_32SC1',4),('CV_32SC2',12),('CV_32SC3',20),('CV_32SC4',28),('CV_32F',5),('CV_32FC1',5),('CV_32FC2',13),('CV_32FC3',21),('CV_32FC4',29),('CV_64F',6),('CV_64FC1',6),('CV_64FC2',14),('CV_64FC3',22),('CV_64FC4',30)])]

    def process_inputs(self, inputs, outputs, parameters):
        src1 = inputs['src1'].value
        src2 = inputs['src2'].value
        mask = inputs['mask'].value
        dtype = parameters['dtype']
        dst = cv2.add(src1=src1, src2=src2, mask=mask, dtype=dtype)
        outputs['dst'] = Data(dst)

### addWeighted ###

class OpenCVAuto2_AddWeighted(NormalElement):
    name = 'AddWeighted'
    comment = '''addWeighted(src1, alpha, src2, beta, gamma[, dst[, dtype]]) -> dst\n@brief Calculates the weighted sum of two arrays.\n\nThe function addWeighted calculates the weighted sum of two arrays as follows:\n\f[\texttt{dst} (I)= \texttt{saturate} ( \texttt{src1} (I)* \texttt{alpha} +  \texttt{src2} (I)* \texttt{beta} +  \texttt{gamma} )\f]\nwhere I is a multi-dimensional index of array elements. In case of multi-channel arrays, each\nchannel is processed independently.\nThe function can be replaced with a matrix expression:\n@code{.cpp}\ndst = src1*alpha + src2*beta + gamma;\n@endcode\n@note Saturation is not applied when the output array has the depth CV_32S. You may even get\nresult of an incorrect sign in the case of overflow.\n@param src1 first input array.\n@param alpha weight of the first array elements.\n@param src2 second input array of the same size and channel number as src1.\n@param beta weight of the second array elements.\n@param gamma scalar added to each sum.\n@param dst output array that has the same size and number of channels as the input arrays.\n@param dtype optional depth of the output array; when both input arrays have the same depth, dtype\ncan be set to -1, which will be equivalent to src1.depth().\n@sa  add, subtract, scaleAdd, Mat::convertTo'''

    def get_attributes(self):
        return [Input('src1', 'src1'),
                Input('src2', 'src2')], \
               [Output('dst', 'dst')], \
               [FloatParameter('alpha', 'alpha'),
                FloatParameter('beta', 'beta'),
                FloatParameter('gamma', 'gamma'),
                ComboboxParameter('dtype', [('NONE',0),('CV_8U',0),('CV_8UC1',0),('CV_8UC2',8),('CV_8UC3',16),('CV_8UC4',24),('CV_8S',1),('CV_8SC1',1),('CV_8SC2',9),('CV_8SC3',17),('CV_8SC4',25),('CV_16U',2),('CV_16UC1',2),('CV_16UC2',10),('CV_16UC3',18),('CV_16UC4',26),('CV_16S',3),('CV_16SC1',3),('CV_16SC2',11),('CV_16SC3',19),('CV_16SC4',27),('CV_32S',4),('CV_32SC1',4),('CV_32SC2',12),('CV_32SC3',20),('CV_32SC4',28),('CV_32F',5),('CV_32FC1',5),('CV_32FC2',13),('CV_32FC3',21),('CV_32FC4',29),('CV_64F',6),('CV_64FC1',6),('CV_64FC2',14),('CV_64FC3',22),('CV_64FC4',30)])]

    def process_inputs(self, inputs, outputs, parameters):
        src1 = inputs['src1'].value
        src2 = inputs['src2'].value
        alpha = parameters['alpha']
        beta = parameters['beta']
        gamma = parameters['gamma']
        dtype = parameters['dtype']
        dst = cv2.addWeighted(src1=src1, src2=src2, alpha=alpha, beta=beta, gamma=gamma, dtype=dtype)
        outputs['dst'] = Data(dst)

### arrowedLine ###

class OpenCVAuto2_ArrowedLine(NormalElement):
    name = 'ArrowedLine'
    comment = '''arrowedLine(img, pt1, pt2, color[, thickness[, line_type[, shift[, tipLength]]]]) -> img\n@brief Draws a arrow segment pointing from the first point to the second one.\n\nThe function cv::arrowedLine draws an arrow between pt1 and pt2 points in the image. See also #line.\n\n@param img Image.\n@param pt1 The point the arrow starts from.\n@param pt2 The point the arrow points to.\n@param color Line color.\n@param thickness Line thickness.\n@param line_type Type of the line. See #LineTypes\n@param shift Number of fractional bits in the point coordinates.\n@param tipLength The length of the arrow tip in relation to the arrow length'''

    def get_attributes(self):
        return [Input('img', 'img')], \
               [Output('img', 'img')], \
               [PointParameter('pt1', 'pt1'),
                PointParameter('pt2', 'pt2'),
                ScalarParameter('color', 'color'),
                IntParameter('thickness', 'thickness')]

    def process_inputs(self, inputs, outputs, parameters):
        img = inputs['img'].value
        pt1 = parameters['pt1']
        pt2 = parameters['pt2']
        color = parameters['color']
        thickness = parameters['thickness']
        img = cv2.arrowedLine(img=img, pt1=pt1, pt2=pt2, color=color, thickness=thickness)
        outputs['img'] = Data(img)

### batchDistance ###

class OpenCVAuto2_BatchDistance(NormalElement):
    name = 'BatchDistance'
    comment = '''batchDistance(src1, src2, dtype[, dist[, nidx[, normType[, K[, mask[, update[, crosscheck]]]]]]]) -> dist, nidx\n@brief naive nearest neighbor finder\n\nsee http://en.wikipedia.org/wiki/Nearest_neighbor_search\n@todo document'''

    def get_attributes(self):
        return [Input('src1', 'src1'),
                Input('src2', 'src2'),
                Input('mask', 'mask', optional=True)], \
               [Output('dist', 'dist'),
                Output('nidx', 'nidx')], \
               [ComboboxParameter('dtype', [('NONE',0),('CV_8U',0),('CV_8UC1',0),('CV_8UC2',8),('CV_8UC3',16),('CV_8UC4',24),('CV_8S',1),('CV_8SC1',1),('CV_8SC2',9),('CV_8SC3',17),('CV_8SC4',25),('CV_16U',2),('CV_16UC1',2),('CV_16UC2',10),('CV_16UC3',18),('CV_16UC4',26),('CV_16S',3),('CV_16SC1',3),('CV_16SC2',11),('CV_16SC3',19),('CV_16SC4',27),('CV_32S',4),('CV_32SC1',4),('CV_32SC2',12),('CV_32SC3',20),('CV_32SC4',28),('CV_32F',5),('CV_32FC1',5),('CV_32FC2',13),('CV_32FC3',21),('CV_32FC4',29),('CV_64F',6),('CV_64FC1',6),('CV_64FC2',14),('CV_64FC3',22),('CV_64FC4',30)])]

    def process_inputs(self, inputs, outputs, parameters):
        src1 = inputs['src1'].value
        src2 = inputs['src2'].value
        mask = inputs['mask'].value
        dtype = parameters['dtype']
        dist, nidx = cv2.batchDistance(src1=src1, src2=src2, mask=mask, dtype=dtype)
        outputs['dist'] = Data(dist)
        outputs['nidx'] = Data(nidx)

### bitwise_and ###

class OpenCVAuto2_Bitwise_and(NormalElement):
    name = 'Bitwise_and'
    comment = '''bitwise_and(src1, src2[, dst[, mask]]) -> dst\n@brief computes bitwise conjunction of the two arrays (dst = src1 & src2)\nCalculates the per-element bit-wise conjunction of two arrays or an\narray and a scalar.\n\nThe function cv::bitwise_and calculates the per-element bit-wise logical conjunction for:\n*   Two arrays when src1 and src2 have the same size:\n\f[\texttt{dst} (I) =  \texttt{src1} (I)  \wedge \texttt{src2} (I) \quad \texttt{if mask} (I) \ne0\f]\n*   An array and a scalar when src2 is constructed from Scalar or has\nthe same number of elements as `src1.channels()`:\n\f[\texttt{dst} (I) =  \texttt{src1} (I)  \wedge \texttt{src2} \quad \texttt{if mask} (I) \ne0\f]\n*   A scalar and an array when src1 is constructed from Scalar or has\nthe same number of elements as `src2.channels()`:\n\f[\texttt{dst} (I) =  \texttt{src1}  \wedge \texttt{src2} (I) \quad \texttt{if mask} (I) \ne0\f]\nIn case of floating-point arrays, their machine-specific bit\nrepresentations (usually IEEE754-compliant) are used for the operation.\nIn case of multi-channel arrays, each channel is processed\nindependently. In the second and third cases above, the scalar is first\nconverted to the array type.\n@param src1 first input array or a scalar.\n@param src2 second input array or a scalar.\n@param dst output array that has the same size and type as the input\narrays.\n@param mask optional operation mask, 8-bit single channel array, that\nspecifies elements of the output array to be changed.'''

    def get_attributes(self):
        return [Input('src1', 'src1'),
                Input('src2', 'src2'),
                Input('mask', 'mask', optional=True)], \
               [Output('dst', 'dst')], \
               []

    def process_inputs(self, inputs, outputs, parameters):
        src1 = inputs['src1'].value
        src2 = inputs['src2'].value
        mask = inputs['mask'].value
        dst = cv2.bitwise_and(src1=src1, src2=src2, mask=mask)
        outputs['dst'] = Data(dst)

### bitwise_not ###

class OpenCVAuto2_Bitwise_not(NormalElement):
    name = 'Bitwise_not'
    comment = '''bitwise_not(src[, dst[, mask]]) -> dst\n@brief  Inverts every bit of an array.\n\nThe function cv::bitwise_not calculates per-element bit-wise inversion of the input\narray:\n\f[\texttt{dst} (I) =  \neg \texttt{src} (I)\f]\nIn case of a floating-point input array, its machine-specific bit\nrepresentation (usually IEEE754-compliant) is used for the operation. In\ncase of multi-channel arrays, each channel is processed independently.\n@param src input array.\n@param dst output array that has the same size and type as the input\narray.\n@param mask optional operation mask, 8-bit single channel array, that\nspecifies elements of the output array to be changed.'''

    def get_attributes(self):
        return [Input('src', 'src'),
                Input('mask', 'mask', optional=True)], \
               [Output('dst', 'dst')], \
               []

    def process_inputs(self, inputs, outputs, parameters):
        src = inputs['src'].value
        mask = inputs['mask'].value
        dst = cv2.bitwise_not(src=src, mask=mask)
        outputs['dst'] = Data(dst)

### bitwise_or ###

class OpenCVAuto2_Bitwise_or(NormalElement):
    name = 'Bitwise_or'
    comment = '''bitwise_or(src1, src2[, dst[, mask]]) -> dst\n@brief Calculates the per-element bit-wise disjunction of two arrays or an\narray and a scalar.\n\nThe function cv::bitwise_or calculates the per-element bit-wise logical disjunction for:\n*   Two arrays when src1 and src2 have the same size:\n\f[\texttt{dst} (I) =  \texttt{src1} (I)  \vee \texttt{src2} (I) \quad \texttt{if mask} (I) \ne0\f]\n*   An array and a scalar when src2 is constructed from Scalar or has\nthe same number of elements as `src1.channels()`:\n\f[\texttt{dst} (I) =  \texttt{src1} (I)  \vee \texttt{src2} \quad \texttt{if mask} (I) \ne0\f]\n*   A scalar and an array when src1 is constructed from Scalar or has\nthe same number of elements as `src2.channels()`:\n\f[\texttt{dst} (I) =  \texttt{src1}  \vee \texttt{src2} (I) \quad \texttt{if mask} (I) \ne0\f]\nIn case of floating-point arrays, their machine-specific bit\nrepresentations (usually IEEE754-compliant) are used for the operation.\nIn case of multi-channel arrays, each channel is processed\nindependently. In the second and third cases above, the scalar is first\nconverted to the array type.\n@param src1 first input array or a scalar.\n@param src2 second input array or a scalar.\n@param dst output array that has the same size and type as the input\narrays.\n@param mask optional operation mask, 8-bit single channel array, that\nspecifies elements of the output array to be changed.'''

    def get_attributes(self):
        return [Input('src1', 'src1'),
                Input('src2', 'src2'),
                Input('mask', 'mask', optional=True)], \
               [Output('dst', 'dst')], \
               []

    def process_inputs(self, inputs, outputs, parameters):
        src1 = inputs['src1'].value
        src2 = inputs['src2'].value
        mask = inputs['mask'].value
        dst = cv2.bitwise_or(src1=src1, src2=src2, mask=mask)
        outputs['dst'] = Data(dst)

### bitwise_xor ###

class OpenCVAuto2_Bitwise_xor(NormalElement):
    name = 'Bitwise_xor'
    comment = '''bitwise_xor(src1, src2[, dst[, mask]]) -> dst\n@brief Calculates the per-element bit-wise "exclusive or" operation on two\narrays or an array and a scalar.\n\nThe function cv::bitwise_xor calculates the per-element bit-wise logical "exclusive-or"\noperation for:\n*   Two arrays when src1 and src2 have the same size:\n\f[\texttt{dst} (I) =  \texttt{src1} (I)  \oplus \texttt{src2} (I) \quad \texttt{if mask} (I) \ne0\f]\n*   An array and a scalar when src2 is constructed from Scalar or has\nthe same number of elements as `src1.channels()`:\n\f[\texttt{dst} (I) =  \texttt{src1} (I)  \oplus \texttt{src2} \quad \texttt{if mask} (I) \ne0\f]\n*   A scalar and an array when src1 is constructed from Scalar or has\nthe same number of elements as `src2.channels()`:\n\f[\texttt{dst} (I) =  \texttt{src1}  \oplus \texttt{src2} (I) \quad \texttt{if mask} (I) \ne0\f]\nIn case of floating-point arrays, their machine-specific bit\nrepresentations (usually IEEE754-compliant) are used for the operation.\nIn case of multi-channel arrays, each channel is processed\nindependently. In the 2nd and 3rd cases above, the scalar is first\nconverted to the array type.\n@param src1 first input array or a scalar.\n@param src2 second input array or a scalar.\n@param dst output array that has the same size and type as the input\narrays.\n@param mask optional operation mask, 8-bit single channel array, that\nspecifies elements of the output array to be changed.'''

    def get_attributes(self):
        return [Input('src1', 'src1'),
                Input('src2', 'src2'),
                Input('mask', 'mask', optional=True)], \
               [Output('dst', 'dst')], \
               []

    def process_inputs(self, inputs, outputs, parameters):
        src1 = inputs['src1'].value
        src2 = inputs['src2'].value
        mask = inputs['mask'].value
        dst = cv2.bitwise_xor(src1=src1, src2=src2, mask=mask)
        outputs['dst'] = Data(dst)

### blur ###

class OpenCVAuto2_Blur(NormalElement):
    name = 'Blur'
    comment = '''blur(src, ksize[, dst[, anchor[, borderType]]]) -> dst\n@brief Blurs an image using the normalized box filter.\n\nThe function smooths an image using the kernel:\n\n\f[\texttt{K} =  \frac{1}{\texttt{ksize.width*ksize.height}} \begin{bmatrix} 1 & 1 & 1 &  \cdots & 1 & 1  \\ 1 & 1 & 1 &  \cdots & 1 & 1  \\ \hdotsfor{6} \\ 1 & 1 & 1 &  \cdots & 1 & 1  \\ \end{bmatrix}\f]\n\nThe call `blur(src, dst, ksize, anchor, borderType)` is equivalent to `boxFilter(src, dst, src.type(),\nanchor, true, borderType)`.\n\n@param src input image; it can have any number of channels, which are processed independently, but\nthe depth should be CV_8U, CV_16U, CV_16S, CV_32F or CV_64F.\n@param dst output image of the same size and type as src.\n@param ksize blurring kernel size.\n@param anchor anchor point; default value Point(-1,-1) means that the anchor is at the kernel\ncenter.\n@param borderType border mode used to extrapolate pixels outside of the image, see #BorderTypes\n@sa  boxFilter, bilateralFilter, GaussianBlur, medianBlur'''

    def get_attributes(self):
        return [Input('src', 'src')], \
               [Output('dst', 'dst')], \
               [SizeParameter('ksize', 'ksize'),
                PointParameter('anchor', 'anchor'),
                ComboboxParameter('borderType', [('BORDER_CONSTANT',0),('BORDER_DEFAULT',4),('BORDER_ISOLATED',16),('BORDER_REFLECT',2),('BORDER_REFLECT101',4),('BORDER_REFLECT_101',4),('BORDER_REPLICATE',1),('BORDER_TRANSPARENT',5),('BORDER_WRAP',3)])]

    def process_inputs(self, inputs, outputs, parameters):
        src = inputs['src'].value
        ksize = parameters['ksize']
        anchor = parameters['anchor']
        borderType = parameters['borderType']
        dst = cv2.blur(src=src, ksize=ksize, anchor=anchor, borderType=borderType)
        outputs['dst'] = Data(dst)

### boxFilter ###

class OpenCVAuto2_BoxFilter(NormalElement):
    name = 'BoxFilter'
    comment = '''boxFilter(src, ddepth, ksize[, dst[, anchor[, normalize[, borderType]]]]) -> dst\n@brief Blurs an image using the box filter.\n\nThe function smooths an image using the kernel:\n\n\f[\texttt{K} =  \alpha \begin{bmatrix} 1 & 1 & 1 &  \cdots & 1 & 1  \\ 1 & 1 & 1 &  \cdots & 1 & 1  \\ \hdotsfor{6} \\ 1 & 1 & 1 &  \cdots & 1 & 1 \end{bmatrix}\f]\n\nwhere\n\n\f[\alpha = \fork{\frac{1}{\texttt{ksize.width*ksize.height}}}{when \texttt{normalize=true}}{1}{otherwise}\f]\n\nUnnormalized box filter is useful for computing various integral characteristics over each pixel\nneighborhood, such as covariance matrices of image derivatives (used in dense optical flow\nalgorithms, and so on). If you need to compute pixel sums over variable-size windows, use #integral.\n\n@param src input image.\n@param dst output image of the same size and type as src.\n@param ddepth the output image depth (-1 to use src.depth()).\n@param ksize blurring kernel size.\n@param anchor anchor point; default value Point(-1,-1) means that the anchor is at the kernel\ncenter.\n@param normalize flag, specifying whether the kernel is normalized by its area or not.\n@param borderType border mode used to extrapolate pixels outside of the image, see #BorderTypes\n@sa  blur, bilateralFilter, GaussianBlur, medianBlur, integral'''

    def get_attributes(self):
        return [Input('src', 'src')], \
               [Output('dst', 'dst')], \
               [ComboboxParameter('ddepth', [('NONE',0),('CV_8U',0),('CV_8UC1',0),('CV_8UC2',8),('CV_8UC3',16),('CV_8UC4',24),('CV_8S',1),('CV_8SC1',1),('CV_8SC2',9),('CV_8SC3',17),('CV_8SC4',25),('CV_16U',2),('CV_16UC1',2),('CV_16UC2',10),('CV_16UC3',18),('CV_16UC4',26),('CV_16S',3),('CV_16SC1',3),('CV_16SC2',11),('CV_16SC3',19),('CV_16SC4',27),('CV_32S',4),('CV_32SC1',4),('CV_32SC2',12),('CV_32SC3',20),('CV_32SC4',28),('CV_32F',5),('CV_32FC1',5),('CV_32FC2',13),('CV_32FC3',21),('CV_32FC4',29),('CV_64F',6),('CV_64FC1',6),('CV_64FC2',14),('CV_64FC3',22),('CV_64FC4',30)]),
                SizeParameter('ksize', 'ksize'),
                PointParameter('anchor', 'anchor'),
                ComboboxParameter('borderType', [('BORDER_CONSTANT',0),('BORDER_DEFAULT',4),('BORDER_ISOLATED',16),('BORDER_REFLECT',2),('BORDER_REFLECT101',4),('BORDER_REFLECT_101',4),('BORDER_REPLICATE',1),('BORDER_TRANSPARENT',5),('BORDER_WRAP',3)])]

    def process_inputs(self, inputs, outputs, parameters):
        src = inputs['src'].value
        ddepth = parameters['ddepth']
        ksize = parameters['ksize']
        anchor = parameters['anchor']
        borderType = parameters['borderType']
        dst = cv2.boxFilter(src=src, ddepth=ddepth, ksize=ksize, anchor=anchor, borderType=borderType)
        outputs['dst'] = Data(dst)

### calibrateCameraExtended ###

class OpenCVAuto2_CalibrateCameraExtended(NormalElement):
    name = 'CalibrateCameraExtended'
    comment = '''calibrateCameraExtended(objectPoints, imagePoints, imageSize, cameraMatrix, distCoeffs[, rvecs[, tvecs[, stdDeviationsIntrinsics[, stdDeviationsExtrinsics[, perViewErrors[, flags[, criteria]]]]]]]) -> retval, cameraMatrix, distCoeffs, rvecs, tvecs, stdDeviationsIntrinsics, stdDeviationsExtrinsics, perViewErrors\n@brief Finds the camera intrinsic and extrinsic parameters from several views of a calibration pattern.\n\n@param objectPoints In the new interface it is a vector of vectors of calibration pattern points in\nthe calibration pattern coordinate space (e.g. std::vector<std::vector<cv::Vec3f>>). The outer\nvector contains as many elements as the number of the pattern views. If the same calibration pattern\nis shown in each view and it is fully visible, all the vectors will be the same. Although, it is\npossible to use partially occluded patterns, or even different patterns in different views. Then,\nthe vectors will be different. The points are 3D, but since they are in a pattern coordinate system,\nthen, if the rig is planar, it may make sense to put the model to a XY coordinate plane so that\nZ-coordinate of each input object point is 0.\nIn the old interface all the vectors of object points from different views are concatenated\ntogether.\n@param imagePoints In the new interface it is a vector of vectors of the projections of calibration\npattern points (e.g. std::vector<std::vector<cv::Vec2f>>). imagePoints.size() and\nobjectPoints.size() and imagePoints[i].size() must be equal to objectPoints[i].size() for each i.\nIn the old interface all the vectors of object points from different views are concatenated\ntogether.\n@param imageSize Size of the image used only to initialize the intrinsic camera matrix.\n@param cameraMatrix Output 3x3 floating-point camera matrix\n\f$A = \vecthreethree{f_x}{0}{c_x}{0}{f_y}{c_y}{0}{0}{1}\f$ . If CV\_CALIB\_USE\_INTRINSIC\_GUESS\nand/or CALIB_FIX_ASPECT_RATIO are specified, some or all of fx, fy, cx, cy must be\ninitialized before calling the function.\n@param distCoeffs Output vector of distortion coefficients\n\f$(k_1, k_2, p_1, p_2[, k_3[, k_4, k_5, k_6 [, s_1, s_2, s_3, s_4[, \tau_x, \tau_y]]]])\f$ of\n4, 5, 8, 12 or 14 elements.\n@param rvecs Output vector of rotation vectors (see Rodrigues ) estimated for each pattern view\n(e.g. std::vector<cv::Mat>>). That is, each k-th rotation vector together with the corresponding\nk-th translation vector (see the next output parameter description) brings the calibration pattern\nfrom the model coordinate space (in which object points are specified) to the world coordinate\nspace, that is, a real position of the calibration pattern in the k-th pattern view (k=0.. *M* -1).\n@param tvecs Output vector of translation vectors estimated for each pattern view.\n@param stdDeviationsIntrinsics Output vector of standard deviations estimated for intrinsic parameters.\nOrder of deviations values:\n\f$(f_x, f_y, c_x, c_y, k_1, k_2, p_1, p_2, k_3, k_4, k_5, k_6 , s_1, s_2, s_3,\ns_4, \tau_x, \tau_y)\f$ If one of parameters is not estimated, it's deviation is equals to zero.\n@param stdDeviationsExtrinsics Output vector of standard deviations estimated for extrinsic parameters.\nOrder of deviations values: \f$(R_1, T_1, \dotsc , R_M, T_M)\f$ where M is number of pattern views,\n\f$R_i, T_i\f$ are concatenated 1x3 vectors.\n@param perViewErrors Output vector of the RMS re-projection error estimated for each pattern view.\n@param flags Different flags that may be zero or a combination of the following values:\n-   **CALIB_USE_INTRINSIC_GUESS** cameraMatrix contains valid initial values of\nfx, fy, cx, cy that are optimized further. Otherwise, (cx, cy) is initially set to the image\ncenter ( imageSize is used), and focal distances are computed in a least-squares fashion.\nNote, that if intrinsic parameters are known, there is no need to use this function just to\nestimate extrinsic parameters. Use solvePnP instead.\n-   **CALIB_FIX_PRINCIPAL_POINT** The principal point is not changed during the global\noptimization. It stays at the center or at a different location specified when\nCALIB_USE_INTRINSIC_GUESS is set too.\n-   **CALIB_FIX_ASPECT_RATIO** The functions considers only fy as a free parameter. The\nratio fx/fy stays the same as in the input cameraMatrix . When\nCALIB_USE_INTRINSIC_GUESS is not set, the actual input values of fx and fy are\nignored, only their ratio is computed and used further.\n-   **CALIB_ZERO_TANGENT_DIST** Tangential distortion coefficients \f$(p_1, p_2)\f$ are set\nto zeros and stay zero.\n-   **CALIB_FIX_K1,...,CALIB_FIX_K6** The corresponding radial distortion\ncoefficient is not changed during the optimization. If CALIB_USE_INTRINSIC_GUESS is\nset, the coefficient from the supplied distCoeffs matrix is used. Otherwise, it is set to 0.\n-   **CALIB_RATIONAL_MODEL** Coefficients k4, k5, and k6 are enabled. To provide the\nbackward compatibility, this extra flag should be explicitly specified to make the\ncalibration function use the rational model and return 8 coefficients. If the flag is not\nset, the function computes and returns only 5 distortion coefficients.\n-   **CALIB_THIN_PRISM_MODEL** Coefficients s1, s2, s3 and s4 are enabled. To provide the\nbackward compatibility, this extra flag should be explicitly specified to make the\ncalibration function use the thin prism model and return 12 coefficients. If the flag is not\nset, the function computes and returns only 5 distortion coefficients.\n-   **CALIB_FIX_S1_S2_S3_S4** The thin prism distortion coefficients are not changed during\nthe optimization. If CALIB_USE_INTRINSIC_GUESS is set, the coefficient from the\nsupplied distCoeffs matrix is used. Otherwise, it is set to 0.\n-   **CALIB_TILTED_MODEL** Coefficients tauX and tauY are enabled. To provide the\nbackward compatibility, this extra flag should be explicitly specified to make the\ncalibration function use the tilted sensor model and return 14 coefficients. If the flag is not\nset, the function computes and returns only 5 distortion coefficients.\n-   **CALIB_FIX_TAUX_TAUY** The coefficients of the tilted sensor model are not changed during\nthe optimization. If CALIB_USE_INTRINSIC_GUESS is set, the coefficient from the\nsupplied distCoeffs matrix is used. Otherwise, it is set to 0.\n@param criteria Termination criteria for the iterative optimization algorithm.\n\n@return the overall RMS re-projection error.\n\nThe function estimates the intrinsic camera parameters and extrinsic parameters for each of the\nviews. The algorithm is based on @cite Zhang2000 and @cite BouguetMCT . The coordinates of 3D object\npoints and their corresponding 2D projections in each view must be specified. That may be achieved\nby using an object with a known geometry and easily detectable feature points. Such an object is\ncalled a calibration rig or calibration pattern, and OpenCV has built-in support for a chessboard as\na calibration rig (see findChessboardCorners ). Currently, initialization of intrinsic parameters\n(when CALIB_USE_INTRINSIC_GUESS is not set) is only implemented for planar calibration\npatterns (where Z-coordinates of the object points must be all zeros). 3D calibration rigs can also\nbe used as long as initial cameraMatrix is provided.\n\nThe algorithm performs the following steps:\n\n-   Compute the initial intrinsic parameters (the option only available for planar calibration\npatterns) or read them from the input parameters. The distortion coefficients are all set to\nzeros initially unless some of CALIB_FIX_K? are specified.\n\n-   Estimate the initial camera pose as if the intrinsic parameters have been already known. This is\ndone using solvePnP .\n\n-   Run the global Levenberg-Marquardt optimization algorithm to minimize the reprojection error,\nthat is, the total sum of squared distances between the observed feature points imagePoints and\nthe projected (using the current estimates for camera parameters and the poses) object points\nobjectPoints. See projectPoints for details.\n\n@note\nIf you use a non-square (=non-NxN) grid and findChessboardCorners for calibration, and\ncalibrateCamera returns bad values (zero distortion coefficients, an image center very far from\n(w/2-0.5,h/2-0.5), and/or large differences between \f$f_x\f$ and \f$f_y\f$ (ratios of 10:1 or more)),\nthen you have probably used patternSize=cvSize(rows,cols) instead of using\npatternSize=cvSize(cols,rows) in findChessboardCorners .\n\n@sa\nfindChessboardCorners, solvePnP, initCameraMatrix2D, stereoCalibrate, undistort'''

    def get_attributes(self):
        return [Input('objectPoints', 'objectPoints'),
                Input('imagePoints', 'imagePoints'),
                Input('imageSize', 'imageSize'),
                Input('cameraMatrix', 'cameraMatrix'),
                Input('distCoeffs', 'distCoeffs')], \
               [Output('cameraMatrix', 'cameraMatrix'),
                Output('distCoeffs', 'distCoeffs'),
                Output('rvecs', 'rvecs'),
                Output('tvecs', 'tvecs'),
                Output('stdDeviationsIntrinsics', 'stdDeviationsIntrinsics'),
                Output('stdDeviationsExtrinsics', 'stdDeviationsExtrinsics'),
                Output('perViewErrors', 'perViewErrors')], \
               [ComboboxParameter('flags', [('CALIB_USE_INTRINSIC_GUESS',1),('CALIB_FIX_PRINCIPAL_POINT',4),('CALIB_FIX_ASPECT_RATIO',2),('CALIB_ZERO_TANGENT_DIST',8),('CALIB_FIX_K1',32),('CALIB_FIX_K6',8192),('CALIB_RATIONAL_MODEL',16384),('CALIB_THIN_PRISM_MODEL',32768),('CALIB_FIX_S1_S2_S3_S4',65536),('CALIB_TILTED_MODEL',262144),('CALIB_FIX_TAUX_TAUY',524288)])]

    def process_inputs(self, inputs, outputs, parameters):
        objectPoints = inputs['objectPoints'].value
        imagePoints = inputs['imagePoints'].value
        imageSize = inputs['imageSize'].value
        cameraMatrix = inputs['cameraMatrix'].value
        distCoeffs = inputs['distCoeffs'].value
        flags = parameters['flags']
        retval, cameraMatrix, distCoeffs, rvecs, tvecs, stdDeviationsIntrinsics, stdDeviationsExtrinsics, perViewErrors = cv2.calibrateCameraExtended(objectPoints=objectPoints, imagePoints=imagePoints, imageSize=imageSize, cameraMatrix=cameraMatrix, distCoeffs=distCoeffs, flags=flags)
        outputs['cameraMatrix'] = Data(cameraMatrix)
        outputs['distCoeffs'] = Data(distCoeffs)
        outputs['rvecs'] = Data(rvecs)
        outputs['tvecs'] = Data(tvecs)
        outputs['stdDeviationsIntrinsics'] = Data(stdDeviationsIntrinsics)
        outputs['stdDeviationsExtrinsics'] = Data(stdDeviationsExtrinsics)
        outputs['perViewErrors'] = Data(perViewErrors)

### circle ###

class OpenCVAuto2_Circle(NormalElement):
    name = 'Circle'
    comment = '''circle(img, center, radius, color[, thickness[, lineType[, shift]]]) -> img\n@brief Draws a circle.\n\nThe function cv::circle draws a simple or filled circle with a given center and radius.\n@param img Image where the circle is drawn.\n@param center Center of the circle.\n@param radius Radius of the circle.\n@param color Circle color.\n@param thickness Thickness of the circle outline, if positive. Negative values, like #FILLED,\nmean that a filled circle is to be drawn.\n@param lineType Type of the circle boundary. See #LineTypes\n@param shift Number of fractional bits in the coordinates of the center and in the radius value.'''

    def get_attributes(self):
        return [Input('img', 'img')], \
               [Output('img', 'img')], \
               [PointParameter('center', 'center'),
                IntParameter('radius', 'radius'),
                ScalarParameter('color', 'color'),
                IntParameter('thickness', 'thickness')]

    def process_inputs(self, inputs, outputs, parameters):
        img = inputs['img'].value
        center = parameters['center']
        radius = parameters['radius']
        color = parameters['color']
        thickness = parameters['thickness']
        img = cv2.circle(img=img, center=center, radius=radius, color=color, thickness=thickness)
        outputs['img'] = Data(img)

### colorChange ###

class OpenCVAuto2_ColorChange(NormalElement):
    name = 'ColorChange'
    comment = '''colorChange(src, mask[, dst[, red_mul[, green_mul[, blue_mul]]]]) -> dst\n@brief Given an original color image, two differently colored versions of this image can be mixed\nseamlessly.\n\n@param src Input 8-bit 3-channel image.\n@param mask Input 8-bit 1 or 3-channel image.\n@param dst Output image with the same size and type as src .\n@param red_mul R-channel multiply factor.\n@param green_mul G-channel multiply factor.\n@param blue_mul B-channel multiply factor.\n\nMultiplication factor is between .5 to 2.5.'''

    def get_attributes(self):
        return [Input('src', 'src'),
                Input('mask', 'mask')], \
               [Output('dst', 'dst')], \
               []

    def process_inputs(self, inputs, outputs, parameters):
        src = inputs['src'].value
        mask = inputs['mask'].value
        dst = cv2.colorChange(src=src, mask=mask)
        outputs['dst'] = Data(dst)

### completeSymm ###

class OpenCVAuto2_CompleteSymm(NormalElement):
    name = 'CompleteSymm'
    comment = '''completeSymm(m[, lowerToUpper]) -> m\n@brief Copies the lower or the upper half of a square matrix to its another half.\n\nThe function cv::completeSymm copies the lower or the upper half of a square matrix to\nits another half. The matrix diagonal remains unchanged:\n- \f$\texttt{m}_{ij}=\texttt{m}_{ji}\f$ for \f$i > j\f$ if\nlowerToUpper=false\n- \f$\texttt{m}_{ij}=\texttt{m}_{ji}\f$ for \f$i < j\f$ if\nlowerToUpper=true\n\n@param m input-output floating-point square matrix.\n@param lowerToUpper operation flag; if true, the lower half is copied to\nthe upper half. Otherwise, the upper half is copied to the lower half.\n@sa flip, transpose'''

    def get_attributes(self):
        return [Input('m', 'm')], \
               [Output('m', 'm')], \
               []

    def process_inputs(self, inputs, outputs, parameters):
        m = inputs['m'].value
        m = cv2.completeSymm(m=m)
        outputs['m'] = Data(m)

### connectedComponents ###

class OpenCVAuto2_ConnectedComponents(NormalElement):
    name = 'ConnectedComponents'
    comment = '''connectedComponents(image[, labels[, connectivity[, ltype]]]) -> retval, labels\n@overload\n\n@param image the 8-bit single-channel image to be labeled\n@param labels destination labeled image\n@param connectivity 8 or 4 for 8-way or 4-way connectivity respectively\n@param ltype output image label type. Currently CV_32S and CV_16U are supported.'''

    def get_attributes(self):
        return [Input('image', 'image')], \
               [Output('labels', 'labels')], \
               []

    def process_inputs(self, inputs, outputs, parameters):
        image = inputs['image'].value
        retval, labels = cv2.connectedComponents(image=image)
        outputs['labels'] = Data(labels)

### connectedComponentsWithStats ###

class OpenCVAuto2_ConnectedComponentsWithStats(NormalElement):
    name = 'ConnectedComponentsWithStats'
    comment = '''connectedComponentsWithStats(image[, labels[, stats[, centroids[, connectivity[, ltype]]]]]) -> retval, labels, stats, centroids\n@overload\n@param image the 8-bit single-channel image to be labeled\n@param labels destination labeled image\n@param stats statistics output for each label, including the background label, see below for\navailable statistics. Statistics are accessed via stats(label, COLUMN) where COLUMN is one of\n#ConnectedComponentsTypes. The data type is CV_32S.\n@param centroids centroid output for each label, including the background label. Centroids are\naccessed via centroids(label, 0) for x and centroids(label, 1) for y. The data type CV_64F.\n@param connectivity 8 or 4 for 8-way or 4-way connectivity respectively\n@param ltype output image label type. Currently CV_32S and CV_16U are supported.'''

    def get_attributes(self):
        return [Input('image', 'image')], \
               [Output('labels', 'labels'),
                Output('stats', 'stats'),
                Output('centroids', 'centroids')], \
               []

    def process_inputs(self, inputs, outputs, parameters):
        image = inputs['image'].value
        retval, labels, stats, centroids = cv2.connectedComponentsWithStats(image=image)
        outputs['labels'] = Data(labels)
        outputs['stats'] = Data(stats)
        outputs['centroids'] = Data(centroids)

### convertFp16 ###

class OpenCVAuto2_ConvertFp16(NormalElement):
    name = 'ConvertFp16'
    comment = '''convertFp16(src[, dst]) -> dst\n@brief Converts an array to half precision floating number.\n\nThis function converts FP32 (single precision floating point) from/to FP16 (half precision floating point). CV_16S format is used to represent FP16 data.\nThere are two use modes (src -> dst): CV_32F -> CV_16S and CV_16S -> CV_32F. The input array has to have type of CV_32F or\nCV_16S to represent the bit depth. If the input array is neither of them, the function will raise an error.\nThe format of half precision floating point is defined in IEEE 754-2008.\n\n@param src input array.\n@param dst output array.'''

    def get_attributes(self):
        return [Input('src', 'src')], \
               [Output('dst', 'dst')], \
               []

    def process_inputs(self, inputs, outputs, parameters):
        src = inputs['src'].value
        dst = cv2.convertFp16(src=src)
        outputs['dst'] = Data(dst)

### convertPointsFromHomogeneous ###

class OpenCVAuto2_ConvertPointsFromHomogeneous(NormalElement):
    name = 'ConvertPointsFromHomogeneous'
    comment = '''convertPointsFromHomogeneous(src[, dst]) -> dst\n@brief Converts points from homogeneous to Euclidean space.\n\n@param src Input vector of N-dimensional points.\n@param dst Output vector of N-1-dimensional points.\n\nThe function converts points homogeneous to Euclidean space using perspective projection. That is,\neach point (x1, x2, ... x(n-1), xn) is converted to (x1/xn, x2/xn, ..., x(n-1)/xn). When xn=0, the\noutput point coordinates will be (0,0,0,...).'''

    def get_attributes(self):
        return [Input('src', 'src')], \
               [Output('dst', 'dst')], \
               []

    def process_inputs(self, inputs, outputs, parameters):
        src = inputs['src'].value
        dst = cv2.convertPointsFromHomogeneous(src=src)
        outputs['dst'] = Data(dst)

### convertPointsToHomogeneous ###

class OpenCVAuto2_ConvertPointsToHomogeneous(NormalElement):
    name = 'ConvertPointsToHomogeneous'
    comment = '''convertPointsToHomogeneous(src[, dst]) -> dst\n@brief Converts points from Euclidean to homogeneous space.\n\n@param src Input vector of N-dimensional points.\n@param dst Output vector of N+1-dimensional points.\n\nThe function converts points from Euclidean to homogeneous space by appending 1's to the tuple of\npoint coordinates. That is, each point (x1, x2, ..., xn) is converted to (x1, x2, ..., xn, 1).'''

    def get_attributes(self):
        return [Input('src', 'src')], \
               [Output('dst', 'dst')], \
               []

    def process_inputs(self, inputs, outputs, parameters):
        src = inputs['src'].value
        dst = cv2.convertPointsToHomogeneous(src=src)
        outputs['dst'] = Data(dst)

### convertScaleAbs ###

class OpenCVAuto2_ConvertScaleAbs(NormalElement):
    name = 'ConvertScaleAbs'
    comment = '''convertScaleAbs(src[, dst[, alpha[, beta]]]) -> dst\n@brief Scales, calculates absolute values, and converts the result to 8-bit.\n\nOn each element of the input array, the function convertScaleAbs\nperforms three operations sequentially: scaling, taking an absolute\nvalue, conversion to an unsigned 8-bit type:\n\f[\texttt{dst} (I)= \texttt{saturate\_cast<uchar>} (| \texttt{src} (I)* \texttt{alpha} +  \texttt{beta} |)\f]\nIn case of multi-channel arrays, the function processes each channel\nindependently. When the output is not 8-bit, the operation can be\nemulated by calling the Mat::convertTo method (or by using matrix\nexpressions) and then by calculating an absolute value of the result.\nFor example:\n@code{.cpp}\nMat_<float> A(30,30);\nrandu(A, Scalar(-100), Scalar(100));\nMat_<float> B = A*5 + 3;\nB = abs(B);\n// Mat_<float> B = abs(A*5+3) will also do the job,\n// but it will allocate a temporary matrix\n@endcode\n@param src input array.\n@param dst output array.\n@param alpha optional scale factor.\n@param beta optional delta added to the scaled values.\n@sa  Mat::convertTo, cv::abs(const Mat&)'''

    def get_attributes(self):
        return [Input('src', 'src')], \
               [Output('dst', 'dst')], \
               [FloatParameter('alpha', 'alpha'),
                FloatParameter('beta', 'beta')]

    def process_inputs(self, inputs, outputs, parameters):
        src = inputs['src'].value
        alpha = parameters['alpha']
        beta = parameters['beta']
        dst = cv2.convertScaleAbs(src=src, alpha=alpha, beta=beta)
        outputs['dst'] = Data(dst)

### convexHull ###

class OpenCVAuto2_ConvexHull(NormalElement):
    name = 'ConvexHull'
    comment = '''convexHull(points[, hull[, clockwise[, returnPoints]]]) -> hull\n@brief Finds the convex hull of a point set.\n\nThe function cv::convexHull finds the convex hull of a 2D point set using the Sklansky's algorithm @cite Sklansky82\nthat has *O(N logN)* complexity in the current implementation.\n\n@param points Input 2D point set, stored in std::vector or Mat.\n@param hull Output convex hull. It is either an integer vector of indices or vector of points. In\nthe first case, the hull elements are 0-based indices of the convex hull points in the original\narray (since the set of convex hull points is a subset of the original point set). In the second\ncase, hull elements are the convex hull points themselves.\n@param clockwise Orientation flag. If it is true, the output convex hull is oriented clockwise.\nOtherwise, it is oriented counter-clockwise. The assumed coordinate system has its X axis pointing\nto the right, and its Y axis pointing upwards.\n@param returnPoints Operation flag. In case of a matrix, when the flag is true, the function\nreturns convex hull points. Otherwise, it returns indices of the convex hull points. When the\noutput array is std::vector, the flag is ignored, and the output depends on the type of the\nvector: std::vector\<int\> implies returnPoints=false, std::vector\<Point\> implies\nreturnPoints=true.\n\n@note `points` and `hull` should be different arrays, inplace processing isn't supported.'''

    def get_attributes(self):
        return [Input('points', 'points'),
                Input('returnPoints', 'returnPoints', optional=True)], \
               [Output('hull', 'hull')], \
               []

    def process_inputs(self, inputs, outputs, parameters):
        points = inputs['points'].value
        returnPoints = inputs['returnPoints'].value
        hull = cv2.convexHull(points=points, returnPoints=returnPoints)
        outputs['hull'] = Data(hull)

### cornerEigenValsAndVecs ###

class OpenCVAuto2_CornerEigenValsAndVecs(NormalElement):
    name = 'CornerEigenValsAndVecs'
    comment = '''cornerEigenValsAndVecs(src, blockSize, ksize[, dst[, borderType]]) -> dst\n@brief Calculates eigenvalues and eigenvectors of image blocks for corner detection.\n\nFor every pixel \f$p\f$ , the function cornerEigenValsAndVecs considers a blockSize \f$\times\f$ blockSize\nneighborhood \f$S(p)\f$ . It calculates the covariation matrix of derivatives over the neighborhood as:\n\n\f[M =  \begin{bmatrix} \sum _{S(p)}(dI/dx)^2 &  \sum _{S(p)}dI/dx dI/dy  \\ \sum _{S(p)}dI/dx dI/dy &  \sum _{S(p)}(dI/dy)^2 \end{bmatrix}\f]\n\nwhere the derivatives are computed using the Sobel operator.\n\nAfter that, it finds eigenvectors and eigenvalues of \f$M\f$ and stores them in the destination image as\n\f$(\lambda_1, \lambda_2, x_1, y_1, x_2, y_2)\f$ where\n\n-   \f$\lambda_1, \lambda_2\f$ are the non-sorted eigenvalues of \f$M\f$\n-   \f$x_1, y_1\f$ are the eigenvectors corresponding to \f$\lambda_1\f$\n-   \f$x_2, y_2\f$ are the eigenvectors corresponding to \f$\lambda_2\f$\n\nThe output of the function can be used for robust edge or corner detection.\n\n@param src Input single-channel 8-bit or floating-point image.\n@param dst Image to store the results. It has the same size as src and the type CV_32FC(6) .\n@param blockSize Neighborhood size (see details below).\n@param ksize Aperture parameter for the Sobel operator.\n@param borderType Pixel extrapolation method. See #BorderTypes.\n\n@sa  cornerMinEigenVal, cornerHarris, preCornerDetect'''

    def get_attributes(self):
        return [Input('src', 'src')], \
               [Output('dst', 'dst')], \
               [SizeParameter('blockSize', 'blockSize'),
                SizeParameter('ksize', 'ksize'),
                ComboboxParameter('borderType', [('BORDER_CONSTANT',0),('BORDER_DEFAULT',4),('BORDER_ISOLATED',16),('BORDER_REFLECT',2),('BORDER_REFLECT101',4),('BORDER_REFLECT_101',4),('BORDER_REPLICATE',1),('BORDER_TRANSPARENT',5),('BORDER_WRAP',3)])]

    def process_inputs(self, inputs, outputs, parameters):
        src = inputs['src'].value
        blockSize = parameters['blockSize']
        ksize = parameters['ksize']
        borderType = parameters['borderType']
        dst = cv2.cornerEigenValsAndVecs(src=src, blockSize=blockSize, ksize=ksize, borderType=borderType)
        outputs['dst'] = Data(dst)

### cornerMinEigenVal ###

class OpenCVAuto2_CornerMinEigenVal(NormalElement):
    name = 'CornerMinEigenVal'
    comment = '''cornerMinEigenVal(src, blockSize[, dst[, ksize[, borderType]]]) -> dst\n@brief Calculates the minimal eigenvalue of gradient matrices for corner detection.\n\nThe function is similar to cornerEigenValsAndVecs but it calculates and stores only the minimal\neigenvalue of the covariance matrix of derivatives, that is, \f$\min(\lambda_1, \lambda_2)\f$ in terms\nof the formulae in the cornerEigenValsAndVecs description.\n\n@param src Input single-channel 8-bit or floating-point image.\n@param dst Image to store the minimal eigenvalues. It has the type CV_32FC1 and the same size as\nsrc .\n@param blockSize Neighborhood size (see the details on #cornerEigenValsAndVecs ).\n@param ksize Aperture parameter for the Sobel operator.\n@param borderType Pixel extrapolation method. See #BorderTypes.'''

    def get_attributes(self):
        return [Input('src', 'src')], \
               [Output('dst', 'dst')], \
               [SizeParameter('blockSize', 'blockSize'),
                SizeParameter('ksize', 'ksize'),
                ComboboxParameter('borderType', [('BORDER_CONSTANT',0),('BORDER_DEFAULT',4),('BORDER_ISOLATED',16),('BORDER_REFLECT',2),('BORDER_REFLECT101',4),('BORDER_REFLECT_101',4),('BORDER_REPLICATE',1),('BORDER_TRANSPARENT',5),('BORDER_WRAP',3)])]

    def process_inputs(self, inputs, outputs, parameters):
        src = inputs['src'].value
        blockSize = parameters['blockSize']
        ksize = parameters['ksize']
        borderType = parameters['borderType']
        dst = cv2.cornerMinEigenVal(src=src, blockSize=blockSize, ksize=ksize, borderType=borderType)
        outputs['dst'] = Data(dst)

### cvtColor ###

class OpenCVAuto2_CvtColor(NormalElement):
    name = 'CvtColor'
    comment = '''cvtColor(src, code[, dst[, dstCn]]) -> dst\n@brief Converts an image from one color space to another.\n\nThe function converts an input image from one color space to another. In case of a transformation\nto-from RGB color space, the order of the channels should be specified explicitly (RGB or BGR). Note\nthat the default color format in OpenCV is often referred to as RGB but it is actually BGR (the\nbytes are reversed). So the first byte in a standard (24-bit) color image will be an 8-bit Blue\ncomponent, the second byte will be Green, and the third byte will be Red. The fourth, fifth, and\nsixth bytes would then be the second pixel (Blue, then Green, then Red), and so on.\n\nThe conventional ranges for R, G, and B channel values are:\n-   0 to 255 for CV_8U images\n-   0 to 65535 for CV_16U images\n-   0 to 1 for CV_32F images\n\nIn case of linear transformations, the range does not matter. But in case of a non-linear\ntransformation, an input RGB image should be normalized to the proper value range to get the correct\nresults, for example, for RGB \f$\rightarrow\f$ L\*u\*v\* transformation. For example, if you have a\n32-bit floating-point image directly converted from an 8-bit image without any scaling, then it will\nhave the 0..255 value range instead of 0..1 assumed by the function. So, before calling #cvtColor ,\nyou need first to scale the image down:\n@code\nimg *= 1./255;\ncvtColor(img, img, COLOR_BGR2Luv);\n@endcode\nIf you use #cvtColor with 8-bit images, the conversion will have some information lost. For many\napplications, this will not be noticeable but it is recommended to use 32-bit images in applications\nthat need the full range of colors or that convert an image before an operation and then convert\nback.\n\nIf conversion adds the alpha channel, its value will set to the maximum of corresponding channel\nrange: 255 for CV_8U, 65535 for CV_16U, 1 for CV_32F.\n\n@param src input image: 8-bit unsigned, 16-bit unsigned ( CV_16UC... ), or single-precision\nfloating-point.\n@param dst output image of the same size and depth as src.\n@param code color space conversion code (see #ColorConversionCodes).\n@param dstCn number of channels in the destination image; if the parameter is 0, the number of the\nchannels is derived automatically from src and code.\n\n@see @ref imgproc_color_conversions'''

    def get_attributes(self):
        return [Input('src', 'src')], \
               [Output('dst', 'dst')], \
               [ComboboxParameter('code', [('COLOR_BGR2BGR555',22),('COLOR_BGR2BGR565',12),('COLOR_BGR2BGRA',0),('COLOR_BGR2GRAY',6),('COLOR_BGR2HLS',52),('COLOR_BGR2HLS_FULL',68),('COLOR_BGR2HSV',40),('COLOR_BGR2HSV_FULL',66),('COLOR_BGR2Lab',44),('COLOR_BGR2LAB',44),('COLOR_BGR2Luv',50),('COLOR_BGR2LUV',50),('COLOR_BGR2RGB',4),('COLOR_BGR2RGBA',2),('COLOR_BGR2XYZ',32),('COLOR_BGR2YCrCb',36),('COLOR_BGR2YCR_CB',36),('COLOR_BGR2YUV',82),('COLOR_BGR2YUV_I420',128),('COLOR_BGR2YUV_IYUV',128),('COLOR_BGR2YUV_YV12',132),('COLOR_BGR5552BGR',24),('COLOR_BGR5552BGRA',28),('COLOR_BGR5552GRAY',31),('COLOR_BGR5552RGB',25),('COLOR_BGR5552RGBA',29),('COLOR_BGR5652BGR',14),('COLOR_BGR5652BGRA',18),('COLOR_BGR5652GRAY',21),('COLOR_BGR5652RGB',15),('COLOR_BGR5652RGBA',19),('COLOR_BGRA2BGR',1),('COLOR_BGRA2BGR555',26),('COLOR_BGRA2BGR565',16),('COLOR_BGRA2GRAY',10),('COLOR_BGRA2RGB',3),('COLOR_BGRA2RGBA',5),('COLOR_BGRA2YUV_I420',130),('COLOR_BGRA2YUV_IYUV',130),('COLOR_BGRA2YUV_YV12',134),('COLOR_BayerBG2BGR',46),('COLOR_BAYER_BG2BGR',46),('COLOR_BayerBG2BGRA',139),('COLOR_BAYER_BG2BGRA',139),('COLOR_BayerBG2BGR_EA',135),('COLOR_BAYER_BG2BGR_EA',135),('COLOR_BayerBG2BGR_VNG',62),('COLOR_BAYER_BG2BGR_VNG',62),('COLOR_BayerBG2GRAY',86),('COLOR_BAYER_BG2GRAY',86),('COLOR_BayerBG2RGB',48),('COLOR_BAYER_BG2RGB',48),('COLOR_BayerBG2RGBA',141),('COLOR_BAYER_BG2RGBA',141),('COLOR_BayerBG2RGB_EA',137),('COLOR_BAYER_BG2RGB_EA',137),('COLOR_BayerBG2RGB_VNG',64),('COLOR_BAYER_BG2RGB_VNG',64),('COLOR_BayerGB2BGR',47),('COLOR_BAYER_GB2BGR',47),('COLOR_BayerGB2BGRA',140),('COLOR_BAYER_GB2BGRA',140),('COLOR_BayerGB2BGR_EA',136),('COLOR_BAYER_GB2BGR_EA',136),('COLOR_BayerGB2BGR_VNG',63),('COLOR_BAYER_GB2BGR_VNG',63),('COLOR_BayerGB2GRAY',87),('COLOR_BAYER_GB2GRAY',87),('COLOR_BayerGB2RGB',49),('COLOR_BAYER_GB2RGB',49),('COLOR_BayerGB2RGBA',142),('COLOR_BAYER_GB2RGBA',142),('COLOR_BayerGB2RGB_EA',138),('COLOR_BAYER_GB2RGB_EA',138),('COLOR_BayerGB2RGB_VNG',65),('COLOR_BAYER_GB2RGB_VNG',65),('COLOR_BayerGR2BGR',49),('COLOR_BAYER_GR2BGR',49),('COLOR_BayerGR2BGRA',142),('COLOR_BAYER_GR2BGRA',142),('COLOR_BayerGR2BGR_EA',138),('COLOR_BAYER_GR2BGR_EA',138),('COLOR_BayerGR2BGR_VNG',65),('COLOR_BAYER_GR2BGR_VNG',65),('COLOR_BayerGR2GRAY',89),('COLOR_BAYER_GR2GRAY',89),('COLOR_BayerGR2RGB',47),('COLOR_BAYER_GR2RGB',47),('COLOR_BayerGR2RGBA',140),('COLOR_BAYER_GR2RGBA',140),('COLOR_BayerGR2RGB_EA',136),('COLOR_BAYER_GR2RGB_EA',136),('COLOR_BayerGR2RGB_VNG',63),('COLOR_BAYER_GR2RGB_VNG',63),('COLOR_BayerRG2BGR',48),('COLOR_BAYER_RG2BGR',48),('COLOR_BayerRG2BGRA',141),('COLOR_BAYER_RG2BGRA',141),('COLOR_BayerRG2BGR_EA',137),('COLOR_BAYER_RG2BGR_EA',137),('COLOR_BayerRG2BGR_VNG',64),('COLOR_BAYER_RG2BGR_VNG',64),('COLOR_BayerRG2GRAY',88),('COLOR_BAYER_RG2GRAY',88),('COLOR_BayerRG2RGB',46),('COLOR_BAYER_RG2RGB',46),('COLOR_BayerRG2RGBA',139),('COLOR_BAYER_RG2RGBA',139),('COLOR_BayerRG2RGB_EA',135),('COLOR_BAYER_RG2RGB_EA',135),('COLOR_BayerRG2RGB_VNG',62),('COLOR_BAYER_RG2RGB_VNG',62),('COLOR_COLORCVT_MAX',143),('COLOR_GRAY2BGR',8),('COLOR_GRAY2BGR555',30),('COLOR_GRAY2BGR565',20),('COLOR_GRAY2BGRA',9),('COLOR_GRAY2RGB',8),('COLOR_GRAY2RGBA',9),('COLOR_HLS2BGR',60),('COLOR_HLS2BGR_FULL',72),('COLOR_HLS2RGB',61),('COLOR_HLS2RGB_FULL',73),('COLOR_HSV2BGR',54),('COLOR_HSV2BGR_FULL',70),('COLOR_HSV2RGB',55),('COLOR_HSV2RGB_FULL',71),('COLOR_LBGR2Lab',74),('COLOR_LBGR2LAB',74),('COLOR_LBGR2Luv',76),('COLOR_LBGR2LUV',76),('COLOR_LRGB2Lab',75),('COLOR_LRGB2LAB',75),('COLOR_LRGB2Luv',77),('COLOR_LRGB2LUV',77),('COLOR_Lab2BGR',56),('COLOR_LAB2BGR',56),('COLOR_Lab2LBGR',78),('COLOR_LAB2LBGR',78),('COLOR_Lab2LRGB',79),('COLOR_LAB2LRGB',79),('COLOR_Lab2RGB',57),('COLOR_LAB2RGB',57),('COLOR_Luv2BGR',58),('COLOR_LUV2BGR',58),('COLOR_Luv2LBGR',80),('COLOR_LUV2LBGR',80),('COLOR_Luv2LRGB',81),('COLOR_LUV2LRGB',81),('COLOR_Luv2RGB',59),('COLOR_LUV2RGB',59),('COLOR_RGB2BGR',4),('COLOR_RGB2BGR555',23),('COLOR_RGB2BGR565',13),('COLOR_RGB2BGRA',2),('COLOR_RGB2GRAY',7),('COLOR_RGB2HLS',53),('COLOR_RGB2HLS_FULL',69),('COLOR_RGB2HSV',41),('COLOR_RGB2HSV_FULL',67),('COLOR_RGB2Lab',45),('COLOR_RGB2LAB',45),('COLOR_RGB2Luv',51),('COLOR_RGB2LUV',51),('COLOR_RGB2RGBA',0),('COLOR_RGB2XYZ',33),('COLOR_RGB2YCrCb',37),('COLOR_RGB2YCR_CB',37),('COLOR_RGB2YUV',83),('COLOR_RGB2YUV_I420',127),('COLOR_RGB2YUV_IYUV',127),('COLOR_RGB2YUV_YV12',131),('COLOR_RGBA2BGR',3),('COLOR_RGBA2BGR555',27),('COLOR_RGBA2BGR565',17),('COLOR_RGBA2BGRA',5),('COLOR_RGBA2GRAY',11),('COLOR_RGBA2RGB',1),('COLOR_RGBA2YUV_I420',129),('COLOR_RGBA2YUV_IYUV',129),('COLOR_RGBA2YUV_YV12',133),('COLOR_RGBA2mRGBA',125),('COLOR_RGBA2M_RGBA',125),('COLOR_XYZ2BGR',34),('COLOR_XYZ2RGB',35),('COLOR_YCrCb2BGR',38),('COLOR_YCR_CB2BGR',38),('COLOR_YCrCb2RGB',39),('COLOR_YCR_CB2RGB',39),('COLOR_YUV2BGR',84),('COLOR_YUV2BGRA_I420',105),('COLOR_YUV2BGRA_IYUV',105),('COLOR_YUV2BGRA_NV12',95),('COLOR_YUV2BGRA_NV21',97),('COLOR_YUV2BGRA_UYNV',112),('COLOR_YUV2BGRA_UYVY',112),('COLOR_YUV2BGRA_Y422',112),('COLOR_YUV2BGRA_YUNV',120),('COLOR_YUV2BGRA_YUY2',120),('COLOR_YUV2BGRA_YUYV',120),('COLOR_YUV2BGRA_YV12',103),('COLOR_YUV2BGRA_YVYU',122),('COLOR_YUV2BGR_I420',101),('COLOR_YUV2BGR_IYUV',101),('COLOR_YUV2BGR_NV12',91),('COLOR_YUV2BGR_NV21',93),('COLOR_YUV2BGR_UYNV',108),('COLOR_YUV2BGR_UYVY',108),('COLOR_YUV2BGR_Y422',108),('COLOR_YUV2BGR_YUNV',116),('COLOR_YUV2BGR_YUY2',116),('COLOR_YUV2BGR_YUYV',116),('COLOR_YUV2BGR_YV12',99),('COLOR_YUV2BGR_YVYU',118),('COLOR_YUV2GRAY_420',106),('COLOR_YUV2GRAY_I420',106),('COLOR_YUV2GRAY_IYUV',106),('COLOR_YUV2GRAY_NV12',106),('COLOR_YUV2GRAY_NV21',106),('COLOR_YUV2GRAY_UYNV',123),('COLOR_YUV2GRAY_UYVY',123),('COLOR_YUV2GRAY_Y422',123),('COLOR_YUV2GRAY_YUNV',124),('COLOR_YUV2GRAY_YUY2',124),('COLOR_YUV2GRAY_YUYV',124),('COLOR_YUV2GRAY_YV12',106),('COLOR_YUV2GRAY_YVYU',124),('COLOR_YUV2RGB',85),('COLOR_YUV2RGBA_I420',104),('COLOR_YUV2RGBA_IYUV',104),('COLOR_YUV2RGBA_NV12',94),('COLOR_YUV2RGBA_NV21',96),('COLOR_YUV2RGBA_UYNV',111),('COLOR_YUV2RGBA_UYVY',111),('COLOR_YUV2RGBA_Y422',111),('COLOR_YUV2RGBA_YUNV',119),('COLOR_YUV2RGBA_YUY2',119),('COLOR_YUV2RGBA_YUYV',119),('COLOR_YUV2RGBA_YV12',102),('COLOR_YUV2RGBA_YVYU',121),('COLOR_YUV2RGB_I420',100),('COLOR_YUV2RGB_IYUV',100),('COLOR_YUV2RGB_NV12',90),('COLOR_YUV2RGB_NV21',92),('COLOR_YUV2RGB_UYNV',107),('COLOR_YUV2RGB_UYVY',107),('COLOR_YUV2RGB_Y422',107),('COLOR_YUV2RGB_YUNV',115),('COLOR_YUV2RGB_YUY2',115),('COLOR_YUV2RGB_YUYV',115),('COLOR_YUV2RGB_YV12',98),('COLOR_YUV2RGB_YVYU',117),('COLOR_YUV420p2BGR',99),('COLOR_YUV420P2BGR',99),('COLOR_YUV420p2BGRA',103),('COLOR_YUV420P2BGRA',103),('COLOR_YUV420p2GRAY',106),('COLOR_YUV420P2GRAY',106),('COLOR_YUV420p2RGB',98),('COLOR_YUV420P2RGB',98),('COLOR_YUV420p2RGBA',102),('COLOR_YUV420P2RGBA',102),('COLOR_YUV420sp2BGR',93),('COLOR_YUV420SP2BGR',93),('COLOR_YUV420sp2BGRA',97),('COLOR_YUV420SP2BGRA',97),('COLOR_YUV420sp2GRAY',106),('COLOR_YUV420SP2GRAY',106),('COLOR_YUV420sp2RGB',92),('COLOR_YUV420SP2RGB',92),('COLOR_YUV420sp2RGBA',96),('COLOR_YUV420SP2RGBA',96),('COLOR_mRGBA2RGBA',126),('COLOR_M_RGBA2RGBA',126)])]

    def process_inputs(self, inputs, outputs, parameters):
        src = inputs['src'].value
        code = parameters['code']
        dst = cv2.cvtColor(src=src, code=code)
        outputs['dst'] = Data(dst)

### cvtColorTwoPlane ###

class OpenCVAuto2_CvtColorTwoPlane(NormalElement):
    name = 'CvtColorTwoPlane'
    comment = '''cvtColorTwoPlane(src1, src2, code[, dst]) -> dst
.'''

    def get_attributes(self):
        return [Input('src1', 'src1'),
                Input('src2', 'src2')], \
               [Output('dst', 'dst')], \
               [ComboboxParameter('code', [('COLOR_BGR2BGR555',22),('COLOR_BGR2BGR565',12),('COLOR_BGR2BGRA',0),('COLOR_BGR2GRAY',6),('COLOR_BGR2HLS',52),('COLOR_BGR2HLS_FULL',68),('COLOR_BGR2HSV',40),('COLOR_BGR2HSV_FULL',66),('COLOR_BGR2Lab',44),('COLOR_BGR2LAB',44),('COLOR_BGR2Luv',50),('COLOR_BGR2LUV',50),('COLOR_BGR2RGB',4),('COLOR_BGR2RGBA',2),('COLOR_BGR2XYZ',32),('COLOR_BGR2YCrCb',36),('COLOR_BGR2YCR_CB',36),('COLOR_BGR2YUV',82),('COLOR_BGR2YUV_I420',128),('COLOR_BGR2YUV_IYUV',128),('COLOR_BGR2YUV_YV12',132),('COLOR_BGR5552BGR',24),('COLOR_BGR5552BGRA',28),('COLOR_BGR5552GRAY',31),('COLOR_BGR5552RGB',25),('COLOR_BGR5552RGBA',29),('COLOR_BGR5652BGR',14),('COLOR_BGR5652BGRA',18),('COLOR_BGR5652GRAY',21),('COLOR_BGR5652RGB',15),('COLOR_BGR5652RGBA',19),('COLOR_BGRA2BGR',1),('COLOR_BGRA2BGR555',26),('COLOR_BGRA2BGR565',16),('COLOR_BGRA2GRAY',10),('COLOR_BGRA2RGB',3),('COLOR_BGRA2RGBA',5),('COLOR_BGRA2YUV_I420',130),('COLOR_BGRA2YUV_IYUV',130),('COLOR_BGRA2YUV_YV12',134),('COLOR_BayerBG2BGR',46),('COLOR_BAYER_BG2BGR',46),('COLOR_BayerBG2BGRA',139),('COLOR_BAYER_BG2BGRA',139),('COLOR_BayerBG2BGR_EA',135),('COLOR_BAYER_BG2BGR_EA',135),('COLOR_BayerBG2BGR_VNG',62),('COLOR_BAYER_BG2BGR_VNG',62),('COLOR_BayerBG2GRAY',86),('COLOR_BAYER_BG2GRAY',86),('COLOR_BayerBG2RGB',48),('COLOR_BAYER_BG2RGB',48),('COLOR_BayerBG2RGBA',141),('COLOR_BAYER_BG2RGBA',141),('COLOR_BayerBG2RGB_EA',137),('COLOR_BAYER_BG2RGB_EA',137),('COLOR_BayerBG2RGB_VNG',64),('COLOR_BAYER_BG2RGB_VNG',64),('COLOR_BayerGB2BGR',47),('COLOR_BAYER_GB2BGR',47),('COLOR_BayerGB2BGRA',140),('COLOR_BAYER_GB2BGRA',140),('COLOR_BayerGB2BGR_EA',136),('COLOR_BAYER_GB2BGR_EA',136),('COLOR_BayerGB2BGR_VNG',63),('COLOR_BAYER_GB2BGR_VNG',63),('COLOR_BayerGB2GRAY',87),('COLOR_BAYER_GB2GRAY',87),('COLOR_BayerGB2RGB',49),('COLOR_BAYER_GB2RGB',49),('COLOR_BayerGB2RGBA',142),('COLOR_BAYER_GB2RGBA',142),('COLOR_BayerGB2RGB_EA',138),('COLOR_BAYER_GB2RGB_EA',138),('COLOR_BayerGB2RGB_VNG',65),('COLOR_BAYER_GB2RGB_VNG',65),('COLOR_BayerGR2BGR',49),('COLOR_BAYER_GR2BGR',49),('COLOR_BayerGR2BGRA',142),('COLOR_BAYER_GR2BGRA',142),('COLOR_BayerGR2BGR_EA',138),('COLOR_BAYER_GR2BGR_EA',138),('COLOR_BayerGR2BGR_VNG',65),('COLOR_BAYER_GR2BGR_VNG',65),('COLOR_BayerGR2GRAY',89),('COLOR_BAYER_GR2GRAY',89),('COLOR_BayerGR2RGB',47),('COLOR_BAYER_GR2RGB',47),('COLOR_BayerGR2RGBA',140),('COLOR_BAYER_GR2RGBA',140),('COLOR_BayerGR2RGB_EA',136),('COLOR_BAYER_GR2RGB_EA',136),('COLOR_BayerGR2RGB_VNG',63),('COLOR_BAYER_GR2RGB_VNG',63),('COLOR_BayerRG2BGR',48),('COLOR_BAYER_RG2BGR',48),('COLOR_BayerRG2BGRA',141),('COLOR_BAYER_RG2BGRA',141),('COLOR_BayerRG2BGR_EA',137),('COLOR_BAYER_RG2BGR_EA',137),('COLOR_BayerRG2BGR_VNG',64),('COLOR_BAYER_RG2BGR_VNG',64),('COLOR_BayerRG2GRAY',88),('COLOR_BAYER_RG2GRAY',88),('COLOR_BayerRG2RGB',46),('COLOR_BAYER_RG2RGB',46),('COLOR_BayerRG2RGBA',139),('COLOR_BAYER_RG2RGBA',139),('COLOR_BayerRG2RGB_EA',135),('COLOR_BAYER_RG2RGB_EA',135),('COLOR_BayerRG2RGB_VNG',62),('COLOR_BAYER_RG2RGB_VNG',62),('COLOR_COLORCVT_MAX',143),('COLOR_GRAY2BGR',8),('COLOR_GRAY2BGR555',30),('COLOR_GRAY2BGR565',20),('COLOR_GRAY2BGRA',9),('COLOR_GRAY2RGB',8),('COLOR_GRAY2RGBA',9),('COLOR_HLS2BGR',60),('COLOR_HLS2BGR_FULL',72),('COLOR_HLS2RGB',61),('COLOR_HLS2RGB_FULL',73),('COLOR_HSV2BGR',54),('COLOR_HSV2BGR_FULL',70),('COLOR_HSV2RGB',55),('COLOR_HSV2RGB_FULL',71),('COLOR_LBGR2Lab',74),('COLOR_LBGR2LAB',74),('COLOR_LBGR2Luv',76),('COLOR_LBGR2LUV',76),('COLOR_LRGB2Lab',75),('COLOR_LRGB2LAB',75),('COLOR_LRGB2Luv',77),('COLOR_LRGB2LUV',77),('COLOR_Lab2BGR',56),('COLOR_LAB2BGR',56),('COLOR_Lab2LBGR',78),('COLOR_LAB2LBGR',78),('COLOR_Lab2LRGB',79),('COLOR_LAB2LRGB',79),('COLOR_Lab2RGB',57),('COLOR_LAB2RGB',57),('COLOR_Luv2BGR',58),('COLOR_LUV2BGR',58),('COLOR_Luv2LBGR',80),('COLOR_LUV2LBGR',80),('COLOR_Luv2LRGB',81),('COLOR_LUV2LRGB',81),('COLOR_Luv2RGB',59),('COLOR_LUV2RGB',59),('COLOR_RGB2BGR',4),('COLOR_RGB2BGR555',23),('COLOR_RGB2BGR565',13),('COLOR_RGB2BGRA',2),('COLOR_RGB2GRAY',7),('COLOR_RGB2HLS',53),('COLOR_RGB2HLS_FULL',69),('COLOR_RGB2HSV',41),('COLOR_RGB2HSV_FULL',67),('COLOR_RGB2Lab',45),('COLOR_RGB2LAB',45),('COLOR_RGB2Luv',51),('COLOR_RGB2LUV',51),('COLOR_RGB2RGBA',0),('COLOR_RGB2XYZ',33),('COLOR_RGB2YCrCb',37),('COLOR_RGB2YCR_CB',37),('COLOR_RGB2YUV',83),('COLOR_RGB2YUV_I420',127),('COLOR_RGB2YUV_IYUV',127),('COLOR_RGB2YUV_YV12',131),('COLOR_RGBA2BGR',3),('COLOR_RGBA2BGR555',27),('COLOR_RGBA2BGR565',17),('COLOR_RGBA2BGRA',5),('COLOR_RGBA2GRAY',11),('COLOR_RGBA2RGB',1),('COLOR_RGBA2YUV_I420',129),('COLOR_RGBA2YUV_IYUV',129),('COLOR_RGBA2YUV_YV12',133),('COLOR_RGBA2mRGBA',125),('COLOR_RGBA2M_RGBA',125),('COLOR_XYZ2BGR',34),('COLOR_XYZ2RGB',35),('COLOR_YCrCb2BGR',38),('COLOR_YCR_CB2BGR',38),('COLOR_YCrCb2RGB',39),('COLOR_YCR_CB2RGB',39),('COLOR_YUV2BGR',84),('COLOR_YUV2BGRA_I420',105),('COLOR_YUV2BGRA_IYUV',105),('COLOR_YUV2BGRA_NV12',95),('COLOR_YUV2BGRA_NV21',97),('COLOR_YUV2BGRA_UYNV',112),('COLOR_YUV2BGRA_UYVY',112),('COLOR_YUV2BGRA_Y422',112),('COLOR_YUV2BGRA_YUNV',120),('COLOR_YUV2BGRA_YUY2',120),('COLOR_YUV2BGRA_YUYV',120),('COLOR_YUV2BGRA_YV12',103),('COLOR_YUV2BGRA_YVYU',122),('COLOR_YUV2BGR_I420',101),('COLOR_YUV2BGR_IYUV',101),('COLOR_YUV2BGR_NV12',91),('COLOR_YUV2BGR_NV21',93),('COLOR_YUV2BGR_UYNV',108),('COLOR_YUV2BGR_UYVY',108),('COLOR_YUV2BGR_Y422',108),('COLOR_YUV2BGR_YUNV',116),('COLOR_YUV2BGR_YUY2',116),('COLOR_YUV2BGR_YUYV',116),('COLOR_YUV2BGR_YV12',99),('COLOR_YUV2BGR_YVYU',118),('COLOR_YUV2GRAY_420',106),('COLOR_YUV2GRAY_I420',106),('COLOR_YUV2GRAY_IYUV',106),('COLOR_YUV2GRAY_NV12',106),('COLOR_YUV2GRAY_NV21',106),('COLOR_YUV2GRAY_UYNV',123),('COLOR_YUV2GRAY_UYVY',123),('COLOR_YUV2GRAY_Y422',123),('COLOR_YUV2GRAY_YUNV',124),('COLOR_YUV2GRAY_YUY2',124),('COLOR_YUV2GRAY_YUYV',124),('COLOR_YUV2GRAY_YV12',106),('COLOR_YUV2GRAY_YVYU',124),('COLOR_YUV2RGB',85),('COLOR_YUV2RGBA_I420',104),('COLOR_YUV2RGBA_IYUV',104),('COLOR_YUV2RGBA_NV12',94),('COLOR_YUV2RGBA_NV21',96),('COLOR_YUV2RGBA_UYNV',111),('COLOR_YUV2RGBA_UYVY',111),('COLOR_YUV2RGBA_Y422',111),('COLOR_YUV2RGBA_YUNV',119),('COLOR_YUV2RGBA_YUY2',119),('COLOR_YUV2RGBA_YUYV',119),('COLOR_YUV2RGBA_YV12',102),('COLOR_YUV2RGBA_YVYU',121),('COLOR_YUV2RGB_I420',100),('COLOR_YUV2RGB_IYUV',100),('COLOR_YUV2RGB_NV12',90),('COLOR_YUV2RGB_NV21',92),('COLOR_YUV2RGB_UYNV',107),('COLOR_YUV2RGB_UYVY',107),('COLOR_YUV2RGB_Y422',107),('COLOR_YUV2RGB_YUNV',115),('COLOR_YUV2RGB_YUY2',115),('COLOR_YUV2RGB_YUYV',115),('COLOR_YUV2RGB_YV12',98),('COLOR_YUV2RGB_YVYU',117),('COLOR_YUV420p2BGR',99),('COLOR_YUV420P2BGR',99),('COLOR_YUV420p2BGRA',103),('COLOR_YUV420P2BGRA',103),('COLOR_YUV420p2GRAY',106),('COLOR_YUV420P2GRAY',106),('COLOR_YUV420p2RGB',98),('COLOR_YUV420P2RGB',98),('COLOR_YUV420p2RGBA',102),('COLOR_YUV420P2RGBA',102),('COLOR_YUV420sp2BGR',93),('COLOR_YUV420SP2BGR',93),('COLOR_YUV420sp2BGRA',97),('COLOR_YUV420SP2BGRA',97),('COLOR_YUV420sp2GRAY',106),('COLOR_YUV420SP2GRAY',106),('COLOR_YUV420sp2RGB',92),('COLOR_YUV420SP2RGB',92),('COLOR_YUV420sp2RGBA',96),('COLOR_YUV420SP2RGBA',96),('COLOR_mRGBA2RGBA',126),('COLOR_M_RGBA2RGBA',126)])]

    def process_inputs(self, inputs, outputs, parameters):
        src1 = inputs['src1'].value
        src2 = inputs['src2'].value
        code = parameters['code']
        dst = cv2.cvtColorTwoPlane(src1=src1, src2=src2, code=code)
        outputs['dst'] = Data(dst)

### decolor ###

class OpenCVAuto2_Decolor(NormalElement):
    name = 'Decolor'
    comment = '''decolor(src[, grayscale[, color_boost]]) -> grayscale, color_boost\n@brief Transforms a color image to a grayscale image. It is a basic tool in digital printing, stylized\nblack-and-white photograph rendering, and in many single channel image processing applications\n@cite CL12 .\n\n@param src Input 8-bit 3-channel image.\n@param grayscale Output 8-bit 1-channel image.\n@param color_boost Output 8-bit 3-channel image.\n\nThis function is to be applied on color images.'''

    def get_attributes(self):
        return [Input('src', 'src')], \
               [Output('grayscale', 'grayscale'),
                Output('color_boost', 'color_boost')], \
               []

    def process_inputs(self, inputs, outputs, parameters):
        src = inputs['src'].value
        grayscale, color_boost = cv2.decolor(src=src)
        outputs['grayscale'] = Data(grayscale)
        outputs['color_boost'] = Data(color_boost)

### decomposeProjectionMatrix ###

class OpenCVAuto2_DecomposeProjectionMatrix(NormalElement):
    name = 'DecomposeProjectionMatrix'
    comment = '''decomposeProjectionMatrix(projMatrix[, cameraMatrix[, rotMatrix[, transVect[, rotMatrixX[, rotMatrixY[, rotMatrixZ[, eulerAngles]]]]]]]) -> cameraMatrix, rotMatrix, transVect, rotMatrixX, rotMatrixY, rotMatrixZ, eulerAngles\n@brief Decomposes a projection matrix into a rotation matrix and a camera matrix.\n\n@param projMatrix 3x4 input projection matrix P.\n@param cameraMatrix Output 3x3 camera matrix K.\n@param rotMatrix Output 3x3 external rotation matrix R.\n@param transVect Output 4x1 translation vector T.\n@param rotMatrixX Optional 3x3 rotation matrix around x-axis.\n@param rotMatrixY Optional 3x3 rotation matrix around y-axis.\n@param rotMatrixZ Optional 3x3 rotation matrix around z-axis.\n@param eulerAngles Optional three-element vector containing three Euler angles of rotation in\ndegrees.\n\nThe function computes a decomposition of a projection matrix into a calibration and a rotation\nmatrix and the position of a camera.\n\nIt optionally returns three rotation matrices, one for each axis, and three Euler angles that could\nbe used in OpenGL. Note, there is always more than one sequence of rotations about the three\nprincipal axes that results in the same orientation of an object, e.g. see @cite Slabaugh . Returned\ntree rotation matrices and corresponding three Euler angles are only one of the possible solutions.\n\nThe function is based on RQDecomp3x3 .'''

    def get_attributes(self):
        return [Input('projMatrix', 'projMatrix')], \
               [Output('cameraMatrix', 'cameraMatrix'),
                Output('rotMatrix', 'rotMatrix'),
                Output('transVect', 'transVect'),
                Output('rotMatrixX', 'rotMatrixX'),
                Output('rotMatrixY', 'rotMatrixY'),
                Output('rotMatrixZ', 'rotMatrixZ'),
                Output('eulerAngles', 'eulerAngles')], \
               []

    def process_inputs(self, inputs, outputs, parameters):
        projMatrix = inputs['projMatrix'].value
        cameraMatrix, rotMatrix, transVect, rotMatrixX, rotMatrixY, rotMatrixZ, eulerAngles = cv2.decomposeProjectionMatrix(projMatrix=projMatrix)
        outputs['cameraMatrix'] = Data(cameraMatrix)
        outputs['rotMatrix'] = Data(rotMatrix)
        outputs['transVect'] = Data(transVect)
        outputs['rotMatrixX'] = Data(rotMatrixX)
        outputs['rotMatrixY'] = Data(rotMatrixY)
        outputs['rotMatrixZ'] = Data(rotMatrixZ)
        outputs['eulerAngles'] = Data(eulerAngles)

### demosaicing ###

class OpenCVAuto2_Demosaicing(NormalElement):
    name = 'Demosaicing'
    comment = '''demosaicing(_src, code[, _dst[, dcn]]) -> _dst
.'''

    def get_attributes(self):
        return [Input('_src', '_src')], \
               [Output('_dst', '_dst')], \
               [ComboboxParameter('code', [('COLOR_BGR2BGR555',22),('COLOR_BGR2BGR565',12),('COLOR_BGR2BGRA',0),('COLOR_BGR2GRAY',6),('COLOR_BGR2HLS',52),('COLOR_BGR2HLS_FULL',68),('COLOR_BGR2HSV',40),('COLOR_BGR2HSV_FULL',66),('COLOR_BGR2Lab',44),('COLOR_BGR2LAB',44),('COLOR_BGR2Luv',50),('COLOR_BGR2LUV',50),('COLOR_BGR2RGB',4),('COLOR_BGR2RGBA',2),('COLOR_BGR2XYZ',32),('COLOR_BGR2YCrCb',36),('COLOR_BGR2YCR_CB',36),('COLOR_BGR2YUV',82),('COLOR_BGR2YUV_I420',128),('COLOR_BGR2YUV_IYUV',128),('COLOR_BGR2YUV_YV12',132),('COLOR_BGR5552BGR',24),('COLOR_BGR5552BGRA',28),('COLOR_BGR5552GRAY',31),('COLOR_BGR5552RGB',25),('COLOR_BGR5552RGBA',29),('COLOR_BGR5652BGR',14),('COLOR_BGR5652BGRA',18),('COLOR_BGR5652GRAY',21),('COLOR_BGR5652RGB',15),('COLOR_BGR5652RGBA',19),('COLOR_BGRA2BGR',1),('COLOR_BGRA2BGR555',26),('COLOR_BGRA2BGR565',16),('COLOR_BGRA2GRAY',10),('COLOR_BGRA2RGB',3),('COLOR_BGRA2RGBA',5),('COLOR_BGRA2YUV_I420',130),('COLOR_BGRA2YUV_IYUV',130),('COLOR_BGRA2YUV_YV12',134),('COLOR_BayerBG2BGR',46),('COLOR_BAYER_BG2BGR',46),('COLOR_BayerBG2BGRA',139),('COLOR_BAYER_BG2BGRA',139),('COLOR_BayerBG2BGR_EA',135),('COLOR_BAYER_BG2BGR_EA',135),('COLOR_BayerBG2BGR_VNG',62),('COLOR_BAYER_BG2BGR_VNG',62),('COLOR_BayerBG2GRAY',86),('COLOR_BAYER_BG2GRAY',86),('COLOR_BayerBG2RGB',48),('COLOR_BAYER_BG2RGB',48),('COLOR_BayerBG2RGBA',141),('COLOR_BAYER_BG2RGBA',141),('COLOR_BayerBG2RGB_EA',137),('COLOR_BAYER_BG2RGB_EA',137),('COLOR_BayerBG2RGB_VNG',64),('COLOR_BAYER_BG2RGB_VNG',64),('COLOR_BayerGB2BGR',47),('COLOR_BAYER_GB2BGR',47),('COLOR_BayerGB2BGRA',140),('COLOR_BAYER_GB2BGRA',140),('COLOR_BayerGB2BGR_EA',136),('COLOR_BAYER_GB2BGR_EA',136),('COLOR_BayerGB2BGR_VNG',63),('COLOR_BAYER_GB2BGR_VNG',63),('COLOR_BayerGB2GRAY',87),('COLOR_BAYER_GB2GRAY',87),('COLOR_BayerGB2RGB',49),('COLOR_BAYER_GB2RGB',49),('COLOR_BayerGB2RGBA',142),('COLOR_BAYER_GB2RGBA',142),('COLOR_BayerGB2RGB_EA',138),('COLOR_BAYER_GB2RGB_EA',138),('COLOR_BayerGB2RGB_VNG',65),('COLOR_BAYER_GB2RGB_VNG',65),('COLOR_BayerGR2BGR',49),('COLOR_BAYER_GR2BGR',49),('COLOR_BayerGR2BGRA',142),('COLOR_BAYER_GR2BGRA',142),('COLOR_BayerGR2BGR_EA',138),('COLOR_BAYER_GR2BGR_EA',138),('COLOR_BayerGR2BGR_VNG',65),('COLOR_BAYER_GR2BGR_VNG',65),('COLOR_BayerGR2GRAY',89),('COLOR_BAYER_GR2GRAY',89),('COLOR_BayerGR2RGB',47),('COLOR_BAYER_GR2RGB',47),('COLOR_BayerGR2RGBA',140),('COLOR_BAYER_GR2RGBA',140),('COLOR_BayerGR2RGB_EA',136),('COLOR_BAYER_GR2RGB_EA',136),('COLOR_BayerGR2RGB_VNG',63),('COLOR_BAYER_GR2RGB_VNG',63),('COLOR_BayerRG2BGR',48),('COLOR_BAYER_RG2BGR',48),('COLOR_BayerRG2BGRA',141),('COLOR_BAYER_RG2BGRA',141),('COLOR_BayerRG2BGR_EA',137),('COLOR_BAYER_RG2BGR_EA',137),('COLOR_BayerRG2BGR_VNG',64),('COLOR_BAYER_RG2BGR_VNG',64),('COLOR_BayerRG2GRAY',88),('COLOR_BAYER_RG2GRAY',88),('COLOR_BayerRG2RGB',46),('COLOR_BAYER_RG2RGB',46),('COLOR_BayerRG2RGBA',139),('COLOR_BAYER_RG2RGBA',139),('COLOR_BayerRG2RGB_EA',135),('COLOR_BAYER_RG2RGB_EA',135),('COLOR_BayerRG2RGB_VNG',62),('COLOR_BAYER_RG2RGB_VNG',62),('COLOR_COLORCVT_MAX',143),('COLOR_GRAY2BGR',8),('COLOR_GRAY2BGR555',30),('COLOR_GRAY2BGR565',20),('COLOR_GRAY2BGRA',9),('COLOR_GRAY2RGB',8),('COLOR_GRAY2RGBA',9),('COLOR_HLS2BGR',60),('COLOR_HLS2BGR_FULL',72),('COLOR_HLS2RGB',61),('COLOR_HLS2RGB_FULL',73),('COLOR_HSV2BGR',54),('COLOR_HSV2BGR_FULL',70),('COLOR_HSV2RGB',55),('COLOR_HSV2RGB_FULL',71),('COLOR_LBGR2Lab',74),('COLOR_LBGR2LAB',74),('COLOR_LBGR2Luv',76),('COLOR_LBGR2LUV',76),('COLOR_LRGB2Lab',75),('COLOR_LRGB2LAB',75),('COLOR_LRGB2Luv',77),('COLOR_LRGB2LUV',77),('COLOR_Lab2BGR',56),('COLOR_LAB2BGR',56),('COLOR_Lab2LBGR',78),('COLOR_LAB2LBGR',78),('COLOR_Lab2LRGB',79),('COLOR_LAB2LRGB',79),('COLOR_Lab2RGB',57),('COLOR_LAB2RGB',57),('COLOR_Luv2BGR',58),('COLOR_LUV2BGR',58),('COLOR_Luv2LBGR',80),('COLOR_LUV2LBGR',80),('COLOR_Luv2LRGB',81),('COLOR_LUV2LRGB',81),('COLOR_Luv2RGB',59),('COLOR_LUV2RGB',59),('COLOR_RGB2BGR',4),('COLOR_RGB2BGR555',23),('COLOR_RGB2BGR565',13),('COLOR_RGB2BGRA',2),('COLOR_RGB2GRAY',7),('COLOR_RGB2HLS',53),('COLOR_RGB2HLS_FULL',69),('COLOR_RGB2HSV',41),('COLOR_RGB2HSV_FULL',67),('COLOR_RGB2Lab',45),('COLOR_RGB2LAB',45),('COLOR_RGB2Luv',51),('COLOR_RGB2LUV',51),('COLOR_RGB2RGBA',0),('COLOR_RGB2XYZ',33),('COLOR_RGB2YCrCb',37),('COLOR_RGB2YCR_CB',37),('COLOR_RGB2YUV',83),('COLOR_RGB2YUV_I420',127),('COLOR_RGB2YUV_IYUV',127),('COLOR_RGB2YUV_YV12',131),('COLOR_RGBA2BGR',3),('COLOR_RGBA2BGR555',27),('COLOR_RGBA2BGR565',17),('COLOR_RGBA2BGRA',5),('COLOR_RGBA2GRAY',11),('COLOR_RGBA2RGB',1),('COLOR_RGBA2YUV_I420',129),('COLOR_RGBA2YUV_IYUV',129),('COLOR_RGBA2YUV_YV12',133),('COLOR_RGBA2mRGBA',125),('COLOR_RGBA2M_RGBA',125),('COLOR_XYZ2BGR',34),('COLOR_XYZ2RGB',35),('COLOR_YCrCb2BGR',38),('COLOR_YCR_CB2BGR',38),('COLOR_YCrCb2RGB',39),('COLOR_YCR_CB2RGB',39),('COLOR_YUV2BGR',84),('COLOR_YUV2BGRA_I420',105),('COLOR_YUV2BGRA_IYUV',105),('COLOR_YUV2BGRA_NV12',95),('COLOR_YUV2BGRA_NV21',97),('COLOR_YUV2BGRA_UYNV',112),('COLOR_YUV2BGRA_UYVY',112),('COLOR_YUV2BGRA_Y422',112),('COLOR_YUV2BGRA_YUNV',120),('COLOR_YUV2BGRA_YUY2',120),('COLOR_YUV2BGRA_YUYV',120),('COLOR_YUV2BGRA_YV12',103),('COLOR_YUV2BGRA_YVYU',122),('COLOR_YUV2BGR_I420',101),('COLOR_YUV2BGR_IYUV',101),('COLOR_YUV2BGR_NV12',91),('COLOR_YUV2BGR_NV21',93),('COLOR_YUV2BGR_UYNV',108),('COLOR_YUV2BGR_UYVY',108),('COLOR_YUV2BGR_Y422',108),('COLOR_YUV2BGR_YUNV',116),('COLOR_YUV2BGR_YUY2',116),('COLOR_YUV2BGR_YUYV',116),('COLOR_YUV2BGR_YV12',99),('COLOR_YUV2BGR_YVYU',118),('COLOR_YUV2GRAY_420',106),('COLOR_YUV2GRAY_I420',106),('COLOR_YUV2GRAY_IYUV',106),('COLOR_YUV2GRAY_NV12',106),('COLOR_YUV2GRAY_NV21',106),('COLOR_YUV2GRAY_UYNV',123),('COLOR_YUV2GRAY_UYVY',123),('COLOR_YUV2GRAY_Y422',123),('COLOR_YUV2GRAY_YUNV',124),('COLOR_YUV2GRAY_YUY2',124),('COLOR_YUV2GRAY_YUYV',124),('COLOR_YUV2GRAY_YV12',106),('COLOR_YUV2GRAY_YVYU',124),('COLOR_YUV2RGB',85),('COLOR_YUV2RGBA_I420',104),('COLOR_YUV2RGBA_IYUV',104),('COLOR_YUV2RGBA_NV12',94),('COLOR_YUV2RGBA_NV21',96),('COLOR_YUV2RGBA_UYNV',111),('COLOR_YUV2RGBA_UYVY',111),('COLOR_YUV2RGBA_Y422',111),('COLOR_YUV2RGBA_YUNV',119),('COLOR_YUV2RGBA_YUY2',119),('COLOR_YUV2RGBA_YUYV',119),('COLOR_YUV2RGBA_YV12',102),('COLOR_YUV2RGBA_YVYU',121),('COLOR_YUV2RGB_I420',100),('COLOR_YUV2RGB_IYUV',100),('COLOR_YUV2RGB_NV12',90),('COLOR_YUV2RGB_NV21',92),('COLOR_YUV2RGB_UYNV',107),('COLOR_YUV2RGB_UYVY',107),('COLOR_YUV2RGB_Y422',107),('COLOR_YUV2RGB_YUNV',115),('COLOR_YUV2RGB_YUY2',115),('COLOR_YUV2RGB_YUYV',115),('COLOR_YUV2RGB_YV12',98),('COLOR_YUV2RGB_YVYU',117),('COLOR_YUV420p2BGR',99),('COLOR_YUV420P2BGR',99),('COLOR_YUV420p2BGRA',103),('COLOR_YUV420P2BGRA',103),('COLOR_YUV420p2GRAY',106),('COLOR_YUV420P2GRAY',106),('COLOR_YUV420p2RGB',98),('COLOR_YUV420P2RGB',98),('COLOR_YUV420p2RGBA',102),('COLOR_YUV420P2RGBA',102),('COLOR_YUV420sp2BGR',93),('COLOR_YUV420SP2BGR',93),('COLOR_YUV420sp2BGRA',97),('COLOR_YUV420SP2BGRA',97),('COLOR_YUV420sp2GRAY',106),('COLOR_YUV420SP2GRAY',106),('COLOR_YUV420sp2RGB',92),('COLOR_YUV420SP2RGB',92),('COLOR_YUV420sp2RGBA',96),('COLOR_YUV420SP2RGBA',96),('COLOR_mRGBA2RGBA',126),('COLOR_M_RGBA2RGBA',126)])]

    def process_inputs(self, inputs, outputs, parameters):
        _src = inputs['_src'].value
        code = parameters['code']
        _dst = cv2.demosaicing(_src=_src, code=code)
        outputs['_dst'] = Data(_dst)

### detailEnhance ###

class OpenCVAuto2_DetailEnhance(NormalElement):
    name = 'DetailEnhance'
    comment = '''detailEnhance(src[, dst[, sigma_s[, sigma_r]]]) -> dst\n@brief This filter enhances the details of a particular image.\n\n@param src Input 8-bit 3-channel image.\n@param dst Output image with the same size and type as src.\n@param sigma_s Range between 0 to 200.\n@param sigma_r Range between 0 to 1.'''

    def get_attributes(self):
        return [Input('src', 'src')], \
               [Output('dst', 'dst')], \
               [FloatParameter('sigma_s', 'sigma_s'),
                FloatParameter('sigma_r', 'sigma_r')]

    def process_inputs(self, inputs, outputs, parameters):
        src = inputs['src'].value
        sigma_s = parameters['sigma_s']
        sigma_r = parameters['sigma_r']
        dst = cv2.detailEnhance(src=src, sigma_s=sigma_s, sigma_r=sigma_r)
        outputs['dst'] = Data(dst)

### divide ###

class OpenCVAuto2_Divide(NormalElement):
    name = 'Divide'
    comment = '''divide(src1, src2[, dst[, scale[, dtype]]]) -> dst\n@brief Performs per-element division of two arrays or a scalar by an array.\n\nThe function cv::divide divides one array by another:\n\f[\texttt{dst(I) = saturate(src1(I)*scale/src2(I))}\f]\nor a scalar by an array when there is no src1 :\n\f[\texttt{dst(I) = saturate(scale/src2(I))}\f]\n\nWhen src2(I) is zero, dst(I) will also be zero. Different channels of\nmulti-channel arrays are processed independently.\n\n@note Saturation is not applied when the output array has the depth CV_32S. You may even get\nresult of an incorrect sign in the case of overflow.\n@param src1 first input array.\n@param src2 second input array of the same size and type as src1.\n@param scale scalar factor.\n@param dst output array of the same size and type as src2.\n@param dtype optional depth of the output array; if -1, dst will have depth src2.depth(), but in\ncase of an array-by-array division, you can only pass -1 when src1.depth()==src2.depth().\n@sa  multiply, add, subtract



divide(scale, src2[, dst[, dtype]]) -> dst\n@overload'''

    def get_attributes(self):
        return [Input('src1', 'src1'),
                Input('src2', 'src2')], \
               [Output('dst', 'dst')], \
               [FloatParameter('scale', 'scale'),
                ComboboxParameter('dtype', [('NONE',0),('CV_8U',0),('CV_8UC1',0),('CV_8UC2',8),('CV_8UC3',16),('CV_8UC4',24),('CV_8S',1),('CV_8SC1',1),('CV_8SC2',9),('CV_8SC3',17),('CV_8SC4',25),('CV_16U',2),('CV_16UC1',2),('CV_16UC2',10),('CV_16UC3',18),('CV_16UC4',26),('CV_16S',3),('CV_16SC1',3),('CV_16SC2',11),('CV_16SC3',19),('CV_16SC4',27),('CV_32S',4),('CV_32SC1',4),('CV_32SC2',12),('CV_32SC3',20),('CV_32SC4',28),('CV_32F',5),('CV_32FC1',5),('CV_32FC2',13),('CV_32FC3',21),('CV_32FC4',29),('CV_64F',6),('CV_64FC1',6),('CV_64FC2',14),('CV_64FC3',22),('CV_64FC4',30)])]

    def process_inputs(self, inputs, outputs, parameters):
        src1 = inputs['src1'].value
        src2 = inputs['src2'].value
        scale = parameters['scale']
        dtype = parameters['dtype']
        dst = cv2.divide(src1=src1, src2=src2, scale=scale, dtype=dtype)
        outputs['dst'] = Data(dst)

### edgePreservingFilter ###

class OpenCVAuto2_EdgePreservingFilter(NormalElement):
    name = 'EdgePreservingFilter'
    comment = '''edgePreservingFilter(src[, dst[, flags[, sigma_s[, sigma_r]]]]) -> dst\n@brief Filtering is the fundamental operation in image and video processing. Edge-preserving smoothing\nfilters are used in many different applications @cite EM11 .\n\n@param src Input 8-bit 3-channel image.\n@param dst Output 8-bit 3-channel image.\n@param flags Edge preserving filters:\n-   **RECURS_FILTER** = 1\n-   **NORMCONV_FILTER** = 2\n@param sigma_s Range between 0 to 200.\n@param sigma_r Range between 0 to 1.'''

    def get_attributes(self):
        return [Input('src', 'src')], \
               [Output('dst', 'dst')], \
               [ComboboxParameter('flags', [('RECURS_FILTER',1),('NORMCONV_FILTER',2)]),
                FloatParameter('sigma_s', 'sigma_s'),
                FloatParameter('sigma_r', 'sigma_r')]

    def process_inputs(self, inputs, outputs, parameters):
        src = inputs['src'].value
        flags = parameters['flags']
        sigma_s = parameters['sigma_s']
        sigma_r = parameters['sigma_r']
        dst = cv2.edgePreservingFilter(src=src, flags=flags, sigma_s=sigma_s, sigma_r=sigma_r)
        outputs['dst'] = Data(dst)

### eigen ###

class OpenCVAuto2_Eigen(NormalElement):
    name = 'Eigen'
    comment = '''eigen(src[, eigenvalues[, eigenvectors]]) -> retval, eigenvalues, eigenvectors\n@brief Calculates eigenvalues and eigenvectors of a symmetric matrix.\n\nThe function cv::eigen calculates just eigenvalues, or eigenvalues and eigenvectors of the symmetric\nmatrix src:\n@code\nsrc*eigenvectors.row(i).t() = eigenvalues.at<srcType>(i)*eigenvectors.row(i).t()\n@endcode\n\n@note Use cv::eigenNonSymmetric for calculation of real eigenvalues and eigenvectors of non-symmetric matrix.\n\n@param src input matrix that must have CV_32FC1 or CV_64FC1 type, square size and be symmetrical\n(src ^T^ == src).\n@param eigenvalues output vector of eigenvalues of the same type as src; the eigenvalues are stored\nin the descending order.\n@param eigenvectors output matrix of eigenvectors; it has the same size and type as src; the\neigenvectors are stored as subsequent matrix rows, in the same order as the corresponding\neigenvalues.\n@sa eigenNonSymmetric, completeSymm , PCA'''

    def get_attributes(self):
        return [Input('src', 'src')], \
               [Output('eigenvalues', 'eigenvalues'),
                Output('eigenvectors', 'eigenvectors')], \
               []

    def process_inputs(self, inputs, outputs, parameters):
        src = inputs['src'].value
        retval, eigenvalues, eigenvectors = cv2.eigen(src=src)
        outputs['eigenvalues'] = Data(eigenvalues)
        outputs['eigenvectors'] = Data(eigenvectors)

### eigenNonSymmetric ###

class OpenCVAuto2_EigenNonSymmetric(NormalElement):
    name = 'EigenNonSymmetric'
    comment = '''eigenNonSymmetric(src[, eigenvalues[, eigenvectors]]) -> eigenvalues, eigenvectors\n@brief Calculates eigenvalues and eigenvectors of a non-symmetric matrix (real eigenvalues only).\n\n@note Assumes real eigenvalues.\n\nThe function calculates eigenvalues and eigenvectors (optional) of the square matrix src:\n@code\nsrc*eigenvectors.row(i).t() = eigenvalues.at<srcType>(i)*eigenvectors.row(i).t()\n@endcode\n\n@param src input matrix (CV_32FC1 or CV_64FC1 type).\n@param eigenvalues output vector of eigenvalues (type is the same type as src).\n@param eigenvectors output matrix of eigenvectors (type is the same type as src). The eigenvectors are stored as subsequent matrix rows, in the same order as the corresponding eigenvalues.\n@sa eigen'''

    def get_attributes(self):
        return [Input('src', 'src')], \
               [Output('eigenvalues', 'eigenvalues'),
                Output('eigenvectors', 'eigenvectors')], \
               []

    def process_inputs(self, inputs, outputs, parameters):
        src = inputs['src'].value
        eigenvalues, eigenvectors = cv2.eigenNonSymmetric(src=src)
        outputs['eigenvalues'] = Data(eigenvalues)
        outputs['eigenvectors'] = Data(eigenvectors)

### equalizeHist ###

class OpenCVAuto2_EqualizeHist(NormalElement):
    name = 'EqualizeHist'
    comment = '''equalizeHist(src[, dst]) -> dst\n@brief Equalizes the histogram of a grayscale image.\n\nThe function equalizes the histogram of the input image using the following algorithm:\n\n- Calculate the histogram \f$H\f$ for src .\n- Normalize the histogram so that the sum of histogram bins is 255.\n- Compute the integral of the histogram:\n\f[H'_i =  \sum _{0  \le j < i} H(j)\f]\n- Transform the image using \f$H'\f$ as a look-up table: \f$\texttt{dst}(x,y) = H'(\texttt{src}(x,y))\f$\n\nThe algorithm normalizes the brightness and increases the contrast of the image.\n\n@param src Source 8-bit single channel image.\n@param dst Destination image of the same size and type as src .'''

    def get_attributes(self):
        return [Input('src', 'src')], \
               [Output('dst', 'dst')], \
               []

    def process_inputs(self, inputs, outputs, parameters):
        src = inputs['src'].value
        dst = cv2.equalizeHist(src=src)
        outputs['dst'] = Data(dst)

### estimateAffine3D ###

class OpenCVAuto2_EstimateAffine3D(NormalElement):
    name = 'EstimateAffine3D'
    comment = '''estimateAffine3D(src, dst[, out[, inliers[, ransacThreshold[, confidence]]]]) -> retval, out, inliers\n@brief Computes an optimal affine transformation between two 3D point sets.\n\nIt computes\n\f[\n\begin{bmatrix}\nx\\\ny\\\nz\\\n\end{bmatrix}\n=\n\begin{bmatrix}\na_{11} & a_{12} & a_{13}\\\na_{21} & a_{22} & a_{23}\\\na_{31} & a_{32} & a_{33}\\\n\end{bmatrix}\n\begin{bmatrix}\nX\\\nY\\\nZ\\\n\end{bmatrix}\n+\n\begin{bmatrix}\nb_1\\\nb_2\\\nb_3\\\n\end{bmatrix}\n\f]\n\n@param src First input 3D point set containing \f$(X,Y,Z)\f$.\n@param dst Second input 3D point set containing \f$(x,y,z)\f$.\n@param out Output 3D affine transformation matrix \f$3 \times 4\f$ of the form\n\f[\n\begin{bmatrix}\na_{11} & a_{12} & a_{13} & b_1\\\na_{21} & a_{22} & a_{23} & b_2\\\na_{31} & a_{32} & a_{33} & b_3\\\n\end{bmatrix}\n\f]\n@param inliers Output vector indicating which points are inliers (1-inlier, 0-outlier).\n@param ransacThreshold Maximum reprojection error in the RANSAC algorithm to consider a point as\nan inlier.\n@param confidence Confidence level, between 0 and 1, for the estimated transformation. Anything\nbetween 0.95 and 0.99 is usually good enough. Values too close to 1 can slow down the estimation\nsignificantly. Values lower than 0.8-0.9 can result in an incorrectly estimated transformation.\n\nThe function estimates an optimal 3D affine transformation between two 3D point sets using the\nRANSAC algorithm.'''

    def get_attributes(self):
        return [Input('src', 'src'),
                Input('dst', 'dst')], \
               [Output('out', 'out'),
                Output('inliers', 'inliers')], \
               [FloatParameter('ransacThreshold', 'ransacThreshold')]

    def process_inputs(self, inputs, outputs, parameters):
        src = inputs['src'].value
        dst = inputs['dst'].value
        ransacThreshold = parameters['ransacThreshold']
        retval, out, inliers = cv2.estimateAffine3D(src=src, dst=dst, ransacThreshold=ransacThreshold)
        outputs['out'] = Data(out)
        outputs['inliers'] = Data(inliers)

### exp ###

class OpenCVAuto2_Exp(NormalElement):
    name = 'Exp'
    comment = '''exp(src[, dst]) -> dst\n@brief Calculates the exponent of every array element.\n\nThe function cv::exp calculates the exponent of every element of the input\narray:\n\f[\texttt{dst} [I] = e^{ src(I) }\f]\n\nThe maximum relative error is about 7e-6 for single-precision input and\nless than 1e-10 for double-precision input. Currently, the function\nconverts denormalized values to zeros on output. Special values (NaN,\nInf) are not handled.\n@param src input array.\n@param dst output array of the same size and type as src.\n@sa log , cartToPolar , polarToCart , phase , pow , sqrt , magnitude'''

    def get_attributes(self):
        return [Input('src', 'src')], \
               [Output('dst', 'dst')], \
               []

    def process_inputs(self, inputs, outputs, parameters):
        src = inputs['src'].value
        dst = cv2.exp(src=src)
        outputs['dst'] = Data(dst)

### fastNlMeansDenoising ###

class OpenCVAuto2_FastNlMeansDenoising(NormalElement):
    name = 'FastNlMeansDenoising'
    comment = '''fastNlMeansDenoising(src[, dst[, h[, templateWindowSize[, searchWindowSize]]]]) -> dst\n@brief Perform image denoising using Non-local Means Denoising algorithm\n<http://www.ipol.im/pub/algo/bcm_non_local_means_denoising/> with several computational\noptimizations. Noise expected to be a gaussian white noise\n\n@param src Input 8-bit 1-channel, 2-channel, 3-channel or 4-channel image.\n@param dst Output image with the same size and type as src .\n@param templateWindowSize Size in pixels of the template patch that is used to compute weights.\nShould be odd. Recommended value 7 pixels\n@param searchWindowSize Size in pixels of the window that is used to compute weighted average for\ngiven pixel. Should be odd. Affect performance linearly: greater searchWindowsSize - greater\ndenoising time. Recommended value 21 pixels\n@param h Parameter regulating filter strength. Big h value perfectly removes noise but also\nremoves image details, smaller h value preserves details but also preserves some noise\n\nThis function expected to be applied to grayscale images. For colored images look at\nfastNlMeansDenoisingColored. Advanced usage of this functions can be manual denoising of colored\nimage in different colorspaces. Such approach is used in fastNlMeansDenoisingColored by converting\nimage to CIELAB colorspace and then separately denoise L and AB components with different h\nparameter.



fastNlMeansDenoising(src, h[, dst[, templateWindowSize[, searchWindowSize[, normType]]]]) -> dst\n@brief Perform image denoising using Non-local Means Denoising algorithm\n<http://www.ipol.im/pub/algo/bcm_non_local_means_denoising/> with several computational\noptimizations. Noise expected to be a gaussian white noise\n\n@param src Input 8-bit or 16-bit (only with NORM_L1) 1-channel,\n2-channel, 3-channel or 4-channel image.\n@param dst Output image with the same size and type as src .\n@param templateWindowSize Size in pixels of the template patch that is used to compute weights.\nShould be odd. Recommended value 7 pixels\n@param searchWindowSize Size in pixels of the window that is used to compute weighted average for\ngiven pixel. Should be odd. Affect performance linearly: greater searchWindowsSize - greater\ndenoising time. Recommended value 21 pixels\n@param h Array of parameters regulating filter strength, either one\nparameter applied to all channels or one per channel in dst. Big h value\nperfectly removes noise but also removes image details, smaller h\nvalue preserves details but also preserves some noise\n@param normType Type of norm used for weight calculation. Can be either NORM_L2 or NORM_L1\n\nThis function expected to be applied to grayscale images. For colored images look at\nfastNlMeansDenoisingColored. Advanced usage of this functions can be manual denoising of colored\nimage in different colorspaces. Such approach is used in fastNlMeansDenoisingColored by converting\nimage to CIELAB colorspace and then separately denoise L and AB components with different h\nparameter.'''

    def get_attributes(self):
        return [Input('src', 'src')], \
               [Output('dst', 'dst')], \
               [SizeParameter('templateWindowSize', 'templateWindowSize'),
                SizeParameter('searchWindowSize', 'searchWindowSize')]

    def process_inputs(self, inputs, outputs, parameters):
        src = inputs['src'].value
        templateWindowSize = parameters['templateWindowSize']
        searchWindowSize = parameters['searchWindowSize']
        dst = cv2.fastNlMeansDenoising(src=src, templateWindowSize=templateWindowSize, searchWindowSize=searchWindowSize)
        outputs['dst'] = Data(dst)

### fastNlMeansDenoisingColored ###

class OpenCVAuto2_FastNlMeansDenoisingColored(NormalElement):
    name = 'FastNlMeansDenoisingColored'
    comment = '''fastNlMeansDenoisingColored(src[, dst[, h[, hColor[, templateWindowSize[, searchWindowSize]]]]]) -> dst\n@brief Modification of fastNlMeansDenoising function for colored images\n\n@param src Input 8-bit 3-channel image.\n@param dst Output image with the same size and type as src .\n@param templateWindowSize Size in pixels of the template patch that is used to compute weights.\nShould be odd. Recommended value 7 pixels\n@param searchWindowSize Size in pixels of the window that is used to compute weighted average for\ngiven pixel. Should be odd. Affect performance linearly: greater searchWindowsSize - greater\ndenoising time. Recommended value 21 pixels\n@param h Parameter regulating filter strength for luminance component. Bigger h value perfectly\nremoves noise but also removes image details, smaller h value preserves details but also preserves\nsome noise\n@param hColor The same as h but for color components. For most images value equals 10\nwill be enough to remove colored noise and do not distort colors\n\nThe function converts image to CIELAB colorspace and then separately denoise L and AB components\nwith given h parameters using fastNlMeansDenoising function.'''

    def get_attributes(self):
        return [Input('src', 'src')], \
               [Output('dst', 'dst')], \
               [SizeParameter('templateWindowSize', 'templateWindowSize'),
                SizeParameter('searchWindowSize', 'searchWindowSize')]

    def process_inputs(self, inputs, outputs, parameters):
        src = inputs['src'].value
        templateWindowSize = parameters['templateWindowSize']
        searchWindowSize = parameters['searchWindowSize']
        dst = cv2.fastNlMeansDenoisingColored(src=src, templateWindowSize=templateWindowSize, searchWindowSize=searchWindowSize)
        outputs['dst'] = Data(dst)

### fillConvexPoly ###

class OpenCVAuto2_FillConvexPoly(NormalElement):
    name = 'FillConvexPoly'
    comment = '''fillConvexPoly(img, points, color[, lineType[, shift]]) -> img\n@brief Fills a convex polygon.\n\nThe function cv::fillConvexPoly draws a filled convex polygon. This function is much faster than the\nfunction #fillPoly . It can fill not only convex polygons but any monotonic polygon without\nself-intersections, that is, a polygon whose contour intersects every horizontal line (scan line)\ntwice at the most (though, its top-most and/or the bottom edge could be horizontal).\n\n@param img Image.\n@param points Polygon vertices.\n@param color Polygon color.\n@param lineType Type of the polygon boundaries. See #LineTypes\n@param shift Number of fractional bits in the vertex coordinates.'''

    def get_attributes(self):
        return [Input('img', 'img'),
                Input('points', 'points')], \
               [Output('img', 'img')], \
               [ScalarParameter('color', 'color')]

    def process_inputs(self, inputs, outputs, parameters):
        img = inputs['img'].value
        points = inputs['points'].value
        color = parameters['color']
        img = cv2.fillConvexPoly(img=img, points=points, color=color)
        outputs['img'] = Data(img)

### findChessboardCorners ###

class OpenCVAuto2_FindChessboardCorners(NormalElement):
    name = 'FindChessboardCorners'
    comment = '''findChessboardCorners(image, patternSize[, corners[, flags]]) -> retval, corners\n@brief Finds the positions of internal corners of the chessboard.\n\n@param image Source chessboard view. It must be an 8-bit grayscale or color image.\n@param patternSize Number of inner corners per a chessboard row and column\n( patternSize = cvSize(points_per_row,points_per_colum) = cvSize(columns,rows) ).\n@param corners Output array of detected corners.\n@param flags Various operation flags that can be zero or a combination of the following values:\n-   **CALIB_CB_ADAPTIVE_THRESH** Use adaptive thresholding to convert the image to black\nand white, rather than a fixed threshold level (computed from the average image brightness).\n-   **CALIB_CB_NORMALIZE_IMAGE** Normalize the image gamma with equalizeHist before\napplying fixed or adaptive thresholding.\n-   **CALIB_CB_FILTER_QUADS** Use additional criteria (like contour area, perimeter,\nsquare-like shape) to filter out false quads extracted at the contour retrieval stage.\n-   **CALIB_CB_FAST_CHECK** Run a fast check on the image that looks for chessboard corners,\nand shortcut the call if none is found. This can drastically speed up the call in the\ndegenerate condition when no chessboard is observed.\n\nThe function attempts to determine whether the input image is a view of the chessboard pattern and\nlocate the internal chessboard corners. The function returns a non-zero value if all of the corners\nare found and they are placed in a certain order (row by row, left to right in every row).\nOtherwise, if the function fails to find all the corners or reorder them, it returns 0. For example,\na regular chessboard has 8 x 8 squares and 7 x 7 internal corners, that is, points where the black\nsquares touch each other. The detected coordinates are approximate, and to determine their positions\nmore accurately, the function calls cornerSubPix. You also may use the function cornerSubPix with\ndifferent parameters if returned coordinates are not accurate enough.\n\nSample usage of detecting and drawing chessboard corners: :\n@code\nSize patternsize(8,6); //interior number of corners\nMat gray = ....; //source image\nvector<Point2f> corners; //this will be filled by the detected corners\n\n//CALIB_CB_FAST_CHECK saves a lot of time on images\n//that do not contain any chessboard corners\nbool patternfound = findChessboardCorners(gray, patternsize, corners,\nCALIB_CB_ADAPTIVE_THRESH + CALIB_CB_NORMALIZE_IMAGE\n+ CALIB_CB_FAST_CHECK);\n\nif(patternfound)\ncornerSubPix(gray, corners, Size(11, 11), Size(-1, -1),\nTermCriteria(CV_TERMCRIT_EPS + CV_TERMCRIT_ITER, 30, 0.1));\n\ndrawChessboardCorners(img, patternsize, Mat(corners), patternfound);\n@endcode\n@note The function requires white space (like a square-thick border, the wider the better) around\nthe board to make the detection more robust in various environments. Otherwise, if there is no\nborder and the background is dark, the outer black squares cannot be segmented properly and so the\nsquare grouping and ordering algorithm fails.'''

    def get_attributes(self):
        return [Input('image', 'image')], \
               [Output('corners', 'corners')], \
               [SizeParameter('patternSize', 'patternSize'),
                ComboboxParameter('flags', [('CALIB_CB_ADAPTIVE_THRESH',1),('CALIB_CB_NORMALIZE_IMAGE',2),('CALIB_CB_FILTER_QUADS',4),('CALIB_CB_FAST_CHECK',8)])]

    def process_inputs(self, inputs, outputs, parameters):
        image = inputs['image'].value
        patternSize = parameters['patternSize']
        flags = parameters['flags']
        retval, corners = cv2.findChessboardCorners(image=image, patternSize=patternSize, flags=flags)
        outputs['corners'] = Data(corners)

### findEssentialMat ###

class OpenCVAuto2_FindEssentialMat(NormalElement):
    name = 'FindEssentialMat'
    comment = '''findEssentialMat(points1, points2, cameraMatrix[, method[, prob[, threshold[, mask]]]]) -> retval, mask\n@brief Calculates an essential matrix from the corresponding points in two images.\n\n@param points1 Array of N (N \>= 5) 2D points from the first image. The point coordinates should\nbe floating-point (single or double precision).\n@param points2 Array of the second image points of the same size and format as points1 .\n@param cameraMatrix Camera matrix \f$K = \vecthreethree{f_x}{0}{c_x}{0}{f_y}{c_y}{0}{0}{1}\f$ .\nNote that this function assumes that points1 and points2 are feature points from cameras with the\nsame camera matrix.\n@param method Method for computing an essential matrix.\n-   **RANSAC** for the RANSAC algorithm.\n-   **LMEDS** for the LMedS algorithm.\n@param prob Parameter used for the RANSAC or LMedS methods only. It specifies a desirable level of\nconfidence (probability) that the estimated matrix is correct.\n@param threshold Parameter used for RANSAC. It is the maximum distance from a point to an epipolar\nline in pixels, beyond which the point is considered an outlier and is not used for computing the\nfinal fundamental matrix. It can be set to something like 1-3, depending on the accuracy of the\npoint localization, image resolution, and the image noise.\n@param mask Output array of N elements, every element of which is set to 0 for outliers and to 1\nfor the other points. The array is computed only in the RANSAC and LMedS methods.\n\nThis function estimates essential matrix based on the five-point algorithm solver in @cite Nister03 .\n@cite SteweniusCFS is also a related. The epipolar geometry is described by the following equation:\n\n\f[[p_2; 1]^T K^{-T} E K^{-1} [p_1; 1] = 0\f]\n\nwhere \f$E\f$ is an essential matrix, \f$p_1\f$ and \f$p_2\f$ are corresponding points in the first and the\nsecond images, respectively. The result of this function may be passed further to\ndecomposeEssentialMat or recoverPose to recover the relative pose between cameras.



findEssentialMat(points1, points2[, focal[, pp[, method[, prob[, threshold[, mask]]]]]]) -> retval, mask\n@overload\n@param points1 Array of N (N \>= 5) 2D points from the first image. The point coordinates should\nbe floating-point (single or double precision).\n@param points2 Array of the second image points of the same size and format as points1 .\n@param focal focal length of the camera. Note that this function assumes that points1 and points2\nare feature points from cameras with same focal length and principal point.\n@param pp principal point of the camera.\n@param method Method for computing a fundamental matrix.\n-   **RANSAC** for the RANSAC algorithm.\n-   **LMEDS** for the LMedS algorithm.\n@param threshold Parameter used for RANSAC. It is the maximum distance from a point to an epipolar\nline in pixels, beyond which the point is considered an outlier and is not used for computing the\nfinal fundamental matrix. It can be set to something like 1-3, depending on the accuracy of the\npoint localization, image resolution, and the image noise.\n@param prob Parameter used for the RANSAC or LMedS methods only. It specifies a desirable level of\nconfidence (probability) that the estimated matrix is correct.\n@param mask Output array of N elements, every element of which is set to 0 for outliers and to 1\nfor the other points. The array is computed only in the RANSAC and LMedS methods.\n\nThis function differs from the one above that it computes camera matrix from focal length and\nprincipal point:\n\n\f[K =\n\begin{bmatrix}\nf & 0 & x_{pp}  \\\n0 & f & y_{pp}  \\\n0 & 0 & 1\n\end{bmatrix}\f]'''

    def get_attributes(self):
        return [Input('points1', 'points1'),
                Input('points2', 'points2'),
                Input('cameraMatrix', 'cameraMatrix')], \
               [Output('mask', 'mask')], \
               [FloatParameter('threshold', 'threshold')]

    def process_inputs(self, inputs, outputs, parameters):
        points1 = inputs['points1'].value
        points2 = inputs['points2'].value
        cameraMatrix = inputs['cameraMatrix'].value
        threshold = parameters['threshold']
        retval, mask = cv2.findEssentialMat(points1=points1, points2=points2, cameraMatrix=cameraMatrix, threshold=threshold)
        outputs['mask'] = Data(mask)

### findHomography ###

class OpenCVAuto2_FindHomography(NormalElement):
    name = 'FindHomography'
    comment = '''findHomography(srcPoints, dstPoints[, method[, ransacReprojThreshold[, mask[, maxIters[, confidence]]]]]) -> retval, mask\n@brief Finds a perspective transformation between two planes.\n\n@param srcPoints Coordinates of the points in the original plane, a matrix of the type CV_32FC2\nor vector\<Point2f\> .\n@param dstPoints Coordinates of the points in the target plane, a matrix of the type CV_32FC2 or\na vector\<Point2f\> .\n@param method Method used to compute a homography matrix. The following methods are possible:\n-   **0** - a regular method using all the points, i.e., the least squares method\n-   **RANSAC** - RANSAC-based robust method\n-   **LMEDS** - Least-Median robust method\n-   **RHO** - PROSAC-based robust method\n@param ransacReprojThreshold Maximum allowed reprojection error to treat a point pair as an inlier\n(used in the RANSAC and RHO methods only). That is, if\n\f[\| \texttt{dstPoints} _i -  \texttt{convertPointsHomogeneous} ( \texttt{H} * \texttt{srcPoints} _i) \|_2  >  \texttt{ransacReprojThreshold}\f]\nthen the point \f$i\f$ is considered as an outlier. If srcPoints and dstPoints are measured in pixels,\nit usually makes sense to set this parameter somewhere in the range of 1 to 10.\n@param mask Optional output mask set by a robust method ( RANSAC or LMEDS ). Note that the input\nmask values are ignored.\n@param maxIters The maximum number of RANSAC iterations.\n@param confidence Confidence level, between 0 and 1.\n\nThe function finds and returns the perspective transformation \f$H\f$ between the source and the\ndestination planes:\n\n\f[s_i  \vecthree{x'_i}{y'_i}{1} \sim H  \vecthree{x_i}{y_i}{1}\f]\n\nso that the back-projection error\n\n\f[\sum _i \left ( x'_i- \frac{h_{11} x_i + h_{12} y_i + h_{13}}{h_{31} x_i + h_{32} y_i + h_{33}} \right )^2+ \left ( y'_i- \frac{h_{21} x_i + h_{22} y_i + h_{23}}{h_{31} x_i + h_{32} y_i + h_{33}} \right )^2\f]\n\nis minimized. If the parameter method is set to the default value 0, the function uses all the point\npairs to compute an initial homography estimate with a simple least-squares scheme.\n\nHowever, if not all of the point pairs ( \f$srcPoints_i\f$, \f$dstPoints_i\f$ ) fit the rigid perspective\ntransformation (that is, there are some outliers), this initial estimate will be poor. In this case,\nyou can use one of the three robust methods. The methods RANSAC, LMeDS and RHO try many different\nrandom subsets of the corresponding point pairs (of four pairs each, collinear pairs are discarded), estimate the homography matrix\nusing this subset and a simple least-squares algorithm, and then compute the quality/goodness of the\ncomputed homography (which is the number of inliers for RANSAC or the least median re-projection error for\nLMeDS). The best subset is then used to produce the initial estimate of the homography matrix and\nthe mask of inliers/outliers.\n\nRegardless of the method, robust or not, the computed homography matrix is refined further (using\ninliers only in case of a robust method) with the Levenberg-Marquardt method to reduce the\nre-projection error even more.\n\nThe methods RANSAC and RHO can handle practically any ratio of outliers but need a threshold to\ndistinguish inliers from outliers. The method LMeDS does not need any threshold but it works\ncorrectly only when there are more than 50% of inliers. Finally, if there are no outliers and the\nnoise is rather small, use the default method (method=0).\n\nThe function is used to find initial intrinsic and extrinsic matrices. Homography matrix is\ndetermined up to a scale. Thus, it is normalized so that \f$h_{33}=1\f$. Note that whenever an \f$H\f$ matrix\ncannot be estimated, an empty one will be returned.\n\n@sa\ngetAffineTransform, estimateAffine2D, estimateAffinePartial2D, getPerspectiveTransform, warpPerspective,\nperspectiveTransform'''

    def get_attributes(self):
        return [Input('srcPoints', 'srcPoints'),
                Input('dstPoints', 'dstPoints')], \
               [Output('mask', 'mask')], \
               [FloatParameter('ransacReprojThreshold', 'ransacReprojThreshold'),
                IntParameter('maxIters', 'maxIters')]

    def process_inputs(self, inputs, outputs, parameters):
        srcPoints = inputs['srcPoints'].value
        dstPoints = inputs['dstPoints'].value
        ransacReprojThreshold = parameters['ransacReprojThreshold']
        maxIters = parameters['maxIters']
        retval, mask = cv2.findHomography(srcPoints=srcPoints, dstPoints=dstPoints, ransacReprojThreshold=ransacReprojThreshold, maxIters=maxIters)
        outputs['mask'] = Data(mask)

### findNonZero ###

class OpenCVAuto2_FindNonZero(NormalElement):
    name = 'FindNonZero'
    comment = '''findNonZero(src[, idx]) -> idx\n@brief Returns the list of locations of non-zero pixels\n\nGiven a binary matrix (likely returned from an operation such\nas threshold(), compare(), >, ==, etc, return all of\nthe non-zero indices as a cv::Mat or std::vector<cv::Point> (x,y)\nFor example:\n@code{.cpp}\ncv::Mat binaryImage; // input, binary image\ncv::Mat locations;   // output, locations of non-zero pixels\ncv::findNonZero(binaryImage, locations);\n\n// access pixel coordinates\nPoint pnt = locations.at<Point>(i);\n@endcode\nor\n@code{.cpp}\ncv::Mat binaryImage; // input, binary image\nvector<Point> locations;   // output, locations of non-zero pixels\ncv::findNonZero(binaryImage, locations);\n\n// access pixel coordinates\nPoint pnt = locations[i];\n@endcode\n@param src single-channel array (type CV_8UC1)\n@param idx the output array, type of cv::Mat or std::vector<Point>, corresponding to non-zero indices in the input'''

    def get_attributes(self):
        return [Input('src', 'src')], \
               [Output('idx', 'idx')], \
               []

    def process_inputs(self, inputs, outputs, parameters):
        src = inputs['src'].value
        idx = cv2.findNonZero(src=src)
        outputs['idx'] = Data(idx)

### findTransformECC ###

class OpenCVAuto2_FindTransformECC(NormalElement):
    name = 'FindTransformECC'
    comment = '''findTransformECC(templateImage, inputImage, warpMatrix[, motionType[, criteria[, inputMask]]]) -> retval, warpMatrix\n@brief Finds the geometric transform (warp) between two images in terms of the ECC criterion @cite EP08 .\n\n@param templateImage single-channel template image; CV_8U or CV_32F array.\n@param inputImage single-channel input image which should be warped with the final warpMatrix in\norder to provide an image similar to templateImage, same type as temlateImage.\n@param warpMatrix floating-point \f$2\times 3\f$ or \f$3\times 3\f$ mapping matrix (warp).\n@param motionType parameter, specifying the type of motion:\n-   **MOTION_TRANSLATION** sets a translational motion model; warpMatrix is \f$2\times 3\f$ with\nthe first \f$2\times 2\f$ part being the unity matrix and the rest two parameters being\nestimated.\n-   **MOTION_EUCLIDEAN** sets a Euclidean (rigid) transformation as motion model; three\nparameters are estimated; warpMatrix is \f$2\times 3\f$.\n-   **MOTION_AFFINE** sets an affine motion model (DEFAULT); six parameters are estimated;\nwarpMatrix is \f$2\times 3\f$.\n-   **MOTION_HOMOGRAPHY** sets a homography as a motion model; eight parameters are\nestimated;\`warpMatrix\` is \f$3\times 3\f$.\n@param criteria parameter, specifying the termination criteria of the ECC algorithm;\ncriteria.epsilon defines the threshold of the increment in the correlation coefficient between two\niterations (a negative criteria.epsilon makes criteria.maxcount the only termination criterion).\nDefault values are shown in the declaration above.\n@param inputMask An optional mask to indicate valid values of inputImage.\n\nThe function estimates the optimum transformation (warpMatrix) with respect to ECC criterion\n(@cite EP08), that is\n\n\f[\texttt{warpMatrix} = \texttt{warpMatrix} = \arg\max_{W} \texttt{ECC}(\texttt{templateImage}(x,y),\texttt{inputImage}(x',y'))\f]\n\nwhere\n\n\f[\begin{bmatrix} x' \\ y' \end{bmatrix} = W \cdot \begin{bmatrix} x \\ y \\ 1 \end{bmatrix}\f]\n\n(the equation holds with homogeneous coordinates for homography). It returns the final enhanced\ncorrelation coefficient, that is the correlation coefficient between the template image and the\nfinal warped input image. When a \f$3\times 3\f$ matrix is given with motionType =0, 1 or 2, the third\nrow is ignored.\n\nUnlike findHomography and estimateRigidTransform, the function findTransformECC implements an\narea-based alignment that builds on intensity similarities. In essence, the function updates the\ninitial transformation that roughly aligns the images. If this information is missing, the identity\nwarp (unity matrix) is used as an initialization. Note that if images undergo strong\ndisplacements/rotations, an initial transformation that roughly aligns the images is necessary\n(e.g., a simple euclidean/similarity transform that allows for the images showing the same image\ncontent approximately). Use inverse warping in the second image to take an image close to the first\none, i.e. use the flag WARP_INVERSE_MAP with warpAffine or warpPerspective. See also the OpenCV\nsample image_alignment.cpp that demonstrates the use of the function. Note that the function throws\nan exception if algorithm does not converges.\n\n@sa\nestimateAffine2D, estimateAffinePartial2D, findHomography'''

    def get_attributes(self):
        return [Input('templateImage', 'templateImage'),
                Input('inputImage', 'inputImage'),
                Input('warpMatrix', 'warpMatrix')], \
               [Output('warpMatrix', 'warpMatrix')], \
               []

    def process_inputs(self, inputs, outputs, parameters):
        templateImage = inputs['templateImage'].value
        inputImage = inputs['inputImage'].value
        warpMatrix = inputs['warpMatrix'].value
        retval, warpMatrix = cv2.findTransformECC(templateImage=templateImage, inputImage=inputImage, warpMatrix=warpMatrix)
        outputs['warpMatrix'] = Data(warpMatrix)

### getDerivKernels ###

class OpenCVAuto2_GetDerivKernels(NormalElement):
    name = 'GetDerivKernels'
    comment = '''getDerivKernels(dx, dy, ksize[, kx[, ky[, normalize[, ktype]]]]) -> kx, ky\n@brief Returns filter coefficients for computing spatial image derivatives.\n\nThe function computes and returns the filter coefficients for spatial image derivatives. When\n`ksize=CV_SCHARR`, the Scharr \f$3 \times 3\f$ kernels are generated (see #Scharr). Otherwise, Sobel\nkernels are generated (see #Sobel). The filters are normally passed to #sepFilter2D or to\n\n@param kx Output matrix of row filter coefficients. It has the type ktype .\n@param ky Output matrix of column filter coefficients. It has the type ktype .\n@param dx Derivative order in respect of x.\n@param dy Derivative order in respect of y.\n@param ksize Aperture size. It can be CV_SCHARR, 1, 3, 5, or 7.\n@param normalize Flag indicating whether to normalize (scale down) the filter coefficients or not.\nTheoretically, the coefficients should have the denominator \f$=2^{ksize*2-dx-dy-2}\f$. If you are\ngoing to filter floating-point images, you are likely to use the normalized kernels. But if you\ncompute derivatives of an 8-bit image, store the results in a 16-bit image, and wish to preserve\nall the fractional bits, you may want to set normalize=false .\n@param ktype Type of filter coefficients. It can be CV_32f or CV_64F .'''

    def get_attributes(self):
        return [], \
               [Output('kx', 'kx'),
                Output('ky', 'ky')], \
               [IntParameter('dx', 'dx'),
                IntParameter('dy', 'dy'),
                SizeParameter('ksize', 'ksize')]

    def process_inputs(self, inputs, outputs, parameters):
        dx = parameters['dx']
        dy = parameters['dy']
        ksize = parameters['ksize']
        kx, ky = cv2.getDerivKernels(dx=dx, dy=dy, ksize=ksize)
        outputs['kx'] = Data(kx)
        outputs['ky'] = Data(ky)

### getOptimalNewCameraMatrix ###

class OpenCVAuto2_GetOptimalNewCameraMatrix(NormalElement):
    name = 'GetOptimalNewCameraMatrix'
    comment = '''getOptimalNewCameraMatrix(cameraMatrix, distCoeffs, imageSize, alpha[, newImgSize[, centerPrincipalPoint]]) -> retval, validPixROI\n@brief Returns the new camera matrix based on the free scaling parameter.\n\n@param cameraMatrix Input camera matrix.\n@param distCoeffs Input vector of distortion coefficients\n\f$(k_1, k_2, p_1, p_2[, k_3[, k_4, k_5, k_6 [, s_1, s_2, s_3, s_4[, \tau_x, \tau_y]]]])\f$ of\n4, 5, 8, 12 or 14 elements. If the vector is NULL/empty, the zero distortion coefficients are\nassumed.\n@param imageSize Original image size.\n@param alpha Free scaling parameter between 0 (when all the pixels in the undistorted image are\nvalid) and 1 (when all the source image pixels are retained in the undistorted image). See\nstereoRectify for details.\n@param newImgSize Image size after rectification. By default, it is set to imageSize .\n@param validPixROI Optional output rectangle that outlines all-good-pixels region in the\nundistorted image. See roi1, roi2 description in stereoRectify .\n@param centerPrincipalPoint Optional flag that indicates whether in the new camera matrix the\nprincipal point should be at the image center or not. By default, the principal point is chosen to\nbest fit a subset of the source image (determined by alpha) to the corrected image.\n@return new_camera_matrix Output new camera matrix.\n\nThe function computes and returns the optimal new camera matrix based on the free scaling parameter.\nBy varying this parameter, you may retrieve only sensible pixels alpha=0 , keep all the original\nimage pixels if there is valuable information in the corners alpha=1 , or get something in between.\nWhen alpha\>0 , the undistorted result is likely to have some black pixels corresponding to\n"virtual" pixels outside of the captured distorted image. The original camera matrix, distortion\ncoefficients, the computed new camera matrix, and newImageSize should be passed to\ninitUndistortRectifyMap to produce the maps for remap .'''

    def get_attributes(self):
        return [Input('cameraMatrix', 'cameraMatrix'),
                Input('distCoeffs', 'distCoeffs'),
                Input('imageSize', 'imageSize')], \
               [Output('validPixROI', 'validPixROI')], \
               [FloatParameter('alpha', 'alpha'),
                SizeParameter('newImgSize', 'newImgSize')]

    def process_inputs(self, inputs, outputs, parameters):
        cameraMatrix = inputs['cameraMatrix'].value
        distCoeffs = inputs['distCoeffs'].value
        imageSize = inputs['imageSize'].value
        alpha = parameters['alpha']
        newImgSize = parameters['newImgSize']
        retval, validPixROI = cv2.getOptimalNewCameraMatrix(cameraMatrix=cameraMatrix, distCoeffs=distCoeffs, imageSize=imageSize, alpha=alpha, newImgSize=newImgSize)
        outputs['validPixROI'] = Data(validPixROI)

### getRectSubPix ###

class OpenCVAuto2_GetRectSubPix(NormalElement):
    name = 'GetRectSubPix'
    comment = '''getRectSubPix(image, patchSize, center[, patch[, patchType]]) -> patch\n@brief Retrieves a pixel rectangle from an image with sub-pixel accuracy.\n\nThe function getRectSubPix extracts pixels from src:\n\n\f[patch(x, y) = src(x +  \texttt{center.x} - ( \texttt{dst.cols} -1)*0.5, y +  \texttt{center.y} - ( \texttt{dst.rows} -1)*0.5)\f]\n\nwhere the values of the pixels at non-integer coordinates are retrieved using bilinear\ninterpolation. Every channel of multi-channel images is processed independently. Also\nthe image should be a single channel or three channel image. While the center of the\nrectangle must be inside the image, parts of the rectangle may be outside.\n\n@param image Source image.\n@param patchSize Size of the extracted patch.\n@param center Floating point coordinates of the center of the extracted rectangle within the\nsource image. The center must be inside the image.\n@param patch Extracted patch that has the size patchSize and the same number of channels as src .\n@param patchType Depth of the extracted pixels. By default, they have the same depth as src .\n\n@sa  warpAffine, warpPerspective'''

    def get_attributes(self):
        return [Input('image', 'image')], \
               [Output('patch', 'patch')], \
               [SizeParameter('patchSize', 'patchSize'),
                PointParameter('center', 'center')]

    def process_inputs(self, inputs, outputs, parameters):
        image = inputs['image'].value
        patchSize = parameters['patchSize']
        center = parameters['center']
        patch = cv2.getRectSubPix(image=image, patchSize=patchSize, center=center)
        outputs['patch'] = Data(patch)

### hconcat ###

class OpenCVAuto2_Hconcat(NormalElement):
    name = 'Hconcat'
    comment = '''hconcat(src[, dst]) -> dst\n@overload\n@code{.cpp}\nstd::vector<cv::Mat> matrices = { cv::Mat(4, 1, CV_8UC1, cv::Scalar(1)),\ncv::Mat(4, 1, CV_8UC1, cv::Scalar(2)),\ncv::Mat(4, 1, CV_8UC1, cv::Scalar(3)),};\n\ncv::Mat out;\ncv::hconcat( matrices, out );\n//out:\n//[1, 2, 3;\n// 1, 2, 3;\n// 1, 2, 3;\n// 1, 2, 3]\n@endcode\n@param src input array or vector of matrices. all of the matrices must have the same number of rows and the same depth.\n@param dst output array. It has the same number of rows and depth as the src, and the sum of cols of the src.\nsame depth.'''

    def get_attributes(self):
        return [Input('src', 'src')], \
               [Output('dst', 'dst')], \
               []

    def process_inputs(self, inputs, outputs, parameters):
        src = inputs['src'].value
        dst = cv2.hconcat(src=src)
        outputs['dst'] = Data(dst)

### illuminationChange ###

class OpenCVAuto2_IlluminationChange(NormalElement):
    name = 'IlluminationChange'
    comment = '''illuminationChange(src, mask[, dst[, alpha[, beta]]]) -> dst\n@brief Applying an appropriate non-linear transformation to the gradient field inside the selection and\nthen integrating back with a Poisson solver, modifies locally the apparent illumination of an image.\n\n@param src Input 8-bit 3-channel image.\n@param mask Input 8-bit 1 or 3-channel image.\n@param dst Output image with the same size and type as src.\n@param alpha Value ranges between 0-2.\n@param beta Value ranges between 0-2.\n\nThis is useful to highlight under-exposed foreground objects or to reduce specular reflections.'''

    def get_attributes(self):
        return [Input('src', 'src'),
                Input('mask', 'mask')], \
               [Output('dst', 'dst')], \
               [FloatParameter('alpha', 'alpha'),
                FloatParameter('beta', 'beta')]

    def process_inputs(self, inputs, outputs, parameters):
        src = inputs['src'].value
        mask = inputs['mask'].value
        alpha = parameters['alpha']
        beta = parameters['beta']
        dst = cv2.illuminationChange(src=src, mask=mask, alpha=alpha, beta=beta)
        outputs['dst'] = Data(dst)

### imreadmulti ###

class OpenCVAuto2_Imreadmulti(NormalElement):
    name = 'Imreadmulti'
    comment = '''imreadmulti(filename[, mats[, flags]]) -> retval, mats\n@brief Loads a multi-page image from a file.\n\nThe function imreadmulti loads a multi-page image from the specified file into a vector of Mat objects.\n@param filename Name of file to be loaded.\n@param flags Flag that can take values of cv::ImreadModes, default with cv::IMREAD_ANYCOLOR.\n@param mats A vector of Mat objects holding each page, if more than one.\n@sa cv::imread'''

    def get_attributes(self):
        return [], \
               [Output('mats', 'mats')], \
               [TextParameter('filename', 'filename'),
                ComboboxParameter('flags', [('IMREAD_ANYCOLOR',4)])]

    def process_inputs(self, inputs, outputs, parameters):
        filename = parameters['filename']
        flags = parameters['flags']
        retval, mats = cv2.imreadmulti(filename=filename, flags=flags)
        outputs['mats'] = Data(mats)

### inpaint ###

class OpenCVAuto2_Inpaint(NormalElement):
    name = 'Inpaint'
    comment = '''inpaint(src, inpaintMask, inpaintRadius, flags[, dst]) -> dst\n@brief Restores the selected region in an image using the region neighborhood.\n\n@param src Input 8-bit, 16-bit unsigned or 32-bit float 1-channel or 8-bit 3-channel image.\n@param inpaintMask Inpainting mask, 8-bit 1-channel image. Non-zero pixels indicate the area that\nneeds to be inpainted.\n@param dst Output image with the same size and type as src .\n@param inpaintRadius Radius of a circular neighborhood of each point inpainted that is considered\nby the algorithm.\n@param flags Inpainting method that could be one of the following:\n-   **INPAINT_NS** Navier-Stokes based method [Navier01]\n-   **INPAINT_TELEA** Method by Alexandru Telea @cite Telea04 .\n\nThe function reconstructs the selected image area from the pixel near the area boundary. The\nfunction may be used to remove dust and scratches from a scanned photo, or to remove undesirable\nobjects from still images or video. See <http://en.wikipedia.org/wiki/Inpainting> for more details.\n\n@note\n-   An example using the inpainting technique can be found at\nopencv_source_code/samples/cpp/inpaint.cpp\n-   (Python) An example using the inpainting technique can be found at\nopencv_source_code/samples/python/inpaint.py'''

    def get_attributes(self):
        return [Input('src', 'src'),
                Input('inpaintMask', 'inpaintMask')], \
               [Output('dst', 'dst')], \
               [IntParameter('inpaintRadius', 'inpaintRadius'),
                ComboboxParameter('flags', [('INPAINT_NS',0),('INPAINT_TELEA',1)])]

    def process_inputs(self, inputs, outputs, parameters):
        src = inputs['src'].value
        inpaintMask = inputs['inpaintMask'].value
        inpaintRadius = parameters['inpaintRadius']
        flags = parameters['flags']
        dst = cv2.inpaint(src=src, inpaintMask=inpaintMask, inpaintRadius=inpaintRadius, flags=flags)
        outputs['dst'] = Data(dst)

### integral ###

class OpenCVAuto2_Integral(NormalElement):
    name = 'Integral'
    comment = '''integral(src[, sum[, sdepth]]) -> sum\n@overload'''

    def get_attributes(self):
        return [Input('src', 'src')], \
               [Output('sum', 'sum')], \
               []

    def process_inputs(self, inputs, outputs, parameters):
        src = inputs['src'].value
        sum = cv2.integral(src=src)
        outputs['sum'] = Data(sum)

### integral2 ###

class OpenCVAuto2_Integral2(NormalElement):
    name = 'Integral2'
    comment = '''integral2(src[, sum[, sqsum[, sdepth[, sqdepth]]]]) -> sum, sqsum\n@overload'''

    def get_attributes(self):
        return [Input('src', 'src')], \
               [Output('sum', 'sum'),
                Output('sqsum', 'sqsum')], \
               []

    def process_inputs(self, inputs, outputs, parameters):
        src = inputs['src'].value
        sum, sqsum = cv2.integral2(src=src)
        outputs['sum'] = Data(sum)
        outputs['sqsum'] = Data(sqsum)

### integral3 ###

class OpenCVAuto2_Integral3(NormalElement):
    name = 'Integral3'
    comment = '''integral3(src[, sum[, sqsum[, tilted[, sdepth[, sqdepth]]]]]) -> sum, sqsum, tilted\n@brief Calculates the integral of an image.\n\nThe function calculates one or more integral images for the source image as follows:\n\n\f[\texttt{sum} (X,Y) =  \sum _{x<X,y<Y}  \texttt{image} (x,y)\f]\n\n\f[\texttt{sqsum} (X,Y) =  \sum _{x<X,y<Y}  \texttt{image} (x,y)^2\f]\n\n\f[\texttt{tilted} (X,Y) =  \sum _{y<Y,abs(x-X+1) \leq Y-y-1}  \texttt{image} (x,y)\f]\n\nUsing these integral images, you can calculate sum, mean, and standard deviation over a specific\nup-right or rotated rectangular region of the image in a constant time, for example:\n\n\f[\sum _{x_1 \leq x < x_2,  \, y_1  \leq y < y_2}  \texttt{image} (x,y) =  \texttt{sum} (x_2,y_2)- \texttt{sum} (x_1,y_2)- \texttt{sum} (x_2,y_1)+ \texttt{sum} (x_1,y_1)\f]\n\nIt makes possible to do a fast blurring or fast block correlation with a variable window size, for\nexample. In case of multi-channel images, sums for each channel are accumulated independently.\n\nAs a practical example, the next figure shows the calculation of the integral of a straight\nrectangle Rect(3,3,3,2) and of a tilted rectangle Rect(5,1,2,3) . The selected pixels in the\noriginal image are shown, as well as the relative pixels in the integral images sum and tilted .\n\n![integral calculation example](pics/integral.png)\n\n@param src input image as \f$W \times H\f$, 8-bit or floating-point (32f or 64f).\n@param sum integral image as \f$(W+1)\times (H+1)\f$ , 32-bit integer or floating-point (32f or 64f).\n@param sqsum integral image for squared pixel values; it is \f$(W+1)\times (H+1)\f$, double-precision\nfloating-point (64f) array.\n@param tilted integral for the image rotated by 45 degrees; it is \f$(W+1)\times (H+1)\f$ array with\nthe same data type as sum.\n@param sdepth desired depth of the integral and the tilted integral images, CV_32S, CV_32F, or\nCV_64F.\n@param sqdepth desired depth of the integral image of squared pixel values, CV_32F or CV_64F.'''

    def get_attributes(self):
        return [Input('src', 'src')], \
               [Output('sum', 'sum'),
                Output('sqsum', 'sqsum'),
                Output('tilted', 'tilted')], \
               []

    def process_inputs(self, inputs, outputs, parameters):
        src = inputs['src'].value
        sum, sqsum, tilted = cv2.integral3(src=src)
        outputs['sum'] = Data(sum)
        outputs['sqsum'] = Data(sqsum)
        outputs['tilted'] = Data(tilted)

### invertAffineTransform ###

class OpenCVAuto2_InvertAffineTransform(NormalElement):
    name = 'InvertAffineTransform'
    comment = '''invertAffineTransform(M[, iM]) -> iM\n@brief Inverts an affine transformation.\n\nThe function computes an inverse affine transformation represented by \f$2 \times 3\f$ matrix M:\n\n\f[\begin{bmatrix} a_{11} & a_{12} & b_1  \\ a_{21} & a_{22} & b_2 \end{bmatrix}\f]\n\nThe result is also a \f$2 \times 3\f$ matrix of the same type as M.\n\n@param M Original affine transformation.\n@param iM Output reverse affine transformation.'''

    def get_attributes(self):
        return [Input('M', 'M')], \
               [Output('iM', 'iM')], \
               []

    def process_inputs(self, inputs, outputs, parameters):
        M = inputs['M'].value
        iM = cv2.invertAffineTransform(M=M)
        outputs['iM'] = Data(iM)

### line ###

class OpenCVAuto2_Line(NormalElement):
    name = 'Line'
    comment = '''line(img, pt1, pt2, color[, thickness[, lineType[, shift]]]) -> img\n@brief Draws a line segment connecting two points.\n\nThe function line draws the line segment between pt1 and pt2 points in the image. The line is\nclipped by the image boundaries. For non-antialiased lines with integer coordinates, the 8-connected\nor 4-connected Bresenham algorithm is used. Thick lines are drawn with rounding endings. Antialiased\nlines are drawn using Gaussian filtering.\n\n@param img Image.\n@param pt1 First point of the line segment.\n@param pt2 Second point of the line segment.\n@param color Line color.\n@param thickness Line thickness.\n@param lineType Type of the line. See #LineTypes.\n@param shift Number of fractional bits in the point coordinates.'''

    def get_attributes(self):
        return [Input('img', 'img')], \
               [Output('img', 'img')], \
               [PointParameter('pt1', 'pt1'),
                PointParameter('pt2', 'pt2'),
                ScalarParameter('color', 'color'),
                IntParameter('thickness', 'thickness')]

    def process_inputs(self, inputs, outputs, parameters):
        img = inputs['img'].value
        pt1 = parameters['pt1']
        pt2 = parameters['pt2']
        color = parameters['color']
        thickness = parameters['thickness']
        img = cv2.line(img=img, pt1=pt1, pt2=pt2, color=color, thickness=thickness)
        outputs['img'] = Data(img)

### linearPolar ###

class OpenCVAuto2_LinearPolar(NormalElement):
    name = 'LinearPolar'
    comment = '''linearPolar(src, center, maxRadius, flags[, dst]) -> dst\n@brief Remaps an image to polar coordinates space.\n\n@deprecated This function produces same result as cv::warpPolar(src, dst, src.size(), center, maxRadius, flags)\n\n@internal\nTransform the source image using the following transformation (See @ref polar_remaps_reference_image "Polar remaps reference image c)"):\n\f[\begin{array}{l}\ndst( \rho , \phi ) = src(x,y) \\\ndst.size() \leftarrow src.size()\n\end{array}\f]\n\nwhere\n\f[\begin{array}{l}\nI = (dx,dy) = (x - center.x,y - center.y) \\\n\rho = Kmag \cdot \texttt{magnitude} (I) ,\\\n\phi = angle \cdot \texttt{angle} (I)\n\end{array}\f]\n\nand\n\f[\begin{array}{l}\nKx = src.cols / maxRadius \\\nKy = src.rows / 2\Pi\n\end{array}\f]\n\n\n@param src Source image\n@param dst Destination image. It will have same size and type as src.\n@param center The transformation center;\n@param maxRadius The radius of the bounding circle to transform. It determines the inverse magnitude scale parameter too.\n@param flags A combination of interpolation methods, see #InterpolationFlags\n\n@note\n-   The function can not operate in-place.\n-   To calculate magnitude and angle in degrees #cartToPolar is used internally thus angles are measured from 0 to 360 with accuracy about 0.3 degrees.\n\n@sa cv::logPolar\n@endinternal'''

    def get_attributes(self):
        return [Input('src', 'src')], \
               [Output('dst', 'dst')], \
               [PointParameter('center', 'center'),
                IntParameter('maxRadius', 'maxRadius'),
                ComboboxParameter('flags', [('INTER_AREA',3),('INTER_BITS',5),('INTER_BITS2',10),('INTER_CUBIC',2),('INTER_LANCZOS4',4),('INTER_LINEAR',1),('INTER_LINEAR_EXACT',5),('INTER_MAX',7),('INTER_NEAREST',0),('INTER_TAB_SIZE',32),('INTER_TAB_SIZE2',1024)])]

    def process_inputs(self, inputs, outputs, parameters):
        src = inputs['src'].value
        center = parameters['center']
        maxRadius = parameters['maxRadius']
        flags = parameters['flags']
        dst = cv2.linearPolar(src=src, center=center, maxRadius=maxRadius, flags=flags)
        outputs['dst'] = Data(dst)

### log ###

class OpenCVAuto2_Log(NormalElement):
    name = 'Log'
    comment = '''log(src[, dst]) -> dst\n@brief Calculates the natural logarithm of every array element.\n\nThe function cv::log calculates the natural logarithm of every element of the input array:\n\f[\texttt{dst} (I) =  \log (\texttt{src}(I)) \f]\n\nOutput on zero, negative and special (NaN, Inf) values is undefined.\n\n@param src input array.\n@param dst output array of the same size and type as src .\n@sa exp, cartToPolar, polarToCart, phase, pow, sqrt, magnitude'''

    def get_attributes(self):
        return [Input('src', 'src')], \
               [Output('dst', 'dst')], \
               []

    def process_inputs(self, inputs, outputs, parameters):
        src = inputs['src'].value
        dst = cv2.log(src=src)
        outputs['dst'] = Data(dst)

### logPolar ###

class OpenCVAuto2_LogPolar(NormalElement):
    name = 'LogPolar'
    comment = '''logPolar(src, center, M, flags[, dst]) -> dst\n@brief Remaps an image to semilog-polar coordinates space.\n\n@deprecated This function produces same result as cv::warpPolar(src, dst, src.size(), center, maxRadius, flags+WARP_POLAR_LOG);\n\n@internal\nTransform the source image using the following transformation (See @ref polar_remaps_reference_image "Polar remaps reference image d)"):\n\f[\begin{array}{l}\ndst( \rho , \phi ) = src(x,y) \\\ndst.size() \leftarrow src.size()\n\end{array}\f]\n\nwhere\n\f[\begin{array}{l}\nI = (dx,dy) = (x - center.x,y - center.y) \\\n\rho = M \cdot log_e(\texttt{magnitude} (I)) ,\\\n\phi = Kangle \cdot \texttt{angle} (I) \\\n\end{array}\f]\n\nand\n\f[\begin{array}{l}\nM = src.cols / log_e(maxRadius) \\\nKangle = src.rows / 2\Pi \\\n\end{array}\f]\n\nThe function emulates the human "foveal" vision and can be used for fast scale and\nrotation-invariant template matching, for object tracking and so forth.\n@param src Source image\n@param dst Destination image. It will have same size and type as src.\n@param center The transformation center; where the output precision is maximal\n@param M Magnitude scale parameter. It determines the radius of the bounding circle to transform too.\n@param flags A combination of interpolation methods, see #InterpolationFlags\n\n@note\n-   The function can not operate in-place.\n-   To calculate magnitude and angle in degrees #cartToPolar is used internally thus angles are measured from 0 to 360 with accuracy about 0.3 degrees.\n\n@sa cv::linearPolar\n@endinternal'''

    def get_attributes(self):
        return [Input('src', 'src'),
                Input('M', 'M')], \
               [Output('dst', 'dst')], \
               [PointParameter('center', 'center'),
                ComboboxParameter('flags', [('INTER_AREA',3),('INTER_BITS',5),('INTER_BITS2',10),('INTER_CUBIC',2),('INTER_LANCZOS4',4),('INTER_LINEAR',1),('INTER_LINEAR_EXACT',5),('INTER_MAX',7),('INTER_NEAREST',0),('INTER_TAB_SIZE',32),('INTER_TAB_SIZE2',1024)])]

    def process_inputs(self, inputs, outputs, parameters):
        src = inputs['src'].value
        M = inputs['M'].value
        center = parameters['center']
        flags = parameters['flags']
        dst = cv2.logPolar(src=src, M=M, center=center, flags=flags)
        outputs['dst'] = Data(dst)

### max ###

class OpenCVAuto2_Max(NormalElement):
    name = 'Max'
    comment = '''max(src1, src2[, dst]) -> dst\n@brief Calculates per-element maximum of two arrays or an array and a scalar.\n\nThe function cv::max calculates the per-element maximum of two arrays:\n\f[\texttt{dst} (I)= \max ( \texttt{src1} (I), \texttt{src2} (I))\f]\nor array and a scalar:\n\f[\texttt{dst} (I)= \max ( \texttt{src1} (I), \texttt{value} )\f]\n@param src1 first input array.\n@param src2 second input array of the same size and type as src1 .\n@param dst output array of the same size and type as src1.\n@sa  min, compare, inRange, minMaxLoc, @ref MatrixExpressions'''

    def get_attributes(self):
        return [Input('src1', 'src1'),
                Input('src2', 'src2')], \
               [Output('dst', 'dst')], \
               []

    def process_inputs(self, inputs, outputs, parameters):
        src1 = inputs['src1'].value
        src2 = inputs['src2'].value
        dst = cv2.max(src1=src1, src2=src2)
        outputs['dst'] = Data(dst)

### meanStdDev ###

class OpenCVAuto2_MeanStdDev(NormalElement):
    name = 'MeanStdDev'
    comment = '''meanStdDev(src[, mean[, stddev[, mask]]]) -> mean, stddev\nCalculates a mean and standard deviation of array elements.\n\nThe function cv::meanStdDev calculates the mean and the standard deviation M\nof array elements independently for each channel and returns it via the\noutput parameters:\n\f[\begin{array}{l} N =  \sum _{I, \texttt{mask} (I)  \ne 0} 1 \\ \texttt{mean} _c =  \frac{\sum_{ I: \; \texttt{mask}(I) \ne 0} \texttt{src} (I)_c}{N} \\ \texttt{stddev} _c =  \sqrt{\frac{\sum_{ I: \; \texttt{mask}(I) \ne 0} \left ( \texttt{src} (I)_c -  \texttt{mean} _c \right )^2}{N}} \end{array}\f]\nWhen all the mask elements are 0's, the function returns\nmean=stddev=Scalar::all(0).\n@note The calculated standard deviation is only the diagonal of the\ncomplete normalized covariance matrix. If the full matrix is needed, you\ncan reshape the multi-channel array M x N to the single-channel array\nM\*N x mtx.channels() (only possible when the matrix is continuous) and\nthen pass the matrix to calcCovarMatrix .\n@param src input array that should have from 1 to 4 channels so that the results can be stored in\nScalar_ 's.\n@param mean output parameter: calculated mean value.\n@param stddev output parameter: calculated standard deviation.\n@param mask optional operation mask.\n@sa  countNonZero, mean, norm, minMaxLoc, calcCovarMatrix'''

    def get_attributes(self):
        return [Input('src', 'src'),
                Input('mask', 'mask', optional=True)], \
               [Output('mean', 'mean'),
                Output('stddev', 'stddev')], \
               []

    def process_inputs(self, inputs, outputs, parameters):
        src = inputs['src'].value
        mask = inputs['mask'].value
        mean, stddev = cv2.meanStdDev(src=src, mask=mask)
        outputs['mean'] = Data(mean)
        outputs['stddev'] = Data(stddev)

### medianBlur ###

class OpenCVAuto2_MedianBlur(NormalElement):
    name = 'MedianBlur'
    comment = '''medianBlur(src, ksize[, dst]) -> dst\n@brief Blurs an image using the median filter.\n\nThe function smoothes an image using the median filter with the \f$\texttt{ksize} \times\n\texttt{ksize}\f$ aperture. Each channel of a multi-channel image is processed independently.\nIn-place operation is supported.\n\n@note The median filter uses #BORDER_REPLICATE internally to cope with border pixels, see #BorderTypes\n\n@param src input 1-, 3-, or 4-channel image; when ksize is 3 or 5, the image depth should be\nCV_8U, CV_16U, or CV_32F, for larger aperture sizes, it can only be CV_8U.\n@param dst destination array of the same size and type as src.\n@param ksize aperture linear size; it must be odd and greater than 1, for example: 3, 5, 7 ...\n@sa  bilateralFilter, blur, boxFilter, GaussianBlur'''

    def get_attributes(self):
        return [Input('src', 'src')], \
               [Output('dst', 'dst')], \
               [SizeParameter('ksize', 'ksize')]

    def process_inputs(self, inputs, outputs, parameters):
        src = inputs['src'].value
        ksize = parameters['ksize']
        dst = cv2.medianBlur(src=src, ksize=ksize)
        outputs['dst'] = Data(dst)

### min ###

class OpenCVAuto2_Min(NormalElement):
    name = 'Min'
    comment = '''min(src1, src2[, dst]) -> dst\n@brief Calculates per-element minimum of two arrays or an array and a scalar.\n\nThe function cv::min calculates the per-element minimum of two arrays:\n\f[\texttt{dst} (I)= \min ( \texttt{src1} (I), \texttt{src2} (I))\f]\nor array and a scalar:\n\f[\texttt{dst} (I)= \min ( \texttt{src1} (I), \texttt{value} )\f]\n@param src1 first input array.\n@param src2 second input array of the same size and type as src1.\n@param dst output array of the same size and type as src1.\n@sa max, compare, inRange, minMaxLoc'''

    def get_attributes(self):
        return [Input('src1', 'src1'),
                Input('src2', 'src2')], \
               [Output('dst', 'dst')], \
               []

    def process_inputs(self, inputs, outputs, parameters):
        src1 = inputs['src1'].value
        src2 = inputs['src2'].value
        dst = cv2.min(src1=src1, src2=src2)
        outputs['dst'] = Data(dst)

### minEnclosingCircle ###

class OpenCVAuto2_MinEnclosingCircle(NormalElement):
    name = 'MinEnclosingCircle'
    comment = '''minEnclosingCircle(points) -> center, radius\n@brief Finds a circle of the minimum area enclosing a 2D point set.\n\nThe function finds the minimal enclosing circle of a 2D point set using an iterative algorithm.\n\n@param points Input vector of 2D points, stored in std::vector\<\> or Mat\n@param center Output center of the circle.\n@param radius Output radius of the circle.'''

    def get_attributes(self):
        return [Input('points', 'points')], \
               [Output('center', 'center'),
                Output('radius', 'radius')], \
               []

    def process_inputs(self, inputs, outputs, parameters):
        points = inputs['points'].value
        center, radius = cv2.minEnclosingCircle(points=points)
        outputs['center'] = Data(center)
        outputs['radius'] = Data(radius)

### minEnclosingTriangle ###

class OpenCVAuto2_MinEnclosingTriangle(NormalElement):
    name = 'MinEnclosingTriangle'
    comment = '''minEnclosingTriangle(points[, triangle]) -> retval, triangle\n@brief Finds a triangle of minimum area enclosing a 2D point set and returns its area.\n\nThe function finds a triangle of minimum area enclosing the given set of 2D points and returns its\narea. The output for a given 2D point set is shown in the image below. 2D points are depicted in\n*red* and the enclosing triangle in *yellow*.\n\n![Sample output of the minimum enclosing triangle function](pics/minenclosingtriangle.png)\n\nThe implementation of the algorithm is based on O'Rourke's @cite ORourke86 and Klee and Laskowski's\n@cite KleeLaskowski85 papers. O'Rourke provides a \f$\theta(n)\f$ algorithm for finding the minimal\nenclosing triangle of a 2D convex polygon with n vertices. Since the #minEnclosingTriangle function\ntakes a 2D point set as input an additional preprocessing step of computing the convex hull of the\n2D point set is required. The complexity of the #convexHull function is \f$O(n log(n))\f$ which is higher\nthan \f$\theta(n)\f$. Thus the overall complexity of the function is \f$O(n log(n))\f$.\n\n@param points Input vector of 2D points with depth CV_32S or CV_32F, stored in std::vector\<\> or Mat\n@param triangle Output vector of three 2D points defining the vertices of the triangle. The depth\nof the OutputArray must be CV_32F.'''

    def get_attributes(self):
        return [Input('points', 'points')], \
               [Output('triangle', 'triangle')], \
               []

    def process_inputs(self, inputs, outputs, parameters):
        points = inputs['points'].value
        retval, triangle = cv2.minEnclosingTriangle(points=points)
        outputs['triangle'] = Data(triangle)

### minMaxLoc ###

class OpenCVAuto2_MinMaxLoc(NormalElement):
    name = 'MinMaxLoc'
    comment = '''minMaxLoc(src[, mask]) -> minVal, maxVal, minLoc, maxLoc\n@brief Finds the global minimum and maximum in an array.\n\nThe function cv::minMaxLoc finds the minimum and maximum element values and their positions. The\nextremums are searched across the whole array or, if mask is not an empty array, in the specified\narray region.\n\nThe function do not work with multi-channel arrays. If you need to find minimum or maximum\nelements across all the channels, use Mat::reshape first to reinterpret the array as\nsingle-channel. Or you may extract the particular channel using either extractImageCOI , or\nmixChannels , or split .\n@param src input single-channel array.\n@param minVal pointer to the returned minimum value; NULL is used if not required.\n@param maxVal pointer to the returned maximum value; NULL is used if not required.\n@param minLoc pointer to the returned minimum location (in 2D case); NULL is used if not required.\n@param maxLoc pointer to the returned maximum location (in 2D case); NULL is used if not required.\n@param mask optional mask used to select a sub-array.\n@sa max, min, compare, inRange, extractImageCOI, mixChannels, split, Mat::reshape'''

    def get_attributes(self):
        return [Input('src', 'src'),
                Input('mask', 'mask', optional=True)], \
               [Output('minVal', 'minVal'),
                Output('maxVal', 'maxVal'),
                Output('minLoc', 'minLoc'),
                Output('maxLoc', 'maxLoc')], \
               []

    def process_inputs(self, inputs, outputs, parameters):
        src = inputs['src'].value
        mask = inputs['mask'].value
        minVal, maxVal, minLoc, maxLoc = cv2.minMaxLoc(src=src, mask=mask)
        outputs['minVal'] = Data(minVal)
        outputs['maxVal'] = Data(maxVal)
        outputs['minLoc'] = Data(minLoc)
        outputs['maxLoc'] = Data(maxLoc)

### multiply ###

class OpenCVAuto2_Multiply(NormalElement):
    name = 'Multiply'
    comment = '''multiply(src1, src2[, dst[, scale[, dtype]]]) -> dst\n@brief Calculates the per-element scaled product of two arrays.\n\nThe function multiply calculates the per-element product of two arrays:\n\n\f[\texttt{dst} (I)= \texttt{saturate} ( \texttt{scale} \cdot \texttt{src1} (I)  \cdot \texttt{src2} (I))\f]\n\nThere is also a @ref MatrixExpressions -friendly variant of the first function. See Mat::mul .\n\nFor a not-per-element matrix product, see gemm .\n\n@note Saturation is not applied when the output array has the depth\nCV_32S. You may even get result of an incorrect sign in the case of\noverflow.\n@param src1 first input array.\n@param src2 second input array of the same size and the same type as src1.\n@param dst output array of the same size and type as src1.\n@param scale optional scale factor.\n@param dtype optional depth of the output array\n@sa add, subtract, divide, scaleAdd, addWeighted, accumulate, accumulateProduct, accumulateSquare,\nMat::convertTo'''

    def get_attributes(self):
        return [Input('src1', 'src1'),
                Input('src2', 'src2')], \
               [Output('dst', 'dst')], \
               [FloatParameter('scale', 'scale'),
                ComboboxParameter('dtype', [('NONE',0),('CV_8U',0),('CV_8UC1',0),('CV_8UC2',8),('CV_8UC3',16),('CV_8UC4',24),('CV_8S',1),('CV_8SC1',1),('CV_8SC2',9),('CV_8SC3',17),('CV_8SC4',25),('CV_16U',2),('CV_16UC1',2),('CV_16UC2',10),('CV_16UC3',18),('CV_16UC4',26),('CV_16S',3),('CV_16SC1',3),('CV_16SC2',11),('CV_16SC3',19),('CV_16SC4',27),('CV_32S',4),('CV_32SC1',4),('CV_32SC2',12),('CV_32SC3',20),('CV_32SC4',28),('CV_32F',5),('CV_32FC1',5),('CV_32FC2',13),('CV_32FC3',21),('CV_32FC4',29),('CV_64F',6),('CV_64FC1',6),('CV_64FC2',14),('CV_64FC3',22),('CV_64FC4',30)])]

    def process_inputs(self, inputs, outputs, parameters):
        src1 = inputs['src1'].value
        src2 = inputs['src2'].value
        scale = parameters['scale']
        dtype = parameters['dtype']
        dst = cv2.multiply(src1=src1, src2=src2, scale=scale, dtype=dtype)
        outputs['dst'] = Data(dst)

### normalize ###

class OpenCVAuto2_Normalize(NormalElement):
    name = 'Normalize'
    comment = '''normalize(src, dst[, alpha[, beta[, norm_type[, dtype[, mask]]]]]) -> dst\n@brief Normalizes the norm or value range of an array.\n\nThe function cv::normalize normalizes scale and shift the input array elements so that\n\f[\| \texttt{dst} \| _{L_p}= \texttt{alpha}\f]\n(where p=Inf, 1 or 2) when normType=NORM_INF, NORM_L1, or NORM_L2, respectively; or so that\n\f[\min _I  \texttt{dst} (I)= \texttt{alpha} , \, \, \max _I  \texttt{dst} (I)= \texttt{beta}\f]\n\nwhen normType=NORM_MINMAX (for dense arrays only). The optional mask specifies a sub-array to be\nnormalized. This means that the norm or min-n-max are calculated over the sub-array, and then this\nsub-array is modified to be normalized. If you want to only use the mask to calculate the norm or\nmin-max but modify the whole array, you can use norm and Mat::convertTo.\n\nIn case of sparse matrices, only the non-zero values are analyzed and transformed. Because of this,\nthe range transformation for sparse matrices is not allowed since it can shift the zero level.\n\nPossible usage with some positive example data:\n@code{.cpp}\nvector<double> positiveData = { 2.0, 8.0, 10.0 };\nvector<double> normalizedData_l1, normalizedData_l2, normalizedData_inf, normalizedData_minmax;\n\n// Norm to probability (total count)\n// sum(numbers) = 20.0\n// 2.0      0.1     (2.0/20.0)\n// 8.0      0.4     (8.0/20.0)\n// 10.0     0.5     (10.0/20.0)\nnormalize(positiveData, normalizedData_l1, 1.0, 0.0, NORM_L1);\n\n// Norm to unit vector: ||positiveData|| = 1.0\n// 2.0      0.15\n// 8.0      0.62\n// 10.0     0.77\nnormalize(positiveData, normalizedData_l2, 1.0, 0.0, NORM_L2);\n\n// Norm to max element\n// 2.0      0.2     (2.0/10.0)\n// 8.0      0.8     (8.0/10.0)\n// 10.0     1.0     (10.0/10.0)\nnormalize(positiveData, normalizedData_inf, 1.0, 0.0, NORM_INF);\n\n// Norm to range [0.0;1.0]\n// 2.0      0.0     (shift to left border)\n// 8.0      0.75    (6.0/8.0)\n// 10.0     1.0     (shift to right border)\nnormalize(positiveData, normalizedData_minmax, 1.0, 0.0, NORM_MINMAX);\n@endcode\n\n@param src input array.\n@param dst output array of the same size as src .\n@param alpha norm value to normalize to or the lower range boundary in case of the range\nnormalization.\n@param beta upper range boundary in case of the range normalization; it is not used for the norm\nnormalization.\n@param norm_type normalization type (see cv::NormTypes).\n@param dtype when negative, the output array has the same type as src; otherwise, it has the same\nnumber of channels as src and the depth =CV_MAT_DEPTH(dtype).\n@param mask optional operation mask.\n@sa norm, Mat::convertTo, SparseMat::convertTo'''

    def get_attributes(self):
        return [Input('src', 'src'),
                Input('dst', 'dst'),
                Input('mask', 'mask', optional=True)], \
               [Output('dst', 'dst')], \
               [FloatParameter('alpha', 'alpha'),
                FloatParameter('beta', 'beta'),
                ComboboxParameter('dtype', [('NONE',0),('CV_8U',0),('CV_8UC1',0),('CV_8UC2',8),('CV_8UC3',16),('CV_8UC4',24),('CV_8S',1),('CV_8SC1',1),('CV_8SC2',9),('CV_8SC3',17),('CV_8SC4',25),('CV_16U',2),('CV_16UC1',2),('CV_16UC2',10),('CV_16UC3',18),('CV_16UC4',26),('CV_16S',3),('CV_16SC1',3),('CV_16SC2',11),('CV_16SC3',19),('CV_16SC4',27),('CV_32S',4),('CV_32SC1',4),('CV_32SC2',12),('CV_32SC3',20),('CV_32SC4',28),('CV_32F',5),('CV_32FC1',5),('CV_32FC2',13),('CV_32FC3',21),('CV_32FC4',29),('CV_64F',6),('CV_64FC1',6),('CV_64FC2',14),('CV_64FC3',22),('CV_64FC4',30)])]

    def process_inputs(self, inputs, outputs, parameters):
        src = inputs['src'].value
        dst = inputs['dst'].value
        mask = inputs['mask'].value
        alpha = parameters['alpha']
        beta = parameters['beta']
        dtype = parameters['dtype']
        dst = cv2.normalize(src=src, dst=dst, mask=mask, alpha=alpha, beta=beta, dtype=dtype)
        outputs['dst'] = Data(dst)

### pencilSketch ###

class OpenCVAuto2_PencilSketch(NormalElement):
    name = 'PencilSketch'
    comment = '''pencilSketch(src[, dst1[, dst2[, sigma_s[, sigma_r[, shade_factor]]]]]) -> dst1, dst2\n@brief Pencil-like non-photorealistic line drawing\n\n@param src Input 8-bit 3-channel image.\n@param dst1 Output 8-bit 1-channel image.\n@param dst2 Output image with the same size and type as src.\n@param sigma_s Range between 0 to 200.\n@param sigma_r Range between 0 to 1.\n@param shade_factor Range between 0 to 0.1.'''

    def get_attributes(self):
        return [Input('src', 'src')], \
               [Output('dst1', 'dst1'),
                Output('dst2', 'dst2')], \
               [FloatParameter('sigma_s', 'sigma_s'),
                FloatParameter('sigma_r', 'sigma_r')]

    def process_inputs(self, inputs, outputs, parameters):
        src = inputs['src'].value
        sigma_s = parameters['sigma_s']
        sigma_r = parameters['sigma_r']
        dst1, dst2 = cv2.pencilSketch(src=src, sigma_s=sigma_s, sigma_r=sigma_r)
        outputs['dst1'] = Data(dst1)
        outputs['dst2'] = Data(dst2)

### perspectiveTransform ###

class OpenCVAuto2_PerspectiveTransform(NormalElement):
    name = 'PerspectiveTransform'
    comment = '''perspectiveTransform(src, m[, dst]) -> dst\n@brief Performs the perspective matrix transformation of vectors.\n\nThe function cv::perspectiveTransform transforms every element of src by\ntreating it as a 2D or 3D vector, in the following way:\n\f[(x, y, z)  \rightarrow (x'/w, y'/w, z'/w)\f]\nwhere\n\f[(x', y', z', w') =  \texttt{mat} \cdot \begin{bmatrix} x & y & z & 1  \end{bmatrix}\f]\nand\n\f[w =  \fork{w'}{if \(w' \ne 0\)}{\infty}{otherwise}\f]\n\nHere a 3D vector transformation is shown. In case of a 2D vector\ntransformation, the z component is omitted.\n\n@note The function transforms a sparse set of 2D or 3D vectors. If you\nwant to transform an image using perspective transformation, use\nwarpPerspective . If you have an inverse problem, that is, you want to\ncompute the most probable perspective transformation out of several\npairs of corresponding points, you can use getPerspectiveTransform or\nfindHomography .\n@param src input two-channel or three-channel floating-point array; each\nelement is a 2D/3D vector to be transformed.\n@param dst output array of the same size and type as src.\n@param m 3x3 or 4x4 floating-point transformation matrix.\n@sa  transform, warpPerspective, getPerspectiveTransform, findHomography'''

    def get_attributes(self):
        return [Input('src', 'src'),
                Input('m', 'm')], \
               [Output('dst', 'dst')], \
               []

    def process_inputs(self, inputs, outputs, parameters):
        src = inputs['src'].value
        m = inputs['m'].value
        dst = cv2.perspectiveTransform(src=src, m=m)
        outputs['dst'] = Data(dst)

### phaseCorrelate ###

class OpenCVAuto2_PhaseCorrelate(NormalElement):
    name = 'PhaseCorrelate'
    comment = '''phaseCorrelate(src1, src2[, window]) -> retval, response\n@brief The function is used to detect translational shifts that occur between two images.\n\nThe operation takes advantage of the Fourier shift theorem for detecting the translational shift in\nthe frequency domain. It can be used for fast image registration as well as motion estimation. For\nmore information please see <http://en.wikipedia.org/wiki/Phase_correlation>\n\nCalculates the cross-power spectrum of two supplied source arrays. The arrays are padded if needed\nwith getOptimalDFTSize.\n\nThe function performs the following equations:\n- First it applies a Hanning window (see <http://en.wikipedia.org/wiki/Hann_function>) to each\nimage to remove possible edge effects. This window is cached until the array size changes to speed\nup processing time.\n- Next it computes the forward DFTs of each source array:\n\f[\mathbf{G}_a = \mathcal{F}\{src_1\}, \; \mathbf{G}_b = \mathcal{F}\{src_2\}\f]\nwhere \f$\mathcal{F}\f$ is the forward DFT.\n- It then computes the cross-power spectrum of each frequency domain array:\n\f[R = \frac{ \mathbf{G}_a \mathbf{G}_b^*}{|\mathbf{G}_a \mathbf{G}_b^*|}\f]\n- Next the cross-correlation is converted back into the time domain via the inverse DFT:\n\f[r = \mathcal{F}^{-1}\{R\}\f]\n- Finally, it computes the peak location and computes a 5x5 weighted centroid around the peak to\nachieve sub-pixel accuracy.\n\f[(\Delta x, \Delta y) = \texttt{weightedCentroid} \{\arg \max_{(x, y)}\{r\}\}\f]\n- If non-zero, the response parameter is computed as the sum of the elements of r within the 5x5\ncentroid around the peak location. It is normalized to a maximum of 1 (meaning there is a single\npeak) and will be smaller when there are multiple peaks.\n\n@param src1 Source floating point array (CV_32FC1 or CV_64FC1)\n@param src2 Source floating point array (CV_32FC1 or CV_64FC1)\n@param window Floating point array with windowing coefficients to reduce edge effects (optional).\n@param response Signal power within the 5x5 centroid around the peak, between 0 and 1 (optional).\n@returns detected phase shift (sub-pixel) between the two arrays.\n\n@sa dft, getOptimalDFTSize, idft, mulSpectrums createHanningWindow'''

    def get_attributes(self):
        return [Input('src1', 'src1'),
                Input('src2', 'src2')], \
               [Output('response', 'response')], \
               []

    def process_inputs(self, inputs, outputs, parameters):
        src1 = inputs['src1'].value
        src2 = inputs['src2'].value
        retval, response = cv2.phaseCorrelate(src1=src1, src2=src2)
        outputs['response'] = Data(response)

### preCornerDetect ###

class OpenCVAuto2_PreCornerDetect(NormalElement):
    name = 'PreCornerDetect'
    comment = '''preCornerDetect(src, ksize[, dst[, borderType]]) -> dst\n@brief Calculates a feature map for corner detection.\n\nThe function calculates the complex spatial derivative-based function of the source image\n\n\f[\texttt{dst} = (D_x  \texttt{src} )^2  \cdot D_{yy}  \texttt{src} + (D_y  \texttt{src} )^2  \cdot D_{xx}  \texttt{src} - 2 D_x  \texttt{src} \cdot D_y  \texttt{src} \cdot D_{xy}  \texttt{src}\f]\n\nwhere \f$D_x\f$,\f$D_y\f$ are the first image derivatives, \f$D_{xx}\f$,\f$D_{yy}\f$ are the second image\nderivatives, and \f$D_{xy}\f$ is the mixed derivative.\n\nThe corners can be found as local maximums of the functions, as shown below:\n@code\nMat corners, dilated_corners;\npreCornerDetect(image, corners, 3);\n// dilation with 3x3 rectangular structuring element\ndilate(corners, dilated_corners, Mat(), 1);\nMat corner_mask = corners == dilated_corners;\n@endcode\n\n@param src Source single-channel 8-bit of floating-point image.\n@param dst Output image that has the type CV_32F and the same size as src .\n@param ksize %Aperture size of the Sobel .\n@param borderType Pixel extrapolation method. See #BorderTypes.'''

    def get_attributes(self):
        return [Input('src', 'src')], \
               [Output('dst', 'dst')], \
               [SizeParameter('ksize', 'ksize'),
                ComboboxParameter('borderType', [('BORDER_CONSTANT',0),('BORDER_DEFAULT',4),('BORDER_ISOLATED',16),('BORDER_REFLECT',2),('BORDER_REFLECT101',4),('BORDER_REFLECT_101',4),('BORDER_REPLICATE',1),('BORDER_TRANSPARENT',5),('BORDER_WRAP',3)])]

    def process_inputs(self, inputs, outputs, parameters):
        src = inputs['src'].value
        ksize = parameters['ksize']
        borderType = parameters['borderType']
        dst = cv2.preCornerDetect(src=src, ksize=ksize, borderType=borderType)
        outputs['dst'] = Data(dst)

### pyrDown ###

class OpenCVAuto2_PyrDown(NormalElement):
    name = 'PyrDown'
    comment = '''pyrDown(src[, dst[, dstsize[, borderType]]]) -> dst\n@brief Blurs an image and downsamples it.\n\nBy default, size of the output image is computed as `Size((src.cols+1)/2, (src.rows+1)/2)`, but in\nany case, the following conditions should be satisfied:\n\n\f[\begin{array}{l} | \texttt{dstsize.width} *2-src.cols| \leq 2 \\ | \texttt{dstsize.height} *2-src.rows| \leq 2 \end{array}\f]\n\nThe function performs the downsampling step of the Gaussian pyramid construction. First, it\nconvolves the source image with the kernel:\n\n\f[\frac{1}{256} \begin{bmatrix} 1 & 4 & 6 & 4 & 1  \\ 4 & 16 & 24 & 16 & 4  \\ 6 & 24 & 36 & 24 & 6  \\ 4 & 16 & 24 & 16 & 4  \\ 1 & 4 & 6 & 4 & 1 \end{bmatrix}\f]\n\nThen, it downsamples the image by rejecting even rows and columns.\n\n@param src input image.\n@param dst output image; it has the specified size and the same type as src.\n@param dstsize size of the output image.\n@param borderType Pixel extrapolation method, see #BorderTypes (#BORDER_CONSTANT isn't supported)'''

    def get_attributes(self):
        return [Input('src', 'src')], \
               [Output('dst', 'dst')], \
               [SizeParameter('dstsize', 'dstsize'),
                ComboboxParameter('borderType', [('BORDER_CONSTANT',0),('BORDER_DEFAULT',4),('BORDER_ISOLATED',16),('BORDER_REFLECT',2),('BORDER_REFLECT101',4),('BORDER_REFLECT_101',4),('BORDER_REPLICATE',1),('BORDER_TRANSPARENT',5),('BORDER_WRAP',3)])]

    def process_inputs(self, inputs, outputs, parameters):
        src = inputs['src'].value
        dstsize = parameters['dstsize']
        borderType = parameters['borderType']
        dst = cv2.pyrDown(src=src, dstsize=dstsize, borderType=borderType)
        outputs['dst'] = Data(dst)

### pyrUp ###

class OpenCVAuto2_PyrUp(NormalElement):
    name = 'PyrUp'
    comment = '''pyrUp(src[, dst[, dstsize[, borderType]]]) -> dst\n@brief Upsamples an image and then blurs it.\n\nBy default, size of the output image is computed as `Size(src.cols\*2, (src.rows\*2)`, but in any\ncase, the following conditions should be satisfied:\n\n\f[\begin{array}{l} | \texttt{dstsize.width} -src.cols*2| \leq  ( \texttt{dstsize.width}   \mod  2)  \\ | \texttt{dstsize.height} -src.rows*2| \leq  ( \texttt{dstsize.height}   \mod  2) \end{array}\f]\n\nThe function performs the upsampling step of the Gaussian pyramid construction, though it can\nactually be used to construct the Laplacian pyramid. First, it upsamples the source image by\ninjecting even zero rows and columns and then convolves the result with the same kernel as in\npyrDown multiplied by 4.\n\n@param src input image.\n@param dst output image. It has the specified size and the same type as src .\n@param dstsize size of the output image.\n@param borderType Pixel extrapolation method, see #BorderTypes (only #BORDER_DEFAULT is supported)'''

    def get_attributes(self):
        return [Input('src', 'src')], \
               [Output('dst', 'dst')], \
               [SizeParameter('dstsize', 'dstsize'),
                ComboboxParameter('borderType', [('BORDER_DEFAULT',4),('BORDER_CONSTANT',0),('BORDER_ISOLATED',16),('BORDER_REFLECT',2),('BORDER_REFLECT101',4),('BORDER_REFLECT_101',4),('BORDER_REPLICATE',1),('BORDER_TRANSPARENT',5),('BORDER_WRAP',3)])]

    def process_inputs(self, inputs, outputs, parameters):
        src = inputs['src'].value
        dstsize = parameters['dstsize']
        borderType = parameters['borderType']
        dst = cv2.pyrUp(src=src, dstsize=dstsize, borderType=borderType)
        outputs['dst'] = Data(dst)

### randShuffle ###

class OpenCVAuto2_RandShuffle(NormalElement):
    name = 'RandShuffle'
    comment = '''randShuffle(dst[, iterFactor]) -> dst\n@brief Shuffles the array elements randomly.\n\nThe function cv::randShuffle shuffles the specified 1D array by randomly choosing pairs of elements and\nswapping them. The number of such swap operations will be dst.rows\*dst.cols\*iterFactor .\n@param dst input/output numerical 1D array.\n@param iterFactor scale factor that determines the number of random swap operations (see the details\nbelow).\n@param rng optional random number generator used for shuffling; if it is zero, theRNG () is used\ninstead.\n@sa RNG, sort'''

    def get_attributes(self):
        return [Input('dst', 'dst')], \
               [Output('dst', 'dst')], \
               []

    def process_inputs(self, inputs, outputs, parameters):
        dst = inputs['dst'].value
        dst = cv2.randShuffle(dst=dst)
        outputs['dst'] = Data(dst)

### rectangle ###

class OpenCVAuto2_Rectangle(NormalElement):
    name = 'Rectangle'
    comment = '''rectangle(img, pt1, pt2, color[, thickness[, lineType[, shift]]]) -> img\n@brief Draws a simple, thick, or filled up-right rectangle.\n\nThe function cv::rectangle draws a rectangle outline or a filled rectangle whose two opposite corners\nare pt1 and pt2.\n\n@param img Image.\n@param pt1 Vertex of the rectangle.\n@param pt2 Vertex of the rectangle opposite to pt1 .\n@param color Rectangle color or brightness (grayscale image).\n@param thickness Thickness of lines that make up the rectangle. Negative values, like #FILLED,\nmean that the function has to draw a filled rectangle.\n@param lineType Type of the line. See #LineTypes\n@param shift Number of fractional bits in the point coordinates.'''

    def get_attributes(self):
        return [Input('img', 'img')], \
               [Output('img', 'img')], \
               [PointParameter('pt1', 'pt1'),
                PointParameter('pt2', 'pt2'),
                ScalarParameter('color', 'color'),
                IntParameter('thickness', 'thickness')]

    def process_inputs(self, inputs, outputs, parameters):
        img = inputs['img'].value
        pt1 = parameters['pt1']
        pt2 = parameters['pt2']
        color = parameters['color']
        thickness = parameters['thickness']
        img = cv2.rectangle(img=img, pt1=pt1, pt2=pt2, color=color, thickness=thickness)
        outputs['img'] = Data(img)

### resize ###

class OpenCVAuto2_Resize(NormalElement):
    name = 'Resize'
    comment = '''resize(src, dsize[, dst[, fx[, fy[, interpolation]]]]) -> dst\n@brief Resizes an image.\n\nThe function resize resizes the image src down to or up to the specified size. Note that the\ninitial dst type or size are not taken into account. Instead, the size and type are derived from\nthe `src`,`dsize`,`fx`, and `fy`. If you want to resize src so that it fits the pre-created dst,\nyou may call the function as follows:\n@code\n// explicitly specify dsize=dst.size(); fx and fy will be computed from that.\nresize(src, dst, dst.size(), 0, 0, interpolation);\n@endcode\nIf you want to decimate the image by factor of 2 in each direction, you can call the function this\nway:\n@code\n// specify fx and fy and let the function compute the destination image size.\nresize(src, dst, Size(), 0.5, 0.5, interpolation);\n@endcode\nTo shrink an image, it will generally look best with #INTER_AREA interpolation, whereas to\nenlarge an image, it will generally look best with c#INTER_CUBIC (slow) or #INTER_LINEAR\n(faster but still looks OK).\n\n@param src input image.\n@param dst output image; it has the size dsize (when it is non-zero) or the size computed from\nsrc.size(), fx, and fy; the type of dst is the same as of src.\n@param dsize output image size; if it equals zero, it is computed as:\n\f[\texttt{dsize = Size(round(fx*src.cols), round(fy*src.rows))}\f]\nEither dsize or both fx and fy must be non-zero.\n@param fx scale factor along the horizontal axis; when it equals 0, it is computed as\n\f[\texttt{(double)dsize.width/src.cols}\f]\n@param fy scale factor along the vertical axis; when it equals 0, it is computed as\n\f[\texttt{(double)dsize.height/src.rows}\f]\n@param interpolation interpolation method, see #InterpolationFlags\n\n@sa  warpAffine, warpPerspective, remap'''

    def get_attributes(self):
        return [Input('src', 'src')], \
               [Output('dst', 'dst')], \
               [SizeParameter('dsize', 'dsize'),
                ComboboxParameter('interpolation', [('INTER_AREA',3),('INTER_BITS',5),('INTER_BITS2',10),('INTER_CUBIC',2),('INTER_LANCZOS4',4),('INTER_LINEAR',1),('INTER_LINEAR_EXACT',5),('INTER_MAX',7),('INTER_NEAREST',0),('INTER_TAB_SIZE',32),('INTER_TAB_SIZE2',1024)])]

    def process_inputs(self, inputs, outputs, parameters):
        src = inputs['src'].value
        dsize = parameters['dsize']
        interpolation = parameters['interpolation']
        dst = cv2.resize(src=src, dsize=dsize, interpolation=interpolation)
        outputs['dst'] = Data(dst)

### scaleAdd ###

class OpenCVAuto2_ScaleAdd(NormalElement):
    name = 'ScaleAdd'
    comment = '''scaleAdd(src1, alpha, src2[, dst]) -> dst\n@brief Calculates the sum of a scaled array and another array.\n\nThe function scaleAdd is one of the classical primitive linear algebra operations, known as DAXPY\nor SAXPY in [BLAS](http://en.wikipedia.org/wiki/Basic_Linear_Algebra_Subprograms). It calculates\nthe sum of a scaled array and another array:\n\f[\texttt{dst} (I)= \texttt{scale} \cdot \texttt{src1} (I) +  \texttt{src2} (I)\f]\nThe function can also be emulated with a matrix expression, for example:\n@code{.cpp}\nMat A(3, 3, CV_64F);\n...\nA.row(0) = A.row(1)*2 + A.row(2);\n@endcode\n@param src1 first input array.\n@param alpha scale factor for the first array.\n@param src2 second input array of the same size and type as src1.\n@param dst output array of the same size and type as src1.\n@sa add, addWeighted, subtract, Mat::dot, Mat::convertTo'''

    def get_attributes(self):
        return [Input('src1', 'src1'),
                Input('src2', 'src2')], \
               [Output('dst', 'dst')], \
               [FloatParameter('alpha', 'alpha')]

    def process_inputs(self, inputs, outputs, parameters):
        src1 = inputs['src1'].value
        src2 = inputs['src2'].value
        alpha = parameters['alpha']
        dst = cv2.scaleAdd(src1=src1, src2=src2, alpha=alpha)
        outputs['dst'] = Data(dst)

### selectROIs ###

class OpenCVAuto2_SelectROIs(NormalElement):
    name = 'SelectROIs'
    comment = '''selectROIs(windowName, img[, showCrosshair[, fromCenter]]) -> boundingBoxes\n@brief Selects ROIs on the given image.\nFunction creates a window and allows user to select a ROIs using mouse.\nControls: use `space` or `enter` to finish current selection and start a new one,\nuse `esc` to terminate multiple ROI selection process.\n\n@param windowName name of the window where selection process will be shown.\n@param img image to select a ROI.\n@param boundingBoxes selected ROIs.\n@param showCrosshair if true crosshair of selection rectangle will be shown.\n@param fromCenter if true center of selection will match initial mouse position. In opposite case a corner of\nselection rectangle will correspont to the initial mouse position.\n\n@note The function sets it's own mouse callback for specified window using cv::setMouseCallback(windowName, ...).\nAfter finish of work an empty callback will be set for the used window.'''

    def get_attributes(self):
        return [Input('img', 'img')], \
               [Output('boundingBoxes', 'boundingBoxes')], \
               [TextParameter('windowName', 'windowName')]

    def process_inputs(self, inputs, outputs, parameters):
        img = inputs['img'].value
        windowName = parameters['windowName']
        boundingBoxes = cv2.selectROIs(img=img, windowName=windowName)
        outputs['boundingBoxes'] = Data(boundingBoxes)

### setIdentity ###

class OpenCVAuto2_SetIdentity(NormalElement):
    name = 'SetIdentity'
    comment = '''setIdentity(mtx[, s]) -> mtx\n@brief Initializes a scaled identity matrix.\n\nThe function cv::setIdentity initializes a scaled identity matrix:\n\f[\texttt{mtx} (i,j)= \fork{\texttt{value}}{ if \(i=j\)}{0}{otherwise}\f]\n\nThe function can also be emulated using the matrix initializers and the\nmatrix expressions:\n@code\nMat A = Mat::eye(4, 3, CV_32F)*5;\n// A will be set to [[5, 0, 0], [0, 5, 0], [0, 0, 5], [0, 0, 0]]\n@endcode\n@param mtx matrix to initialize (not necessarily square).\n@param s value to assign to diagonal elements.\n@sa Mat::zeros, Mat::ones, Mat::setTo, Mat::operator='''

    def get_attributes(self):
        return [Input('mtx', 'mtx')], \
               [Output('mtx', 'mtx')], \
               []

    def process_inputs(self, inputs, outputs, parameters):
        mtx = inputs['mtx'].value
        mtx = cv2.setIdentity(mtx=mtx)
        outputs['mtx'] = Data(mtx)

### solveCubic ###

class OpenCVAuto2_SolveCubic(NormalElement):
    name = 'SolveCubic'
    comment = '''solveCubic(coeffs[, roots]) -> retval, roots\n@brief Finds the real roots of a cubic equation.\n\nThe function solveCubic finds the real roots of a cubic equation:\n-   if coeffs is a 4-element vector:\n\f[\texttt{coeffs} [0] x^3 +  \texttt{coeffs} [1] x^2 +  \texttt{coeffs} [2] x +  \texttt{coeffs} [3] = 0\f]\n-   if coeffs is a 3-element vector:\n\f[x^3 +  \texttt{coeffs} [0] x^2 +  \texttt{coeffs} [1] x +  \texttt{coeffs} [2] = 0\f]\n\nThe roots are stored in the roots array.\n@param coeffs equation coefficients, an array of 3 or 4 elements.\n@param roots output array of real roots that has 1 or 3 elements.\n@return number of real roots. It can be 0, 1 or 2.'''

    def get_attributes(self):
        return [Input('coeffs', 'coeffs')], \
               [Output('roots', 'roots')], \
               []

    def process_inputs(self, inputs, outputs, parameters):
        coeffs = inputs['coeffs'].value
        retval, roots = cv2.solveCubic(coeffs=coeffs)
        outputs['roots'] = Data(roots)

### solveP3P ###

class OpenCVAuto2_SolveP3P(NormalElement):
    name = 'SolveP3P'
    comment = '''solveP3P(objectPoints, imagePoints, cameraMatrix, distCoeffs, flags[, rvecs[, tvecs]]) -> retval, rvecs, tvecs\n@brief Finds an object pose from 3 3D-2D point correspondences.\n\n@param objectPoints Array of object points in the object coordinate space, 3x3 1-channel or\n1x3/3x1 3-channel. vector\<Point3f\> can be also passed here.\n@param imagePoints Array of corresponding image points, 3x2 1-channel or 1x3/3x1 2-channel.\nvector\<Point2f\> can be also passed here.\n@param cameraMatrix Input camera matrix \f$A = \vecthreethree{fx}{0}{cx}{0}{fy}{cy}{0}{0}{1}\f$ .\n@param distCoeffs Input vector of distortion coefficients\n\f$(k_1, k_2, p_1, p_2[, k_3[, k_4, k_5, k_6 [, s_1, s_2, s_3, s_4[, \tau_x, \tau_y]]]])\f$ of\n4, 5, 8, 12 or 14 elements. If the vector is NULL/empty, the zero distortion coefficients are\nassumed.\n@param rvecs Output rotation vectors (see Rodrigues ) that, together with tvecs , brings points from\nthe model coordinate system to the camera coordinate system. A P3P problem has up to 4 solutions.\n@param tvecs Output translation vectors.\n@param flags Method for solving a P3P problem:\n-   **SOLVEPNP_P3P** Method is based on the paper of X.S. Gao, X.-R. Hou, J. Tang, H.-F. Chang\n"Complete Solution Classification for the Perspective-Three-Point Problem" (@cite gao2003complete).\n-   **SOLVEPNP_AP3P** Method is based on the paper of Tong Ke and Stergios I. Roumeliotis.\n"An Efficient Algebraic Solution to the Perspective-Three-Point Problem" (@cite Ke17).\n\nThe function estimates the object pose given 3 object points, their corresponding image\nprojections, as well as the camera matrix and the distortion coefficients.'''

    def get_attributes(self):
        return [Input('objectPoints', 'objectPoints'),
                Input('imagePoints', 'imagePoints'),
                Input('cameraMatrix', 'cameraMatrix'),
                Input('distCoeffs', 'distCoeffs')], \
               [Output('rvecs', 'rvecs'),
                Output('tvecs', 'tvecs')], \
               [ComboboxParameter('flags', [('SOLVEPNP_P3P',2),('SOLVEPNP_AP3P',5)])]

    def process_inputs(self, inputs, outputs, parameters):
        objectPoints = inputs['objectPoints'].value
        imagePoints = inputs['imagePoints'].value
        cameraMatrix = inputs['cameraMatrix'].value
        distCoeffs = inputs['distCoeffs'].value
        flags = parameters['flags']
        retval, rvecs, tvecs = cv2.solveP3P(objectPoints=objectPoints, imagePoints=imagePoints, cameraMatrix=cameraMatrix, distCoeffs=distCoeffs, flags=flags)
        outputs['rvecs'] = Data(rvecs)
        outputs['tvecs'] = Data(tvecs)

### solvePnP ###

class OpenCVAuto2_SolvePnP(NormalElement):
    name = 'SolvePnP'
    comment = '''solvePnP(objectPoints, imagePoints, cameraMatrix, distCoeffs[, rvec[, tvec[, useExtrinsicGuess[, flags]]]]) -> retval, rvec, tvec\n@brief Finds an object pose from 3D-2D point correspondences.\n\n@param objectPoints Array of object points in the object coordinate space, Nx3 1-channel or\n1xN/Nx1 3-channel, where N is the number of points. vector\<Point3f\> can be also passed here.\n@param imagePoints Array of corresponding image points, Nx2 1-channel or 1xN/Nx1 2-channel,\nwhere N is the number of points. vector\<Point2f\> can be also passed here.\n@param cameraMatrix Input camera matrix \f$A = \vecthreethree{fx}{0}{cx}{0}{fy}{cy}{0}{0}{1}\f$ .\n@param distCoeffs Input vector of distortion coefficients\n\f$(k_1, k_2, p_1, p_2[, k_3[, k_4, k_5, k_6 [, s_1, s_2, s_3, s_4[, \tau_x, \tau_y]]]])\f$ of\n4, 5, 8, 12 or 14 elements. If the vector is NULL/empty, the zero distortion coefficients are\nassumed.\n@param rvec Output rotation vector (see @ref Rodrigues ) that, together with tvec , brings points from\nthe model coordinate system to the camera coordinate system.\n@param tvec Output translation vector.\n@param useExtrinsicGuess Parameter used for #SOLVEPNP_ITERATIVE. If true (1), the function uses\nthe provided rvec and tvec values as initial approximations of the rotation and translation\nvectors, respectively, and further optimizes them.\n@param flags Method for solving a PnP problem:\n-   **SOLVEPNP_ITERATIVE** Iterative method is based on Levenberg-Marquardt optimization. In\nthis case the function finds such a pose that minimizes reprojection error, that is the sum\nof squared distances between the observed projections imagePoints and the projected (using\nprojectPoints ) objectPoints .\n-   **SOLVEPNP_P3P** Method is based on the paper of X.S. Gao, X.-R. Hou, J. Tang, H.-F. Chang\n"Complete Solution Classification for the Perspective-Three-Point Problem" (@cite gao2003complete).\nIn this case the function requires exactly four object and image points.\n-   **SOLVEPNP_AP3P** Method is based on the paper of T. Ke, S. Roumeliotis\n"An Efficient Algebraic Solution to the Perspective-Three-Point Problem" (@cite Ke17).\nIn this case the function requires exactly four object and image points.\n-   **SOLVEPNP_EPNP** Method has been introduced by F.Moreno-Noguer, V.Lepetit and P.Fua in the\npaper "EPnP: Efficient Perspective-n-Point Camera Pose Estimation" (@cite lepetit2009epnp).\n-   **SOLVEPNP_DLS** Method is based on the paper of Joel A. Hesch and Stergios I. Roumeliotis.\n"A Direct Least-Squares (DLS) Method for PnP" (@cite hesch2011direct).\n-   **SOLVEPNP_UPNP** Method is based on the paper of A.Penate-Sanchez, J.Andrade-Cetto,\nF.Moreno-Noguer. "Exhaustive Linearization for Robust Camera Pose and Focal Length\nEstimation" (@cite penate2013exhaustive). In this case the function also estimates the parameters \f$f_x\f$ and \f$f_y\f$\nassuming that both have the same value. Then the cameraMatrix is updated with the estimated\nfocal length.\n-   **SOLVEPNP_AP3P** Method is based on the paper of Tong Ke and Stergios I. Roumeliotis.\n"An Efficient Algebraic Solution to the Perspective-Three-Point Problem" (@cite Ke17). In this case the\nfunction requires exactly four object and image points.\n\nThe function estimates the object pose given a set of object points, their corresponding image\nprojections, as well as the camera matrix and the distortion coefficients, see the figure below\n(more precisely, the X-axis of the camera frame is pointing to the right, the Y-axis downward\nand the Z-axis forward).\n\n![](pnp.jpg)\n\nPoints expressed in the world frame \f$ \bf{X}_w \f$ are projected into the image plane \f$ \left[ u, v \right] \f$\nusing the perspective projection model \f$ \Pi \f$ and the camera intrinsic parameters matrix \f$ \bf{A} \f$:\n\n\f[\n\begin{align*}\n\begin{bmatrix}\nu \\\nv \\\n1\n\end{bmatrix} &=\n\bf{A} \hspace{0.1em} \Pi \hspace{0.2em} ^{c}\bf{M}_w\n\begin{bmatrix}\nX_{w} \\\nY_{w} \\\nZ_{w} \\\n1\n\end{bmatrix} \\\n\begin{bmatrix}\nu \\\nv \\\n1\n\end{bmatrix} &=\n\begin{bmatrix}\nf_x & 0 & c_x \\\n0 & f_y & c_y \\\n0 & 0 & 1\n\end{bmatrix}\n\begin{bmatrix}\n1 & 0 & 0 & 0 \\\n0 & 1 & 0 & 0 \\\n0 & 0 & 1 & 0\n\end{bmatrix}\n\begin{bmatrix}\nr_{11} & r_{12} & r_{13} & t_x \\\nr_{21} & r_{22} & r_{23} & t_y \\\nr_{31} & r_{32} & r_{33} & t_z \\\n0 & 0 & 0 & 1\n\end{bmatrix}\n\begin{bmatrix}\nX_{w} \\\nY_{w} \\\nZ_{w} \\\n1\n\end{bmatrix}\n\end{align*}\n\f]\n\nThe estimated pose is thus the rotation (`rvec`) and the translation (`tvec`) vectors that allow to transform\na 3D point expressed in the world frame into the camera frame:\n\n\f[\n\begin{align*}\n\begin{bmatrix}\nX_c \\\nY_c \\\nZ_c \\\n1\n\end{bmatrix} &=\n\hspace{0.2em} ^{c}\bf{M}_w\n\begin{bmatrix}\nX_{w} \\\nY_{w} \\\nZ_{w} \\\n1\n\end{bmatrix} \\\n\begin{bmatrix}\nX_c \\\nY_c \\\nZ_c \\\n1\n\end{bmatrix} &=\n\begin{bmatrix}\nr_{11} & r_{12} & r_{13} & t_x \\\nr_{21} & r_{22} & r_{23} & t_y \\\nr_{31} & r_{32} & r_{33} & t_z \\\n0 & 0 & 0 & 1\n\end{bmatrix}\n\begin{bmatrix}\nX_{w} \\\nY_{w} \\\nZ_{w} \\\n1\n\end{bmatrix}\n\end{align*}\n\f]\n\n@note\n-   An example of how to use solvePnP for planar augmented reality can be found at\nopencv_source_code/samples/python/plane_ar.py\n-   If you are using Python:\n- Numpy array slices won't work as input because solvePnP requires contiguous\narrays (enforced by the assertion using cv::Mat::checkVector() around line 55 of\nmodules/calib3d/src/solvepnp.cpp version 2.4.9)\n- The P3P algorithm requires image points to be in an array of shape (N,1,2) due\nto its calling of cv::undistortPoints (around line 75 of modules/calib3d/src/solvepnp.cpp version 2.4.9)\nwhich requires 2-channel information.\n- Thus, given some data D = np.array(...) where D.shape = (N,M), in order to use a subset of\nit as, e.g., imagePoints, one must effectively copy it into a new array: imagePoints =\nnp.ascontiguousarray(D[:,:2]).reshape((N,1,2))\n-   The methods **SOLVEPNP_DLS** and **SOLVEPNP_UPNP** cannot be used as the current implementations are\nunstable and sometimes give completely wrong results. If you pass one of these two\nflags, **SOLVEPNP_EPNP** method will be used instead.\n-   The minimum number of points is 4 in the general case. In the case of **SOLVEPNP_P3P** and **SOLVEPNP_AP3P**\nmethods, it is required to use exactly 4 points (the first 3 points are used to estimate all the solutions\nof the P3P problem, the last one is used to retain the best solution that minimizes the reprojection error).\n-   With **SOLVEPNP_ITERATIVE** method and `useExtrinsicGuess=true`, the minimum number of points is 3 (3 points\nare sufficient to compute a pose but there are up to 4 solutions). The initial solution should be close to the\nglobal solution to converge.'''

    def get_attributes(self):
        return [Input('objectPoints', 'objectPoints'),
                Input('imagePoints', 'imagePoints'),
                Input('cameraMatrix', 'cameraMatrix'),
                Input('distCoeffs', 'distCoeffs')], \
               [Output('rvec', 'rvec'),
                Output('tvec', 'tvec')], \
               [ComboboxParameter('flags', [('SOLVEPNP_ITERATIVE',0),('SOLVEPNP_P3P',2),('SOLVEPNP_AP3P',5),('SOLVEPNP_EPNP',1),('SOLVEPNP_DLS',3),('SOLVEPNP_UPNP',4)])]

    def process_inputs(self, inputs, outputs, parameters):
        objectPoints = inputs['objectPoints'].value
        imagePoints = inputs['imagePoints'].value
        cameraMatrix = inputs['cameraMatrix'].value
        distCoeffs = inputs['distCoeffs'].value
        flags = parameters['flags']
        retval, rvec, tvec = cv2.solvePnP(objectPoints=objectPoints, imagePoints=imagePoints, cameraMatrix=cameraMatrix, distCoeffs=distCoeffs, flags=flags)
        outputs['rvec'] = Data(rvec)
        outputs['tvec'] = Data(tvec)

### solvePoly ###

class OpenCVAuto2_SolvePoly(NormalElement):
    name = 'SolvePoly'
    comment = '''solvePoly(coeffs[, roots[, maxIters]]) -> retval, roots\n@brief Finds the real or complex roots of a polynomial equation.\n\nThe function cv::solvePoly finds real and complex roots of a polynomial equation:\n\f[\texttt{coeffs} [n] x^{n} +  \texttt{coeffs} [n-1] x^{n-1} + ... +  \texttt{coeffs} [1] x +  \texttt{coeffs} [0] = 0\f]\n@param coeffs array of polynomial coefficients.\n@param roots output (complex) array of roots.\n@param maxIters maximum number of iterations the algorithm does.'''

    def get_attributes(self):
        return [Input('coeffs', 'coeffs')], \
               [Output('roots', 'roots')], \
               [IntParameter('maxIters', 'maxIters')]

    def process_inputs(self, inputs, outputs, parameters):
        coeffs = inputs['coeffs'].value
        maxIters = parameters['maxIters']
        retval, roots = cv2.solvePoly(coeffs=coeffs, maxIters=maxIters)
        outputs['roots'] = Data(roots)

### spatialGradient ###

class OpenCVAuto2_SpatialGradient(NormalElement):
    name = 'SpatialGradient'
    comment = '''spatialGradient(src[, dx[, dy[, ksize[, borderType]]]]) -> dx, dy\n@brief Calculates the first order image derivative in both x and y using a Sobel operator\n\nEquivalent to calling:\n\n@code\nSobel( src, dx, CV_16SC1, 1, 0, 3 );\nSobel( src, dy, CV_16SC1, 0, 1, 3 );\n@endcode\n\n@param src input image.\n@param dx output image with first-order derivative in x.\n@param dy output image with first-order derivative in y.\n@param ksize size of Sobel kernel. It must be 3.\n@param borderType pixel extrapolation method, see #BorderTypes\n\n@sa Sobel'''

    def get_attributes(self):
        return [Input('src', 'src')], \
               [Output('dx', 'dx'),
                Output('dy', 'dy')], \
               [SizeParameter('ksize', 'ksize'),
                ComboboxParameter('borderType', [('BORDER_CONSTANT',0),('BORDER_DEFAULT',4),('BORDER_ISOLATED',16),('BORDER_REFLECT',2),('BORDER_REFLECT101',4),('BORDER_REFLECT_101',4),('BORDER_REPLICATE',1),('BORDER_TRANSPARENT',5),('BORDER_WRAP',3)])]

    def process_inputs(self, inputs, outputs, parameters):
        src = inputs['src'].value
        ksize = parameters['ksize']
        borderType = parameters['borderType']
        dx, dy = cv2.spatialGradient(src=src, ksize=ksize, borderType=borderType)
        outputs['dx'] = Data(dx)
        outputs['dy'] = Data(dy)

### split ###

class OpenCVAuto2_Split(NormalElement):
    name = 'Split'
    comment = '''split(m[, mv]) -> mv\n@overload\n@param m input multi-channel array.\n@param mv output vector of arrays; the arrays themselves are reallocated, if needed.'''

    def get_attributes(self):
        return [Input('m', 'm')], \
               [Output('mv', 'mv')], \
               []

    def process_inputs(self, inputs, outputs, parameters):
        m = inputs['m'].value
        mv = cv2.split(m=m)
        outputs['mv'] = Data(mv)

### sqrBoxFilter ###

class OpenCVAuto2_SqrBoxFilter(NormalElement):
    name = 'SqrBoxFilter'
    comment = '''sqrBoxFilter(_src, ddepth, ksize[, _dst[, anchor[, normalize[, borderType]]]]) -> _dst\n@brief Calculates the normalized sum of squares of the pixel values overlapping the filter.\n\nFor every pixel \f$ (x, y) \f$ in the source image, the function calculates the sum of squares of those neighboring\npixel values which overlap the filter placed over the pixel \f$ (x, y) \f$.\n\nThe unnormalized square box filter can be useful in computing local image statistics such as the the local\nvariance and standard deviation around the neighborhood of a pixel.\n\n@param _src input image\n@param _dst output image of the same size and type as _src\n@param ddepth the output image depth (-1 to use src.depth())\n@param ksize kernel size\n@param anchor kernel anchor point. The default value of Point(-1, -1) denotes that the anchor is at the kernel\ncenter.\n@param normalize flag, specifying whether the kernel is to be normalized by it's area or not.\n@param borderType border mode used to extrapolate pixels outside of the image, see #BorderTypes\n@sa boxFilter'''

    def get_attributes(self):
        return [Input('_src', '_src')], \
               [Output('_dst', '_dst')], \
               [ComboboxParameter('ddepth', [('NONE',0),('CV_8U',0),('CV_8UC1',0),('CV_8UC2',8),('CV_8UC3',16),('CV_8UC4',24),('CV_8S',1),('CV_8SC1',1),('CV_8SC2',9),('CV_8SC3',17),('CV_8SC4',25),('CV_16U',2),('CV_16UC1',2),('CV_16UC2',10),('CV_16UC3',18),('CV_16UC4',26),('CV_16S',3),('CV_16SC1',3),('CV_16SC2',11),('CV_16SC3',19),('CV_16SC4',27),('CV_32S',4),('CV_32SC1',4),('CV_32SC2',12),('CV_32SC3',20),('CV_32SC4',28),('CV_32F',5),('CV_32FC1',5),('CV_32FC2',13),('CV_32FC3',21),('CV_32FC4',29),('CV_64F',6),('CV_64FC1',6),('CV_64FC2',14),('CV_64FC3',22),('CV_64FC4',30)]),
                SizeParameter('ksize', 'ksize'),
                PointParameter('anchor', 'anchor'),
                ComboboxParameter('borderType', [('BORDER_CONSTANT',0),('BORDER_DEFAULT',4),('BORDER_ISOLATED',16),('BORDER_REFLECT',2),('BORDER_REFLECT101',4),('BORDER_REFLECT_101',4),('BORDER_REPLICATE',1),('BORDER_TRANSPARENT',5),('BORDER_WRAP',3)])]

    def process_inputs(self, inputs, outputs, parameters):
        _src = inputs['_src'].value
        ddepth = parameters['ddepth']
        ksize = parameters['ksize']
        anchor = parameters['anchor']
        borderType = parameters['borderType']
        _dst = cv2.sqrBoxFilter(_src=_src, ddepth=ddepth, ksize=ksize, anchor=anchor, borderType=borderType)
        outputs['_dst'] = Data(_dst)

### sqrt ###

class OpenCVAuto2_Sqrt(NormalElement):
    name = 'Sqrt'
    comment = '''sqrt(src[, dst]) -> dst\n@brief Calculates a square root of array elements.\n\nThe function cv::sqrt calculates a square root of each input array element.\nIn case of multi-channel arrays, each channel is processed\nindependently. The accuracy is approximately the same as of the built-in\nstd::sqrt .\n@param src input floating-point array.\n@param dst output array of the same size and type as src.'''

    def get_attributes(self):
        return [Input('src', 'src')], \
               [Output('dst', 'dst')], \
               []

    def process_inputs(self, inputs, outputs, parameters):
        src = inputs['src'].value
        dst = cv2.sqrt(src=src)
        outputs['dst'] = Data(dst)

### stylization ###

class OpenCVAuto2_Stylization(NormalElement):
    name = 'Stylization'
    comment = '''stylization(src[, dst[, sigma_s[, sigma_r]]]) -> dst\n@brief Stylization aims to produce digital imagery with a wide variety of effects not focused on\nphotorealism. Edge-aware filters are ideal for stylization, as they can abstract regions of low\ncontrast while preserving, or enhancing, high-contrast features.\n\n@param src Input 8-bit 3-channel image.\n@param dst Output image with the same size and type as src.\n@param sigma_s Range between 0 to 200.\n@param sigma_r Range between 0 to 1.'''

    def get_attributes(self):
        return [Input('src', 'src')], \
               [Output('dst', 'dst')], \
               [FloatParameter('sigma_s', 'sigma_s'),
                FloatParameter('sigma_r', 'sigma_r')]

    def process_inputs(self, inputs, outputs, parameters):
        src = inputs['src'].value
        sigma_s = parameters['sigma_s']
        sigma_r = parameters['sigma_r']
        dst = cv2.stylization(src=src, sigma_s=sigma_s, sigma_r=sigma_r)
        outputs['dst'] = Data(dst)

### subtract ###

class OpenCVAuto2_Subtract(NormalElement):
    name = 'Subtract'
    comment = '''subtract(src1, src2[, dst[, mask[, dtype]]]) -> dst\n@brief Calculates the per-element difference between two arrays or array and a scalar.\n\nThe function subtract calculates:\n- Difference between two arrays, when both input arrays have the same size and the same number of\nchannels:\n\f[\texttt{dst}(I) =  \texttt{saturate} ( \texttt{src1}(I) -  \texttt{src2}(I)) \quad \texttt{if mask}(I) \ne0\f]\n- Difference between an array and a scalar, when src2 is constructed from Scalar or has the same\nnumber of elements as `src1.channels()`:\n\f[\texttt{dst}(I) =  \texttt{saturate} ( \texttt{src1}(I) -  \texttt{src2} ) \quad \texttt{if mask}(I) \ne0\f]\n- Difference between a scalar and an array, when src1 is constructed from Scalar or has the same\nnumber of elements as `src2.channels()`:\n\f[\texttt{dst}(I) =  \texttt{saturate} ( \texttt{src1} -  \texttt{src2}(I) ) \quad \texttt{if mask}(I) \ne0\f]\n- The reverse difference between a scalar and an array in the case of `SubRS`:\n\f[\texttt{dst}(I) =  \texttt{saturate} ( \texttt{src2} -  \texttt{src1}(I) ) \quad \texttt{if mask}(I) \ne0\f]\nwhere I is a multi-dimensional index of array elements. In case of multi-channel arrays, each\nchannel is processed independently.\n\nThe first function in the list above can be replaced with matrix expressions:\n@code{.cpp}\ndst = src1 - src2;\ndst -= src1; // equivalent to subtract(dst, src1, dst);\n@endcode\nThe input arrays and the output array can all have the same or different depths. For example, you\ncan subtract to 8-bit unsigned arrays and store the difference in a 16-bit signed array. Depth of\nthe output array is determined by dtype parameter. In the second and third cases above, as well as\nin the first case, when src1.depth() == src2.depth(), dtype can be set to the default -1. In this\ncase the output array will have the same depth as the input array, be it src1, src2 or both.\n@note Saturation is not applied when the output array has the depth CV_32S. You may even get\nresult of an incorrect sign in the case of overflow.\n@param src1 first input array or a scalar.\n@param src2 second input array or a scalar.\n@param dst output array of the same size and the same number of channels as the input array.\n@param mask optional operation mask; this is an 8-bit single channel array that specifies elements\nof the output array to be changed.\n@param dtype optional depth of the output array\n@sa  add, addWeighted, scaleAdd, Mat::convertTo'''

    def get_attributes(self):
        return [Input('src1', 'src1'),
                Input('src2', 'src2'),
                Input('mask', 'mask', optional=True)], \
               [Output('dst', 'dst')], \
               [ComboboxParameter('dtype', [('NONE',0),('CV_8U',0),('CV_8UC1',0),('CV_8UC2',8),('CV_8UC3',16),('CV_8UC4',24),('CV_8S',1),('CV_8SC1',1),('CV_8SC2',9),('CV_8SC3',17),('CV_8SC4',25),('CV_16U',2),('CV_16UC1',2),('CV_16UC2',10),('CV_16UC3',18),('CV_16UC4',26),('CV_16S',3),('CV_16SC1',3),('CV_16SC2',11),('CV_16SC3',19),('CV_16SC4',27),('CV_32S',4),('CV_32SC1',4),('CV_32SC2',12),('CV_32SC3',20),('CV_32SC4',28),('CV_32F',5),('CV_32FC1',5),('CV_32FC2',13),('CV_32FC3',21),('CV_32FC4',29),('CV_64F',6),('CV_64FC1',6),('CV_64FC2',14),('CV_64FC3',22),('CV_64FC4',30)])]

    def process_inputs(self, inputs, outputs, parameters):
        src1 = inputs['src1'].value
        src2 = inputs['src2'].value
        mask = inputs['mask'].value
        dtype = parameters['dtype']
        dst = cv2.subtract(src1=src1, src2=src2, mask=mask, dtype=dtype)
        outputs['dst'] = Data(dst)

### textureFlattening ###

class OpenCVAuto2_TextureFlattening(NormalElement):
    name = 'TextureFlattening'
    comment = '''textureFlattening(src, mask[, dst[, low_threshold[, high_threshold[, kernel_size]]]]) -> dst\n@brief By retaining only the gradients at edge locations, before integrating with the Poisson solver, one\nwashes out the texture of the selected region, giving its contents a flat aspect. Here Canny Edge\nDetector is used.\n\n@param src Input 8-bit 3-channel image.\n@param mask Input 8-bit 1 or 3-channel image.\n@param dst Output image with the same size and type as src.\n@param low_threshold Range from 0 to 100.\n@param high_threshold Value \> 100.\n@param kernel_size The size of the Sobel kernel to be used.\n\n**NOTE:**\n\nThe algorithm assumes that the color of the source image is close to that of the destination. This\nassumption means that when the colors don't match, the source image color gets tinted toward the\ncolor of the destination image.'''

    def get_attributes(self):
        return [Input('src', 'src'),
                Input('mask', 'mask')], \
               [Output('dst', 'dst')], \
               [FloatParameter('low_threshold', 'low_threshold'),
                FloatParameter('high_threshold', 'high_threshold'),
                SizeParameter('kernel_size', 'kernel_size')]

    def process_inputs(self, inputs, outputs, parameters):
        src = inputs['src'].value
        mask = inputs['mask'].value
        low_threshold = parameters['low_threshold']
        high_threshold = parameters['high_threshold']
        kernel_size = parameters['kernel_size']
        dst = cv2.textureFlattening(src=src, mask=mask, low_threshold=low_threshold, high_threshold=high_threshold, kernel_size=kernel_size)
        outputs['dst'] = Data(dst)

### threshold ###

class OpenCVAuto2_Threshold(NormalElement):
    name = 'Threshold'
    comment = '''threshold(src, thresh, maxval, type[, dst]) -> retval, dst\n@brief Applies a fixed-level threshold to each array element.\n\nThe function applies fixed-level thresholding to a multiple-channel array. The function is typically\nused to get a bi-level (binary) image out of a grayscale image ( #compare could be also used for\nthis purpose) or for removing a noise, that is, filtering out pixels with too small or too large\nvalues. There are several types of thresholding supported by the function. They are determined by\ntype parameter.\n\nAlso, the special values #THRESH_OTSU or #THRESH_TRIANGLE may be combined with one of the\nabove values. In these cases, the function determines the optimal threshold value using the Otsu's\nor Triangle algorithm and uses it instead of the specified thresh.\n\n@note Currently, the Otsu's and Triangle methods are implemented only for 8-bit single-channel images.\n\n@param src input array (multiple-channel, 8-bit or 32-bit floating point).\n@param dst output array of the same size  and type and the same number of channels as src.\n@param thresh threshold value.\n@param maxval maximum value to use with the #THRESH_BINARY and #THRESH_BINARY_INV thresholding\ntypes.\n@param type thresholding type (see #ThresholdTypes).\n@return the computed threshold value if Otsu's or Triangle methods used.\n\n@sa  adaptiveThreshold, findContours, compare, min, max'''

    def get_attributes(self):
        return [Input('src', 'src')], \
               [Output('dst', 'dst')], \
               [FloatParameter('thresh', 'thresh'),
                FloatParameter('maxval', 'maxval'),
                ComboboxParameter('type', [('THRESH_BINARY',0),('THRESH_BINARY_INV',1),('THRESH_MASK',7),('THRESH_OTSU',8),('THRESH_TOZERO',3),('THRESH_TOZERO_INV',4),('THRESH_TRIANGLE',16),('THRESH_TRUNC',2)])]

    def process_inputs(self, inputs, outputs, parameters):
        src = inputs['src'].value
        thresh = parameters['thresh']
        maxval = parameters['maxval']
        type = parameters['type']
        retval, dst = cv2.threshold(src=src, thresh=thresh, maxval=maxval, type=type)
        outputs['dst'] = Data(dst)

### transform ###

class OpenCVAuto2_Transform(NormalElement):
    name = 'Transform'
    comment = '''transform(src, m[, dst]) -> dst\n@brief Performs the matrix transformation of every array element.\n\nThe function cv::transform performs the matrix transformation of every\nelement of the array src and stores the results in dst :\n\f[\texttt{dst} (I) =  \texttt{m} \cdot \texttt{src} (I)\f]\n(when m.cols=src.channels() ), or\n\f[\texttt{dst} (I) =  \texttt{m} \cdot [ \texttt{src} (I); 1]\f]\n(when m.cols=src.channels()+1 )\n\nEvery element of the N -channel array src is interpreted as N -element\nvector that is transformed using the M x N or M x (N+1) matrix m to\nM-element vector - the corresponding element of the output array dst .\n\nThe function may be used for geometrical transformation of\nN -dimensional points, arbitrary linear color space transformation (such\nas various kinds of RGB to YUV transforms), shuffling the image\nchannels, and so forth.\n@param src input array that must have as many channels (1 to 4) as\nm.cols or m.cols-1.\n@param dst output array of the same size and depth as src; it has as\nmany channels as m.rows.\n@param m transformation 2x2 or 2x3 floating-point matrix.\n@sa perspectiveTransform, getAffineTransform, estimateAffine2D, warpAffine, warpPerspective'''

    def get_attributes(self):
        return [Input('src', 'src'),
                Input('m', 'm')], \
               [Output('dst', 'dst')], \
               []

    def process_inputs(self, inputs, outputs, parameters):
        src = inputs['src'].value
        m = inputs['m'].value
        dst = cv2.transform(src=src, m=m)
        outputs['dst'] = Data(dst)

### transpose ###

class OpenCVAuto2_Transpose(NormalElement):
    name = 'Transpose'
    comment = '''transpose(src[, dst]) -> dst\n@brief Transposes a matrix.\n\nThe function cv::transpose transposes the matrix src :\n\f[\texttt{dst} (i,j) =  \texttt{src} (j,i)\f]\n@note No complex conjugation is done in case of a complex matrix. It\nshould be done separately if needed.\n@param src input array.\n@param dst output array of the same type as src.'''

    def get_attributes(self):
        return [Input('src', 'src')], \
               [Output('dst', 'dst')], \
               []

    def process_inputs(self, inputs, outputs, parameters):
        src = inputs['src'].value
        dst = cv2.transpose(src=src)
        outputs['dst'] = Data(dst)

### triangulatePoints ###

class OpenCVAuto2_TriangulatePoints(NormalElement):
    name = 'TriangulatePoints'
    comment = '''triangulatePoints(projMatr1, projMatr2, projPoints1, projPoints2[, points4D]) -> points4D\n@brief Reconstructs points by triangulation.\n\n@param projMatr1 3x4 projection matrix of the first camera.\n@param projMatr2 3x4 projection matrix of the second camera.\n@param projPoints1 2xN array of feature points in the first image. In case of c++ version it can\nbe also a vector of feature points or two-channel matrix of size 1xN or Nx1.\n@param projPoints2 2xN array of corresponding points in the second image. In case of c++ version\nit can be also a vector of feature points or two-channel matrix of size 1xN or Nx1.\n@param points4D 4xN array of reconstructed points in homogeneous coordinates.\n\nThe function reconstructs 3-dimensional points (in homogeneous coordinates) by using their\nobservations with a stereo camera. Projections matrices can be obtained from stereoRectify.\n\n@note\nKeep in mind that all input data should be of float type in order for this function to work.\n\n@sa\nreprojectImageTo3D'''

    def get_attributes(self):
        return [Input('projMatr1', 'projMatr1'),
                Input('projMatr2', 'projMatr2'),
                Input('projPoints1', 'projPoints1'),
                Input('projPoints2', 'projPoints2')], \
               [Output('points4D', 'points4D')], \
               []

    def process_inputs(self, inputs, outputs, parameters):
        projMatr1 = inputs['projMatr1'].value
        projMatr2 = inputs['projMatr2'].value
        projPoints1 = inputs['projPoints1'].value
        projPoints2 = inputs['projPoints2'].value
        points4D = cv2.triangulatePoints(projMatr1=projMatr1, projMatr2=projMatr2, projPoints1=projPoints1, projPoints2=projPoints2)
        outputs['points4D'] = Data(points4D)

### undistort ###

class OpenCVAuto2_Undistort(NormalElement):
    name = 'Undistort'
    comment = '''undistort(src, cameraMatrix, distCoeffs[, dst[, newCameraMatrix]]) -> dst\n@brief Transforms an image to compensate for lens distortion.\n\nThe function transforms an image to compensate radial and tangential lens distortion.\n\nThe function is simply a combination of #initUndistortRectifyMap (with unity R ) and #remap\n(with bilinear interpolation). See the former function for details of the transformation being\nperformed.\n\nThose pixels in the destination image, for which there is no correspondent pixels in the source\nimage, are filled with zeros (black color).\n\nA particular subset of the source image that will be visible in the corrected image can be regulated\nby newCameraMatrix. You can use #getOptimalNewCameraMatrix to compute the appropriate\nnewCameraMatrix depending on your requirements.\n\nThe camera matrix and the distortion parameters can be determined using #calibrateCamera. If\nthe resolution of images is different from the resolution used at the calibration stage, \f$f_x,\nf_y, c_x\f$ and \f$c_y\f$ need to be scaled accordingly, while the distortion coefficients remain\nthe same.\n\n@param src Input (distorted) image.\n@param dst Output (corrected) image that has the same size and type as src .\n@param cameraMatrix Input camera matrix \f$A = \vecthreethree{f_x}{0}{c_x}{0}{f_y}{c_y}{0}{0}{1}\f$ .\n@param distCoeffs Input vector of distortion coefficients\n\f$(k_1, k_2, p_1, p_2[, k_3[, k_4, k_5, k_6[, s_1, s_2, s_3, s_4[, \tau_x, \tau_y]]]])\f$\nof 4, 5, 8, 12 or 14 elements. If the vector is NULL/empty, the zero distortion coefficients are assumed.\n@param newCameraMatrix Camera matrix of the distorted image. By default, it is the same as\ncameraMatrix but you may additionally scale and shift the result by using a different matrix.'''

    def get_attributes(self):
        return [Input('src', 'src'),
                Input('cameraMatrix', 'cameraMatrix'),
                Input('distCoeffs', 'distCoeffs'),
                Input('newCameraMatrix', 'newCameraMatrix', optional=True)], \
               [Output('dst', 'dst')], \
               []

    def process_inputs(self, inputs, outputs, parameters):
        src = inputs['src'].value
        cameraMatrix = inputs['cameraMatrix'].value
        distCoeffs = inputs['distCoeffs'].value
        newCameraMatrix = inputs['newCameraMatrix'].value
        dst = cv2.undistort(src=src, cameraMatrix=cameraMatrix, distCoeffs=distCoeffs, newCameraMatrix=newCameraMatrix)
        outputs['dst'] = Data(dst)

### undistortPoints ###

class OpenCVAuto2_UndistortPoints(NormalElement):
    name = 'UndistortPoints'
    comment = '''undistortPoints(src, cameraMatrix, distCoeffs[, dst[, R[, P]]]) -> dst\n@brief Computes the ideal point coordinates from the observed point coordinates.\n\nThe function is similar to #undistort and #initUndistortRectifyMap but it operates on a\nsparse set of points instead of a raster image. Also the function performs a reverse transformation\nto projectPoints. In case of a 3D object, it does not reconstruct its 3D coordinates, but for a\nplanar object, it does, up to a translation vector, if the proper R is specified.\n\nFor each observed point coordinate \f$(u, v)\f$ the function computes:\n\f[\n\begin{array}{l}\nx^{"}  \leftarrow (u - c_x)/f_x  \\\ny^{"}  \leftarrow (v - c_y)/f_y  \\\n(x',y') = undistort(x^{"},y^{"}, \texttt{distCoeffs}) \\\n{[X\,Y\,W]} ^T  \leftarrow R*[x' \, y' \, 1]^T  \\\nx  \leftarrow X/W  \\\ny  \leftarrow Y/W  \\\n\text{only performed if P is specified:} \\\nu'  \leftarrow x {f'}_x + {c'}_x  \\\nv'  \leftarrow y {f'}_y + {c'}_y\n\end{array}\n\f]\n\nwhere *undistort* is an approximate iterative algorithm that estimates the normalized original\npoint coordinates out of the normalized distorted point coordinates ("normalized" means that the\ncoordinates do not depend on the camera matrix).\n\nThe function can be used for both a stereo camera head or a monocular camera (when R is empty).\n\n@param src Observed point coordinates, 1xN or Nx1 2-channel (CV_32FC2 or CV_64FC2).\n@param dst Output ideal point coordinates after undistortion and reverse perspective\ntransformation. If matrix P is identity or omitted, dst will contain normalized point coordinates.\n@param cameraMatrix Camera matrix \f$\vecthreethree{f_x}{0}{c_x}{0}{f_y}{c_y}{0}{0}{1}\f$ .\n@param distCoeffs Input vector of distortion coefficients\n\f$(k_1, k_2, p_1, p_2[, k_3[, k_4, k_5, k_6[, s_1, s_2, s_3, s_4[, \tau_x, \tau_y]]]])\f$\nof 4, 5, 8, 12 or 14 elements. If the vector is NULL/empty, the zero distortion coefficients are assumed.\n@param R Rectification transformation in the object space (3x3 matrix). R1 or R2 computed by\n#stereoRectify can be passed here. If the matrix is empty, the identity transformation is used.\n@param P New camera matrix (3x3) or new projection matrix (3x4) \f$\begin{bmatrix} {f'}_x & 0 & {c'}_x & t_x \\ 0 & {f'}_y & {c'}_y & t_y \\ 0 & 0 & 1 & t_z \end{bmatrix}\f$. P1 or P2 computed by\n#stereoRectify can be passed here. If the matrix is empty, the identity new camera matrix is used.'''

    def get_attributes(self):
        return [Input('src', 'src'),
                Input('cameraMatrix', 'cameraMatrix'),
                Input('distCoeffs', 'distCoeffs')], \
               [Output('dst', 'dst')], \
               []

    def process_inputs(self, inputs, outputs, parameters):
        src = inputs['src'].value
        cameraMatrix = inputs['cameraMatrix'].value
        distCoeffs = inputs['distCoeffs'].value
        dst = cv2.undistortPoints(src=src, cameraMatrix=cameraMatrix, distCoeffs=distCoeffs)
        outputs['dst'] = Data(dst)

### vconcat ###

class OpenCVAuto2_Vconcat(NormalElement):
    name = 'Vconcat'
    comment = '''vconcat(src[, dst]) -> dst\n@overload\n@code{.cpp}\nstd::vector<cv::Mat> matrices = { cv::Mat(1, 4, CV_8UC1, cv::Scalar(1)),\ncv::Mat(1, 4, CV_8UC1, cv::Scalar(2)),\ncv::Mat(1, 4, CV_8UC1, cv::Scalar(3)),};\n\ncv::Mat out;\ncv::vconcat( matrices, out );\n//out:\n//[1,   1,   1,   1;\n// 2,   2,   2,   2;\n// 3,   3,   3,   3]\n@endcode\n@param src input array or vector of matrices. all of the matrices must have the same number of cols and the same depth\n@param dst output array. It has the same number of cols and depth as the src, and the sum of rows of the src.\nsame depth.'''

    def get_attributes(self):
        return [Input('src', 'src')], \
               [Output('dst', 'dst')], \
               []

    def process_inputs(self, inputs, outputs, parameters):
        src = inputs['src'].value
        dst = cv2.vconcat(src=src)
        outputs['dst'] = Data(dst)

### warpAffine ###

class OpenCVAuto2_WarpAffine(NormalElement):
    name = 'WarpAffine'
    comment = '''warpAffine(src, M, dsize[, dst[, flags[, borderMode[, borderValue]]]]) -> dst\n@brief Applies an affine transformation to an image.\n\nThe function warpAffine transforms the source image using the specified matrix:\n\n\f[\texttt{dst} (x,y) =  \texttt{src} ( \texttt{M} _{11} x +  \texttt{M} _{12} y +  \texttt{M} _{13}, \texttt{M} _{21} x +  \texttt{M} _{22} y +  \texttt{M} _{23})\f]\n\nwhen the flag #WARP_INVERSE_MAP is set. Otherwise, the transformation is first inverted\nwith #invertAffineTransform and then put in the formula above instead of M. The function cannot\noperate in-place.\n\n@param src input image.\n@param dst output image that has the size dsize and the same type as src .\n@param M \f$2\times 3\f$ transformation matrix.\n@param dsize size of the output image.\n@param flags combination of interpolation methods (see #InterpolationFlags) and the optional\nflag #WARP_INVERSE_MAP that means that M is the inverse transformation (\n\f$\texttt{dst}\rightarrow\texttt{src}\f$ ).\n@param borderMode pixel extrapolation method (see #BorderTypes); when\nborderMode=#BORDER_TRANSPARENT, it means that the pixels in the destination image corresponding to\nthe "outliers" in the source image are not modified by the function.\n@param borderValue value used in case of a constant border; by default, it is 0.\n\n@sa  warpPerspective, resize, remap, getRectSubPix, transform'''

    def get_attributes(self):
        return [Input('src', 'src'),
                Input('M', 'M')], \
               [Output('dst', 'dst')], \
               [SizeParameter('dsize', 'dsize'),
                ComboboxParameter('flags', [('WARP_INVERSE_MAP',16),('INTER_AREA',3),('INTER_BITS',5),('INTER_BITS2',10),('INTER_CUBIC',2),('INTER_LANCZOS4',4),('INTER_LINEAR',1),('INTER_LINEAR_EXACT',5),('INTER_MAX',7),('INTER_NEAREST',0),('INTER_TAB_SIZE',32),('INTER_TAB_SIZE2',1024)]),
                ComboboxParameter('borderMode', [('BORDER_TRANSPARENT',5),('BORDER_CONSTANT',0),('BORDER_DEFAULT',4),('BORDER_ISOLATED',16),('BORDER_REFLECT',2),('BORDER_REFLECT101',4),('BORDER_REFLECT_101',4),('BORDER_REPLICATE',1),('BORDER_WRAP',3)]),
                ScalarParameter('borderValue', 'borderValue')]

    def process_inputs(self, inputs, outputs, parameters):
        src = inputs['src'].value
        M = inputs['M'].value
        dsize = parameters['dsize']
        flags = parameters['flags']
        borderMode = parameters['borderMode']
        borderValue = parameters['borderValue']
        dst = cv2.warpAffine(src=src, M=M, dsize=dsize, flags=flags, borderMode=borderMode, borderValue=borderValue)
        outputs['dst'] = Data(dst)

### warpPerspective ###

class OpenCVAuto2_WarpPerspective(NormalElement):
    name = 'WarpPerspective'
    comment = '''warpPerspective(src, M, dsize[, dst[, flags[, borderMode[, borderValue]]]]) -> dst\n@brief Applies a perspective transformation to an image.\n\nThe function warpPerspective transforms the source image using the specified matrix:\n\n\f[\texttt{dst} (x,y) =  \texttt{src} \left ( \frac{M_{11} x + M_{12} y + M_{13}}{M_{31} x + M_{32} y + M_{33}} ,\n\frac{M_{21} x + M_{22} y + M_{23}}{M_{31} x + M_{32} y + M_{33}} \right )\f]\n\nwhen the flag #WARP_INVERSE_MAP is set. Otherwise, the transformation is first inverted with invert\nand then put in the formula above instead of M. The function cannot operate in-place.\n\n@param src input image.\n@param dst output image that has the size dsize and the same type as src .\n@param M \f$3\times 3\f$ transformation matrix.\n@param dsize size of the output image.\n@param flags combination of interpolation methods (#INTER_LINEAR or #INTER_NEAREST) and the\noptional flag #WARP_INVERSE_MAP, that sets M as the inverse transformation (\n\f$\texttt{dst}\rightarrow\texttt{src}\f$ ).\n@param borderMode pixel extrapolation method (#BORDER_CONSTANT or #BORDER_REPLICATE).\n@param borderValue value used in case of a constant border; by default, it equals 0.\n\n@sa  warpAffine, resize, remap, getRectSubPix, perspectiveTransform'''

    def get_attributes(self):
        return [Input('src', 'src'),
                Input('M', 'M')], \
               [Output('dst', 'dst')], \
               [SizeParameter('dsize', 'dsize'),
                ComboboxParameter('flags', [('INTER_LINEAR',1),('INTER_NEAREST',0),('WARP_INVERSE_MAP',16)]),
                ComboboxParameter('borderMode', [('BORDER_CONSTANT',0),('BORDER_REPLICATE',1)]),
                ScalarParameter('borderValue', 'borderValue')]

    def process_inputs(self, inputs, outputs, parameters):
        src = inputs['src'].value
        M = inputs['M'].value
        dsize = parameters['dsize']
        flags = parameters['flags']
        borderMode = parameters['borderMode']
        borderValue = parameters['borderValue']
        dst = cv2.warpPerspective(src=src, M=M, dsize=dsize, flags=flags, borderMode=borderMode, borderValue=borderValue)
        outputs['dst'] = Data(dst)

### warpPolar ###

class OpenCVAuto2_WarpPolar(NormalElement):
    name = 'WarpPolar'
    comment = '''warpPolar(src, dsize, center, maxRadius, flags[, dst]) -> dst\n\brief Remaps an image to polar or semilog-polar coordinates space\n\n@anchor polar_remaps_reference_image\n![Polar remaps reference](pics/polar_remap_doc.png)\n\nTransform the source image using the following transformation:\n\f[\ndst(\rho , \phi ) = src(x,y)\n\f]\n\nwhere\n\f[\n\begin{array}{l}\n\vec{I} = (x - center.x, \;y - center.y) \\\n\phi = Kangle \cdot \texttt{angle} (\vec{I}) \\\n\rho = \left\{\begin{matrix}\nKlin \cdot \texttt{magnitude} (\vec{I}) & default \\\nKlog \cdot log_e(\texttt{magnitude} (\vec{I})) & if \; semilog \\\n\end{matrix}\right.\n\end{array}\n\f]\n\nand\n\f[\n\begin{array}{l}\nKangle = dsize.height / 2\Pi \\\nKlin = dsize.width / maxRadius \\\nKlog = dsize.width / log_e(maxRadius) \\\n\end{array}\n\f]\n\n\n\par Linear vs semilog mapping\n\nPolar mapping can be linear or semi-log. Add one of #WarpPolarMode to `flags` to specify the polar mapping mode.\n\nLinear is the default mode.\n\nThe semilog mapping emulates the human "foveal" vision that permit very high acuity on the line of sight (central vision)\nin contrast to peripheral vision where acuity is minor.\n\n\par Option on `dsize`:\n\n- if both values in `dsize <=0 ` (default),\nthe destination image will have (almost) same area of source bounding circle:\n\f[\begin{array}{l}\ndsize.area  \leftarrow (maxRadius^2 \cdot \Pi) \\\ndsize.width = \texttt{cvRound}(maxRadius) \\\ndsize.height = \texttt{cvRound}(maxRadius \cdot \Pi) \\\n\end{array}\f]\n\n\n- if only `dsize.height <= 0`,\nthe destination image area will be proportional to the bounding circle area but scaled by `Kx * Kx`:\n\f[\begin{array}{l}\ndsize.height = \texttt{cvRound}(dsize.width \cdot \Pi) \\\n\end{array}\n\f]\n\n- if both values in `dsize > 0 `,\nthe destination image will have the given size therefore the area of the bounding circle will be scaled to `dsize`.\n\n\n\par Reverse mapping\n\nYou can get reverse mapping adding #WARP_INVERSE_MAP to `flags`\n\snippet polar_transforms.cpp InverseMap\n\nIn addiction, to calculate the original coordinate from a polar mapped coordinate \f$(rho, phi)->(x, y)\f$:\n\snippet polar_transforms.cpp InverseCoordinate\n\n@param src Source image.\n@param dst Destination image. It will have same type as src.\n@param dsize The destination image size (see description for valid options).\n@param center The transformation center.\n@param maxRadius The radius of the bounding circle to transform. It determines the inverse magnitude scale parameter too.\n@param flags A combination of interpolation methods, #InterpolationFlags + #WarpPolarMode.\n- Add #WARP_POLAR_LINEAR to select linear polar mapping (default)\n- Add #WARP_POLAR_LOG to select semilog polar mapping\n- Add #WARP_INVERSE_MAP for reverse mapping.\n@note\n-  The function can not operate in-place.\n-  To calculate magnitude and angle in degrees #cartToPolar is used internally thus angles are measured from 0 to 360 with accuracy about 0.3 degrees.\n-  This function uses #remap. Due to current implementation limitations the size of an input and output images should be less than 32767x32767.\n\n@sa cv::remap'''

    def get_attributes(self):
        return [Input('src', 'src')], \
               [Output('dst', 'dst')], \
               [SizeParameter('dsize', 'dsize'),
                PointParameter('center', 'center'),
                IntParameter('maxRadius', 'maxRadius'),
                ComboboxParameter('flags', [('WARP_POLAR_LINEAR',0),('WARP_POLAR_LOG',256),('WARP_INVERSE_MAP',16),('INTER_AREA',3),('INTER_BITS',5),('INTER_BITS2',10),('INTER_CUBIC',2),('INTER_LANCZOS4',4),('INTER_LINEAR',1),('INTER_LINEAR_EXACT',5),('INTER_MAX',7),('INTER_NEAREST',0),('INTER_TAB_SIZE',32),('INTER_TAB_SIZE2',1024)])]

    def process_inputs(self, inputs, outputs, parameters):
        src = inputs['src'].value
        dsize = parameters['dsize']
        center = parameters['center']
        maxRadius = parameters['maxRadius']
        flags = parameters['flags']
        dst = cv2.warpPolar(src=src, dsize=dsize, center=center, maxRadius=maxRadius, flags=flags)
        outputs['dst'] = Data(dst)

### watershed ###

class OpenCVAuto2_Watershed(NormalElement):
    name = 'Watershed'
    comment = '''watershed(image, markers) -> markers\n@brief Performs a marker-based image segmentation using the watershed algorithm.\n\nThe function implements one of the variants of watershed, non-parametric marker-based segmentation\nalgorithm, described in @cite Meyer92 .\n\nBefore passing the image to the function, you have to roughly outline the desired regions in the\nimage markers with positive (\>0) indices. So, every region is represented as one or more connected\ncomponents with the pixel values 1, 2, 3, and so on. Such markers can be retrieved from a binary\nmask using #findContours and #drawContours (see the watershed.cpp demo). The markers are "seeds" of\nthe future image regions. All the other pixels in markers , whose relation to the outlined regions\nis not known and should be defined by the algorithm, should be set to 0's. In the function output,\neach pixel in markers is set to a value of the "seed" components or to -1 at boundaries between the\nregions.\n\n@note Any two neighbor connected components are not necessarily separated by a watershed boundary\n(-1's pixels); for example, they can touch each other in the initial marker image passed to the\nfunction.\n\n@param image Input 8-bit 3-channel image.\n@param markers Input/output 32-bit single-channel image (map) of markers. It should have the same\nsize as image .\n\n@sa findContours\n\n@ingroup imgproc_misc'''

    def get_attributes(self):
        return [Input('image', 'image'),
                Input('markers', 'markers')], \
               [Output('markers', 'markers')], \
               []

    def process_inputs(self, inputs, outputs, parameters):
        image = inputs['image'].value
        markers = inputs['markers'].value
        markers = cv2.watershed(image=image, markers=markers)
        outputs['markers'] = Data(markers)



register_elements_auto(__name__, locals(), "OpenCV autogenerated 2", 15)

