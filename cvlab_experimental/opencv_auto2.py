# Source generated with cvlab/tools/generate_opencv.py
# See: https://github.com/cvlab-ai/cvlab
   
import cv2
from cvlab.diagram.elements.base import *


# cv2.GaussianBlur
class OpenCVAuto2_GaussianBlur(NormalElement):
    name = 'Gaussian Blur'
    comment = '''GaussianBlur(src, ksize, sigmaX[, dst[, sigmaY[, borderType]]]) -> dst\n@brief Blurs an image using a Gaussian filter.\n\nThe function convolves the source image with the specified Gaussian kernel. In-place filtering is\nsupported.\n\n@param src input image; the image can have any number of channels, which are processed\nindependently, but the depth should be CV_8U, CV_16U, CV_16S, CV_32F or CV_64F.\n@param dst output image of the same size and type as src.\n@param ksize Gaussian kernel size. ksize.width and ksize.height can differ but they both must be\npositive and odd. Or, they can be zero's and then they are computed from sigma.\n@param sigmaX Gaussian kernel standard deviation in X direction.\n@param sigmaY Gaussian kernel standard deviation in Y direction; if sigmaY is zero, it is set to be\nequal to sigmaX, if both sigmas are zeros, they are computed from ksize.width and ksize.height,\nrespectively (see #getGaussianKernel for details); to fully control the result regardless of\npossible future modifications of all this semantics, it is recommended to specify all of ksize,\nsigmaX, and sigmaY.\n@param borderType pixel extrapolation method, see #BorderTypes\n\n@sa  sepFilter2D, filter2D, blur, boxFilter, bilateralFilter, medianBlur'''
    package = "Filters"

    def get_attributes(self):
        return [Input('src', 'src')], \
               [Output('dst', 'dst')], \
               [SizeParameter('ksize', 'ksize'),
                FloatParameter('sigmaX', 'Sigma X'),
                FloatParameter('sigmaY', 'Sigma Y'),
                ComboboxParameter('borderType', name='Border Type', values=[('BORDER_CONSTANT',0),('BORDER_REPLICATE',1),('BORDER_REFLECT',2),('BORDER_WRAP',3),('BORDER_DEFAULT',4),('BORDER_REFLECT101',4),('BORDER_REFLECT_101',4),('BORDER_TRANSPARENT',5),('BORDER_ISOLATED',16)])]

    def process_inputs(self, inputs, outputs, parameters):
        src = inputs['src'].value
        ksize = parameters['ksize']
        sigmaX = parameters['sigmaX']
        sigmaY = parameters['sigmaY']
        borderType = parameters['borderType']
        dst = cv2.GaussianBlur(src=src, ksize=ksize, sigmaX=sigmaX, sigmaY=sigmaY, borderType=borderType)
        outputs['dst'] = Data(dst)

# cv2.HoughCircles
class OpenCVAuto2_HoughCircles(NormalElement):
    name = 'Hough Circles'
    comment = '''HoughCircles(image, method, dp, minDist[, circles[, param1[, param2[, minRadius[, maxRadius]]]]]) -> circles\n@brief Finds circles in a grayscale image using the Hough transform.\n\nThe function finds circles in a grayscale image using a modification of the Hough transform.\n\nExample: :\n@include snippets/imgproc_HoughLinesCircles.cpp\n\n@note Usually the function detects the centers of circles well. However, it may fail to find correct\nradii. You can assist to the function by specifying the radius range ( minRadius and maxRadius ) if\nyou know it. Or, you may set maxRadius to a negative number to return centers only without radius\nsearch, and find the correct radius using an additional procedure.\n\n@param image 8-bit, single-channel, grayscale input image.\n@param circles Output vector of found circles. Each vector is encoded as  3 or 4 element\nfloating-point vector \f$(x, y, radius)\f$ or \f$(x, y, radius, votes)\f$ .\n@param method Detection method, see #HoughModes. Currently, the only implemented method is #HOUGH_GRADIENT\n@param dp Inverse ratio of the accumulator resolution to the image resolution. For example, if\ndp=1 , the accumulator has the same resolution as the input image. If dp=2 , the accumulator has\nhalf as big width and height.\n@param minDist Minimum distance between the centers of the detected circles. If the parameter is\ntoo small, multiple neighbor circles may be falsely detected in addition to a true one. If it is\ntoo large, some circles may be missed.\n@param param1 First method-specific parameter. In case of #HOUGH_GRADIENT , it is the higher\nthreshold of the two passed to the Canny edge detector (the lower one is twice smaller).\n@param param2 Second method-specific parameter. In case of #HOUGH_GRADIENT , it is the\naccumulator threshold for the circle centers at the detection stage. The smaller it is, the more\nfalse circles may be detected. Circles, corresponding to the larger accumulator values, will be\nreturned first.\n@param minRadius Minimum circle radius.\n@param maxRadius Maximum circle radius. If <= 0, uses the maximum image dimension. If < 0, returns\ncenters without finding the radius.\n\n@sa fitEllipse, minEnclosingCircle'''
    package = "Contours"

    def get_attributes(self):
        return [Input('image', 'image')], \
               [Output('circles', 'circles')], \
               [IntParameter('method', 'method'),
                FloatParameter('dp', 'dp'),
                FloatParameter('minDist', 'Min Dist'),
                FloatParameter('param1', 'param 1'),
                FloatParameter('param2', 'param 2'),
                IntParameter('minRadius', 'Min Radius', min_=0),
                IntParameter('maxRadius', 'Max Radius', min_=0)]

    def process_inputs(self, inputs, outputs, parameters):
        image = inputs['image'].value
        method = parameters['method']
        dp = parameters['dp']
        minDist = parameters['minDist']
        param1 = parameters['param1']
        param2 = parameters['param2']
        minRadius = parameters['minRadius']
        maxRadius = parameters['maxRadius']
        circles = cv2.HoughCircles(image=image, method=method, dp=dp, minDist=minDist, param1=param1, param2=param2, minRadius=minRadius, maxRadius=maxRadius)
        outputs['circles'] = Data(circles)

# cv2.HoughLines
class OpenCVAuto2_HoughLines(NormalElement):
    name = 'Hough Lines'
    comment = '''HoughLines(image, rho, theta, threshold[, lines[, srn[, stn[, min_theta[, max_theta]]]]]) -> lines\n@brief Finds lines in a binary image using the standard Hough transform.\n\nThe function implements the standard or standard multi-scale Hough transform algorithm for line\ndetection. See <http://homepages.inf.ed.ac.uk/rbf/HIPR2/hough.htm> for a good explanation of Hough\ntransform.\n\n@param image 8-bit, single-channel binary source image. The image may be modified by the function.\n@param lines Output vector of lines. Each line is represented by a 2 or 3 element vector\n\f$(\rho, \theta)\f$ or \f$(\rho, \theta, \votes)\f$ . \f$\rho\f$ is the distance from the coordinate origin \f$(0,0)\f$ (top-left corner of\nthe image). \f$\theta\f$ is the line rotation angle in radians (\n\f$0 \sim \textrm{vertical line}, \pi/2 \sim \textrm{horizontal line}\f$ ).\n\f$\votes\f$ is the value of accumulator.\n@param rho Distance resolution of the accumulator in pixels.\n@param theta Angle resolution of the accumulator in radians.\n@param threshold Accumulator threshold parameter. Only those lines are returned that get enough\nvotes ( \f$>\texttt{threshold}\f$ ).\n@param srn For the multi-scale Hough transform, it is a divisor for the distance resolution rho .\nThe coarse accumulator distance resolution is rho and the accurate accumulator resolution is\nrho/srn . If both srn=0 and stn=0 , the classical Hough transform is used. Otherwise, both these\nparameters should be positive.\n@param stn For the multi-scale Hough transform, it is a divisor for the distance resolution theta.\n@param min_theta For standard and multi-scale Hough transform, minimum angle to check for lines.\nMust fall between 0 and max_theta.\n@param max_theta For standard and multi-scale Hough transform, maximum angle to check for lines.\nMust fall between min_theta and CV_PI.'''
    package = "Contours"

    def get_attributes(self):
        return [Input('image', 'image')], \
               [Output('lines', 'lines')], \
               [FloatParameter('rho', 'rho'),
                FloatParameter('theta', 'theta'),
                FloatParameter('threshold', 'threshold'),
                FloatParameter('srn', 'srn'),
                FloatParameter('stn', 'stn'),
                FloatParameter('min_theta', 'min theta'),
                FloatParameter('max_theta', 'max theta')]

    def process_inputs(self, inputs, outputs, parameters):
        image = inputs['image'].value
        rho = parameters['rho']
        theta = parameters['theta']
        threshold = parameters['threshold']
        srn = parameters['srn']
        stn = parameters['stn']
        min_theta = parameters['min_theta']
        max_theta = parameters['max_theta']
        lines = cv2.HoughLines(image=image, rho=rho, theta=theta, threshold=threshold, srn=srn, stn=stn, min_theta=min_theta, max_theta=max_theta)
        outputs['lines'] = Data(lines)

# cv2.HoughLinesP
class OpenCVAuto2_HoughLinesP(NormalElement):
    name = 'Hough Lines P'
    comment = '''HoughLinesP(image, rho, theta, threshold[, lines[, minLineLength[, maxLineGap]]]) -> lines\n@brief Finds line segments in a binary image using the probabilistic Hough transform.\n\nThe function implements the probabilistic Hough transform algorithm for line detection, described\nin @cite Matas00\n\nSee the line detection example below:\n@include snippets/imgproc_HoughLinesP.cpp\nThis is a sample picture the function parameters have been tuned for:\n\n![image](pics/building.jpg)\n\nAnd this is the output of the above program in case of the probabilistic Hough transform:\n\n![image](pics/houghp.png)\n\n@param image 8-bit, single-channel binary source image. The image may be modified by the function.\n@param lines Output vector of lines. Each line is represented by a 4-element vector\n\f$(x_1, y_1, x_2, y_2)\f$ , where \f$(x_1,y_1)\f$ and \f$(x_2, y_2)\f$ are the ending points of each detected\nline segment.\n@param rho Distance resolution of the accumulator in pixels.\n@param theta Angle resolution of the accumulator in radians.\n@param threshold Accumulator threshold parameter. Only those lines are returned that get enough\nvotes ( \f$>\texttt{threshold}\f$ ).\n@param minLineLength Minimum line length. Line segments shorter than that are rejected.\n@param maxLineGap Maximum allowed gap between points on the same line to link them.\n\n@sa LineSegmentDetector'''
    package = "Contours"

    def get_attributes(self):
        return [Input('image', 'image')], \
               [Output('lines', 'lines')], \
               [FloatParameter('rho', 'rho'),
                FloatParameter('theta', 'theta'),
                FloatParameter('threshold', 'threshold'),
                FloatParameter('minLineLength', 'Min Line Length'),
                FloatParameter('maxLineGap', 'Max Line Gap')]

    def process_inputs(self, inputs, outputs, parameters):
        image = inputs['image'].value
        rho = parameters['rho']
        theta = parameters['theta']
        threshold = parameters['threshold']
        minLineLength = parameters['minLineLength']
        maxLineGap = parameters['maxLineGap']
        lines = cv2.HoughLinesP(image=image, rho=rho, theta=theta, threshold=threshold, minLineLength=minLineLength, maxLineGap=maxLineGap)
        outputs['lines'] = Data(lines)

# cv2.HoughLinesPointSet
class OpenCVAuto2_HoughLinesPointSet(NormalElement):
    name = 'Hough Lines Point Set'
    comment = '''HoughLinesPointSet(_point, lines_max, threshold, min_rho, max_rho, rho_step, min_theta, max_theta, theta_step[, _lines]) -> _lines\n@brief Finds lines in a set of points using the standard Hough transform.\n\nThe function finds lines in a set of points using a modification of the Hough transform.\n@include snippets/imgproc_HoughLinesPointSet.cpp\n@param _point Input vector of points. Each vector must be encoded as a Point vector \f$(x,y)\f$. Type must be CV_32FC2 or CV_32SC2.\n@param _lines Output vector of found lines. Each vector is encoded as a vector<Vec3d> \f$(votes, rho, theta)\f$.\nThe larger the value of 'votes', the higher the reliability of the Hough line.\n@param lines_max Max count of hough lines.\n@param threshold Accumulator threshold parameter. Only those lines are returned that get enough\nvotes ( \f$>\texttt{threshold}\f$ )\n@param min_rho Minimum Distance value of the accumulator in pixels.\n@param max_rho Maximum Distance value of the accumulator in pixels.\n@param rho_step Distance resolution of the accumulator in pixels.\n@param min_theta Minimum angle value of the accumulator in radians.\n@param max_theta Maximum angle value of the accumulator in radians.\n@param theta_step Angle resolution of the accumulator in radians.'''
    package = "Contours"

    def get_attributes(self):
        return [Input('_point', 'point')], \
               [Output('_lines', 'lines')], \
               [IntParameter('lines_max', 'lines max'),
                FloatParameter('threshold', 'threshold'),
                FloatParameter('min_rho', 'min rho'),
                FloatParameter('max_rho', 'max rho'),
                FloatParameter('rho_step', 'rho step'),
                FloatParameter('min_theta', 'min theta'),
                FloatParameter('max_theta', 'max theta'),
                FloatParameter('theta_step', 'theta step')]

    def process_inputs(self, inputs, outputs, parameters):
        _point = inputs['_point'].value
        lines_max = parameters['lines_max']
        threshold = parameters['threshold']
        min_rho = parameters['min_rho']
        max_rho = parameters['max_rho']
        rho_step = parameters['rho_step']
        min_theta = parameters['min_theta']
        max_theta = parameters['max_theta']
        theta_step = parameters['theta_step']
        _lines = cv2.HoughLinesPointSet(_point=_point, lines_max=lines_max, threshold=threshold, min_rho=min_rho, max_rho=max_rho, rho_step=rho_step, min_theta=min_theta, max_theta=max_theta, theta_step=theta_step)
        outputs['_lines'] = Data(_lines)

# cv2.HuMoments
class OpenCVAuto2_HuMoments(NormalElement):
    name = 'Hu Moments'
    comment = '''HuMoments(m[, hu]) -> hu\n@overload'''
    package = "Shape"

    def get_attributes(self):
        return [Input('m', 'm')], \
               [Output('hu', 'hu')], \
               []

    def process_inputs(self, inputs, outputs, parameters):
        m = inputs['m'].value
        hu = cv2.HuMoments(m=m)
        outputs['hu'] = Data(hu)

# cv2.KeyPoint_convert
class OpenCVAuto2_KeyPoint_convert(NormalElement):
    name = 'Key Point convert'
    comment = '''KeyPoint_convert(keypoints[, keypointIndexes]) -> points2f\nThis method converts vector of keypoints to vector of points or the reverse, where each keypoint is\nassigned the same size and the same orientation.\n\n@param keypoints Keypoints obtained from any feature detection algorithm like SIFT/SURF/ORB\n@param points2f Array of (x,y) coordinates of each keypoint\n@param keypointIndexes Array of indexes of keypoints to be converted to points. (Acts like a mask to\nconvert only specified keypoints)



KeyPoint_convert(points2f[, size[, response[, octave[, class_id]]]]) -> keypoints\n@overload\n@param points2f Array of (x,y) coordinates of each keypoint\n@param keypoints Keypoints obtained from any feature detection algorithm like SIFT/SURF/ORB\n@param size keypoint diameter\n@param response keypoint detector response on the keypoint (that is, strength of the keypoint)\n@param octave pyramid octave in which the keypoint has been detected\n@param class_id object id'''
    

    def get_attributes(self):
        return [Input('keypoints', 'keypoints')], \
               [Output('points2f', 'points 2 f')], \
               []

    def process_inputs(self, inputs, outputs, parameters):
        keypoints = inputs['keypoints'].value
        points2f = cv2.KeyPoint_convert(keypoints=keypoints)
        outputs['points2f'] = Data(points2f)

# cv2.LUT
class OpenCVAuto2_LUT(NormalElement):
    name = 'LUT'
    comment = '''LUT(src, lut[, dst]) -> dst\n@brief Performs a look-up table transform of an array.\n\nThe function LUT fills the output array with values from the look-up table. Indices of the entries\nare taken from the input array. That is, the function processes each element of src as follows:\n\f[\texttt{dst} (I)  \leftarrow \texttt{lut(src(I) + d)}\f]\nwhere\n\f[d =  \fork{0}{if \(\texttt{src}\) has depth \(\texttt{CV_8U}\)}{128}{if \(\texttt{src}\) has depth \(\texttt{CV_8S}\)}\f]\n@param src input array of 8-bit elements.\n@param lut look-up table of 256 elements; in case of multi-channel input array, the table should\neither have a single channel (in this case the same table is used for all channels) or the same\nnumber of channels as in the input array.\n@param dst output array of the same size and number of channels as src, and the same depth as lut.\n@sa  convertScaleAbs, Mat::convertTo'''
    package = "Color"

    def get_attributes(self):
        return [Input('src', 'src'),
                Input('lut', 'lut')], \
               [Output('dst', 'dst')], \
               []

    def process_inputs(self, inputs, outputs, parameters):
        src = inputs['src'].value
        lut = inputs['lut'].value
        dst = cv2.LUT(src=src, lut=lut)
        outputs['dst'] = Data(dst)

# cv2.Laplacian
class OpenCVAuto2_Laplacian(NormalElement):
    name = 'Laplacian'
    comment = '''Laplacian(src, ddepth[, dst[, ksize[, scale[, delta[, borderType]]]]]) -> dst\n@brief Calculates the Laplacian of an image.\n\nThe function calculates the Laplacian of the source image by adding up the second x and y\nderivatives calculated using the Sobel operator:\n\n\f[\texttt{dst} =  \Delta \texttt{src} =  \frac{\partial^2 \texttt{src}}{\partial x^2} +  \frac{\partial^2 \texttt{src}}{\partial y^2}\f]\n\nThis is done when `ksize > 1`. When `ksize == 1`, the Laplacian is computed by filtering the image\nwith the following \f$3 \times 3\f$ aperture:\n\n\f[\vecthreethree {0}{1}{0}{1}{-4}{1}{0}{1}{0}\f]\n\n@param src Source image.\n@param dst Destination image of the same size and the same number of channels as src .\n@param ddepth Desired depth of the destination image.\n@param ksize Aperture size used to compute the second-derivative filters. See #getDerivKernels for\ndetails. The size must be positive and odd.\n@param scale Optional scale factor for the computed Laplacian values. By default, no scaling is\napplied. See #getDerivKernels for details.\n@param delta Optional delta value that is added to the results prior to storing them in dst .\n@param borderType Pixel extrapolation method, see #BorderTypes\n@sa  Sobel, Scharr'''
    package = "Math"

    def get_attributes(self):
        return [Input('src', 'src')], \
               [Output('dst', 'dst')], \
               [ComboboxParameter('ddepth', name='ddepth', values=[('NONE',-1),('CV_8U',0),('CV_8UC1',0),('CV_8S',1),('CV_8SC1',1),('CV_16U',2),('CV_16UC1',2),('CV_16S',3),('CV_16SC1',3),('CV_32S',4),('CV_32SC1',4),('CV_32F',5),('CV_32FC1',5),('CV_64F',6),('CV_64FC1',6),('CV_8UC2',8),('CV_8SC2',9),('CV_16UC2',10),('CV_16SC2',11),('CV_32SC2',12),('CV_32FC2',13),('CV_64FC2',14),('CV_8UC3',16),('CV_8SC3',17),('CV_16UC3',18),('CV_16SC3',19),('CV_32SC3',20),('CV_32FC3',21),('CV_64FC3',22),('CV_8UC4',24),('CV_8SC4',25),('CV_16UC4',26),('CV_16SC4',27),('CV_32SC4',28),('CV_32FC4',29),('CV_64FC4',30)]),
                SizeParameter('ksize', 'ksize'),
                FloatParameter('scale', 'scale'),
                FloatParameter('delta', 'delta'),
                ComboboxParameter('borderType', name='Border Type', values=[('BORDER_CONSTANT',0),('BORDER_REPLICATE',1),('BORDER_REFLECT',2),('BORDER_WRAP',3),('BORDER_DEFAULT',4),('BORDER_REFLECT101',4),('BORDER_REFLECT_101',4),('BORDER_TRANSPARENT',5),('BORDER_ISOLATED',16)])]

    def process_inputs(self, inputs, outputs, parameters):
        src = inputs['src'].value
        ddepth = parameters['ddepth']
        ksize = parameters['ksize']
        scale = parameters['scale']
        delta = parameters['delta']
        borderType = parameters['borderType']
        dst = cv2.Laplacian(src=src, ddepth=ddepth, ksize=ksize, scale=scale, delta=delta, borderType=borderType)
        outputs['dst'] = Data(dst)

# cv2.PCABackProject
class OpenCVAuto2_PCABackProject(NormalElement):
    name = 'PCA Back Project'
    comment = '''PCABackProject(data, mean, eigenvectors[, result]) -> result\nwrap PCA::backProject'''
    package = "Principal Component Analysis"

    def get_attributes(self):
        return [Input('data', 'data'),
                Input('mean', 'mean'),
                Input('eigenvectors', 'eigenvectors')], \
               [Output('result', 'result')], \
               []

    def process_inputs(self, inputs, outputs, parameters):
        data = inputs['data'].value
        mean = inputs['mean'].value
        eigenvectors = inputs['eigenvectors'].value
        result = cv2.PCABackProject(data=data, mean=mean, eigenvectors=eigenvectors)
        outputs['result'] = Data(result)

# cv2.PCACompute
class OpenCVAuto2_PCACompute(NormalElement):
    name = 'PCA Compute'
    comment = '''PCACompute(data, mean[, eigenvectors[, maxComponents]]) -> mean, eigenvectors\nwrap PCA::operator()



PCACompute(data, mean, retainedVariance[, eigenvectors]) -> mean, eigenvectors\nwrap PCA::operator()'''
    package = "Principal Component Analysis"

    def get_attributes(self):
        return [Input('data', 'data'),
                Input('mean', 'mean')], \
               [Output('mean', 'mean'),
                Output('eigenvectors', 'eigenvectors')], \
               [IntParameter('maxComponents', 'Max Components')]

    def process_inputs(self, inputs, outputs, parameters):
        data = inputs['data'].value
        mean = inputs['mean'].value.copy()
        maxComponents = parameters['maxComponents']
        mean, eigenvectors = cv2.PCACompute(data=data, mean=mean, maxComponents=maxComponents)
        outputs['mean'] = Data(mean)
        outputs['eigenvectors'] = Data(eigenvectors)

# cv2.PCAProject
class OpenCVAuto2_PCAProject(NormalElement):
    name = 'PCA Project'
    comment = '''PCAProject(data, mean, eigenvectors[, result]) -> result\nwrap PCA::project'''
    package = "Principal Component Analysis"

    def get_attributes(self):
        return [Input('data', 'data'),
                Input('mean', 'mean'),
                Input('eigenvectors', 'eigenvectors')], \
               [Output('result', 'result')], \
               []

    def process_inputs(self, inputs, outputs, parameters):
        data = inputs['data'].value
        mean = inputs['mean'].value
        eigenvectors = inputs['eigenvectors'].value
        result = cv2.PCAProject(data=data, mean=mean, eigenvectors=eigenvectors)
        outputs['result'] = Data(result)

# cv2.RQDecomp3x3
class OpenCVAuto2_RQDecomp3x3(NormalElement):
    name = 'RQ Decomp 3 x 3'
    comment = '''RQDecomp3x3(src[, mtxR[, mtxQ[, Qx[, Qy[, Qz]]]]]) -> retval, mtxR, mtxQ, Qx, Qy, Qz\n@brief Computes an RQ decomposition of 3x3 matrices.\n\n@param src 3x3 input matrix.\n@param mtxR Output 3x3 upper-triangular matrix.\n@param mtxQ Output 3x3 orthogonal matrix.\n@param Qx Optional output 3x3 rotation matrix around x-axis.\n@param Qy Optional output 3x3 rotation matrix around y-axis.\n@param Qz Optional output 3x3 rotation matrix around z-axis.\n\nThe function computes a RQ decomposition using the given rotations. This function is used in\ndecomposeProjectionMatrix to decompose the left 3x3 submatrix of a projection matrix into a camera\nand a rotation matrix.\n\nIt optionally returns three rotation matrices, one for each axis, and the three Euler angles in\ndegrees (as the return value) that could be used in OpenGL. Note, there is always more than one\nsequence of rotations about the three principal axes that results in the same orientation of an\nobject, e.g. see @cite Slabaugh . Returned tree rotation matrices and corresponding three Euler angles\nare only one of the possible solutions.'''
    package = "3D calibration & reconstruction"

    def get_attributes(self):
        return [Input('src', 'src')], \
               [Output('mtxR', 'Mtx R'),
                Output('mtxQ', 'Mtx Q'),
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

# cv2.Rodrigues
class OpenCVAuto2_Rodrigues(NormalElement):
    name = 'Rodrigues'
    comment = '''Rodrigues(src[, dst[, jacobian]]) -> dst, jacobian\n@brief Converts a rotation matrix to a rotation vector or vice versa.\n\n@param src Input rotation vector (3x1 or 1x3) or rotation matrix (3x3).\n@param dst Output rotation matrix (3x3) or rotation vector (3x1 or 1x3), respectively.\n@param jacobian Optional output Jacobian matrix, 3x9 or 9x3, which is a matrix of partial\nderivatives of the output array components with respect to the input array components.\n\n\f[\begin{array}{l} \theta \leftarrow norm(r) \\ r  \leftarrow r/ \theta \\ R =  \cos{\theta} I + (1- \cos{\theta} ) r r^T +  \sin{\theta} \vecthreethree{0}{-r_z}{r_y}{r_z}{0}{-r_x}{-r_y}{r_x}{0} \end{array}\f]\n\nInverse transformation can be also done easily, since\n\n\f[\sin ( \theta ) \vecthreethree{0}{-r_z}{r_y}{r_z}{0}{-r_x}{-r_y}{r_x}{0} = \frac{R - R^T}{2}\f]\n\nA rotation vector is a convenient and most compact representation of a rotation matrix (since any\nrotation matrix has just 3 degrees of freedom). The representation is used in the global 3D geometry\noptimization procedures like calibrateCamera, stereoCalibrate, or solvePnP .'''
    package = "3D calibration & reconstruction"

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

# cv2.SVBackSubst
class OpenCVAuto2_SVBackSubst(NormalElement):
    name = 'SV Back Subst'
    comment = '''SVBackSubst(w, u, vt, rhs[, dst]) -> dst\nwrap SVD::backSubst'''
    package = "Miscellaneous"

    def get_attributes(self):
        return [Input('w', 'w'),
                Input('u', 'u'),
                Input('vt', 'vt'),
                Input('rhs', 'rhs')], \
               [Output('dst', 'dst')], \
               []

    def process_inputs(self, inputs, outputs, parameters):
        w = inputs['w'].value
        u = inputs['u'].value
        vt = inputs['vt'].value
        rhs = inputs['rhs'].value
        dst = cv2.SVBackSubst(w=w, u=u, vt=vt, rhs=rhs)
        outputs['dst'] = Data(dst)

# cv2.Scharr
class OpenCVAuto2_Scharr(NormalElement):
    name = 'Scharr'
    comment = '''Scharr(src, ddepth, dx, dy[, dst[, scale[, delta[, borderType]]]]) -> dst\n@brief Calculates the first x- or y- image derivative using Scharr operator.\n\nThe function computes the first x- or y- spatial image derivative using the Scharr operator. The\ncall\n\n\f[\texttt{Scharr(src, dst, ddepth, dx, dy, scale, delta, borderType)}\f]\n\nis equivalent to\n\n\f[\texttt{Sobel(src, dst, ddepth, dx, dy, CV_SCHARR, scale, delta, borderType)} .\f]\n\n@param src input image.\n@param dst output image of the same size and the same number of channels as src.\n@param ddepth output image depth, see @ref filter_depths "combinations"\n@param dx order of the derivative x.\n@param dy order of the derivative y.\n@param scale optional scale factor for the computed derivative values; by default, no scaling is\napplied (see #getDerivKernels for details).\n@param delta optional delta value that is added to the results prior to storing them in dst.\n@param borderType pixel extrapolation method, see #BorderTypes\n@sa  cartToPolar'''
    package = "Filters"

    def get_attributes(self):
        return [Input('src', 'src')], \
               [Output('dst', 'dst')], \
               [ComboboxParameter('ddepth', name='ddepth', values=[('NONE',-1),('CV_8U',0),('CV_8UC1',0),('CV_8S',1),('CV_8SC1',1),('CV_16U',2),('CV_16UC1',2),('CV_16S',3),('CV_16SC1',3),('CV_32S',4),('CV_32SC1',4),('CV_32F',5),('CV_32FC1',5),('CV_64F',6),('CV_64FC1',6),('CV_8UC2',8),('CV_8SC2',9),('CV_16UC2',10),('CV_16SC2',11),('CV_32SC2',12),('CV_32FC2',13),('CV_64FC2',14),('CV_8UC3',16),('CV_8SC3',17),('CV_16UC3',18),('CV_16SC3',19),('CV_32SC3',20),('CV_32FC3',21),('CV_64FC3',22),('CV_8UC4',24),('CV_8SC4',25),('CV_16UC4',26),('CV_16SC4',27),('CV_32SC4',28),('CV_32FC4',29),('CV_64FC4',30)]),
                IntParameter('dx', 'dx', min_=0),
                IntParameter('dy', 'dy', min_=0),
                FloatParameter('scale', 'scale'),
                FloatParameter('delta', 'delta'),
                ComboboxParameter('borderType', name='Border Type', values=[('BORDER_CONSTANT',0),('BORDER_REPLICATE',1),('BORDER_REFLECT',2),('BORDER_WRAP',3),('BORDER_DEFAULT',4),('BORDER_REFLECT101',4),('BORDER_REFLECT_101',4),('BORDER_TRANSPARENT',5),('BORDER_ISOLATED',16)])]

    def process_inputs(self, inputs, outputs, parameters):
        src = inputs['src'].value
        ddepth = parameters['ddepth']
        dx = parameters['dx']
        dy = parameters['dy']
        scale = parameters['scale']
        delta = parameters['delta']
        borderType = parameters['borderType']
        dst = cv2.Scharr(src=src, ddepth=ddepth, dx=dx, dy=dy, scale=scale, delta=delta, borderType=borderType)
        outputs['dst'] = Data(dst)

# cv2.Sobel
class OpenCVAuto2_Sobel(NormalElement):
    name = 'Sobel'
    comment = '''Sobel(src, ddepth, dx, dy[, dst[, ksize[, scale[, delta[, borderType]]]]]) -> dst\n@brief Calculates the first, second, third, or mixed image derivatives using an extended Sobel operator.\n\nIn all cases except one, the \f$\texttt{ksize} \times \texttt{ksize}\f$ separable kernel is used to\ncalculate the derivative. When \f$\texttt{ksize = 1}\f$, the \f$3 \times 1\f$ or \f$1 \times 3\f$\nkernel is used (that is, no Gaussian smoothing is done). `ksize = 1` can only be used for the first\nor the second x- or y- derivatives.\n\nThere is also the special value `ksize = #CV_SCHARR (-1)` that corresponds to the \f$3\times3\f$ Scharr\nfilter that may give more accurate results than the \f$3\times3\f$ Sobel. The Scharr aperture is\n\n\f[\vecthreethree{-3}{0}{3}{-10}{0}{10}{-3}{0}{3}\f]\n\nfor the x-derivative, or transposed for the y-derivative.\n\nThe function calculates an image derivative by convolving the image with the appropriate kernel:\n\n\f[\texttt{dst} =  \frac{\partial^{xorder+yorder} \texttt{src}}{\partial x^{xorder} \partial y^{yorder}}\f]\n\nThe Sobel operators combine Gaussian smoothing and differentiation, so the result is more or less\nresistant to the noise. Most often, the function is called with ( xorder = 1, yorder = 0, ksize = 3)\nor ( xorder = 0, yorder = 1, ksize = 3) to calculate the first x- or y- image derivative. The first\ncase corresponds to a kernel of:\n\n\f[\vecthreethree{-1}{0}{1}{-2}{0}{2}{-1}{0}{1}\f]\n\nThe second case corresponds to a kernel of:\n\n\f[\vecthreethree{-1}{-2}{-1}{0}{0}{0}{1}{2}{1}\f]\n\n@param src input image.\n@param dst output image of the same size and the same number of channels as src .\n@param ddepth output image depth, see @ref filter_depths "combinations"; in the case of\n8-bit input images it will result in truncated derivatives.\n@param dx order of the derivative x.\n@param dy order of the derivative y.\n@param ksize size of the extended Sobel kernel; it must be 1, 3, 5, or 7.\n@param scale optional scale factor for the computed derivative values; by default, no scaling is\napplied (see #getDerivKernels for details).\n@param delta optional delta value that is added to the results prior to storing them in dst.\n@param borderType pixel extrapolation method, see #BorderTypes\n@sa  Scharr, Laplacian, sepFilter2D, filter2D, GaussianBlur, cartToPolar'''
    package = "Filters"

    def get_attributes(self):
        return [Input('src', 'src')], \
               [Output('dst', 'dst')], \
               [ComboboxParameter('ddepth', name='ddepth', values=[('NONE',-1),('CV_8U',0),('CV_8UC1',0),('CV_8S',1),('CV_8SC1',1),('CV_16U',2),('CV_16UC1',2),('CV_16S',3),('CV_16SC1',3),('CV_32S',4),('CV_32SC1',4),('CV_32F',5),('CV_32FC1',5),('CV_64F',6),('CV_64FC1',6),('CV_8UC2',8),('CV_8SC2',9),('CV_16UC2',10),('CV_16SC2',11),('CV_32SC2',12),('CV_32FC2',13),('CV_64FC2',14),('CV_8UC3',16),('CV_8SC3',17),('CV_16UC3',18),('CV_16SC3',19),('CV_32SC3',20),('CV_32FC3',21),('CV_64FC3',22),('CV_8UC4',24),('CV_8SC4',25),('CV_16UC4',26),('CV_16SC4',27),('CV_32SC4',28),('CV_32FC4',29),('CV_64FC4',30)]),
                IntParameter('dx', 'dx', min_=0),
                IntParameter('dy', 'dy', min_=0),
                SizeParameter('ksize', 'ksize'),
                FloatParameter('scale', 'scale'),
                FloatParameter('delta', 'delta'),
                ComboboxParameter('borderType', name='Border Type', values=[('BORDER_CONSTANT',0),('BORDER_REPLICATE',1),('BORDER_REFLECT',2),('BORDER_WRAP',3),('BORDER_DEFAULT',4),('BORDER_REFLECT101',4),('BORDER_REFLECT_101',4),('BORDER_TRANSPARENT',5),('BORDER_ISOLATED',16)])]

    def process_inputs(self, inputs, outputs, parameters):
        src = inputs['src'].value
        ddepth = parameters['ddepth']
        dx = parameters['dx']
        dy = parameters['dy']
        ksize = parameters['ksize']
        scale = parameters['scale']
        delta = parameters['delta']
        borderType = parameters['borderType']
        dst = cv2.Sobel(src=src, ddepth=ddepth, dx=dx, dy=dy, ksize=ksize, scale=scale, delta=delta, borderType=borderType)
        outputs['dst'] = Data(dst)

# cv2.absdiff
class OpenCVAuto2_Absdiff(NormalElement):
    name = 'Absdiff'
    comment = '''absdiff(src1, src2[, dst]) -> dst\n@brief Calculates the per-element absolute difference between two arrays or between an array and a scalar.\n\nThe function cv::absdiff calculates:\n*   Absolute difference between two arrays when they have the same\nsize and type:\n\f[\texttt{dst}(I) =  \texttt{saturate} (| \texttt{src1}(I) -  \texttt{src2}(I)|)\f]\n*   Absolute difference between an array and a scalar when the second\narray is constructed from Scalar or has as many elements as the\nnumber of channels in `src1`:\n\f[\texttt{dst}(I) =  \texttt{saturate} (| \texttt{src1}(I) -  \texttt{src2} |)\f]\n*   Absolute difference between a scalar and an array when the first\narray is constructed from Scalar or has as many elements as the\nnumber of channels in `src2`:\n\f[\texttt{dst}(I) =  \texttt{saturate} (| \texttt{src1} -  \texttt{src2}(I) |)\f]\nwhere I is a multi-dimensional index of array elements. In case of\nmulti-channel arrays, each channel is processed independently.\n@note Saturation is not applied when the arrays have the depth CV_32S.\nYou may even get a negative value in the case of overflow.\n@param src1 first input array or a scalar.\n@param src2 second input array or a scalar.\n@param dst output array that has the same size and type as input arrays.\n@sa cv::abs(const Mat&)'''
    package = "Operators"

    def get_attributes(self):
        return [Input('src1', 'src 1'),
                Input('src2', 'src 2')], \
               [Output('dst', 'dst')], \
               []

    def process_inputs(self, inputs, outputs, parameters):
        src1 = inputs['src1'].value
        src2 = inputs['src2'].value
        dst = cv2.absdiff(src1=src1, src2=src2)
        outputs['dst'] = Data(dst)

# cv2.accumulate
class OpenCVAuto2_Accumulate(NormalElement):
    name = 'Accumulate'
    comment = '''accumulate(src, dst[, mask]) -> dst\n@brief Adds an image to the accumulator image.\n\nThe function adds src or some of its elements to dst :\n\n\f[\texttt{dst} (x,y)  \leftarrow \texttt{dst} (x,y) +  \texttt{src} (x,y)  \quad \text{if} \quad \texttt{mask} (x,y)  \ne 0\f]\n\nThe function supports multi-channel images. Each channel is processed independently.\n\nThe function cv::accumulate can be used, for example, to collect statistics of a scene background\nviewed by a still camera and for the further foreground-background segmentation.\n\n@param src Input image of type CV_8UC(n), CV_16UC(n), CV_32FC(n) or CV_64FC(n), where n is a positive integer.\n@param dst %Accumulator image with the same number of channels as input image, and a depth of CV_32F or CV_64F.\n@param mask Optional operation mask.\n\n@sa  accumulateSquare, accumulateProduct, accumulateWeighted'''
    package = "Accumulators"

    def get_attributes(self):
        return [Input('src', 'src'),
                Input('dst', 'dst'),
                Input('mask', 'mask', optional=True)], \
               [Output('dst', 'dst')], \
               []

    def process_inputs(self, inputs, outputs, parameters):
        src = inputs['src'].value
        dst = inputs['dst'].value.copy()
        mask = inputs['mask'].value
        dst = cv2.accumulate(src=src, dst=dst, mask=mask)
        outputs['dst'] = Data(dst)

# cv2.accumulateProduct
class OpenCVAuto2_AccumulateProduct(NormalElement):
    name = 'Accumulate Product'
    comment = '''accumulateProduct(src1, src2, dst[, mask]) -> dst\n@brief Adds the per-element product of two input images to the accumulator image.\n\nThe function adds the product of two images or their selected regions to the accumulator dst :\n\n\f[\texttt{dst} (x,y)  \leftarrow \texttt{dst} (x,y) +  \texttt{src1} (x,y)  \cdot \texttt{src2} (x,y)  \quad \text{if} \quad \texttt{mask} (x,y)  \ne 0\f]\n\nThe function supports multi-channel images. Each channel is processed independently.\n\n@param src1 First input image, 1- or 3-channel, 8-bit or 32-bit floating point.\n@param src2 Second input image of the same type and the same size as src1 .\n@param dst %Accumulator image with the same number of channels as input images, 32-bit or 64-bit\nfloating-point.\n@param mask Optional operation mask.\n\n@sa  accumulate, accumulateSquare, accumulateWeighted'''
    package = "Accumulators"

    def get_attributes(self):
        return [Input('src1', 'src 1'),
                Input('src2', 'src 2'),
                Input('dst', 'dst'),
                Input('mask', 'mask', optional=True)], \
               [Output('dst', 'dst')], \
               []

    def process_inputs(self, inputs, outputs, parameters):
        src1 = inputs['src1'].value
        src2 = inputs['src2'].value
        dst = inputs['dst'].value.copy()
        mask = inputs['mask'].value
        dst = cv2.accumulateProduct(src1=src1, src2=src2, dst=dst, mask=mask)
        outputs['dst'] = Data(dst)

# cv2.accumulateSquare
class OpenCVAuto2_AccumulateSquare(NormalElement):
    name = 'Accumulate Square'
    comment = '''accumulateSquare(src, dst[, mask]) -> dst\n@brief Adds the square of a source image to the accumulator image.\n\nThe function adds the input image src or its selected region, raised to a power of 2, to the\naccumulator dst :\n\n\f[\texttt{dst} (x,y)  \leftarrow \texttt{dst} (x,y) +  \texttt{src} (x,y)^2  \quad \text{if} \quad \texttt{mask} (x,y)  \ne 0\f]\n\nThe function supports multi-channel images. Each channel is processed independently.\n\n@param src Input image as 1- or 3-channel, 8-bit or 32-bit floating point.\n@param dst %Accumulator image with the same number of channels as input image, 32-bit or 64-bit\nfloating-point.\n@param mask Optional operation mask.\n\n@sa  accumulateSquare, accumulateProduct, accumulateWeighted'''
    package = "Accumulators"

    def get_attributes(self):
        return [Input('src', 'src'),
                Input('dst', 'dst'),
                Input('mask', 'mask', optional=True)], \
               [Output('dst', 'dst')], \
               []

    def process_inputs(self, inputs, outputs, parameters):
        src = inputs['src'].value
        dst = inputs['dst'].value.copy()
        mask = inputs['mask'].value
        dst = cv2.accumulateSquare(src=src, dst=dst, mask=mask)
        outputs['dst'] = Data(dst)

# cv2.accumulateWeighted
class OpenCVAuto2_AccumulateWeighted(NormalElement):
    name = 'Accumulate Weighted'
    comment = '''accumulateWeighted(src, dst, alpha[, mask]) -> dst\n@brief Updates a running average.\n\nThe function calculates the weighted sum of the input image src and the accumulator dst so that dst\nbecomes a running average of a frame sequence:\n\n\f[\texttt{dst} (x,y)  \leftarrow (1- \texttt{alpha} )  \cdot \texttt{dst} (x,y) +  \texttt{alpha} \cdot \texttt{src} (x,y)  \quad \text{if} \quad \texttt{mask} (x,y)  \ne 0\f]\n\nThat is, alpha regulates the update speed (how fast the accumulator "forgets" about earlier images).\nThe function supports multi-channel images. Each channel is processed independently.\n\n@param src Input image as 1- or 3-channel, 8-bit or 32-bit floating point.\n@param dst %Accumulator image with the same number of channels as input image, 32-bit or 64-bit\nfloating-point.\n@param alpha Weight of the input image.\n@param mask Optional operation mask.\n\n@sa  accumulate, accumulateSquare, accumulateProduct'''
    package = "Accumulators"

    def get_attributes(self):
        return [Input('src', 'src'),
                Input('dst', 'dst'),
                Input('mask', 'mask', optional=True)], \
               [Output('dst', 'dst')], \
               [FloatParameter('alpha', 'alpha')]

    def process_inputs(self, inputs, outputs, parameters):
        src = inputs['src'].value
        dst = inputs['dst'].value.copy()
        mask = inputs['mask'].value
        alpha = parameters['alpha']
        dst = cv2.accumulateWeighted(src=src, dst=dst, mask=mask, alpha=alpha)
        outputs['dst'] = Data(dst)

# cv2.adaptiveThreshold
class OpenCVAuto2_AdaptiveThreshold(NormalElement):
    name = 'Adaptive Threshold'
    comment = '''adaptiveThreshold(src, maxValue, adaptiveMethod, thresholdType, blockSize, C[, dst]) -> dst\n@brief Applies an adaptive threshold to an array.\n\nThe function transforms a grayscale image to a binary image according to the formulae:\n-   **THRESH_BINARY**\n\f[dst(x,y) =  \fork{\texttt{maxValue}}{if \(src(x,y) > T(x,y)\)}{0}{otherwise}\f]\n-   **THRESH_BINARY_INV**\n\f[dst(x,y) =  \fork{0}{if \(src(x,y) > T(x,y)\)}{\texttt{maxValue}}{otherwise}\f]\nwhere \f$T(x,y)\f$ is a threshold calculated individually for each pixel (see adaptiveMethod parameter).\n\nThe function can process the image in-place.\n\n@param src Source 8-bit single-channel image.\n@param dst Destination image of the same size and the same type as src.\n@param maxValue Non-zero value assigned to the pixels for which the condition is satisfied\n@param adaptiveMethod Adaptive thresholding algorithm to use, see #AdaptiveThresholdTypes.\nThe #BORDER_REPLICATE | #BORDER_ISOLATED is used to process boundaries.\n@param thresholdType Thresholding type that must be either #THRESH_BINARY or #THRESH_BINARY_INV,\nsee #ThresholdTypes.\n@param blockSize Size of a pixel neighborhood that is used to calculate a threshold value for the\npixel: 3, 5, 7, and so on.\n@param C Constant subtracted from the mean or weighted mean (see the details below). Normally, it\nis positive but may be zero or negative as well.\n\n@sa  threshold, blur, GaussianBlur'''
    package = "Threshold"

    def get_attributes(self):
        return [Input('src', 'src')], \
               [Output('dst', 'dst')], \
               [FloatParameter('maxValue', 'Max Value'),
                IntParameter('adaptiveMethod', 'Adaptive Method'),
                FloatParameter('thresholdType', 'Threshold Type'),
                SizeParameter('blockSize', 'Block Size'),
                FloatParameter('C', 'C')]

    def process_inputs(self, inputs, outputs, parameters):
        src = inputs['src'].value
        maxValue = parameters['maxValue']
        adaptiveMethod = parameters['adaptiveMethod']
        thresholdType = parameters['thresholdType']
        blockSize = parameters['blockSize']
        C = parameters['C']
        dst = cv2.adaptiveThreshold(src=src, maxValue=maxValue, adaptiveMethod=adaptiveMethod, thresholdType=thresholdType, blockSize=blockSize, C=C)
        outputs['dst'] = Data(dst)

# cv2.add
class OpenCVAuto2_Add(NormalElement):
    name = 'Add'
    comment = '''add(src1, src2[, dst[, mask[, dtype]]]) -> dst\n@brief Calculates the per-element sum of two arrays or an array and a scalar.\n\nThe function add calculates:\n- Sum of two arrays when both input arrays have the same size and the same number of channels:\n\f[\texttt{dst}(I) =  \texttt{saturate} ( \texttt{src1}(I) +  \texttt{src2}(I)) \quad \texttt{if mask}(I) \ne0\f]\n- Sum of an array and a scalar when src2 is constructed from Scalar or has the same number of\nelements as `src1.channels()`:\n\f[\texttt{dst}(I) =  \texttt{saturate} ( \texttt{src1}(I) +  \texttt{src2} ) \quad \texttt{if mask}(I) \ne0\f]\n- Sum of a scalar and an array when src1 is constructed from Scalar or has the same number of\nelements as `src2.channels()`:\n\f[\texttt{dst}(I) =  \texttt{saturate} ( \texttt{src1} +  \texttt{src2}(I) ) \quad \texttt{if mask}(I) \ne0\f]\nwhere `I` is a multi-dimensional index of array elements. In case of multi-channel arrays, each\nchannel is processed independently.\n\nThe first function in the list above can be replaced with matrix expressions:\n@code{.cpp}\ndst = src1 + src2;\ndst += src1; // equivalent to add(dst, src1, dst);\n@endcode\nThe input arrays and the output array can all have the same or different depths. For example, you\ncan add a 16-bit unsigned array to a 8-bit signed array and store the sum as a 32-bit\nfloating-point array. Depth of the output array is determined by the dtype parameter. In the second\nand third cases above, as well as in the first case, when src1.depth() == src2.depth(), dtype can\nbe set to the default -1. In this case, the output array will have the same depth as the input\narray, be it src1, src2 or both.\n@note Saturation is not applied when the output array has the depth CV_32S. You may even get\nresult of an incorrect sign in the case of overflow.\n@param src1 first input array or a scalar.\n@param src2 second input array or a scalar.\n@param dst output array that has the same size and number of channels as the input array(s); the\ndepth is defined by dtype or src1/src2.\n@param mask optional operation mask - 8-bit single channel array, that specifies elements of the\noutput array to be changed.\n@param dtype optional depth of the output array (see the discussion below).\n@sa subtract, addWeighted, scaleAdd, Mat::convertTo'''
    package = "Operators"

    def get_attributes(self):
        return [Input('src1', 'src 1'),
                Input('src2', 'src 2'),
                Input('mask', 'mask', optional=True)], \
               [Output('dst', 'dst')], \
               [ComboboxParameter('dtype', name='dtype', values=[('NONE',-1),('CV_8U',0),('CV_8UC1',0),('CV_8S',1),('CV_8SC1',1),('CV_16U',2),('CV_16UC1',2),('CV_16S',3),('CV_16SC1',3),('CV_32S',4),('CV_32SC1',4),('CV_32F',5),('CV_32FC1',5),('CV_64F',6),('CV_64FC1',6),('CV_8UC2',8),('CV_8SC2',9),('CV_16UC2',10),('CV_16SC2',11),('CV_32SC2',12),('CV_32FC2',13),('CV_64FC2',14),('CV_8UC3',16),('CV_8SC3',17),('CV_16UC3',18),('CV_16SC3',19),('CV_32SC3',20),('CV_32FC3',21),('CV_64FC3',22),('CV_8UC4',24),('CV_8SC4',25),('CV_16UC4',26),('CV_16SC4',27),('CV_32SC4',28),('CV_32FC4',29),('CV_64FC4',30)])]

    def process_inputs(self, inputs, outputs, parameters):
        src1 = inputs['src1'].value
        src2 = inputs['src2'].value
        mask = inputs['mask'].value
        dtype = parameters['dtype']
        dst = cv2.add(src1=src1, src2=src2, mask=mask, dtype=dtype)
        outputs['dst'] = Data(dst)

# cv2.addWeighted
class OpenCVAuto2_AddWeighted(NormalElement):
    name = 'Add Weighted'
    comment = '''addWeighted(src1, alpha, src2, beta, gamma[, dst[, dtype]]) -> dst\n@brief Calculates the weighted sum of two arrays.\n\nThe function addWeighted calculates the weighted sum of two arrays as follows:\n\f[\texttt{dst} (I)= \texttt{saturate} ( \texttt{src1} (I)* \texttt{alpha} +  \texttt{src2} (I)* \texttt{beta} +  \texttt{gamma} )\f]\nwhere I is a multi-dimensional index of array elements. In case of multi-channel arrays, each\nchannel is processed independently.\nThe function can be replaced with a matrix expression:\n@code{.cpp}\ndst = src1*alpha + src2*beta + gamma;\n@endcode\n@note Saturation is not applied when the output array has the depth CV_32S. You may even get\nresult of an incorrect sign in the case of overflow.\n@param src1 first input array.\n@param alpha weight of the first array elements.\n@param src2 second input array of the same size and channel number as src1.\n@param beta weight of the second array elements.\n@param gamma scalar added to each sum.\n@param dst output array that has the same size and number of channels as the input arrays.\n@param dtype optional depth of the output array; when both input arrays have the same depth, dtype\ncan be set to -1, which will be equivalent to src1.depth().\n@sa  add, subtract, scaleAdd, Mat::convertTo'''
    package = "Operators"

    def get_attributes(self):
        return [Input('src1', 'src 1'),
                Input('src2', 'src 2')], \
               [Output('dst', 'dst')], \
               [FloatParameter('alpha', 'alpha'),
                FloatParameter('beta', 'beta'),
                FloatParameter('gamma', 'gamma'),
                ComboboxParameter('dtype', name='dtype', values=[('NONE',-1),('CV_8U',0),('CV_8UC1',0),('CV_8S',1),('CV_8SC1',1),('CV_16U',2),('CV_16UC1',2),('CV_16S',3),('CV_16SC1',3),('CV_32S',4),('CV_32SC1',4),('CV_32F',5),('CV_32FC1',5),('CV_64F',6),('CV_64FC1',6),('CV_8UC2',8),('CV_8SC2',9),('CV_16UC2',10),('CV_16SC2',11),('CV_32SC2',12),('CV_32FC2',13),('CV_64FC2',14),('CV_8UC3',16),('CV_8SC3',17),('CV_16UC3',18),('CV_16SC3',19),('CV_32SC3',20),('CV_32FC3',21),('CV_64FC3',22),('CV_8UC4',24),('CV_8SC4',25),('CV_16UC4',26),('CV_16SC4',27),('CV_32SC4',28),('CV_32FC4',29),('CV_64FC4',30)])]

    def process_inputs(self, inputs, outputs, parameters):
        src1 = inputs['src1'].value
        src2 = inputs['src2'].value
        alpha = parameters['alpha']
        beta = parameters['beta']
        gamma = parameters['gamma']
        dtype = parameters['dtype']
        dst = cv2.addWeighted(src1=src1, src2=src2, alpha=alpha, beta=beta, gamma=gamma, dtype=dtype)
        outputs['dst'] = Data(dst)

# cv2.applyColorMap
class OpenCVAuto2_ApplyColorMap(NormalElement):
    name = 'Apply Color Map'
    comment = '''applyColorMap(src, colormap[, dst]) -> dst\n@brief Applies a GNU Octave/MATLAB equivalent colormap on a given image.\n\n@param src The source image, grayscale or colored of type CV_8UC1 or CV_8UC3.\n@param dst The result is the colormapped source image. Note: Mat::create is called on dst.\n@param colormap The colormap to apply, see #ColormapTypes



applyColorMap(src, userColor[, dst]) -> dst\n@brief Applies a user colormap on a given image.\n\n@param src The source image, grayscale or colored of type CV_8UC1 or CV_8UC3.\n@param dst The result is the colormapped source image. Note: Mat::create is called on dst.\n@param userColor The colormap to apply of type CV_8UC1 or CV_8UC3 and size 256'''
    package = "Color"

    def get_attributes(self):
        return [Input('src', 'src')], \
               [Output('dst', 'dst')], \
               [IntParameter('colormap', 'colormap')]

    def process_inputs(self, inputs, outputs, parameters):
        src = inputs['src'].value
        colormap = parameters['colormap']
        dst = cv2.applyColorMap(src=src, colormap=colormap)
        outputs['dst'] = Data(dst)

# cv2.arrowedLine
class OpenCVAuto2_ArrowedLine(NormalElement):
    name = 'Arrowed Line'
    comment = '''arrowedLine(img, pt1, pt2, color[, thickness[, line_type[, shift[, tipLength]]]]) -> img\n@brief Draws a arrow segment pointing from the first point to the second one.\n\nThe function cv::arrowedLine draws an arrow between pt1 and pt2 points in the image. See also #line.\n\n@param img Image.\n@param pt1 The point the arrow starts from.\n@param pt2 The point the arrow points to.\n@param color Line color.\n@param thickness Line thickness.\n@param line_type Type of the line. See #LineTypes\n@param shift Number of fractional bits in the point coordinates.\n@param tipLength The length of the arrow tip in relation to the arrow length'''
    package = "Drawing"

    def get_attributes(self):
        return [Input('img', 'img')], \
               [Output('img', 'img')], \
               [PointParameter('pt1', 'pt 1'),
                PointParameter('pt2', 'pt 2'),
                ScalarParameter('color', 'color'),
                IntParameter('thickness', 'thickness', min_=-1, max_=100),
                IntParameter('line_type', 'line type'),
                IntParameter('shift', 'shift'),
                FloatParameter('tipLength', 'Tip Length')]

    def process_inputs(self, inputs, outputs, parameters):
        img = inputs['img'].value.copy()
        pt1 = parameters['pt1']
        pt2 = parameters['pt2']
        color = parameters['color']
        thickness = parameters['thickness']
        line_type = parameters['line_type']
        shift = parameters['shift']
        tipLength = parameters['tipLength']
        img = cv2.arrowedLine(img=img, pt1=pt1, pt2=pt2, color=color, thickness=thickness, line_type=line_type, shift=shift, tipLength=tipLength)
        outputs['img'] = Data(img)

# cv2.bilateralFilter
class OpenCVAuto2_BilateralFilter(NormalElement):
    name = 'Bilateral Filter'
    comment = '''bilateralFilter(src, d, sigmaColor, sigmaSpace[, dst[, borderType]]) -> dst\n@brief Applies the bilateral filter to an image.\n\nThe function applies bilateral filtering to the input image, as described in\nhttp://www.dai.ed.ac.uk/CVonline/LOCAL_COPIES/MANDUCHI1/Bilateral_Filtering.html\nbilateralFilter can reduce unwanted noise very well while keeping edges fairly sharp. However, it is\nvery slow compared to most filters.\n\n_Sigma values_: For simplicity, you can set the 2 sigma values to be the same. If they are small (\<\n10), the filter will not have much effect, whereas if they are large (\> 150), they will have a very\nstrong effect, making the image look "cartoonish".\n\n_Filter size_: Large filters (d \> 5) are very slow, so it is recommended to use d=5 for real-time\napplications, and perhaps d=9 for offline applications that need heavy noise filtering.\n\nThis filter does not work inplace.\n@param src Source 8-bit or floating-point, 1-channel or 3-channel image.\n@param dst Destination image of the same size and type as src .\n@param d Diameter of each pixel neighborhood that is used during filtering. If it is non-positive,\nit is computed from sigmaSpace.\n@param sigmaColor Filter sigma in the color space. A larger value of the parameter means that\nfarther colors within the pixel neighborhood (see sigmaSpace) will be mixed together, resulting\nin larger areas of semi-equal color.\n@param sigmaSpace Filter sigma in the coordinate space. A larger value of the parameter means that\nfarther pixels will influence each other as long as their colors are close enough (see sigmaColor\n). When d\>0, it specifies the neighborhood size regardless of sigmaSpace. Otherwise, d is\nproportional to sigmaSpace.\n@param borderType border mode used to extrapolate pixels outside of the image, see #BorderTypes'''
    package = "Filters"

    def get_attributes(self):
        return [Input('src', 'src')], \
               [Output('dst', 'dst')], \
               [IntParameter('d', 'd'),
                FloatParameter('sigmaColor', 'Sigma Color'),
                FloatParameter('sigmaSpace', 'Sigma Space'),
                ComboboxParameter('borderType', name='Border Type', values=[('BORDER_CONSTANT',0),('BORDER_REPLICATE',1),('BORDER_REFLECT',2),('BORDER_WRAP',3),('BORDER_DEFAULT',4),('BORDER_REFLECT101',4),('BORDER_REFLECT_101',4),('BORDER_TRANSPARENT',5),('BORDER_ISOLATED',16)])]

    def process_inputs(self, inputs, outputs, parameters):
        src = inputs['src'].value
        d = parameters['d']
        sigmaColor = parameters['sigmaColor']
        sigmaSpace = parameters['sigmaSpace']
        borderType = parameters['borderType']
        dst = cv2.bilateralFilter(src=src, d=d, sigmaColor=sigmaColor, sigmaSpace=sigmaSpace, borderType=borderType)
        outputs['dst'] = Data(dst)

# cv2.bitwise_and
class OpenCVAuto2_Bitwise_and(NormalElement):
    name = 'Bitwise and'
    comment = '''bitwise_and(src1, src2[, dst[, mask]]) -> dst\n@brief computes bitwise conjunction of the two arrays (dst = src1 & src2)\nCalculates the per-element bit-wise conjunction of two arrays or an\narray and a scalar.\n\nThe function cv::bitwise_and calculates the per-element bit-wise logical conjunction for:\n*   Two arrays when src1 and src2 have the same size:\n\f[\texttt{dst} (I) =  \texttt{src1} (I)  \wedge \texttt{src2} (I) \quad \texttt{if mask} (I) \ne0\f]\n*   An array and a scalar when src2 is constructed from Scalar or has\nthe same number of elements as `src1.channels()`:\n\f[\texttt{dst} (I) =  \texttt{src1} (I)  \wedge \texttt{src2} \quad \texttt{if mask} (I) \ne0\f]\n*   A scalar and an array when src1 is constructed from Scalar or has\nthe same number of elements as `src2.channels()`:\n\f[\texttt{dst} (I) =  \texttt{src1}  \wedge \texttt{src2} (I) \quad \texttt{if mask} (I) \ne0\f]\nIn case of floating-point arrays, their machine-specific bit\nrepresentations (usually IEEE754-compliant) are used for the operation.\nIn case of multi-channel arrays, each channel is processed\nindependently. In the second and third cases above, the scalar is first\nconverted to the array type.\n@param src1 first input array or a scalar.\n@param src2 second input array or a scalar.\n@param dst output array that has the same size and type as the input\narrays.\n@param mask optional operation mask, 8-bit single channel array, that\nspecifies elements of the output array to be changed.'''
    package = "Logic"

    def get_attributes(self):
        return [Input('src1', 'src 1'),
                Input('src2', 'src 2'),
                Input('mask', 'mask', optional=True)], \
               [Output('dst', 'dst')], \
               []

    def process_inputs(self, inputs, outputs, parameters):
        src1 = inputs['src1'].value
        src2 = inputs['src2'].value
        mask = inputs['mask'].value
        dst = cv2.bitwise_and(src1=src1, src2=src2, mask=mask)
        outputs['dst'] = Data(dst)

# cv2.bitwise_not
class OpenCVAuto2_Bitwise_not(NormalElement):
    name = 'Bitwise not'
    comment = '''bitwise_not(src[, dst[, mask]]) -> dst\n@brief  Inverts every bit of an array.\n\nThe function cv::bitwise_not calculates per-element bit-wise inversion of the input\narray:\n\f[\texttt{dst} (I) =  \neg \texttt{src} (I)\f]\nIn case of a floating-point input array, its machine-specific bit\nrepresentation (usually IEEE754-compliant) is used for the operation. In\ncase of multi-channel arrays, each channel is processed independently.\n@param src input array.\n@param dst output array that has the same size and type as the input\narray.\n@param mask optional operation mask, 8-bit single channel array, that\nspecifies elements of the output array to be changed.'''
    package = "Logic"

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

# cv2.bitwise_or
class OpenCVAuto2_Bitwise_or(NormalElement):
    name = 'Bitwise or'
    comment = '''bitwise_or(src1, src2[, dst[, mask]]) -> dst\n@brief Calculates the per-element bit-wise disjunction of two arrays or an\narray and a scalar.\n\nThe function cv::bitwise_or calculates the per-element bit-wise logical disjunction for:\n*   Two arrays when src1 and src2 have the same size:\n\f[\texttt{dst} (I) =  \texttt{src1} (I)  \vee \texttt{src2} (I) \quad \texttt{if mask} (I) \ne0\f]\n*   An array and a scalar when src2 is constructed from Scalar or has\nthe same number of elements as `src1.channels()`:\n\f[\texttt{dst} (I) =  \texttt{src1} (I)  \vee \texttt{src2} \quad \texttt{if mask} (I) \ne0\f]\n*   A scalar and an array when src1 is constructed from Scalar or has\nthe same number of elements as `src2.channels()`:\n\f[\texttt{dst} (I) =  \texttt{src1}  \vee \texttt{src2} (I) \quad \texttt{if mask} (I) \ne0\f]\nIn case of floating-point arrays, their machine-specific bit\nrepresentations (usually IEEE754-compliant) are used for the operation.\nIn case of multi-channel arrays, each channel is processed\nindependently. In the second and third cases above, the scalar is first\nconverted to the array type.\n@param src1 first input array or a scalar.\n@param src2 second input array or a scalar.\n@param dst output array that has the same size and type as the input\narrays.\n@param mask optional operation mask, 8-bit single channel array, that\nspecifies elements of the output array to be changed.'''
    package = "Logic"

    def get_attributes(self):
        return [Input('src1', 'src 1'),
                Input('src2', 'src 2'),
                Input('mask', 'mask', optional=True)], \
               [Output('dst', 'dst')], \
               []

    def process_inputs(self, inputs, outputs, parameters):
        src1 = inputs['src1'].value
        src2 = inputs['src2'].value
        mask = inputs['mask'].value
        dst = cv2.bitwise_or(src1=src1, src2=src2, mask=mask)
        outputs['dst'] = Data(dst)

# cv2.bitwise_xor
class OpenCVAuto2_Bitwise_xor(NormalElement):
    name = 'Bitwise xor'
    comment = '''bitwise_xor(src1, src2[, dst[, mask]]) -> dst\n@brief Calculates the per-element bit-wise "exclusive or" operation on two\narrays or an array and a scalar.\n\nThe function cv::bitwise_xor calculates the per-element bit-wise logical "exclusive-or"\noperation for:\n*   Two arrays when src1 and src2 have the same size:\n\f[\texttt{dst} (I) =  \texttt{src1} (I)  \oplus \texttt{src2} (I) \quad \texttt{if mask} (I) \ne0\f]\n*   An array and a scalar when src2 is constructed from Scalar or has\nthe same number of elements as `src1.channels()`:\n\f[\texttt{dst} (I) =  \texttt{src1} (I)  \oplus \texttt{src2} \quad \texttt{if mask} (I) \ne0\f]\n*   A scalar and an array when src1 is constructed from Scalar or has\nthe same number of elements as `src2.channels()`:\n\f[\texttt{dst} (I) =  \texttt{src1}  \oplus \texttt{src2} (I) \quad \texttt{if mask} (I) \ne0\f]\nIn case of floating-point arrays, their machine-specific bit\nrepresentations (usually IEEE754-compliant) are used for the operation.\nIn case of multi-channel arrays, each channel is processed\nindependently. In the 2nd and 3rd cases above, the scalar is first\nconverted to the array type.\n@param src1 first input array or a scalar.\n@param src2 second input array or a scalar.\n@param dst output array that has the same size and type as the input\narrays.\n@param mask optional operation mask, 8-bit single channel array, that\nspecifies elements of the output array to be changed.'''
    package = "Logic"

    def get_attributes(self):
        return [Input('src1', 'src 1'),
                Input('src2', 'src 2'),
                Input('mask', 'mask', optional=True)], \
               [Output('dst', 'dst')], \
               []

    def process_inputs(self, inputs, outputs, parameters):
        src1 = inputs['src1'].value
        src2 = inputs['src2'].value
        mask = inputs['mask'].value
        dst = cv2.bitwise_xor(src1=src1, src2=src2, mask=mask)
        outputs['dst'] = Data(dst)

# cv2.blur
class OpenCVAuto2_Blur(NormalElement):
    name = 'Blur'
    comment = '''blur(src, ksize[, dst[, anchor[, borderType]]]) -> dst\n@brief Blurs an image using the normalized box filter.\n\nThe function smooths an image using the kernel:\n\n\f[\texttt{K} =  \frac{1}{\texttt{ksize.width*ksize.height}} \begin{bmatrix} 1 & 1 & 1 &  \cdots & 1 & 1  \\ 1 & 1 & 1 &  \cdots & 1 & 1  \\ \hdotsfor{6} \\ 1 & 1 & 1 &  \cdots & 1 & 1  \\ \end{bmatrix}\f]\n\nThe call `blur(src, dst, ksize, anchor, borderType)` is equivalent to `boxFilter(src, dst, src.type(),\nanchor, true, borderType)`.\n\n@param src input image; it can have any number of channels, which are processed independently, but\nthe depth should be CV_8U, CV_16U, CV_16S, CV_32F or CV_64F.\n@param dst output image of the same size and type as src.\n@param ksize blurring kernel size.\n@param anchor anchor point; default value Point(-1,-1) means that the anchor is at the kernel\ncenter.\n@param borderType border mode used to extrapolate pixels outside of the image, see #BorderTypes\n@sa  boxFilter, bilateralFilter, GaussianBlur, medianBlur'''
    package = "Filters"

    def get_attributes(self):
        return [Input('src', 'src')], \
               [Output('dst', 'dst')], \
               [SizeParameter('ksize', 'ksize'),
                PointParameter('anchor', 'anchor'),
                ComboboxParameter('borderType', name='Border Type', values=[('BORDER_CONSTANT',0),('BORDER_REPLICATE',1),('BORDER_REFLECT',2),('BORDER_WRAP',3),('BORDER_DEFAULT',4),('BORDER_REFLECT101',4),('BORDER_REFLECT_101',4),('BORDER_TRANSPARENT',5),('BORDER_ISOLATED',16)])]

    def process_inputs(self, inputs, outputs, parameters):
        src = inputs['src'].value
        ksize = parameters['ksize']
        anchor = parameters['anchor']
        borderType = parameters['borderType']
        dst = cv2.blur(src=src, ksize=ksize, anchor=anchor, borderType=borderType)
        outputs['dst'] = Data(dst)

# cv2.calcBackProject
class OpenCVAuto2_CalcBackProject(NormalElement):
    name = 'Calc Back Project'
    comment = '''calcBackProject(images, channels, hist, ranges, scale[, dst]) -> dst\n@overload'''
    package = "Histogram"

    def get_attributes(self):
        return [Input('images', 'images'),
                Input('channels', 'channels'),
                Input('hist', 'hist'),
                Input('ranges', 'ranges')], \
               [Output('dst', 'dst')], \
               [FloatParameter('scale', 'scale')]

    def process_inputs(self, inputs, outputs, parameters):
        images = inputs['images'].value
        channels = inputs['channels'].value
        hist = inputs['hist'].value
        ranges = inputs['ranges'].value
        scale = parameters['scale']
        dst = cv2.calcBackProject(images=images, channels=channels, hist=hist, ranges=ranges, scale=scale)
        outputs['dst'] = Data(dst)

# cv2.calcOpticalFlowFarneback
class OpenCVAuto2_CalcOpticalFlowFarneback(NormalElement):
    name = 'Calc Optical Flow Farneback'
    comment = '''calcOpticalFlowFarneback(prev, next, flow, pyr_scale, levels, winsize, iterations, poly_n, poly_sigma, flags) -> flow\n@brief Computes a dense optical flow using the Gunnar Farneback's algorithm.\n\n@param prev first 8-bit single-channel input image.\n@param next second input image of the same size and the same type as prev.\n@param flow computed flow image that has the same size as prev and type CV_32FC2.\n@param pyr_scale parameter, specifying the image scale (\<1) to build pyramids for each image;\npyr_scale=0.5 means a classical pyramid, where each next layer is twice smaller than the previous\none.\n@param levels number of pyramid layers including the initial image; levels=1 means that no extra\nlayers are created and only the original images are used.\n@param winsize averaging window size; larger values increase the algorithm robustness to image\nnoise and give more chances for fast motion detection, but yield more blurred motion field.\n@param iterations number of iterations the algorithm does at each pyramid level.\n@param poly_n size of the pixel neighborhood used to find polynomial expansion in each pixel;\nlarger values mean that the image will be approximated with smoother surfaces, yielding more\nrobust algorithm and more blurred motion field, typically poly_n =5 or 7.\n@param poly_sigma standard deviation of the Gaussian that is used to smooth derivatives used as a\nbasis for the polynomial expansion; for poly_n=5, you can set poly_sigma=1.1, for poly_n=7, a\ngood value would be poly_sigma=1.5.\n@param flags operation flags that can be a combination of the following:\n-   **OPTFLOW_USE_INITIAL_FLOW** uses the input flow as an initial flow approximation.\n-   **OPTFLOW_FARNEBACK_GAUSSIAN** uses the Gaussian \f$\texttt{winsize}\times\texttt{winsize}\f$\nfilter instead of a box filter of the same size for optical flow estimation; usually, this\noption gives z more accurate flow than with a box filter, at the cost of lower speed;\nnormally, winsize for a Gaussian window should be set to a larger value to achieve the same\nlevel of robustness.\n\nThe function finds an optical flow for each prev pixel using the @cite Farneback2003 algorithm so that\n\n\f[\texttt{prev} (y,x)  \sim \texttt{next} ( y + \texttt{flow} (y,x)[1],  x + \texttt{flow} (y,x)[0])\f]\n\n@note\n\n-   An example using the optical flow algorithm described by Gunnar Farneback can be found at\nopencv_source_code/samples/cpp/fback.cpp\n-   (Python) An example using the optical flow algorithm described by Gunnar Farneback can be\nfound at opencv_source_code/samples/python/opt_flow.py'''
    package = "Object tracking"

    def get_attributes(self):
        return [Input('prev', 'prev'),
                Input('next', 'next'),
                Input('flow', 'flow')], \
               [Output('flow', 'flow')], \
               [FloatParameter('pyr_scale', 'pyr scale'),
                IntParameter('levels', 'levels'),
                SizeParameter('winsize', 'winsize'),
                IntParameter('iterations', 'iterations', min_=0),
                IntParameter('poly_n', 'poly n'),
                FloatParameter('poly_sigma', 'poly sigma'),
                ComboboxParameter('flags', name='flags', values=[('OPTFLOW_USE_INITIAL_FLOW',4),('OPTFLOW_LK_GET_MIN_EIGENVALS',8),('OPTFLOW_FARNEBACK_GAUSSIAN',256)])]

    def process_inputs(self, inputs, outputs, parameters):
        prev = inputs['prev'].value
        next = inputs['next'].value
        flow = inputs['flow'].value.copy()
        pyr_scale = parameters['pyr_scale']
        levels = parameters['levels']
        winsize = parameters['winsize']
        iterations = parameters['iterations']
        poly_n = parameters['poly_n']
        poly_sigma = parameters['poly_sigma']
        flags = parameters['flags']
        flow = cv2.calcOpticalFlowFarneback(prev=prev, next=next, flow=flow, pyr_scale=pyr_scale, levels=levels, winsize=winsize, iterations=iterations, poly_n=poly_n, poly_sigma=poly_sigma, flags=flags)
        outputs['flow'] = Data(flow)

# cv2.calibrateCameraExtended
class OpenCVAuto2_CalibrateCameraExtended(NormalElement):
    name = 'Calibrate Camera Extended'
    comment = '''calibrateCameraExtended(objectPoints, imagePoints, imageSize, cameraMatrix, distCoeffs[, rvecs[, tvecs[, stdDeviationsIntrinsics[, stdDeviationsExtrinsics[, perViewErrors[, flags[, criteria]]]]]]]) -> retval, cameraMatrix, distCoeffs, rvecs, tvecs, stdDeviationsIntrinsics, stdDeviationsExtrinsics, perViewErrors\n@brief Finds the camera intrinsic and extrinsic parameters from several views of a calibration pattern.\n\n@param objectPoints In the new interface it is a vector of vectors of calibration pattern points in\nthe calibration pattern coordinate space (e.g. std::vector<std::vector<cv::Vec3f>>). The outer\nvector contains as many elements as the number of the pattern views. If the same calibration pattern\nis shown in each view and it is fully visible, all the vectors will be the same. Although, it is\npossible to use partially occluded patterns, or even different patterns in different views. Then,\nthe vectors will be different. The points are 3D, but since they are in a pattern coordinate system,\nthen, if the rig is planar, it may make sense to put the model to a XY coordinate plane so that\nZ-coordinate of each input object point is 0.\nIn the old interface all the vectors of object points from different views are concatenated\ntogether.\n@param imagePoints In the new interface it is a vector of vectors of the projections of calibration\npattern points (e.g. std::vector<std::vector<cv::Vec2f>>). imagePoints.size() and\nobjectPoints.size() and imagePoints[i].size() must be equal to objectPoints[i].size() for each i.\nIn the old interface all the vectors of object points from different views are concatenated\ntogether.\n@param imageSize Size of the image used only to initialize the intrinsic camera matrix.\n@param cameraMatrix Output 3x3 floating-point camera matrix\n\f$A = \vecthreethree{f_x}{0}{c_x}{0}{f_y}{c_y}{0}{0}{1}\f$ . If CV\_CALIB\_USE\_INTRINSIC\_GUESS\nand/or CALIB_FIX_ASPECT_RATIO are specified, some or all of fx, fy, cx, cy must be\ninitialized before calling the function.\n@param distCoeffs Output vector of distortion coefficients\n\f$(k_1, k_2, p_1, p_2[, k_3[, k_4, k_5, k_6 [, s_1, s_2, s_3, s_4[, \tau_x, \tau_y]]]])\f$ of\n4, 5, 8, 12 or 14 elements.\n@param rvecs Output vector of rotation vectors (see Rodrigues ) estimated for each pattern view\n(e.g. std::vector<cv::Mat>>). That is, each k-th rotation vector together with the corresponding\nk-th translation vector (see the next output parameter description) brings the calibration pattern\nfrom the model coordinate space (in which object points are specified) to the world coordinate\nspace, that is, a real position of the calibration pattern in the k-th pattern view (k=0.. *M* -1).\n@param tvecs Output vector of translation vectors estimated for each pattern view.\n@param stdDeviationsIntrinsics Output vector of standard deviations estimated for intrinsic parameters.\nOrder of deviations values:\n\f$(f_x, f_y, c_x, c_y, k_1, k_2, p_1, p_2, k_3, k_4, k_5, k_6 , s_1, s_2, s_3,\ns_4, \tau_x, \tau_y)\f$ If one of parameters is not estimated, it's deviation is equals to zero.\n@param stdDeviationsExtrinsics Output vector of standard deviations estimated for extrinsic parameters.\nOrder of deviations values: \f$(R_1, T_1, \dotsc , R_M, T_M)\f$ where M is number of pattern views,\n\f$R_i, T_i\f$ are concatenated 1x3 vectors.\n@param perViewErrors Output vector of the RMS re-projection error estimated for each pattern view.\n@param flags Different flags that may be zero or a combination of the following values:\n-   **CALIB_USE_INTRINSIC_GUESS** cameraMatrix contains valid initial values of\nfx, fy, cx, cy that are optimized further. Otherwise, (cx, cy) is initially set to the image\ncenter ( imageSize is used), and focal distances are computed in a least-squares fashion.\nNote, that if intrinsic parameters are known, there is no need to use this function just to\nestimate extrinsic parameters. Use solvePnP instead.\n-   **CALIB_FIX_PRINCIPAL_POINT** The principal point is not changed during the global\noptimization. It stays at the center or at a different location specified when\nCALIB_USE_INTRINSIC_GUESS is set too.\n-   **CALIB_FIX_ASPECT_RATIO** The functions considers only fy as a free parameter. The\nratio fx/fy stays the same as in the input cameraMatrix . When\nCALIB_USE_INTRINSIC_GUESS is not set, the actual input values of fx and fy are\nignored, only their ratio is computed and used further.\n-   **CALIB_ZERO_TANGENT_DIST** Tangential distortion coefficients \f$(p_1, p_2)\f$ are set\nto zeros and stay zero.\n-   **CALIB_FIX_K1,...,CALIB_FIX_K6** The corresponding radial distortion\ncoefficient is not changed during the optimization. If CALIB_USE_INTRINSIC_GUESS is\nset, the coefficient from the supplied distCoeffs matrix is used. Otherwise, it is set to 0.\n-   **CALIB_RATIONAL_MODEL** Coefficients k4, k5, and k6 are enabled. To provide the\nbackward compatibility, this extra flag should be explicitly specified to make the\ncalibration function use the rational model and return 8 coefficients. If the flag is not\nset, the function computes and returns only 5 distortion coefficients.\n-   **CALIB_THIN_PRISM_MODEL** Coefficients s1, s2, s3 and s4 are enabled. To provide the\nbackward compatibility, this extra flag should be explicitly specified to make the\ncalibration function use the thin prism model and return 12 coefficients. If the flag is not\nset, the function computes and returns only 5 distortion coefficients.\n-   **CALIB_FIX_S1_S2_S3_S4** The thin prism distortion coefficients are not changed during\nthe optimization. If CALIB_USE_INTRINSIC_GUESS is set, the coefficient from the\nsupplied distCoeffs matrix is used. Otherwise, it is set to 0.\n-   **CALIB_TILTED_MODEL** Coefficients tauX and tauY are enabled. To provide the\nbackward compatibility, this extra flag should be explicitly specified to make the\ncalibration function use the tilted sensor model and return 14 coefficients. If the flag is not\nset, the function computes and returns only 5 distortion coefficients.\n-   **CALIB_FIX_TAUX_TAUY** The coefficients of the tilted sensor model are not changed during\nthe optimization. If CALIB_USE_INTRINSIC_GUESS is set, the coefficient from the\nsupplied distCoeffs matrix is used. Otherwise, it is set to 0.\n@param criteria Termination criteria for the iterative optimization algorithm.\n\n@return the overall RMS re-projection error.\n\nThe function estimates the intrinsic camera parameters and extrinsic parameters for each of the\nviews. The algorithm is based on @cite Zhang2000 and @cite BouguetMCT . The coordinates of 3D object\npoints and their corresponding 2D projections in each view must be specified. That may be achieved\nby using an object with a known geometry and easily detectable feature points. Such an object is\ncalled a calibration rig or calibration pattern, and OpenCV has built-in support for a chessboard as\na calibration rig (see findChessboardCorners ). Currently, initialization of intrinsic parameters\n(when CALIB_USE_INTRINSIC_GUESS is not set) is only implemented for planar calibration\npatterns (where Z-coordinates of the object points must be all zeros). 3D calibration rigs can also\nbe used as long as initial cameraMatrix is provided.\n\nThe algorithm performs the following steps:\n\n-   Compute the initial intrinsic parameters (the option only available for planar calibration\npatterns) or read them from the input parameters. The distortion coefficients are all set to\nzeros initially unless some of CALIB_FIX_K? are specified.\n\n-   Estimate the initial camera pose as if the intrinsic parameters have been already known. This is\ndone using solvePnP .\n\n-   Run the global Levenberg-Marquardt optimization algorithm to minimize the reprojection error,\nthat is, the total sum of squared distances between the observed feature points imagePoints and\nthe projected (using the current estimates for camera parameters and the poses) object points\nobjectPoints. See projectPoints for details.\n\n@note\nIf you use a non-square (=non-NxN) grid and findChessboardCorners for calibration, and\ncalibrateCamera returns bad values (zero distortion coefficients, an image center very far from\n(w/2-0.5,h/2-0.5), and/or large differences between \f$f_x\f$ and \f$f_y\f$ (ratios of 10:1 or more)),\nthen you have probably used patternSize=cvSize(rows,cols) instead of using\npatternSize=cvSize(cols,rows) in findChessboardCorners .\n\n@sa\nfindChessboardCorners, solvePnP, initCameraMatrix2D, stereoCalibrate, undistort'''
    package = "3D calibration & reconstruction"

    def get_attributes(self):
        return [Input('objectPoints', 'Object Points'),
                Input('imagePoints', 'Image Points'),
                Input('imageSize', 'Image Size'),
                Input('cameraMatrix', 'Camera Matrix'),
                Input('distCoeffs', 'Dist Coeffs')], \
               [Output('cameraMatrix', 'Camera Matrix'),
                Output('distCoeffs', 'Dist Coeffs'),
                Output('rvecs', 'rvecs'),
                Output('tvecs', 'tvecs'),
                Output('stdDeviationsIntrinsics', 'Std Deviations Intrinsics'),
                Output('stdDeviationsExtrinsics', 'Std Deviations Extrinsics'),
                Output('perViewErrors', 'Per View Errors')], \
               [ComboboxParameter('flags', name='flags', values=[('CALIB_USE_INTRINSIC_GUESS',1),('CALIB_CB_ADAPTIVE_THRESH',1),('CALIB_CB_SYMMETRIC_GRID',1),('CALIB_FIX_ASPECT_RATIO',2),('CALIB_CB_ASYMMETRIC_GRID',2),('CALIB_CB_NORMALIZE_IMAGE',2),('CALIB_FIX_PRINCIPAL_POINT',4),('CALIB_CB_CLUSTERING',4),('CALIB_CB_FILTER_QUADS',4),('CALIB_ZERO_TANGENT_DIST',8),('CALIB_CB_FAST_CHECK',8),('CALIB_FIX_FOCAL_LENGTH',16),('CALIB_FIX_K1',32),('CALIB_FIX_K2',64),('CALIB_FIX_K3',128),('CALIB_FIX_INTRINSIC',256),('CALIB_SAME_FOCAL_LENGTH',512),('CALIB_ZERO_DISPARITY',1024),('CALIB_FIX_K4',2048),('CALIB_FIX_K5',4096),('CALIB_FIX_K6',8192),('CALIB_RATIONAL_MODEL',16384),('CALIB_THIN_PRISM_MODEL',32768),('CALIB_FIX_S1_S2_S3_S4',65536),('CALIB_USE_LU',131072),('CALIB_TILTED_MODEL',262144),('CALIB_FIX_TAUX_TAUY',524288),('CALIB_USE_QR',1048576),('CALIB_FIX_TANGENT_DIST',2097152),('CALIB_USE_EXTRINSIC_GUESS',4194304)])]

    def process_inputs(self, inputs, outputs, parameters):
        objectPoints = inputs['objectPoints'].value
        imagePoints = inputs['imagePoints'].value
        imageSize = inputs['imageSize'].value
        cameraMatrix = inputs['cameraMatrix'].value.copy()
        distCoeffs = inputs['distCoeffs'].value.copy()
        flags = parameters['flags']
        retval, cameraMatrix, distCoeffs, rvecs, tvecs, stdDeviationsIntrinsics, stdDeviationsExtrinsics, perViewErrors = cv2.calibrateCameraExtended(objectPoints=objectPoints, imagePoints=imagePoints, imageSize=imageSize, cameraMatrix=cameraMatrix, distCoeffs=distCoeffs, flags=flags)
        outputs['cameraMatrix'] = Data(cameraMatrix)
        outputs['distCoeffs'] = Data(distCoeffs)
        outputs['rvecs'] = Data(rvecs)
        outputs['tvecs'] = Data(tvecs)
        outputs['stdDeviationsIntrinsics'] = Data(stdDeviationsIntrinsics)
        outputs['stdDeviationsExtrinsics'] = Data(stdDeviationsExtrinsics)
        outputs['perViewErrors'] = Data(perViewErrors)

# cv2.calibrationMatrixValues
class OpenCVAuto2_CalibrationMatrixValues(NormalElement):
    name = 'Calibration Matrix Values'
    comment = '''calibrationMatrixValues(cameraMatrix, imageSize, apertureWidth, apertureHeight) -> fovx, fovy, focalLength, principalPoint, aspectRatio\n@brief Computes useful camera characteristics from the camera matrix.\n\n@param cameraMatrix Input camera matrix that can be estimated by calibrateCamera or\nstereoCalibrate .\n@param imageSize Input image size in pixels.\n@param apertureWidth Physical width in mm of the sensor.\n@param apertureHeight Physical height in mm of the sensor.\n@param fovx Output field of view in degrees along the horizontal sensor axis.\n@param fovy Output field of view in degrees along the vertical sensor axis.\n@param focalLength Focal length of the lens in mm.\n@param principalPoint Principal point in mm.\n@param aspectRatio \f$f_y/f_x\f$\n\nThe function computes various useful camera characteristics from the previously estimated camera\nmatrix.\n\n@note\nDo keep in mind that the unity measure 'mm' stands for whatever unit of measure one chooses for\nthe chessboard pitch (it can thus be any value).'''
    package = "3D calibration & reconstruction"

    def get_attributes(self):
        return [Input('cameraMatrix', 'Camera Matrix'),
                Input('imageSize', 'Image Size')], \
               [Output('fovx', 'fovx'),
                Output('fovy', 'fovy'),
                Output('focalLength', 'Focal Length'),
                Output('principalPoint', 'Principal Point'),
                Output('aspectRatio', 'Aspect Ratio')], \
               [FloatParameter('apertureWidth', 'Aperture Width'),
                FloatParameter('apertureHeight', 'Aperture Height')]

    def process_inputs(self, inputs, outputs, parameters):
        cameraMatrix = inputs['cameraMatrix'].value
        imageSize = inputs['imageSize'].value
        apertureWidth = parameters['apertureWidth']
        apertureHeight = parameters['apertureHeight']
        fovx, fovy, focalLength, principalPoint, aspectRatio = cv2.calibrationMatrixValues(cameraMatrix=cameraMatrix, imageSize=imageSize, apertureWidth=apertureWidth, apertureHeight=apertureHeight)
        outputs['fovx'] = Data(fovx)
        outputs['fovy'] = Data(fovy)
        outputs['focalLength'] = Data(focalLength)
        outputs['principalPoint'] = Data(principalPoint)
        outputs['aspectRatio'] = Data(aspectRatio)

# cv2.circle
class OpenCVAuto2_Circle(NormalElement):
    name = 'Circle'
    comment = '''circle(img, center, radius, color[, thickness[, lineType[, shift]]]) -> img\n@brief Draws a circle.\n\nThe function cv::circle draws a simple or filled circle with a given center and radius.\n@param img Image where the circle is drawn.\n@param center Center of the circle.\n@param radius Radius of the circle.\n@param color Circle color.\n@param thickness Thickness of the circle outline, if positive. Negative values, like #FILLED,\nmean that a filled circle is to be drawn.\n@param lineType Type of the circle boundary. See #LineTypes\n@param shift Number of fractional bits in the coordinates of the center and in the radius value.'''
    package = "Drawing"

    def get_attributes(self):
        return [Input('img', 'img')], \
               [Output('img', 'img')], \
               [PointParameter('center', 'center'),
                IntParameter('radius', 'radius', min_=1, max_=1000),
                ScalarParameter('color', 'color'),
                IntParameter('thickness', 'thickness', min_=-1, max_=100),
                ComboboxParameter('lineType', name='Line Type', values=[('LINE_4',4),('LINE_8',8),('LINE_AA',16)]),
                IntParameter('shift', 'shift')]

    def process_inputs(self, inputs, outputs, parameters):
        img = inputs['img'].value.copy()
        center = parameters['center']
        radius = parameters['radius']
        color = parameters['color']
        thickness = parameters['thickness']
        lineType = parameters['lineType']
        shift = parameters['shift']
        img = cv2.circle(img=img, center=center, radius=radius, color=color, thickness=thickness, lineType=lineType, shift=shift)
        outputs['img'] = Data(img)

# cv2.colorChange
class OpenCVAuto2_ColorChange(NormalElement):
    name = 'Color Change'
    comment = '''colorChange(src, mask[, dst[, red_mul[, green_mul[, blue_mul]]]]) -> dst\n@brief Given an original color image, two differently colored versions of this image can be mixed\nseamlessly.\n\n@param src Input 8-bit 3-channel image.\n@param mask Input 8-bit 1 or 3-channel image.\n@param dst Output image with the same size and type as src .\n@param red_mul R-channel multiply factor.\n@param green_mul G-channel multiply factor.\n@param blue_mul B-channel multiply factor.\n\nMultiplication factor is between .5 to 2.5.'''
    package = "Color"

    def get_attributes(self):
        return [Input('src', 'src'),
                Input('mask', 'mask')], \
               [Output('dst', 'dst')], \
               [FloatParameter('red_mul', 'red mul'),
                FloatParameter('green_mul', 'green mul'),
                FloatParameter('blue_mul', 'blue mul')]

    def process_inputs(self, inputs, outputs, parameters):
        src = inputs['src'].value
        mask = inputs['mask'].value
        red_mul = parameters['red_mul']
        green_mul = parameters['green_mul']
        blue_mul = parameters['blue_mul']
        dst = cv2.colorChange(src=src, mask=mask, red_mul=red_mul, green_mul=green_mul, blue_mul=blue_mul)
        outputs['dst'] = Data(dst)

# cv2.compare
class OpenCVAuto2_Compare(NormalElement):
    name = 'Compare'
    comment = '''compare(src1, src2, cmpop[, dst]) -> dst\n@brief Performs the per-element comparison of two arrays or an array and scalar value.\n\nThe function compares:\n*   Elements of two arrays when src1 and src2 have the same size:\n\f[\texttt{dst} (I) =  \texttt{src1} (I)  \,\texttt{cmpop}\, \texttt{src2} (I)\f]\n*   Elements of src1 with a scalar src2 when src2 is constructed from\nScalar or has a single element:\n\f[\texttt{dst} (I) =  \texttt{src1}(I) \,\texttt{cmpop}\,  \texttt{src2}\f]\n*   src1 with elements of src2 when src1 is constructed from Scalar or\nhas a single element:\n\f[\texttt{dst} (I) =  \texttt{src1}  \,\texttt{cmpop}\, \texttt{src2} (I)\f]\nWhen the comparison result is true, the corresponding element of output\narray is set to 255. The comparison operations can be replaced with the\nequivalent matrix expressions:\n@code{.cpp}\nMat dst1 = src1 >= src2;\nMat dst2 = src1 < 8;\n...\n@endcode\n@param src1 first input array or a scalar; when it is an array, it must have a single channel.\n@param src2 second input array or a scalar; when it is an array, it must have a single channel.\n@param dst output array of type ref CV_8U that has the same size and the same number of channels as\nthe input arrays.\n@param cmpop a flag, that specifies correspondence between the arrays (cv::CmpTypes)\n@sa checkRange, min, max, threshold'''
    package = "Matrix miscellaneous"

    def get_attributes(self):
        return [Input('src1', 'src 1'),
                Input('src2', 'src 2')], \
               [Output('dst', 'dst')], \
               [IntParameter('cmpop', 'cmpop')]

    def process_inputs(self, inputs, outputs, parameters):
        src1 = inputs['src1'].value
        src2 = inputs['src2'].value
        cmpop = parameters['cmpop']
        dst = cv2.compare(src1=src1, src2=src2, cmpop=cmpop)
        outputs['dst'] = Data(dst)

# cv2.computeCorrespondEpilines
class OpenCVAuto2_ComputeCorrespondEpilines(NormalElement):
    name = 'Compute Correspond Epilines'
    comment = '''computeCorrespondEpilines(points, whichImage, F[, lines]) -> lines\n@brief For points in an image of a stereo pair, computes the corresponding epilines in the other image.\n\n@param points Input points. \f$N \times 1\f$ or \f$1 \times N\f$ matrix of type CV_32FC2 or\nvector\<Point2f\> .\n@param whichImage Index of the image (1 or 2) that contains the points .\n@param F Fundamental matrix that can be estimated using findFundamentalMat or stereoRectify .\n@param lines Output vector of the epipolar lines corresponding to the points in the other image.\nEach line \f$ax + by + c=0\f$ is encoded by 3 numbers \f$(a, b, c)\f$ .\n\nFor every point in one of the two images of a stereo pair, the function finds the equation of the\ncorresponding epipolar line in the other image.\n\nFrom the fundamental matrix definition (see findFundamentalMat ), line \f$l^{(2)}_i\f$ in the second\nimage for the point \f$p^{(1)}_i\f$ in the first image (when whichImage=1 ) is computed as:\n\n\f[l^{(2)}_i = F p^{(1)}_i\f]\n\nAnd vice versa, when whichImage=2, \f$l^{(1)}_i\f$ is computed from \f$p^{(2)}_i\f$ as:\n\n\f[l^{(1)}_i = F^T p^{(2)}_i\f]\n\nLine coefficients are defined up to a scale. They are normalized so that \f$a_i^2+b_i^2=1\f$ .'''
    package = "3D calibration & reconstruction"

    def get_attributes(self):
        return [Input('points', 'points'),
                Input('whichImage', 'Which Image'),
                Input('F', 'F')], \
               [Output('lines', 'lines')], \
               []

    def process_inputs(self, inputs, outputs, parameters):
        points = inputs['points'].value
        whichImage = inputs['whichImage'].value
        F = inputs['F'].value
        lines = cv2.computeCorrespondEpilines(points=points, whichImage=whichImage, F=F)
        outputs['lines'] = Data(lines)

# cv2.connectedComponents
class OpenCVAuto2_ConnectedComponents(NormalElement):
    name = 'Connected Components'
    comment = '''connectedComponents(image[, labels[, connectivity[, ltype]]]) -> retval, labels\n@overload\n\n@param image the 8-bit single-channel image to be labeled\n@param labels destination labeled image\n@param connectivity 8 or 4 for 8-way or 4-way connectivity respectively\n@param ltype output image label type. Currently CV_32S and CV_16U are supported.'''
    package = "Shape"

    def get_attributes(self):
        return [Input('image', 'image')], \
               [Output('labels', 'labels')], \
               [IntParameter('connectivity', 'connectivity'),
                IntParameter('ltype', 'ltype')]

    def process_inputs(self, inputs, outputs, parameters):
        image = inputs['image'].value
        connectivity = parameters['connectivity']
        ltype = parameters['ltype']
        retval, labels = cv2.connectedComponents(image=image, connectivity=connectivity, ltype=ltype)
        outputs['labels'] = Data(labels)

# cv2.connectedComponentsWithStats
class OpenCVAuto2_ConnectedComponentsWithStats(NormalElement):
    name = 'Connected Components With Stats'
    comment = '''connectedComponentsWithStats(image[, labels[, stats[, centroids[, connectivity[, ltype]]]]]) -> retval, labels, stats, centroids\n@overload\n@param image the 8-bit single-channel image to be labeled\n@param labels destination labeled image\n@param stats statistics output for each label, including the background label, see below for\navailable statistics. Statistics are accessed via stats(label, COLUMN) where COLUMN is one of\n#ConnectedComponentsTypes. The data type is CV_32S.\n@param centroids centroid output for each label, including the background label. Centroids are\naccessed via centroids(label, 0) for x and centroids(label, 1) for y. The data type CV_64F.\n@param connectivity 8 or 4 for 8-way or 4-way connectivity respectively\n@param ltype output image label type. Currently CV_32S and CV_16U are supported.'''
    package = "Shape"

    def get_attributes(self):
        return [Input('image', 'image')], \
               [Output('labels', 'labels'),
                Output('stats', 'stats'),
                Output('centroids', 'centroids')], \
               [IntParameter('connectivity', 'connectivity'),
                IntParameter('ltype', 'ltype')]

    def process_inputs(self, inputs, outputs, parameters):
        image = inputs['image'].value
        connectivity = parameters['connectivity']
        ltype = parameters['ltype']
        retval, labels, stats, centroids = cv2.connectedComponentsWithStats(image=image, connectivity=connectivity, ltype=ltype)
        outputs['labels'] = Data(labels)
        outputs['stats'] = Data(stats)
        outputs['centroids'] = Data(centroids)

# cv2.convertFp16
class OpenCVAuto2_ConvertFp16(NormalElement):
    name = 'Convert Fp 16'
    comment = '''convertFp16(src[, dst]) -> dst\n@brief Converts an array to half precision floating number.\n\nThis function converts FP32 (single precision floating point) from/to FP16 (half precision floating point). CV_16S format is used to represent FP16 data.\nThere are two use modes (src -> dst): CV_32F -> CV_16S and CV_16S -> CV_32F. The input array has to have type of CV_32F or\nCV_16S to represent the bit depth. If the input array is neither of them, the function will raise an error.\nThe format of half precision floating point is defined in IEEE 754-2008.\n\n@param src input array.\n@param dst output array.'''
    package = "Type conversion"

    def get_attributes(self):
        return [Input('src', 'src')], \
               [Output('dst', 'dst')], \
               []

    def process_inputs(self, inputs, outputs, parameters):
        src = inputs['src'].value
        dst = cv2.convertFp16(src=src)
        outputs['dst'] = Data(dst)

# cv2.convertPointsFromHomogeneous
class OpenCVAuto2_ConvertPointsFromHomogeneous(NormalElement):
    name = 'Convert Points From Homogeneous'
    comment = '''convertPointsFromHomogeneous(src[, dst]) -> dst\n@brief Converts points from homogeneous to Euclidean space.\n\n@param src Input vector of N-dimensional points.\n@param dst Output vector of N-1-dimensional points.\n\nThe function converts points homogeneous to Euclidean space using perspective projection. That is,\neach point (x1, x2, ... x(n-1), xn) is converted to (x1/xn, x2/xn, ..., x(n-1)/xn). When xn=0, the\noutput point coordinates will be (0,0,0,...).'''
    package = "3D calibration & reconstruction"

    def get_attributes(self):
        return [Input('src', 'src')], \
               [Output('dst', 'dst')], \
               []

    def process_inputs(self, inputs, outputs, parameters):
        src = inputs['src'].value
        dst = cv2.convertPointsFromHomogeneous(src=src)
        outputs['dst'] = Data(dst)

# cv2.convertPointsToHomogeneous
class OpenCVAuto2_ConvertPointsToHomogeneous(NormalElement):
    name = 'Convert Points To Homogeneous'
    comment = '''convertPointsToHomogeneous(src[, dst]) -> dst\n@brief Converts points from Euclidean to homogeneous space.\n\n@param src Input vector of N-dimensional points.\n@param dst Output vector of N+1-dimensional points.\n\nThe function converts points from Euclidean to homogeneous space by appending 1's to the tuple of\npoint coordinates. That is, each point (x1, x2, ..., xn) is converted to (x1, x2, ..., xn, 1).'''
    package = "3D calibration & reconstruction"

    def get_attributes(self):
        return [Input('src', 'src')], \
               [Output('dst', 'dst')], \
               []

    def process_inputs(self, inputs, outputs, parameters):
        src = inputs['src'].value
        dst = cv2.convertPointsToHomogeneous(src=src)
        outputs['dst'] = Data(dst)

# cv2.convertScaleAbs
class OpenCVAuto2_ConvertScaleAbs(NormalElement):
    name = 'Convert Scale Abs'
    comment = '''convertScaleAbs(src[, dst[, alpha[, beta]]]) -> dst\n@brief Scales, calculates absolute values, and converts the result to 8-bit.\n\nOn each element of the input array, the function convertScaleAbs\nperforms three operations sequentially: scaling, taking an absolute\nvalue, conversion to an unsigned 8-bit type:\n\f[\texttt{dst} (I)= \texttt{saturate\_cast<uchar>} (| \texttt{src} (I)* \texttt{alpha} +  \texttt{beta} |)\f]\nIn case of multi-channel arrays, the function processes each channel\nindependently. When the output is not 8-bit, the operation can be\nemulated by calling the Mat::convertTo method (or by using matrix\nexpressions) and then by calculating an absolute value of the result.\nFor example:\n@code{.cpp}\nMat_<float> A(30,30);\nrandu(A, Scalar(-100), Scalar(100));\nMat_<float> B = A*5 + 3;\nB = abs(B);\n// Mat_<float> B = abs(A*5+3) will also do the job,\n// but it will allocate a temporary matrix\n@endcode\n@param src input array.\n@param dst output array.\n@param alpha optional scale factor.\n@param beta optional delta added to the scaled values.\n@sa  Mat::convertTo, cv::abs(const Mat&)'''
    package = "Type conversion"

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

# cv2.convexityDefects
class OpenCVAuto2_ConvexityDefects(NormalElement):
    name = 'Convexity Defects'
    comment = '''convexityDefects(contour, convexhull[, convexityDefects]) -> convexityDefects\n@brief Finds the convexity defects of a contour.\n\nThe figure below displays convexity defects of a hand contour:\n\n![image](pics/defects.png)\n\n@param contour Input contour.\n@param convexhull Convex hull obtained using convexHull that should contain indices of the contour\npoints that make the hull.\n@param convexityDefects The output vector of convexity defects. In C++ and the new Python/Java\ninterface each convexity defect is represented as 4-element integer vector (a.k.a. #Vec4i):\n(start_index, end_index, farthest_pt_index, fixpt_depth), where indices are 0-based indices\nin the original contour of the convexity defect beginning, end and the farthest point, and\nfixpt_depth is fixed-point approximation (with 8 fractional bits) of the distance between the\nfarthest contour point and the hull. That is, to get the floating-point value of the depth will be\nfixpt_depth/256.0.'''
    package = "Shape"

    def get_attributes(self):
        return [Input('contour', 'contour'),
                Input('convexhull', 'convexhull')], \
               [Output('convexityDefects', 'Convexity Defects')], \
               []

    def process_inputs(self, inputs, outputs, parameters):
        contour = inputs['contour'].value
        convexhull = inputs['convexhull'].value
        convexityDefects = cv2.convexityDefects(contour=contour, convexhull=convexhull)
        outputs['convexityDefects'] = Data(convexityDefects)

# cv2.cornerEigenValsAndVecs
class OpenCVAuto2_CornerEigenValsAndVecs(NormalElement):
    name = 'Corner Eigen Vals And Vecs'
    comment = '''cornerEigenValsAndVecs(src, blockSize, ksize[, dst[, borderType]]) -> dst\n@brief Calculates eigenvalues and eigenvectors of image blocks for corner detection.\n\nFor every pixel \f$p\f$ , the function cornerEigenValsAndVecs considers a blockSize \f$\times\f$ blockSize\nneighborhood \f$S(p)\f$ . It calculates the covariation matrix of derivatives over the neighborhood as:\n\n\f[M =  \begin{bmatrix} \sum _{S(p)}(dI/dx)^2 &  \sum _{S(p)}dI/dx dI/dy  \\ \sum _{S(p)}dI/dx dI/dy &  \sum _{S(p)}(dI/dy)^2 \end{bmatrix}\f]\n\nwhere the derivatives are computed using the Sobel operator.\n\nAfter that, it finds eigenvectors and eigenvalues of \f$M\f$ and stores them in the destination image as\n\f$(\lambda_1, \lambda_2, x_1, y_1, x_2, y_2)\f$ where\n\n-   \f$\lambda_1, \lambda_2\f$ are the non-sorted eigenvalues of \f$M\f$\n-   \f$x_1, y_1\f$ are the eigenvectors corresponding to \f$\lambda_1\f$\n-   \f$x_2, y_2\f$ are the eigenvectors corresponding to \f$\lambda_2\f$\n\nThe output of the function can be used for robust edge or corner detection.\n\n@param src Input single-channel 8-bit or floating-point image.\n@param dst Image to store the results. It has the same size as src and the type CV_32FC(6) .\n@param blockSize Neighborhood size (see details below).\n@param ksize Aperture parameter for the Sobel operator.\n@param borderType Pixel extrapolation method. See #BorderTypes.\n\n@sa  cornerMinEigenVal, cornerHarris, preCornerDetect'''
    package = "Features 2D"

    def get_attributes(self):
        return [Input('src', 'src')], \
               [Output('dst', 'dst')], \
               [SizeParameter('blockSize', 'Block Size'),
                SizeParameter('ksize', 'ksize'),
                ComboboxParameter('borderType', name='Border Type', values=[('BORDER_CONSTANT',0),('BORDER_REPLICATE',1),('BORDER_REFLECT',2),('BORDER_WRAP',3),('BORDER_DEFAULT',4),('BORDER_REFLECT101',4),('BORDER_REFLECT_101',4),('BORDER_TRANSPARENT',5),('BORDER_ISOLATED',16)])]

    def process_inputs(self, inputs, outputs, parameters):
        src = inputs['src'].value
        blockSize = parameters['blockSize']
        ksize = parameters['ksize']
        borderType = parameters['borderType']
        dst = cv2.cornerEigenValsAndVecs(src=src, blockSize=blockSize, ksize=ksize, borderType=borderType)
        outputs['dst'] = Data(dst)

# cv2.cornerHarris
class OpenCVAuto2_CornerHarris(NormalElement):
    name = 'Corner Harris'
    comment = '''cornerHarris(src, blockSize, ksize, k[, dst[, borderType]]) -> dst\n@brief Harris corner detector.\n\nThe function runs the Harris corner detector on the image. Similarly to cornerMinEigenVal and\ncornerEigenValsAndVecs , for each pixel \f$(x, y)\f$ it calculates a \f$2\times2\f$ gradient covariance\nmatrix \f$M^{(x,y)}\f$ over a \f$\texttt{blockSize} \times \texttt{blockSize}\f$ neighborhood. Then, it\ncomputes the following characteristic:\n\n\f[\texttt{dst} (x,y) =  \mathrm{det} M^{(x,y)} - k  \cdot \left ( \mathrm{tr} M^{(x,y)} \right )^2\f]\n\nCorners in the image can be found as the local maxima of this response map.\n\n@param src Input single-channel 8-bit or floating-point image.\n@param dst Image to store the Harris detector responses. It has the type CV_32FC1 and the same\nsize as src .\n@param blockSize Neighborhood size (see the details on #cornerEigenValsAndVecs ).\n@param ksize Aperture parameter for the Sobel operator.\n@param k Harris detector free parameter. See the formula below.\n@param borderType Pixel extrapolation method. See #BorderTypes.'''
    package = "Features 2D"

    def get_attributes(self):
        return [Input('src', 'src')], \
               [Output('dst', 'dst')], \
               [SizeParameter('blockSize', 'Block Size'),
                SizeParameter('ksize', 'ksize'),
                FloatParameter('k', 'k'),
                ComboboxParameter('borderType', name='Border Type', values=[('BORDER_CONSTANT',0),('BORDER_REPLICATE',1),('BORDER_REFLECT',2),('BORDER_WRAP',3),('BORDER_DEFAULT',4),('BORDER_REFLECT101',4),('BORDER_REFLECT_101',4),('BORDER_TRANSPARENT',5),('BORDER_ISOLATED',16)])]

    def process_inputs(self, inputs, outputs, parameters):
        src = inputs['src'].value
        blockSize = parameters['blockSize']
        ksize = parameters['ksize']
        k = parameters['k']
        borderType = parameters['borderType']
        dst = cv2.cornerHarris(src=src, blockSize=blockSize, ksize=ksize, k=k, borderType=borderType)
        outputs['dst'] = Data(dst)

# cv2.cornerMinEigenVal
class OpenCVAuto2_CornerMinEigenVal(NormalElement):
    name = 'Corner Min Eigen Val'
    comment = '''cornerMinEigenVal(src, blockSize[, dst[, ksize[, borderType]]]) -> dst\n@brief Calculates the minimal eigenvalue of gradient matrices for corner detection.\n\nThe function is similar to cornerEigenValsAndVecs but it calculates and stores only the minimal\neigenvalue of the covariance matrix of derivatives, that is, \f$\min(\lambda_1, \lambda_2)\f$ in terms\nof the formulae in the cornerEigenValsAndVecs description.\n\n@param src Input single-channel 8-bit or floating-point image.\n@param dst Image to store the minimal eigenvalues. It has the type CV_32FC1 and the same size as\nsrc .\n@param blockSize Neighborhood size (see the details on #cornerEigenValsAndVecs ).\n@param ksize Aperture parameter for the Sobel operator.\n@param borderType Pixel extrapolation method. See #BorderTypes.'''
    package = "Features 2D"

    def get_attributes(self):
        return [Input('src', 'src')], \
               [Output('dst', 'dst')], \
               [SizeParameter('blockSize', 'Block Size'),
                SizeParameter('ksize', 'ksize'),
                ComboboxParameter('borderType', name='Border Type', values=[('BORDER_CONSTANT',0),('BORDER_REPLICATE',1),('BORDER_REFLECT',2),('BORDER_WRAP',3),('BORDER_DEFAULT',4),('BORDER_REFLECT101',4),('BORDER_REFLECT_101',4),('BORDER_TRANSPARENT',5),('BORDER_ISOLATED',16)])]

    def process_inputs(self, inputs, outputs, parameters):
        src = inputs['src'].value
        blockSize = parameters['blockSize']
        ksize = parameters['ksize']
        borderType = parameters['borderType']
        dst = cv2.cornerMinEigenVal(src=src, blockSize=blockSize, ksize=ksize, borderType=borderType)
        outputs['dst'] = Data(dst)

# cv2.correctMatches
class OpenCVAuto2_CorrectMatches(NormalElement):
    name = 'Correct Matches'
    comment = '''correctMatches(F, points1, points2[, newPoints1[, newPoints2]]) -> newPoints1, newPoints2\n@brief Refines coordinates of corresponding points.\n\n@param F 3x3 fundamental matrix.\n@param points1 1xN array containing the first set of points.\n@param points2 1xN array containing the second set of points.\n@param newPoints1 The optimized points1.\n@param newPoints2 The optimized points2.\n\nThe function implements the Optimal Triangulation Method (see Multiple View Geometry for details).\nFor each given point correspondence points1[i] \<-\> points2[i], and a fundamental matrix F, it\ncomputes the corrected correspondences newPoints1[i] \<-\> newPoints2[i] that minimize the geometric\nerror \f$d(points1[i], newPoints1[i])^2 + d(points2[i],newPoints2[i])^2\f$ (where \f$d(a,b)\f$ is the\ngeometric distance between points \f$a\f$ and \f$b\f$ ) subject to the epipolar constraint\n\f$newPoints2^T * F * newPoints1 = 0\f$ .'''
    package = "3D calibration & reconstruction"

    def get_attributes(self):
        return [Input('F', 'F'),
                Input('points1', 'points 1'),
                Input('points2', 'points 2')], \
               [Output('newPoints1', 'New Points 1'),
                Output('newPoints2', 'New Points 2')], \
               []

    def process_inputs(self, inputs, outputs, parameters):
        F = inputs['F'].value
        points1 = inputs['points1'].value
        points2 = inputs['points2'].value
        newPoints1, newPoints2 = cv2.correctMatches(F=F, points1=points1, points2=points2)
        outputs['newPoints1'] = Data(newPoints1)
        outputs['newPoints2'] = Data(newPoints2)

# cv2.createHanningWindow
class OpenCVAuto2_CreateHanningWindow(NormalElement):
    name = 'Create Hanning Window'
    comment = '''createHanningWindow(winSize, type[, dst]) -> dst\n@brief This function computes a Hanning window coefficients in two dimensions.\n\nSee (http://en.wikipedia.org/wiki/Hann_function) and (http://en.wikipedia.org/wiki/Window_function)\nfor more information.\n\nAn example is shown below:\n@code\n// create hanning window of size 100x100 and type CV_32F\nMat hann;\ncreateHanningWindow(hann, Size(100, 100), CV_32F);\n@endcode\n@param dst Destination array to place Hann coefficients in\n@param winSize The window size specifications (both width and height must be > 1)\n@param type Created array type'''
    package = "3D calibration & reconstruction"

    def get_attributes(self):
        return [], \
               [Output('dst', 'dst')], \
               [SizeParameter('winSize', 'Win Size'),
                IntParameter('type', 'type')]

    def process_inputs(self, inputs, outputs, parameters):
        winSize = parameters['winSize']
        type = parameters['type']
        dst = cv2.createHanningWindow(winSize=winSize, type=type)
        outputs['dst'] = Data(dst)

# cv2.cvtColor
class OpenCVAuto2_CvtColor(NormalElement):
    name = 'Cvt Color'
    comment = '''cvtColor(src, code[, dst[, dstCn]]) -> dst\n@brief Converts an image from one color space to another.\n\nThe function converts an input image from one color space to another. In case of a transformation\nto-from RGB color space, the order of the channels should be specified explicitly (RGB or BGR). Note\nthat the default color format in OpenCV is often referred to as RGB but it is actually BGR (the\nbytes are reversed). So the first byte in a standard (24-bit) color image will be an 8-bit Blue\ncomponent, the second byte will be Green, and the third byte will be Red. The fourth, fifth, and\nsixth bytes would then be the second pixel (Blue, then Green, then Red), and so on.\n\nThe conventional ranges for R, G, and B channel values are:\n-   0 to 255 for CV_8U images\n-   0 to 65535 for CV_16U images\n-   0 to 1 for CV_32F images\n\nIn case of linear transformations, the range does not matter. But in case of a non-linear\ntransformation, an input RGB image should be normalized to the proper value range to get the correct\nresults, for example, for RGB \f$\rightarrow\f$ L\*u\*v\* transformation. For example, if you have a\n32-bit floating-point image directly converted from an 8-bit image without any scaling, then it will\nhave the 0..255 value range instead of 0..1 assumed by the function. So, before calling #cvtColor ,\nyou need first to scale the image down:\n@code\nimg *= 1./255;\ncvtColor(img, img, COLOR_BGR2Luv);\n@endcode\nIf you use #cvtColor with 8-bit images, the conversion will have some information lost. For many\napplications, this will not be noticeable but it is recommended to use 32-bit images in applications\nthat need the full range of colors or that convert an image before an operation and then convert\nback.\n\nIf conversion adds the alpha channel, its value will set to the maximum of corresponding channel\nrange: 255 for CV_8U, 65535 for CV_16U, 1 for CV_32F.\n\n@param src input image: 8-bit unsigned, 16-bit unsigned ( CV_16UC... ), or single-precision\nfloating-point.\n@param dst output image of the same size and depth as src.\n@param code color space conversion code (see #ColorConversionCodes).\n@param dstCn number of channels in the destination image; if the parameter is 0, the number of the\nchannels is derived automatically from src and code.\n\n@see @ref imgproc_color_conversions'''
    package = "Color"

    def get_attributes(self):
        return [Input('src', 'src')], \
               [Output('dst', 'dst')], \
               [ComboboxParameter('code', name='code', values=[('COLOR_BGR2BGRA',0),('COLOR_RGB2RGBA',0),('COLOR_BGRA2BGR',1),('COLOR_RGBA2RGB',1),('COLOR_BGR2RGBA',2),('COLOR_RGB2BGRA',2),('COLOR_BGRA2RGB',3),('COLOR_RGBA2BGR',3),('COLOR_BGR2RGB',4),('COLOR_RGB2BGR',4),('COLOR_BGRA2RGBA',5),('COLOR_RGBA2BGRA',5),('COLOR_BGR2GRAY',6),('COLOR_RGB2GRAY',7),('COLOR_GRAY2BGR',8),('COLOR_GRAY2RGB',8),('COLOR_GRAY2BGRA',9),('COLOR_GRAY2RGBA',9),('COLOR_BGRA2GRAY',10),('COLOR_RGBA2GRAY',11),('COLOR_BGR2BGR565',12),('COLOR_RGB2BGR565',13),('COLOR_BGR5652BGR',14),('COLOR_BGR5652RGB',15),('COLOR_BGRA2BGR565',16),('COLOR_RGBA2BGR565',17),('COLOR_BGR5652BGRA',18),('COLOR_BGR5652RGBA',19),('COLOR_GRAY2BGR565',20),('COLOR_BGR5652GRAY',21),('COLOR_BGR2BGR555',22),('COLOR_RGB2BGR555',23),('COLOR_BGR5552BGR',24),('COLOR_BGR5552RGB',25),('COLOR_BGRA2BGR555',26),('COLOR_RGBA2BGR555',27),('COLOR_BGR5552BGRA',28),('COLOR_BGR5552RGBA',29),('COLOR_GRAY2BGR555',30),('COLOR_BGR5552GRAY',31),('COLOR_BGR2XYZ',32),('COLOR_RGB2XYZ',33),('COLOR_XYZ2BGR',34),('COLOR_XYZ2RGB',35),('COLOR_BGR2YCrCb',36),('COLOR_BGR2YCR_CB',36),('COLOR_RGB2YCrCb',37),('COLOR_RGB2YCR_CB',37),('COLOR_YCrCb2BGR',38),('COLOR_YCR_CB2BGR',38),('COLOR_YCrCb2RGB',39),('COLOR_YCR_CB2RGB',39),('COLOR_BGR2HSV',40),('COLOR_RGB2HSV',41),('COLOR_BGR2Lab',44),('COLOR_BGR2LAB',44),('COLOR_RGB2Lab',45),('COLOR_RGB2LAB',45),('COLOR_BayerBG2BGR',46),('COLOR_BAYER_BG2BGR',46),('COLOR_BayerRG2RGB',46),('COLOR_BAYER_RG2RGB',46),('COLOR_BayerGB2BGR',47),('COLOR_BAYER_GB2BGR',47),('COLOR_BayerGR2RGB',47),('COLOR_BAYER_GR2RGB',47),('COLOR_BayerBG2RGB',48),('COLOR_BAYER_BG2RGB',48),('COLOR_BayerRG2BGR',48),('COLOR_BAYER_RG2BGR',48),('COLOR_BayerGB2RGB',49),('COLOR_BAYER_GB2RGB',49),('COLOR_BayerGR2BGR',49),('COLOR_BAYER_GR2BGR',49),('COLOR_BGR2Luv',50),('COLOR_BGR2LUV',50),('COLOR_RGB2Luv',51),('COLOR_RGB2LUV',51),('COLOR_BGR2HLS',52),('COLOR_RGB2HLS',53),('COLOR_HSV2BGR',54),('COLOR_HSV2RGB',55),('COLOR_Lab2BGR',56),('COLOR_LAB2BGR',56),('COLOR_Lab2RGB',57),('COLOR_LAB2RGB',57),('COLOR_Luv2BGR',58),('COLOR_LUV2BGR',58),('COLOR_Luv2RGB',59),('COLOR_LUV2RGB',59),('COLOR_HLS2BGR',60),('COLOR_HLS2RGB',61),('COLOR_BayerBG2BGR_VNG',62),('COLOR_BAYER_BG2BGR_VNG',62),('COLOR_BayerRG2RGB_VNG',62),('COLOR_BAYER_RG2RGB_VNG',62),('COLOR_BayerGB2BGR_VNG',63),('COLOR_BAYER_GB2BGR_VNG',63),('COLOR_BayerGR2RGB_VNG',63),('COLOR_BAYER_GR2RGB_VNG',63),('COLOR_BayerBG2RGB_VNG',64),('COLOR_BAYER_BG2RGB_VNG',64),('COLOR_BayerRG2BGR_VNG',64),('COLOR_BAYER_RG2BGR_VNG',64),('COLOR_BayerGB2RGB_VNG',65),('COLOR_BAYER_GB2RGB_VNG',65),('COLOR_BayerGR2BGR_VNG',65),('COLOR_BAYER_GR2BGR_VNG',65),('COLOR_BGR2HSV_FULL',66),('COLOR_RGB2HSV_FULL',67),('COLOR_BGR2HLS_FULL',68),('COLOR_RGB2HLS_FULL',69),('COLOR_HSV2BGR_FULL',70),('COLOR_HSV2RGB_FULL',71),('COLOR_HLS2BGR_FULL',72),('COLOR_HLS2RGB_FULL',73),('COLOR_LBGR2Lab',74),('COLOR_LBGR2LAB',74),('COLOR_LRGB2Lab',75),('COLOR_LRGB2LAB',75),('COLOR_LBGR2Luv',76),('COLOR_LBGR2LUV',76),('COLOR_LRGB2Luv',77),('COLOR_LRGB2LUV',77),('COLOR_Lab2LBGR',78),('COLOR_LAB2LBGR',78),('COLOR_Lab2LRGB',79),('COLOR_LAB2LRGB',79),('COLOR_Luv2LBGR',80),('COLOR_LUV2LBGR',80),('COLOR_Luv2LRGB',81),('COLOR_LUV2LRGB',81),('COLOR_BGR2YUV',82),('COLOR_RGB2YUV',83),('COLOR_YUV2BGR',84),('COLOR_YUV2RGB',85),('COLOR_BayerBG2GRAY',86),('COLOR_BAYER_BG2GRAY',86),('COLOR_BayerGB2GRAY',87),('COLOR_BAYER_GB2GRAY',87),('COLOR_BayerRG2GRAY',88),('COLOR_BAYER_RG2GRAY',88),('COLOR_BayerGR2GRAY',89),('COLOR_BAYER_GR2GRAY',89),('COLOR_YUV2RGB_NV12',90),('COLOR_YUV2BGR_NV12',91),('COLOR_YUV2RGB_NV21',92),('COLOR_YUV420sp2RGB',92),('COLOR_YUV420SP2RGB',92),('COLOR_YUV2BGR_NV21',93),('COLOR_YUV420sp2BGR',93),('COLOR_YUV420SP2BGR',93),('COLOR_YUV2RGBA_NV12',94),('COLOR_YUV2BGRA_NV12',95),('COLOR_YUV2RGBA_NV21',96),('COLOR_YUV420sp2RGBA',96),('COLOR_YUV420SP2RGBA',96),('COLOR_YUV2BGRA_NV21',97),('COLOR_YUV420sp2BGRA',97),('COLOR_YUV420SP2BGRA',97),('COLOR_YUV2RGB_YV12',98),('COLOR_YUV420p2RGB',98),('COLOR_YUV420P2RGB',98),('COLOR_YUV2BGR_YV12',99),('COLOR_YUV420p2BGR',99),('COLOR_YUV420P2BGR',99),('COLOR_YUV2RGB_I420',100),('COLOR_YUV2RGB_IYUV',100),('COLOR_YUV2BGR_I420',101),('COLOR_YUV2BGR_IYUV',101),('COLOR_YUV2RGBA_YV12',102),('COLOR_YUV420p2RGBA',102),('COLOR_YUV420P2RGBA',102),('COLOR_YUV2BGRA_YV12',103),('COLOR_YUV420p2BGRA',103),('COLOR_YUV420P2BGRA',103),('COLOR_YUV2RGBA_I420',104),('COLOR_YUV2RGBA_IYUV',104),('COLOR_YUV2BGRA_I420',105),('COLOR_YUV2BGRA_IYUV',105),('COLOR_YUV2GRAY_420',106),('COLOR_YUV2GRAY_I420',106),('COLOR_YUV2GRAY_IYUV',106),('COLOR_YUV2GRAY_NV12',106),('COLOR_YUV2GRAY_NV21',106),('COLOR_YUV2GRAY_YV12',106),('COLOR_YUV420p2GRAY',106),('COLOR_YUV420P2GRAY',106),('COLOR_YUV420sp2GRAY',106),('COLOR_YUV420SP2GRAY',106),('COLOR_YUV2RGB_UYNV',107),('COLOR_YUV2RGB_UYVY',107),('COLOR_YUV2RGB_Y422',107),('COLOR_YUV2BGR_UYNV',108),('COLOR_YUV2BGR_UYVY',108),('COLOR_YUV2BGR_Y422',108),('COLOR_YUV2RGBA_UYNV',111),('COLOR_YUV2RGBA_UYVY',111),('COLOR_YUV2RGBA_Y422',111),('COLOR_YUV2BGRA_UYNV',112),('COLOR_YUV2BGRA_UYVY',112),('COLOR_YUV2BGRA_Y422',112),('COLOR_YUV2RGB_YUNV',115),('COLOR_YUV2RGB_YUY2',115),('COLOR_YUV2RGB_YUYV',115),('COLOR_YUV2BGR_YUNV',116),('COLOR_YUV2BGR_YUY2',116),('COLOR_YUV2BGR_YUYV',116),('COLOR_YUV2RGB_YVYU',117),('COLOR_YUV2BGR_YVYU',118),('COLOR_YUV2RGBA_YUNV',119),('COLOR_YUV2RGBA_YUY2',119),('COLOR_YUV2RGBA_YUYV',119),('COLOR_YUV2BGRA_YUNV',120),('COLOR_YUV2BGRA_YUY2',120),('COLOR_YUV2BGRA_YUYV',120),('COLOR_YUV2RGBA_YVYU',121),('COLOR_YUV2BGRA_YVYU',122),('COLOR_YUV2GRAY_UYNV',123),('COLOR_YUV2GRAY_UYVY',123),('COLOR_YUV2GRAY_Y422',123),('COLOR_YUV2GRAY_YUNV',124),('COLOR_YUV2GRAY_YUY2',124),('COLOR_YUV2GRAY_YUYV',124),('COLOR_YUV2GRAY_YVYU',124),('COLOR_RGBA2mRGBA',125),('COLOR_RGBA2M_RGBA',125),('COLOR_mRGBA2RGBA',126),('COLOR_M_RGBA2RGBA',126),('COLOR_RGB2YUV_I420',127),('COLOR_RGB2YUV_IYUV',127),('COLOR_BGR2YUV_I420',128),('COLOR_BGR2YUV_IYUV',128),('COLOR_RGBA2YUV_I420',129),('COLOR_RGBA2YUV_IYUV',129),('COLOR_BGRA2YUV_I420',130),('COLOR_BGRA2YUV_IYUV',130),('COLOR_RGB2YUV_YV12',131),('COLOR_BGR2YUV_YV12',132),('COLOR_RGBA2YUV_YV12',133),('COLOR_BGRA2YUV_YV12',134),('COLOR_BayerBG2BGR_EA',135),('COLOR_BAYER_BG2BGR_EA',135),('COLOR_BayerRG2RGB_EA',135),('COLOR_BAYER_RG2RGB_EA',135),('COLOR_BayerGB2BGR_EA',136),('COLOR_BAYER_GB2BGR_EA',136),('COLOR_BayerGR2RGB_EA',136),('COLOR_BAYER_GR2RGB_EA',136),('COLOR_BayerBG2RGB_EA',137),('COLOR_BAYER_BG2RGB_EA',137),('COLOR_BayerRG2BGR_EA',137),('COLOR_BAYER_RG2BGR_EA',137),('COLOR_BayerGB2RGB_EA',138),('COLOR_BAYER_GB2RGB_EA',138),('COLOR_BayerGR2BGR_EA',138),('COLOR_BAYER_GR2BGR_EA',138),('COLOR_BayerBG2BGRA',139),('COLOR_BAYER_BG2BGRA',139),('COLOR_BayerRG2RGBA',139),('COLOR_BAYER_RG2RGBA',139),('COLOR_BayerGB2BGRA',140),('COLOR_BAYER_GB2BGRA',140),('COLOR_BayerGR2RGBA',140),('COLOR_BAYER_GR2RGBA',140),('COLOR_BayerBG2RGBA',141),('COLOR_BAYER_BG2RGBA',141),('COLOR_BayerRG2BGRA',141),('COLOR_BAYER_RG2BGRA',141),('COLOR_BayerGB2RGBA',142),('COLOR_BAYER_GB2RGBA',142),('COLOR_BayerGR2BGRA',142),('COLOR_BAYER_GR2BGRA',142),('COLOR_COLORCVT_MAX',143)]),
                IntParameter('dstCn', 'Dst Cn')]

    def process_inputs(self, inputs, outputs, parameters):
        src = inputs['src'].value
        code = parameters['code']
        dstCn = parameters['dstCn']
        dst = cv2.cvtColor(src=src, code=code, dstCn=dstCn)
        outputs['dst'] = Data(dst)

# cv2.cvtColorTwoPlane
class OpenCVAuto2_CvtColorTwoPlane(NormalElement):
    name = 'Cvt Color Two Plane'
    comment = '''cvtColorTwoPlane(src1, src2, code[, dst]) -> dst
.'''
    package = "Color"

    def get_attributes(self):
        return [Input('src1', 'src 1'),
                Input('src2', 'src 2')], \
               [Output('dst', 'dst')], \
               [ComboboxParameter('code', name='code', values=[('COLOR_BGR2BGRA',0),('COLOR_RGB2RGBA',0),('COLOR_BGRA2BGR',1),('COLOR_RGBA2RGB',1),('COLOR_BGR2RGBA',2),('COLOR_RGB2BGRA',2),('COLOR_BGRA2RGB',3),('COLOR_RGBA2BGR',3),('COLOR_BGR2RGB',4),('COLOR_RGB2BGR',4),('COLOR_BGRA2RGBA',5),('COLOR_RGBA2BGRA',5),('COLOR_BGR2GRAY',6),('COLOR_RGB2GRAY',7),('COLOR_GRAY2BGR',8),('COLOR_GRAY2RGB',8),('COLOR_GRAY2BGRA',9),('COLOR_GRAY2RGBA',9),('COLOR_BGRA2GRAY',10),('COLOR_RGBA2GRAY',11),('COLOR_BGR2BGR565',12),('COLOR_RGB2BGR565',13),('COLOR_BGR5652BGR',14),('COLOR_BGR5652RGB',15),('COLOR_BGRA2BGR565',16),('COLOR_RGBA2BGR565',17),('COLOR_BGR5652BGRA',18),('COLOR_BGR5652RGBA',19),('COLOR_GRAY2BGR565',20),('COLOR_BGR5652GRAY',21),('COLOR_BGR2BGR555',22),('COLOR_RGB2BGR555',23),('COLOR_BGR5552BGR',24),('COLOR_BGR5552RGB',25),('COLOR_BGRA2BGR555',26),('COLOR_RGBA2BGR555',27),('COLOR_BGR5552BGRA',28),('COLOR_BGR5552RGBA',29),('COLOR_GRAY2BGR555',30),('COLOR_BGR5552GRAY',31),('COLOR_BGR2XYZ',32),('COLOR_RGB2XYZ',33),('COLOR_XYZ2BGR',34),('COLOR_XYZ2RGB',35),('COLOR_BGR2YCrCb',36),('COLOR_BGR2YCR_CB',36),('COLOR_RGB2YCrCb',37),('COLOR_RGB2YCR_CB',37),('COLOR_YCrCb2BGR',38),('COLOR_YCR_CB2BGR',38),('COLOR_YCrCb2RGB',39),('COLOR_YCR_CB2RGB',39),('COLOR_BGR2HSV',40),('COLOR_RGB2HSV',41),('COLOR_BGR2Lab',44),('COLOR_BGR2LAB',44),('COLOR_RGB2Lab',45),('COLOR_RGB2LAB',45),('COLOR_BayerBG2BGR',46),('COLOR_BAYER_BG2BGR',46),('COLOR_BayerRG2RGB',46),('COLOR_BAYER_RG2RGB',46),('COLOR_BayerGB2BGR',47),('COLOR_BAYER_GB2BGR',47),('COLOR_BayerGR2RGB',47),('COLOR_BAYER_GR2RGB',47),('COLOR_BayerBG2RGB',48),('COLOR_BAYER_BG2RGB',48),('COLOR_BayerRG2BGR',48),('COLOR_BAYER_RG2BGR',48),('COLOR_BayerGB2RGB',49),('COLOR_BAYER_GB2RGB',49),('COLOR_BayerGR2BGR',49),('COLOR_BAYER_GR2BGR',49),('COLOR_BGR2Luv',50),('COLOR_BGR2LUV',50),('COLOR_RGB2Luv',51),('COLOR_RGB2LUV',51),('COLOR_BGR2HLS',52),('COLOR_RGB2HLS',53),('COLOR_HSV2BGR',54),('COLOR_HSV2RGB',55),('COLOR_Lab2BGR',56),('COLOR_LAB2BGR',56),('COLOR_Lab2RGB',57),('COLOR_LAB2RGB',57),('COLOR_Luv2BGR',58),('COLOR_LUV2BGR',58),('COLOR_Luv2RGB',59),('COLOR_LUV2RGB',59),('COLOR_HLS2BGR',60),('COLOR_HLS2RGB',61),('COLOR_BayerBG2BGR_VNG',62),('COLOR_BAYER_BG2BGR_VNG',62),('COLOR_BayerRG2RGB_VNG',62),('COLOR_BAYER_RG2RGB_VNG',62),('COLOR_BayerGB2BGR_VNG',63),('COLOR_BAYER_GB2BGR_VNG',63),('COLOR_BayerGR2RGB_VNG',63),('COLOR_BAYER_GR2RGB_VNG',63),('COLOR_BayerBG2RGB_VNG',64),('COLOR_BAYER_BG2RGB_VNG',64),('COLOR_BayerRG2BGR_VNG',64),('COLOR_BAYER_RG2BGR_VNG',64),('COLOR_BayerGB2RGB_VNG',65),('COLOR_BAYER_GB2RGB_VNG',65),('COLOR_BayerGR2BGR_VNG',65),('COLOR_BAYER_GR2BGR_VNG',65),('COLOR_BGR2HSV_FULL',66),('COLOR_RGB2HSV_FULL',67),('COLOR_BGR2HLS_FULL',68),('COLOR_RGB2HLS_FULL',69),('COLOR_HSV2BGR_FULL',70),('COLOR_HSV2RGB_FULL',71),('COLOR_HLS2BGR_FULL',72),('COLOR_HLS2RGB_FULL',73),('COLOR_LBGR2Lab',74),('COLOR_LBGR2LAB',74),('COLOR_LRGB2Lab',75),('COLOR_LRGB2LAB',75),('COLOR_LBGR2Luv',76),('COLOR_LBGR2LUV',76),('COLOR_LRGB2Luv',77),('COLOR_LRGB2LUV',77),('COLOR_Lab2LBGR',78),('COLOR_LAB2LBGR',78),('COLOR_Lab2LRGB',79),('COLOR_LAB2LRGB',79),('COLOR_Luv2LBGR',80),('COLOR_LUV2LBGR',80),('COLOR_Luv2LRGB',81),('COLOR_LUV2LRGB',81),('COLOR_BGR2YUV',82),('COLOR_RGB2YUV',83),('COLOR_YUV2BGR',84),('COLOR_YUV2RGB',85),('COLOR_BayerBG2GRAY',86),('COLOR_BAYER_BG2GRAY',86),('COLOR_BayerGB2GRAY',87),('COLOR_BAYER_GB2GRAY',87),('COLOR_BayerRG2GRAY',88),('COLOR_BAYER_RG2GRAY',88),('COLOR_BayerGR2GRAY',89),('COLOR_BAYER_GR2GRAY',89),('COLOR_YUV2RGB_NV12',90),('COLOR_YUV2BGR_NV12',91),('COLOR_YUV2RGB_NV21',92),('COLOR_YUV420sp2RGB',92),('COLOR_YUV420SP2RGB',92),('COLOR_YUV2BGR_NV21',93),('COLOR_YUV420sp2BGR',93),('COLOR_YUV420SP2BGR',93),('COLOR_YUV2RGBA_NV12',94),('COLOR_YUV2BGRA_NV12',95),('COLOR_YUV2RGBA_NV21',96),('COLOR_YUV420sp2RGBA',96),('COLOR_YUV420SP2RGBA',96),('COLOR_YUV2BGRA_NV21',97),('COLOR_YUV420sp2BGRA',97),('COLOR_YUV420SP2BGRA',97),('COLOR_YUV2RGB_YV12',98),('COLOR_YUV420p2RGB',98),('COLOR_YUV420P2RGB',98),('COLOR_YUV2BGR_YV12',99),('COLOR_YUV420p2BGR',99),('COLOR_YUV420P2BGR',99),('COLOR_YUV2RGB_I420',100),('COLOR_YUV2RGB_IYUV',100),('COLOR_YUV2BGR_I420',101),('COLOR_YUV2BGR_IYUV',101),('COLOR_YUV2RGBA_YV12',102),('COLOR_YUV420p2RGBA',102),('COLOR_YUV420P2RGBA',102),('COLOR_YUV2BGRA_YV12',103),('COLOR_YUV420p2BGRA',103),('COLOR_YUV420P2BGRA',103),('COLOR_YUV2RGBA_I420',104),('COLOR_YUV2RGBA_IYUV',104),('COLOR_YUV2BGRA_I420',105),('COLOR_YUV2BGRA_IYUV',105),('COLOR_YUV2GRAY_420',106),('COLOR_YUV2GRAY_I420',106),('COLOR_YUV2GRAY_IYUV',106),('COLOR_YUV2GRAY_NV12',106),('COLOR_YUV2GRAY_NV21',106),('COLOR_YUV2GRAY_YV12',106),('COLOR_YUV420p2GRAY',106),('COLOR_YUV420P2GRAY',106),('COLOR_YUV420sp2GRAY',106),('COLOR_YUV420SP2GRAY',106),('COLOR_YUV2RGB_UYNV',107),('COLOR_YUV2RGB_UYVY',107),('COLOR_YUV2RGB_Y422',107),('COLOR_YUV2BGR_UYNV',108),('COLOR_YUV2BGR_UYVY',108),('COLOR_YUV2BGR_Y422',108),('COLOR_YUV2RGBA_UYNV',111),('COLOR_YUV2RGBA_UYVY',111),('COLOR_YUV2RGBA_Y422',111),('COLOR_YUV2BGRA_UYNV',112),('COLOR_YUV2BGRA_UYVY',112),('COLOR_YUV2BGRA_Y422',112),('COLOR_YUV2RGB_YUNV',115),('COLOR_YUV2RGB_YUY2',115),('COLOR_YUV2RGB_YUYV',115),('COLOR_YUV2BGR_YUNV',116),('COLOR_YUV2BGR_YUY2',116),('COLOR_YUV2BGR_YUYV',116),('COLOR_YUV2RGB_YVYU',117),('COLOR_YUV2BGR_YVYU',118),('COLOR_YUV2RGBA_YUNV',119),('COLOR_YUV2RGBA_YUY2',119),('COLOR_YUV2RGBA_YUYV',119),('COLOR_YUV2BGRA_YUNV',120),('COLOR_YUV2BGRA_YUY2',120),('COLOR_YUV2BGRA_YUYV',120),('COLOR_YUV2RGBA_YVYU',121),('COLOR_YUV2BGRA_YVYU',122),('COLOR_YUV2GRAY_UYNV',123),('COLOR_YUV2GRAY_UYVY',123),('COLOR_YUV2GRAY_Y422',123),('COLOR_YUV2GRAY_YUNV',124),('COLOR_YUV2GRAY_YUY2',124),('COLOR_YUV2GRAY_YUYV',124),('COLOR_YUV2GRAY_YVYU',124),('COLOR_RGBA2mRGBA',125),('COLOR_RGBA2M_RGBA',125),('COLOR_mRGBA2RGBA',126),('COLOR_M_RGBA2RGBA',126),('COLOR_RGB2YUV_I420',127),('COLOR_RGB2YUV_IYUV',127),('COLOR_BGR2YUV_I420',128),('COLOR_BGR2YUV_IYUV',128),('COLOR_RGBA2YUV_I420',129),('COLOR_RGBA2YUV_IYUV',129),('COLOR_BGRA2YUV_I420',130),('COLOR_BGRA2YUV_IYUV',130),('COLOR_RGB2YUV_YV12',131),('COLOR_BGR2YUV_YV12',132),('COLOR_RGBA2YUV_YV12',133),('COLOR_BGRA2YUV_YV12',134),('COLOR_BayerBG2BGR_EA',135),('COLOR_BAYER_BG2BGR_EA',135),('COLOR_BayerRG2RGB_EA',135),('COLOR_BAYER_RG2RGB_EA',135),('COLOR_BayerGB2BGR_EA',136),('COLOR_BAYER_GB2BGR_EA',136),('COLOR_BayerGR2RGB_EA',136),('COLOR_BAYER_GR2RGB_EA',136),('COLOR_BayerBG2RGB_EA',137),('COLOR_BAYER_BG2RGB_EA',137),('COLOR_BayerRG2BGR_EA',137),('COLOR_BAYER_RG2BGR_EA',137),('COLOR_BayerGB2RGB_EA',138),('COLOR_BAYER_GB2RGB_EA',138),('COLOR_BayerGR2BGR_EA',138),('COLOR_BAYER_GR2BGR_EA',138),('COLOR_BayerBG2BGRA',139),('COLOR_BAYER_BG2BGRA',139),('COLOR_BayerRG2RGBA',139),('COLOR_BAYER_RG2RGBA',139),('COLOR_BayerGB2BGRA',140),('COLOR_BAYER_GB2BGRA',140),('COLOR_BayerGR2RGBA',140),('COLOR_BAYER_GR2RGBA',140),('COLOR_BayerBG2RGBA',141),('COLOR_BAYER_BG2RGBA',141),('COLOR_BayerRG2BGRA',141),('COLOR_BAYER_RG2BGRA',141),('COLOR_BayerGB2RGBA',142),('COLOR_BAYER_GB2RGBA',142),('COLOR_BayerGR2BGRA',142),('COLOR_BAYER_GR2BGRA',142),('COLOR_COLORCVT_MAX',143)])]

    def process_inputs(self, inputs, outputs, parameters):
        src1 = inputs['src1'].value
        src2 = inputs['src2'].value
        code = parameters['code']
        dst = cv2.cvtColorTwoPlane(src1=src1, src2=src2, code=code)
        outputs['dst'] = Data(dst)

# cv2.decolor
class OpenCVAuto2_Decolor(NormalElement):
    name = 'Decolor'
    comment = '''decolor(src[, grayscale[, color_boost]]) -> grayscale, color_boost\n@brief Transforms a color image to a grayscale image. It is a basic tool in digital printing, stylized\nblack-and-white photograph rendering, and in many single channel image processing applications\n@cite CL12 .\n\n@param src Input 8-bit 3-channel image.\n@param grayscale Output 8-bit 1-channel image.\n@param color_boost Output 8-bit 3-channel image.\n\nThis function is to be applied on color images.'''
    package = "Color"

    def get_attributes(self):
        return [Input('src', 'src')], \
               [Output('grayscale', 'grayscale'),
                Output('color_boost', 'color boost')], \
               []

    def process_inputs(self, inputs, outputs, parameters):
        src = inputs['src'].value
        grayscale, color_boost = cv2.decolor(src=src)
        outputs['grayscale'] = Data(grayscale)
        outputs['color_boost'] = Data(color_boost)

# cv2.decomposeEssentialMat
class OpenCVAuto2_DecomposeEssentialMat(NormalElement):
    name = 'Decompose Essential Mat'
    comment = '''decomposeEssentialMat(E[, R1[, R2[, t]]]) -> R1, R2, t\n@brief Decompose an essential matrix to possible rotations and translation.\n\n@param E The input essential matrix.\n@param R1 One possible rotation matrix.\n@param R2 Another possible rotation matrix.\n@param t One possible translation.\n\nThis function decompose an essential matrix E using svd decomposition @cite HartleyZ00 . Generally 4\npossible poses exists for a given E. They are \f$[R_1, t]\f$, \f$[R_1, -t]\f$, \f$[R_2, t]\f$, \f$[R_2, -t]\f$. By\ndecomposing E, you can only get the direction of the translation, so the function returns unit t.'''
    package = "3D calibration & reconstruction"

    def get_attributes(self):
        return [Input('E', 'E')], \
               [Output('R1', 'R1'),
                Output('R2', 'R2'),
                Output('t', 't')], \
               []

    def process_inputs(self, inputs, outputs, parameters):
        E = inputs['E'].value
        R1, R2, t = cv2.decomposeEssentialMat(E=E)
        outputs['R1'] = Data(R1)
        outputs['R2'] = Data(R2)
        outputs['t'] = Data(t)

# cv2.decomposeHomographyMat
class OpenCVAuto2_DecomposeHomographyMat(NormalElement):
    name = 'Decompose Homography Mat'
    comment = '''decomposeHomographyMat(H, K[, rotations[, translations[, normals]]]) -> retval, rotations, translations, normals\n@brief Decompose a homography matrix to rotation(s), translation(s) and plane normal(s).\n\n@param H The input homography matrix between two images.\n@param K The input intrinsic camera calibration matrix.\n@param rotations Array of rotation matrices.\n@param translations Array of translation matrices.\n@param normals Array of plane normal matrices.\n\nThis function extracts relative camera motion between two views observing a planar object from the\nhomography H induced by the plane. The intrinsic camera matrix K must also be provided. The function\nmay return up to four mathematical solution sets. At least two of the solutions may further be\ninvalidated if point correspondences are available by applying positive depth constraint (all points\nmust be in front of the camera). The decomposition method is described in detail in @cite Malis .'''
    package = "3D calibration & reconstruction"

    def get_attributes(self):
        return [Input('H', 'H'),
                Input('K', 'K')], \
               [Output('rotations', 'rotations'),
                Output('translations', 'translations'),
                Output('normals', 'normals')], \
               []

    def process_inputs(self, inputs, outputs, parameters):
        H = inputs['H'].value
        K = inputs['K'].value
        retval, rotations, translations, normals = cv2.decomposeHomographyMat(H=H, K=K)
        outputs['rotations'] = Data(rotations)
        outputs['translations'] = Data(translations)
        outputs['normals'] = Data(normals)

# cv2.decomposeProjectionMatrix
class OpenCVAuto2_DecomposeProjectionMatrix(NormalElement):
    name = 'Decompose Projection Matrix'
    comment = '''decomposeProjectionMatrix(projMatrix[, cameraMatrix[, rotMatrix[, transVect[, rotMatrixX[, rotMatrixY[, rotMatrixZ[, eulerAngles]]]]]]]) -> cameraMatrix, rotMatrix, transVect, rotMatrixX, rotMatrixY, rotMatrixZ, eulerAngles\n@brief Decomposes a projection matrix into a rotation matrix and a camera matrix.\n\n@param projMatrix 3x4 input projection matrix P.\n@param cameraMatrix Output 3x3 camera matrix K.\n@param rotMatrix Output 3x3 external rotation matrix R.\n@param transVect Output 4x1 translation vector T.\n@param rotMatrixX Optional 3x3 rotation matrix around x-axis.\n@param rotMatrixY Optional 3x3 rotation matrix around y-axis.\n@param rotMatrixZ Optional 3x3 rotation matrix around z-axis.\n@param eulerAngles Optional three-element vector containing three Euler angles of rotation in\ndegrees.\n\nThe function computes a decomposition of a projection matrix into a calibration and a rotation\nmatrix and the position of a camera.\n\nIt optionally returns three rotation matrices, one for each axis, and three Euler angles that could\nbe used in OpenGL. Note, there is always more than one sequence of rotations about the three\nprincipal axes that results in the same orientation of an object, e.g. see @cite Slabaugh . Returned\ntree rotation matrices and corresponding three Euler angles are only one of the possible solutions.\n\nThe function is based on RQDecomp3x3 .'''
    package = "3D calibration & reconstruction"

    def get_attributes(self):
        return [Input('projMatrix', 'Proj Matrix')], \
               [Output('cameraMatrix', 'Camera Matrix'),
                Output('rotMatrix', 'Rot Matrix'),
                Output('transVect', 'Trans Vect'),
                Output('rotMatrixX', 'Rot Matrix X'),
                Output('rotMatrixY', 'Rot Matrix Y'),
                Output('rotMatrixZ', 'Rot Matrix Z'),
                Output('eulerAngles', 'Euler Angles')], \
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

# cv2.demosaicing
class OpenCVAuto2_Demosaicing(NormalElement):
    name = 'Demosaicing'
    comment = '''demosaicing(_src, code[, _dst[, dcn]]) -> _dst
.'''
    package = "Photo"

    def get_attributes(self):
        return [Input('_src', 'src')], \
               [Output('_dst', 'dst')], \
               [ComboboxParameter('code', name='code', values=[('COLOR_BGR2BGRA',0),('COLOR_RGB2RGBA',0),('COLOR_BGRA2BGR',1),('COLOR_RGBA2RGB',1),('COLOR_BGR2RGBA',2),('COLOR_RGB2BGRA',2),('COLOR_BGRA2RGB',3),('COLOR_RGBA2BGR',3),('COLOR_BGR2RGB',4),('COLOR_RGB2BGR',4),('COLOR_BGRA2RGBA',5),('COLOR_RGBA2BGRA',5),('COLOR_BGR2GRAY',6),('COLOR_RGB2GRAY',7),('COLOR_GRAY2BGR',8),('COLOR_GRAY2RGB',8),('COLOR_GRAY2BGRA',9),('COLOR_GRAY2RGBA',9),('COLOR_BGRA2GRAY',10),('COLOR_RGBA2GRAY',11),('COLOR_BGR2BGR565',12),('COLOR_RGB2BGR565',13),('COLOR_BGR5652BGR',14),('COLOR_BGR5652RGB',15),('COLOR_BGRA2BGR565',16),('COLOR_RGBA2BGR565',17),('COLOR_BGR5652BGRA',18),('COLOR_BGR5652RGBA',19),('COLOR_GRAY2BGR565',20),('COLOR_BGR5652GRAY',21),('COLOR_BGR2BGR555',22),('COLOR_RGB2BGR555',23),('COLOR_BGR5552BGR',24),('COLOR_BGR5552RGB',25),('COLOR_BGRA2BGR555',26),('COLOR_RGBA2BGR555',27),('COLOR_BGR5552BGRA',28),('COLOR_BGR5552RGBA',29),('COLOR_GRAY2BGR555',30),('COLOR_BGR5552GRAY',31),('COLOR_BGR2XYZ',32),('COLOR_RGB2XYZ',33),('COLOR_XYZ2BGR',34),('COLOR_XYZ2RGB',35),('COLOR_BGR2YCrCb',36),('COLOR_BGR2YCR_CB',36),('COLOR_RGB2YCrCb',37),('COLOR_RGB2YCR_CB',37),('COLOR_YCrCb2BGR',38),('COLOR_YCR_CB2BGR',38),('COLOR_YCrCb2RGB',39),('COLOR_YCR_CB2RGB',39),('COLOR_BGR2HSV',40),('COLOR_RGB2HSV',41),('COLOR_BGR2Lab',44),('COLOR_BGR2LAB',44),('COLOR_RGB2Lab',45),('COLOR_RGB2LAB',45),('COLOR_BayerBG2BGR',46),('COLOR_BAYER_BG2BGR',46),('COLOR_BayerRG2RGB',46),('COLOR_BAYER_RG2RGB',46),('COLOR_BayerGB2BGR',47),('COLOR_BAYER_GB2BGR',47),('COLOR_BayerGR2RGB',47),('COLOR_BAYER_GR2RGB',47),('COLOR_BayerBG2RGB',48),('COLOR_BAYER_BG2RGB',48),('COLOR_BayerRG2BGR',48),('COLOR_BAYER_RG2BGR',48),('COLOR_BayerGB2RGB',49),('COLOR_BAYER_GB2RGB',49),('COLOR_BayerGR2BGR',49),('COLOR_BAYER_GR2BGR',49),('COLOR_BGR2Luv',50),('COLOR_BGR2LUV',50),('COLOR_RGB2Luv',51),('COLOR_RGB2LUV',51),('COLOR_BGR2HLS',52),('COLOR_RGB2HLS',53),('COLOR_HSV2BGR',54),('COLOR_HSV2RGB',55),('COLOR_Lab2BGR',56),('COLOR_LAB2BGR',56),('COLOR_Lab2RGB',57),('COLOR_LAB2RGB',57),('COLOR_Luv2BGR',58),('COLOR_LUV2BGR',58),('COLOR_Luv2RGB',59),('COLOR_LUV2RGB',59),('COLOR_HLS2BGR',60),('COLOR_HLS2RGB',61),('COLOR_BayerBG2BGR_VNG',62),('COLOR_BAYER_BG2BGR_VNG',62),('COLOR_BayerRG2RGB_VNG',62),('COLOR_BAYER_RG2RGB_VNG',62),('COLOR_BayerGB2BGR_VNG',63),('COLOR_BAYER_GB2BGR_VNG',63),('COLOR_BayerGR2RGB_VNG',63),('COLOR_BAYER_GR2RGB_VNG',63),('COLOR_BayerBG2RGB_VNG',64),('COLOR_BAYER_BG2RGB_VNG',64),('COLOR_BayerRG2BGR_VNG',64),('COLOR_BAYER_RG2BGR_VNG',64),('COLOR_BayerGB2RGB_VNG',65),('COLOR_BAYER_GB2RGB_VNG',65),('COLOR_BayerGR2BGR_VNG',65),('COLOR_BAYER_GR2BGR_VNG',65),('COLOR_BGR2HSV_FULL',66),('COLOR_RGB2HSV_FULL',67),('COLOR_BGR2HLS_FULL',68),('COLOR_RGB2HLS_FULL',69),('COLOR_HSV2BGR_FULL',70),('COLOR_HSV2RGB_FULL',71),('COLOR_HLS2BGR_FULL',72),('COLOR_HLS2RGB_FULL',73),('COLOR_LBGR2Lab',74),('COLOR_LBGR2LAB',74),('COLOR_LRGB2Lab',75),('COLOR_LRGB2LAB',75),('COLOR_LBGR2Luv',76),('COLOR_LBGR2LUV',76),('COLOR_LRGB2Luv',77),('COLOR_LRGB2LUV',77),('COLOR_Lab2LBGR',78),('COLOR_LAB2LBGR',78),('COLOR_Lab2LRGB',79),('COLOR_LAB2LRGB',79),('COLOR_Luv2LBGR',80),('COLOR_LUV2LBGR',80),('COLOR_Luv2LRGB',81),('COLOR_LUV2LRGB',81),('COLOR_BGR2YUV',82),('COLOR_RGB2YUV',83),('COLOR_YUV2BGR',84),('COLOR_YUV2RGB',85),('COLOR_BayerBG2GRAY',86),('COLOR_BAYER_BG2GRAY',86),('COLOR_BayerGB2GRAY',87),('COLOR_BAYER_GB2GRAY',87),('COLOR_BayerRG2GRAY',88),('COLOR_BAYER_RG2GRAY',88),('COLOR_BayerGR2GRAY',89),('COLOR_BAYER_GR2GRAY',89),('COLOR_YUV2RGB_NV12',90),('COLOR_YUV2BGR_NV12',91),('COLOR_YUV2RGB_NV21',92),('COLOR_YUV420sp2RGB',92),('COLOR_YUV420SP2RGB',92),('COLOR_YUV2BGR_NV21',93),('COLOR_YUV420sp2BGR',93),('COLOR_YUV420SP2BGR',93),('COLOR_YUV2RGBA_NV12',94),('COLOR_YUV2BGRA_NV12',95),('COLOR_YUV2RGBA_NV21',96),('COLOR_YUV420sp2RGBA',96),('COLOR_YUV420SP2RGBA',96),('COLOR_YUV2BGRA_NV21',97),('COLOR_YUV420sp2BGRA',97),('COLOR_YUV420SP2BGRA',97),('COLOR_YUV2RGB_YV12',98),('COLOR_YUV420p2RGB',98),('COLOR_YUV420P2RGB',98),('COLOR_YUV2BGR_YV12',99),('COLOR_YUV420p2BGR',99),('COLOR_YUV420P2BGR',99),('COLOR_YUV2RGB_I420',100),('COLOR_YUV2RGB_IYUV',100),('COLOR_YUV2BGR_I420',101),('COLOR_YUV2BGR_IYUV',101),('COLOR_YUV2RGBA_YV12',102),('COLOR_YUV420p2RGBA',102),('COLOR_YUV420P2RGBA',102),('COLOR_YUV2BGRA_YV12',103),('COLOR_YUV420p2BGRA',103),('COLOR_YUV420P2BGRA',103),('COLOR_YUV2RGBA_I420',104),('COLOR_YUV2RGBA_IYUV',104),('COLOR_YUV2BGRA_I420',105),('COLOR_YUV2BGRA_IYUV',105),('COLOR_YUV2GRAY_420',106),('COLOR_YUV2GRAY_I420',106),('COLOR_YUV2GRAY_IYUV',106),('COLOR_YUV2GRAY_NV12',106),('COLOR_YUV2GRAY_NV21',106),('COLOR_YUV2GRAY_YV12',106),('COLOR_YUV420p2GRAY',106),('COLOR_YUV420P2GRAY',106),('COLOR_YUV420sp2GRAY',106),('COLOR_YUV420SP2GRAY',106),('COLOR_YUV2RGB_UYNV',107),('COLOR_YUV2RGB_UYVY',107),('COLOR_YUV2RGB_Y422',107),('COLOR_YUV2BGR_UYNV',108),('COLOR_YUV2BGR_UYVY',108),('COLOR_YUV2BGR_Y422',108),('COLOR_YUV2RGBA_UYNV',111),('COLOR_YUV2RGBA_UYVY',111),('COLOR_YUV2RGBA_Y422',111),('COLOR_YUV2BGRA_UYNV',112),('COLOR_YUV2BGRA_UYVY',112),('COLOR_YUV2BGRA_Y422',112),('COLOR_YUV2RGB_YUNV',115),('COLOR_YUV2RGB_YUY2',115),('COLOR_YUV2RGB_YUYV',115),('COLOR_YUV2BGR_YUNV',116),('COLOR_YUV2BGR_YUY2',116),('COLOR_YUV2BGR_YUYV',116),('COLOR_YUV2RGB_YVYU',117),('COLOR_YUV2BGR_YVYU',118),('COLOR_YUV2RGBA_YUNV',119),('COLOR_YUV2RGBA_YUY2',119),('COLOR_YUV2RGBA_YUYV',119),('COLOR_YUV2BGRA_YUNV',120),('COLOR_YUV2BGRA_YUY2',120),('COLOR_YUV2BGRA_YUYV',120),('COLOR_YUV2RGBA_YVYU',121),('COLOR_YUV2BGRA_YVYU',122),('COLOR_YUV2GRAY_UYNV',123),('COLOR_YUV2GRAY_UYVY',123),('COLOR_YUV2GRAY_Y422',123),('COLOR_YUV2GRAY_YUNV',124),('COLOR_YUV2GRAY_YUY2',124),('COLOR_YUV2GRAY_YUYV',124),('COLOR_YUV2GRAY_YVYU',124),('COLOR_RGBA2mRGBA',125),('COLOR_RGBA2M_RGBA',125),('COLOR_mRGBA2RGBA',126),('COLOR_M_RGBA2RGBA',126),('COLOR_RGB2YUV_I420',127),('COLOR_RGB2YUV_IYUV',127),('COLOR_BGR2YUV_I420',128),('COLOR_BGR2YUV_IYUV',128),('COLOR_RGBA2YUV_I420',129),('COLOR_RGBA2YUV_IYUV',129),('COLOR_BGRA2YUV_I420',130),('COLOR_BGRA2YUV_IYUV',130),('COLOR_RGB2YUV_YV12',131),('COLOR_BGR2YUV_YV12',132),('COLOR_RGBA2YUV_YV12',133),('COLOR_BGRA2YUV_YV12',134),('COLOR_BayerBG2BGR_EA',135),('COLOR_BAYER_BG2BGR_EA',135),('COLOR_BayerRG2RGB_EA',135),('COLOR_BAYER_RG2RGB_EA',135),('COLOR_BayerGB2BGR_EA',136),('COLOR_BAYER_GB2BGR_EA',136),('COLOR_BayerGR2RGB_EA',136),('COLOR_BAYER_GR2RGB_EA',136),('COLOR_BayerBG2RGB_EA',137),('COLOR_BAYER_BG2RGB_EA',137),('COLOR_BayerRG2BGR_EA',137),('COLOR_BAYER_RG2BGR_EA',137),('COLOR_BayerGB2RGB_EA',138),('COLOR_BAYER_GB2RGB_EA',138),('COLOR_BayerGR2BGR_EA',138),('COLOR_BAYER_GR2BGR_EA',138),('COLOR_BayerBG2BGRA',139),('COLOR_BAYER_BG2BGRA',139),('COLOR_BayerRG2RGBA',139),('COLOR_BAYER_RG2RGBA',139),('COLOR_BayerGB2BGRA',140),('COLOR_BAYER_GB2BGRA',140),('COLOR_BayerGR2RGBA',140),('COLOR_BAYER_GR2RGBA',140),('COLOR_BayerBG2RGBA',141),('COLOR_BAYER_BG2RGBA',141),('COLOR_BayerRG2BGRA',141),('COLOR_BAYER_RG2BGRA',141),('COLOR_BayerGB2RGBA',142),('COLOR_BAYER_GB2RGBA',142),('COLOR_BayerGR2BGRA',142),('COLOR_BAYER_GR2BGRA',142),('COLOR_COLORCVT_MAX',143)])]

    def process_inputs(self, inputs, outputs, parameters):
        _src = inputs['_src'].value
        code = parameters['code']
        _dst = cv2.demosaicing(_src=_src, code=code)
        outputs['_dst'] = Data(_dst)

# cv2.detailEnhance
class OpenCVAuto2_DetailEnhance(NormalElement):
    name = 'Detail Enhance'
    comment = '''detailEnhance(src[, dst[, sigma_s[, sigma_r]]]) -> dst\n@brief This filter enhances the details of a particular image.\n\n@param src Input 8-bit 3-channel image.\n@param dst Output image with the same size and type as src.\n@param sigma_s Range between 0 to 200.\n@param sigma_r Range between 0 to 1.'''
    package = "Photo"

    def get_attributes(self):
        return [Input('src', 'src')], \
               [Output('dst', 'dst')], \
               [FloatParameter('sigma_s', 'sigma s'),
                FloatParameter('sigma_r', 'sigma r')]

    def process_inputs(self, inputs, outputs, parameters):
        src = inputs['src'].value
        sigma_s = parameters['sigma_s']
        sigma_r = parameters['sigma_r']
        dst = cv2.detailEnhance(src=src, sigma_s=sigma_s, sigma_r=sigma_r)
        outputs['dst'] = Data(dst)

# cv2.dilate
class OpenCVAuto2_Dilate(NormalElement):
    name = 'Dilate'
    comment = '''dilate(src, kernel[, dst[, anchor[, iterations[, borderType[, borderValue]]]]]) -> dst\n@brief Dilates an image by using a specific structuring element.\n\nThe function dilates the source image using the specified structuring element that determines the\nshape of a pixel neighborhood over which the maximum is taken:\n\f[\texttt{dst} (x,y) =  \max _{(x',y'):  \, \texttt{element} (x',y') \ne0 } \texttt{src} (x+x',y+y')\f]\n\nThe function supports the in-place mode. Dilation can be applied several ( iterations ) times. In\ncase of multi-channel images, each channel is processed independently.\n\n@param src input image; the number of channels can be arbitrary, but the depth should be one of\nCV_8U, CV_16U, CV_16S, CV_32F or CV_64F.\n@param dst output image of the same size and type as src.\n@param kernel structuring element used for dilation; if elemenat=Mat(), a 3 x 3 rectangular\nstructuring element is used. Kernel can be created using #getStructuringElement\n@param anchor position of the anchor within the element; default value (-1, -1) means that the\nanchor is at the element center.\n@param iterations number of times dilation is applied.\n@param borderType pixel extrapolation method, see #BorderTypes\n@param borderValue border value in case of a constant border\n@sa  erode, morphologyEx, getStructuringElement'''
    package = "Filters"

    def get_attributes(self):
        return [Input('src', 'src'),
                Input('kernel', 'kernel')], \
               [Output('dst', 'dst')], \
               [PointParameter('anchor', 'anchor'),
                IntParameter('iterations', 'iterations', min_=0),
                ComboboxParameter('borderType', name='Border Type', values=[('BORDER_CONSTANT',0),('BORDER_REPLICATE',1),('BORDER_REFLECT',2),('BORDER_WRAP',3),('BORDER_DEFAULT',4),('BORDER_REFLECT101',4),('BORDER_REFLECT_101',4),('BORDER_TRANSPARENT',5),('BORDER_ISOLATED',16)]),
                ScalarParameter('borderValue', 'Border Value')]

    def process_inputs(self, inputs, outputs, parameters):
        src = inputs['src'].value
        kernel = inputs['kernel'].value
        anchor = parameters['anchor']
        iterations = parameters['iterations']
        borderType = parameters['borderType']
        borderValue = parameters['borderValue']
        dst = cv2.dilate(src=src, kernel=kernel, anchor=anchor, iterations=iterations, borderType=borderType, borderValue=borderValue)
        outputs['dst'] = Data(dst)

# cv2.distanceTransform
class OpenCVAuto2_DistanceTransform(NormalElement):
    name = 'Distance Transform'
    comment = '''distanceTransform(src, distanceType, maskSize[, dst[, dstType]]) -> dst\n@overload\n@param src 8-bit, single-channel (binary) source image.\n@param dst Output image with calculated distances. It is a 8-bit or 32-bit floating-point,\nsingle-channel image of the same size as src .\n@param distanceType Type of distance, see #DistanceTypes\n@param maskSize Size of the distance transform mask, see #DistanceTransformMasks. In case of the\n#DIST_L1 or #DIST_C distance type, the parameter is forced to 3 because a \f$3\times 3\f$ mask gives\nthe same result as \f$5\times 5\f$ or any larger aperture.\n@param dstType Type of output image. It can be CV_8U or CV_32F. Type CV_8U can be used only for\nthe first variant of the function and distanceType == #DIST_L1.'''
    package = "Transforms"

    def get_attributes(self):
        return [Input('src', 'src')], \
               [Output('dst', 'dst')], \
               [IntParameter('distanceType', 'Distance Type'),
                SizeParameter('maskSize', 'Mask Size'),
                IntParameter('dstType', 'Dst Type')]

    def process_inputs(self, inputs, outputs, parameters):
        src = inputs['src'].value
        distanceType = parameters['distanceType']
        maskSize = parameters['maskSize']
        dstType = parameters['dstType']
        dst = cv2.distanceTransform(src=src, distanceType=distanceType, maskSize=maskSize, dstType=dstType)
        outputs['dst'] = Data(dst)

# cv2.divide
class OpenCVAuto2_Divide(NormalElement):
    name = 'Divide'
    comment = '''divide(src1, src2[, dst[, scale[, dtype]]]) -> dst\n@brief Performs per-element division of two arrays or a scalar by an array.\n\nThe function cv::divide divides one array by another:\n\f[\texttt{dst(I) = saturate(src1(I)*scale/src2(I))}\f]\nor a scalar by an array when there is no src1 :\n\f[\texttt{dst(I) = saturate(scale/src2(I))}\f]\n\nWhen src2(I) is zero, dst(I) will also be zero. Different channels of\nmulti-channel arrays are processed independently.\n\n@note Saturation is not applied when the output array has the depth CV_32S. You may even get\nresult of an incorrect sign in the case of overflow.\n@param src1 first input array.\n@param src2 second input array of the same size and type as src1.\n@param scale scalar factor.\n@param dst output array of the same size and type as src2.\n@param dtype optional depth of the output array; if -1, dst will have depth src2.depth(), but in\ncase of an array-by-array division, you can only pass -1 when src1.depth()==src2.depth().\n@sa  multiply, add, subtract



divide(scale, src2[, dst[, dtype]]) -> dst\n@overload'''
    package = "Operators"

    def get_attributes(self):
        return [Input('src1', 'src 1'),
                Input('src2', 'src 2')], \
               [Output('dst', 'dst')], \
               [FloatParameter('scale', 'scale'),
                ComboboxParameter('dtype', name='dtype', values=[('NONE',-1),('CV_8U',0),('CV_8UC1',0),('CV_8S',1),('CV_8SC1',1),('CV_16U',2),('CV_16UC1',2),('CV_16S',3),('CV_16SC1',3),('CV_32S',4),('CV_32SC1',4),('CV_32F',5),('CV_32FC1',5),('CV_64F',6),('CV_64FC1',6),('CV_8UC2',8),('CV_8SC2',9),('CV_16UC2',10),('CV_16SC2',11),('CV_32SC2',12),('CV_32FC2',13),('CV_64FC2',14),('CV_8UC3',16),('CV_8SC3',17),('CV_16UC3',18),('CV_16SC3',19),('CV_32SC3',20),('CV_32FC3',21),('CV_64FC3',22),('CV_8UC4',24),('CV_8SC4',25),('CV_16UC4',26),('CV_16SC4',27),('CV_32SC4',28),('CV_32FC4',29),('CV_64FC4',30)])]

    def process_inputs(self, inputs, outputs, parameters):
        src1 = inputs['src1'].value
        src2 = inputs['src2'].value
        scale = parameters['scale']
        dtype = parameters['dtype']
        dst = cv2.divide(src1=src1, src2=src2, scale=scale, dtype=dtype)
        outputs['dst'] = Data(dst)

# cv2.drawContours
class OpenCVAuto2_DrawContours(NormalElement):
    name = 'Draw Contours'
    comment = '''drawContours(image, contours, contourIdx, color[, thickness[, lineType[, hierarchy[, maxLevel[, offset]]]]]) -> image\n@brief Draws contours outlines or filled contours.\n\nThe function draws contour outlines in the image if \f$\texttt{thickness} \ge 0\f$ or fills the area\nbounded by the contours if \f$\texttt{thickness}<0\f$ . The example below shows how to retrieve\nconnected components from the binary image and label them: :\n@include snippets/imgproc_drawContours.cpp\n\n@param image Destination image.\n@param contours All the input contours. Each contour is stored as a point vector.\n@param contourIdx Parameter indicating a contour to draw. If it is negative, all the contours are drawn.\n@param color Color of the contours.\n@param thickness Thickness of lines the contours are drawn with. If it is negative (for example,\nthickness=#FILLED ), the contour interiors are drawn.\n@param lineType Line connectivity. See #LineTypes\n@param hierarchy Optional information about hierarchy. It is only needed if you want to draw only\nsome of the contours (see maxLevel ).\n@param maxLevel Maximal level for drawn contours. If it is 0, only the specified contour is drawn.\nIf it is 1, the function draws the contour(s) and all the nested contours. If it is 2, the function\ndraws the contours, all the nested contours, all the nested-to-nested contours, and so on. This\nparameter is only taken into account when there is hierarchy available.\n@param offset Optional contour shift parameter. Shift all the drawn contours by the specified\n\f$\texttt{offset}=(dx,dy)\f$ .\n@note When thickness=#FILLED, the function is designed to handle connected components with holes correctly\neven when no hierarchy date is provided. This is done by analyzing all the outlines together\nusing even-odd rule. This may give incorrect results if you have a joint collection of separately retrieved\ncontours. In order to solve this problem, you need to call #drawContours separately for each sub-group\nof contours, or iterate over the collection using contourIdx parameter.'''
    package = "Drawing"

    def get_attributes(self):
        return [Input('image', 'image'),
                Input('contours', 'contours'),
                Input('hierarchy', 'hierarchy', optional=True)], \
               [Output('image', 'image')], \
               [IntParameter('contourIdx', 'Contour Idx'),
                ScalarParameter('color', 'color'),
                IntParameter('thickness', 'thickness', min_=-1, max_=100),
                ComboboxParameter('lineType', name='Line Type', values=[('LINE_4',4),('LINE_8',8),('LINE_AA',16)]),
                IntParameter('maxLevel', 'Max Level'),
                PointParameter('offset', 'offset')]

    def process_inputs(self, inputs, outputs, parameters):
        image = inputs['image'].value.copy()
        contours = inputs['contours'].value
        hierarchy = inputs['hierarchy'].value
        contourIdx = parameters['contourIdx']
        color = parameters['color']
        thickness = parameters['thickness']
        lineType = parameters['lineType']
        maxLevel = parameters['maxLevel']
        offset = parameters['offset']
        image = cv2.drawContours(image=image, contours=contours, hierarchy=hierarchy, contourIdx=contourIdx, color=color, thickness=thickness, lineType=lineType, maxLevel=maxLevel, offset=offset)
        outputs['image'] = Data(image)

# cv2.drawMarker
class OpenCVAuto2_DrawMarker(NormalElement):
    name = 'Draw Marker'
    comment = '''drawMarker(img, position, color[, markerType[, markerSize[, thickness[, line_type]]]]) -> img\n@brief Draws a marker on a predefined position in an image.\n\nThe function cv::drawMarker draws a marker on a given position in the image. For the moment several\nmarker types are supported, see #MarkerTypes for more information.\n\n@param img Image.\n@param position The point where the crosshair is positioned.\n@param color Line color.\n@param markerType The specific type of marker you want to use, see #MarkerTypes\n@param thickness Line thickness.\n@param line_type Type of the line, See #LineTypes\n@param markerSize The length of the marker axis [default = 20 pixels]'''
    package = "Drawing"

    def get_attributes(self):
        return [Input('img', 'img')], \
               [Output('img', 'img')], \
               [PointParameter('position', 'position'),
                ScalarParameter('color', 'color'),
                IntParameter('markerType', 'Marker Type'),
                SizeParameter('markerSize', 'Marker Size'),
                IntParameter('thickness', 'thickness', min_=-1, max_=100),
                IntParameter('line_type', 'line type')]

    def process_inputs(self, inputs, outputs, parameters):
        img = inputs['img'].value.copy()
        position = parameters['position']
        color = parameters['color']
        markerType = parameters['markerType']
        markerSize = parameters['markerSize']
        thickness = parameters['thickness']
        line_type = parameters['line_type']
        img = cv2.drawMarker(img=img, position=position, color=color, markerType=markerType, markerSize=markerSize, thickness=thickness, line_type=line_type)
        outputs['img'] = Data(img)

# cv2.edgePreservingFilter
class OpenCVAuto2_EdgePreservingFilter(NormalElement):
    name = 'Edge Preserving Filter'
    comment = '''edgePreservingFilter(src[, dst[, flags[, sigma_s[, sigma_r]]]]) -> dst\n@brief Filtering is the fundamental operation in image and video processing. Edge-preserving smoothing\nfilters are used in many different applications @cite EM11 .\n\n@param src Input 8-bit 3-channel image.\n@param dst Output 8-bit 3-channel image.\n@param flags Edge preserving filters:\n-   **RECURS_FILTER** = 1\n-   **NORMCONV_FILTER** = 2\n@param sigma_s Range between 0 to 200.\n@param sigma_r Range between 0 to 1.'''
    package = "Filters"

    def get_attributes(self):
        return [Input('src', 'src')], \
               [Output('dst', 'dst')], \
               [ComboboxParameter('flags', name='flags', values=[('RECURS_FILTER',1),('NORMCONV_FILTER',2)]),
                FloatParameter('sigma_s', 'sigma s'),
                FloatParameter('sigma_r', 'sigma r')]

    def process_inputs(self, inputs, outputs, parameters):
        src = inputs['src'].value
        flags = parameters['flags']
        sigma_s = parameters['sigma_s']
        sigma_r = parameters['sigma_r']
        dst = cv2.edgePreservingFilter(src=src, flags=flags, sigma_s=sigma_s, sigma_r=sigma_r)
        outputs['dst'] = Data(dst)

# cv2.eigen
class OpenCVAuto2_Eigen(NormalElement):
    name = 'Eigen'
    comment = '''eigen(src[, eigenvalues[, eigenvectors]]) -> retval, eigenvalues, eigenvectors\n@brief Calculates eigenvalues and eigenvectors of a symmetric matrix.\n\nThe function cv::eigen calculates just eigenvalues, or eigenvalues and eigenvectors of the symmetric\nmatrix src:\n@code\nsrc*eigenvectors.row(i).t() = eigenvalues.at<srcType>(i)*eigenvectors.row(i).t()\n@endcode\n\n@note Use cv::eigenNonSymmetric for calculation of real eigenvalues and eigenvectors of non-symmetric matrix.\n\n@param src input matrix that must have CV_32FC1 or CV_64FC1 type, square size and be symmetrical\n(src ^T^ == src).\n@param eigenvalues output vector of eigenvalues of the same type as src; the eigenvalues are stored\nin the descending order.\n@param eigenvectors output matrix of eigenvectors; it has the same size and type as src; the\neigenvectors are stored as subsequent matrix rows, in the same order as the corresponding\neigenvalues.\n@sa eigenNonSymmetric, completeSymm , PCA'''
    package = "Math"

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

# cv2.eigenNonSymmetric
class OpenCVAuto2_EigenNonSymmetric(NormalElement):
    name = 'Eigen Non Symmetric'
    comment = '''eigenNonSymmetric(src[, eigenvalues[, eigenvectors]]) -> eigenvalues, eigenvectors\n@brief Calculates eigenvalues and eigenvectors of a non-symmetric matrix (real eigenvalues only).\n\n@note Assumes real eigenvalues.\n\nThe function calculates eigenvalues and eigenvectors (optional) of the square matrix src:\n@code\nsrc*eigenvectors.row(i).t() = eigenvalues.at<srcType>(i)*eigenvectors.row(i).t()\n@endcode\n\n@param src input matrix (CV_32FC1 or CV_64FC1 type).\n@param eigenvalues output vector of eigenvalues (type is the same type as src).\n@param eigenvectors output matrix of eigenvectors (type is the same type as src). The eigenvectors are stored as subsequent matrix rows, in the same order as the corresponding eigenvalues.\n@sa eigen'''
    package = "Math"

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

# cv2.ellipse
class OpenCVAuto2_Ellipse(NormalElement):
    name = 'Ellipse'
    comment = '''ellipse(img, center, axes, angle, startAngle, endAngle, color[, thickness[, lineType[, shift]]]) -> img\n@brief Draws a simple or thick elliptic arc or fills an ellipse sector.\n\nThe function cv::ellipse with more parameters draws an ellipse outline, a filled ellipse, an elliptic\narc, or a filled ellipse sector. The drawing code uses general parametric form.\nA piecewise-linear curve is used to approximate the elliptic arc\nboundary. If you need more control of the ellipse rendering, you can retrieve the curve using\n#ellipse2Poly and then render it with #polylines or fill it with #fillPoly. If you use the first\nvariant of the function and want to draw the whole ellipse, not an arc, pass `startAngle=0` and\n`endAngle=360`. If `startAngle` is greater than `endAngle`, they are swapped. The figure below explains\nthe meaning of the parameters to draw the blue arc.\n\n![Parameters of Elliptic Arc](pics/ellipse.svg)\n\n@param img Image.\n@param center Center of the ellipse.\n@param axes Half of the size of the ellipse main axes.\n@param angle Ellipse rotation angle in degrees.\n@param startAngle Starting angle of the elliptic arc in degrees.\n@param endAngle Ending angle of the elliptic arc in degrees.\n@param color Ellipse color.\n@param thickness Thickness of the ellipse arc outline, if positive. Otherwise, this indicates that\na filled ellipse sector is to be drawn.\n@param lineType Type of the ellipse boundary. See #LineTypes\n@param shift Number of fractional bits in the coordinates of the center and values of axes.



ellipse(img, box, color[, thickness[, lineType]]) -> img\n@overload\n@param img Image.\n@param box Alternative ellipse representation via RotatedRect. This means that the function draws\nan ellipse inscribed in the rotated rectangle.\n@param color Ellipse color.\n@param thickness Thickness of the ellipse arc outline, if positive. Otherwise, this indicates that\na filled ellipse sector is to be drawn.\n@param lineType Type of the ellipse boundary. See #LineTypes'''
    package = "Drawing"

    def get_attributes(self):
        return [Input('img', 'img')], \
               [Output('img', 'img')], \
               [PointParameter('center', 'center'),
                SizeParameter('axes', 'axes'),
                FloatParameter('angle', 'angle'),
                FloatParameter('startAngle', 'Start Angle'),
                FloatParameter('endAngle', 'End Angle'),
                ScalarParameter('color', 'color'),
                IntParameter('thickness', 'thickness', min_=-1, max_=100),
                ComboboxParameter('lineType', name='Line Type', values=[('LINE_4',4),('LINE_8',8),('LINE_AA',16)]),
                IntParameter('shift', 'shift')]

    def process_inputs(self, inputs, outputs, parameters):
        img = inputs['img'].value.copy()
        center = parameters['center']
        axes = parameters['axes']
        angle = parameters['angle']
        startAngle = parameters['startAngle']
        endAngle = parameters['endAngle']
        color = parameters['color']
        thickness = parameters['thickness']
        lineType = parameters['lineType']
        shift = parameters['shift']
        img = cv2.ellipse(img=img, center=center, axes=axes, angle=angle, startAngle=startAngle, endAngle=endAngle, color=color, thickness=thickness, lineType=lineType, shift=shift)
        outputs['img'] = Data(img)

# cv2.ellipse2Poly
class OpenCVAuto2_Ellipse2Poly(NormalElement):
    name = 'Ellipse 2 Poly'
    comment = '''ellipse2Poly(center, axes, angle, arcStart, arcEnd, delta) -> pts\n@brief Approximates an elliptic arc with a polyline.\n\nThe function ellipse2Poly computes the vertices of a polyline that approximates the specified\nelliptic arc. It is used by #ellipse. If `arcStart` is greater than `arcEnd`, they are swapped.\n\n@param center Center of the arc.\n@param axes Half of the size of the ellipse main axes. See #ellipse for details.\n@param angle Rotation angle of the ellipse in degrees. See #ellipse for details.\n@param arcStart Starting angle of the elliptic arc in degrees.\n@param arcEnd Ending angle of the elliptic arc in degrees.\n@param delta Angle between the subsequent polyline vertices. It defines the approximation\naccuracy.\n@param pts Output vector of polyline vertices.'''
    package = "Drawing"

    def get_attributes(self):
        return [], \
               [Output('pts', 'pts')], \
               [PointParameter('center', 'center'),
                SizeParameter('axes', 'axes'),
                FloatParameter('angle', 'angle'),
                IntParameter('arcStart', 'Arc Start'),
                IntParameter('arcEnd', 'Arc End'),
                IntParameter('delta', 'delta')]

    def process_inputs(self, inputs, outputs, parameters):
        center = parameters['center']
        axes = parameters['axes']
        angle = parameters['angle']
        arcStart = parameters['arcStart']
        arcEnd = parameters['arcEnd']
        delta = parameters['delta']
        pts = cv2.ellipse2Poly(center=center, axes=axes, angle=angle, arcStart=arcStart, arcEnd=arcEnd, delta=delta)
        outputs['pts'] = Data(pts)

# cv2.equalizeHist
class OpenCVAuto2_EqualizeHist(NormalElement):
    name = 'Equalize Hist'
    comment = '''equalizeHist(src[, dst]) -> dst\n@brief Equalizes the histogram of a grayscale image.\n\nThe function equalizes the histogram of the input image using the following algorithm:\n\n- Calculate the histogram \f$H\f$ for src .\n- Normalize the histogram so that the sum of histogram bins is 255.\n- Compute the integral of the histogram:\n\f[H'_i =  \sum _{0  \le j < i} H(j)\f]\n- Transform the image using \f$H'\f$ as a look-up table: \f$\texttt{dst}(x,y) = H'(\texttt{src}(x,y))\f$\n\nThe algorithm normalizes the brightness and increases the contrast of the image.\n\n@param src Source 8-bit single channel image.\n@param dst Destination image of the same size and type as src .'''
    package = "Histogram"

    def get_attributes(self):
        return [Input('src', 'src')], \
               [Output('dst', 'dst')], \
               []

    def process_inputs(self, inputs, outputs, parameters):
        src = inputs['src'].value
        dst = cv2.equalizeHist(src=src)
        outputs['dst'] = Data(dst)

# cv2.erode
class OpenCVAuto2_Erode(NormalElement):
    name = 'Erode'
    comment = '''erode(src, kernel[, dst[, anchor[, iterations[, borderType[, borderValue]]]]]) -> dst\n@brief Erodes an image by using a specific structuring element.\n\nThe function erodes the source image using the specified structuring element that determines the\nshape of a pixel neighborhood over which the minimum is taken:\n\n\f[\texttt{dst} (x,y) =  \min _{(x',y'):  \, \texttt{element} (x',y') \ne0 } \texttt{src} (x+x',y+y')\f]\n\nThe function supports the in-place mode. Erosion can be applied several ( iterations ) times. In\ncase of multi-channel images, each channel is processed independently.\n\n@param src input image; the number of channels can be arbitrary, but the depth should be one of\nCV_8U, CV_16U, CV_16S, CV_32F or CV_64F.\n@param dst output image of the same size and type as src.\n@param kernel structuring element used for erosion; if `element=Mat()`, a `3 x 3` rectangular\nstructuring element is used. Kernel can be created using #getStructuringElement.\n@param anchor position of the anchor within the element; default value (-1, -1) means that the\nanchor is at the element center.\n@param iterations number of times erosion is applied.\n@param borderType pixel extrapolation method, see #BorderTypes\n@param borderValue border value in case of a constant border\n@sa  dilate, morphologyEx, getStructuringElement'''
    package = "Filters"

    def get_attributes(self):
        return [Input('src', 'src'),
                Input('kernel', 'kernel')], \
               [Output('dst', 'dst')], \
               [PointParameter('anchor', 'anchor'),
                IntParameter('iterations', 'iterations', min_=0),
                ComboboxParameter('borderType', name='Border Type', values=[('BORDER_CONSTANT',0),('BORDER_REPLICATE',1),('BORDER_REFLECT',2),('BORDER_WRAP',3),('BORDER_DEFAULT',4),('BORDER_REFLECT101',4),('BORDER_REFLECT_101',4),('BORDER_TRANSPARENT',5),('BORDER_ISOLATED',16)]),
                ScalarParameter('borderValue', 'Border Value')]

    def process_inputs(self, inputs, outputs, parameters):
        src = inputs['src'].value
        kernel = inputs['kernel'].value
        anchor = parameters['anchor']
        iterations = parameters['iterations']
        borderType = parameters['borderType']
        borderValue = parameters['borderValue']
        dst = cv2.erode(src=src, kernel=kernel, anchor=anchor, iterations=iterations, borderType=borderType, borderValue=borderValue)
        outputs['dst'] = Data(dst)

# cv2.estimateAffine2D
class OpenCVAuto2_EstimateAffine2D(NormalElement):
    name = 'Estimate Affine 2D'
    comment = '''estimateAffine2D(from, to[, inliers[, method[, ransacReprojThreshold[, maxIters[, confidence[, refineIters]]]]]]) -> retval, inliers\n@brief Computes an optimal affine transformation between two 2D point sets.\n\nIt computes\n\f[\n\begin{bmatrix}\nx\\\ny\\\n\end{bmatrix}\n=\n\begin{bmatrix}\na_{11} & a_{12}\\\na_{21} & a_{22}\\\n\end{bmatrix}\n\begin{bmatrix}\nX\\\nY\\\n\end{bmatrix}\n+\n\begin{bmatrix}\nb_1\\\nb_2\\\n\end{bmatrix}\n\f]\n\n@param from First input 2D point set containing \f$(X,Y)\f$.\n@param to Second input 2D point set containing \f$(x,y)\f$.\n@param inliers Output vector indicating which points are inliers (1-inlier, 0-outlier).\n@param method Robust method used to compute transformation. The following methods are possible:\n-   cv::RANSAC - RANSAC-based robust method\n-   cv::LMEDS - Least-Median robust method\nRANSAC is the default method.\n@param ransacReprojThreshold Maximum reprojection error in the RANSAC algorithm to consider\na point as an inlier. Applies only to RANSAC.\n@param maxIters The maximum number of robust method iterations.\n@param confidence Confidence level, between 0 and 1, for the estimated transformation. Anything\nbetween 0.95 and 0.99 is usually good enough. Values too close to 1 can slow down the estimation\nsignificantly. Values lower than 0.8-0.9 can result in an incorrectly estimated transformation.\n@param refineIters Maximum number of iterations of refining algorithm (Levenberg-Marquardt).\nPassing 0 will disable refining, so the output matrix will be output of robust method.\n\n@return Output 2D affine transformation matrix \f$2 \times 3\f$ or empty matrix if transformation\ncould not be estimated. The returned matrix has the following form:\n\f[\n\begin{bmatrix}\na_{11} & a_{12} & b_1\\\na_{21} & a_{22} & b_2\\\n\end{bmatrix}\n\f]\n\nThe function estimates an optimal 2D affine transformation between two 2D point sets using the\nselected robust algorithm.\n\nThe computed transformation is then refined further (using only inliers) with the\nLevenberg-Marquardt method to reduce the re-projection error even more.\n\n@note\nThe RANSAC method can handle practically any ratio of outliers but needs a threshold to\ndistinguish inliers from outliers. The method LMeDS does not need any threshold but it works\ncorrectly only when there are more than 50% of inliers.\n\n@sa estimateAffinePartial2D, getAffineTransform'''
    package = "3D calibration & reconstruction"

    def get_attributes(self):
        return [Input('from_', 'from'),
                Input('to', 'to')], \
               [Output('inliers', 'inliers')], \
               [IntParameter('method', 'method'),
                FloatParameter('ransacReprojThreshold', 'Ransac Reproj Threshold'),
                IntParameter('maxIters', 'Max Iters', min_=0),
                FloatParameter('confidence', 'confidence'),
                IntParameter('refineIters', 'Refine Iters', min_=0)]

    def process_inputs(self, inputs, outputs, parameters):
        from_ = inputs['from_'].value
        to = inputs['to'].value
        method = parameters['method']
        ransacReprojThreshold = parameters['ransacReprojThreshold']
        maxIters = parameters['maxIters']
        confidence = parameters['confidence']
        refineIters = parameters['refineIters']
        retval, inliers = cv2.estimateAffine2D(from_=from_, to=to, method=method, ransacReprojThreshold=ransacReprojThreshold, maxIters=maxIters, confidence=confidence, refineIters=refineIters)
        outputs['inliers'] = Data(inliers)

# cv2.estimateAffine3D
class OpenCVAuto2_EstimateAffine3D(NormalElement):
    name = 'Estimate Affine 3D'
    comment = '''estimateAffine3D(src, dst[, out[, inliers[, ransacThreshold[, confidence]]]]) -> retval, out, inliers\n@brief Computes an optimal affine transformation between two 3D point sets.\n\nIt computes\n\f[\n\begin{bmatrix}\nx\\\ny\\\nz\\\n\end{bmatrix}\n=\n\begin{bmatrix}\na_{11} & a_{12} & a_{13}\\\na_{21} & a_{22} & a_{23}\\\na_{31} & a_{32} & a_{33}\\\n\end{bmatrix}\n\begin{bmatrix}\nX\\\nY\\\nZ\\\n\end{bmatrix}\n+\n\begin{bmatrix}\nb_1\\\nb_2\\\nb_3\\\n\end{bmatrix}\n\f]\n\n@param src First input 3D point set containing \f$(X,Y,Z)\f$.\n@param dst Second input 3D point set containing \f$(x,y,z)\f$.\n@param out Output 3D affine transformation matrix \f$3 \times 4\f$ of the form\n\f[\n\begin{bmatrix}\na_{11} & a_{12} & a_{13} & b_1\\\na_{21} & a_{22} & a_{23} & b_2\\\na_{31} & a_{32} & a_{33} & b_3\\\n\end{bmatrix}\n\f]\n@param inliers Output vector indicating which points are inliers (1-inlier, 0-outlier).\n@param ransacThreshold Maximum reprojection error in the RANSAC algorithm to consider a point as\nan inlier.\n@param confidence Confidence level, between 0 and 1, for the estimated transformation. Anything\nbetween 0.95 and 0.99 is usually good enough. Values too close to 1 can slow down the estimation\nsignificantly. Values lower than 0.8-0.9 can result in an incorrectly estimated transformation.\n\nThe function estimates an optimal 3D affine transformation between two 3D point sets using the\nRANSAC algorithm.'''
    package = "3D calibration & reconstruction"

    def get_attributes(self):
        return [Input('src', 'src'),
                Input('dst', 'dst')], \
               [Output('out', 'out'),
                Output('inliers', 'inliers')], \
               [FloatParameter('ransacThreshold', 'Ransac Threshold'),
                FloatParameter('confidence', 'confidence')]

    def process_inputs(self, inputs, outputs, parameters):
        src = inputs['src'].value
        dst = inputs['dst'].value
        ransacThreshold = parameters['ransacThreshold']
        confidence = parameters['confidence']
        retval, out, inliers = cv2.estimateAffine3D(src=src, dst=dst, ransacThreshold=ransacThreshold, confidence=confidence)
        outputs['out'] = Data(out)
        outputs['inliers'] = Data(inliers)

# cv2.estimateAffinePartial2D
class OpenCVAuto2_EstimateAffinePartial2D(NormalElement):
    name = 'Estimate Affine Partial 2D'
    comment = '''estimateAffinePartial2D(from, to[, inliers[, method[, ransacReprojThreshold[, maxIters[, confidence[, refineIters]]]]]]) -> retval, inliers\n@brief Computes an optimal limited affine transformation with 4 degrees of freedom between\ntwo 2D point sets.\n\n@param from First input 2D point set.\n@param to Second input 2D point set.\n@param inliers Output vector indicating which points are inliers.\n@param method Robust method used to compute transformation. The following methods are possible:\n-   cv::RANSAC - RANSAC-based robust method\n-   cv::LMEDS - Least-Median robust method\nRANSAC is the default method.\n@param ransacReprojThreshold Maximum reprojection error in the RANSAC algorithm to consider\na point as an inlier. Applies only to RANSAC.\n@param maxIters The maximum number of robust method iterations.\n@param confidence Confidence level, between 0 and 1, for the estimated transformation. Anything\nbetween 0.95 and 0.99 is usually good enough. Values too close to 1 can slow down the estimation\nsignificantly. Values lower than 0.8-0.9 can result in an incorrectly estimated transformation.\n@param refineIters Maximum number of iterations of refining algorithm (Levenberg-Marquardt).\nPassing 0 will disable refining, so the output matrix will be output of robust method.\n\n@return Output 2D affine transformation (4 degrees of freedom) matrix \f$2 \times 3\f$ or\nempty matrix if transformation could not be estimated.\n\nThe function estimates an optimal 2D affine transformation with 4 degrees of freedom limited to\ncombinations of translation, rotation, and uniform scaling. Uses the selected algorithm for robust\nestimation.\n\nThe computed transformation is then refined further (using only inliers) with the\nLevenberg-Marquardt method to reduce the re-projection error even more.\n\nEstimated transformation matrix is:\n\f[ \begin{bmatrix} \cos(\theta) \cdot s & -\sin(\theta) \cdot s & t_x \\\n\sin(\theta) \cdot s & \cos(\theta) \cdot s & t_y\n\end{bmatrix} \f]\nWhere \f$ \theta \f$ is the rotation angle, \f$ s \f$ the scaling factor and \f$ t_x, t_y \f$ are\ntranslations in \f$ x, y \f$ axes respectively.\n\n@note\nThe RANSAC method can handle practically any ratio of outliers but need a threshold to\ndistinguish inliers from outliers. The method LMeDS does not need any threshold but it works\ncorrectly only when there are more than 50% of inliers.\n\n@sa estimateAffine2D, getAffineTransform'''
    package = "3D calibration & reconstruction"

    def get_attributes(self):
        return [Input('from_', 'from'),
                Input('to', 'to')], \
               [Output('inliers', 'inliers')], \
               [IntParameter('method', 'method'),
                FloatParameter('ransacReprojThreshold', 'Ransac Reproj Threshold'),
                IntParameter('maxIters', 'Max Iters', min_=0),
                FloatParameter('confidence', 'confidence'),
                IntParameter('refineIters', 'Refine Iters', min_=0)]

    def process_inputs(self, inputs, outputs, parameters):
        from_ = inputs['from_'].value
        to = inputs['to'].value
        method = parameters['method']
        ransacReprojThreshold = parameters['ransacReprojThreshold']
        maxIters = parameters['maxIters']
        confidence = parameters['confidence']
        refineIters = parameters['refineIters']
        retval, inliers = cv2.estimateAffinePartial2D(from_=from_, to=to, method=method, ransacReprojThreshold=ransacReprojThreshold, maxIters=maxIters, confidence=confidence, refineIters=refineIters)
        outputs['inliers'] = Data(inliers)

# cv2.exp
class OpenCVAuto2_Exp(NormalElement):
    name = 'Exp'
    comment = '''exp(src[, dst]) -> dst\n@brief Calculates the exponent of every array element.\n\nThe function cv::exp calculates the exponent of every element of the input\narray:\n\f[\texttt{dst} [I] = e^{ src(I) }\f]\n\nThe maximum relative error is about 7e-6 for single-precision input and\nless than 1e-10 for double-precision input. Currently, the function\nconverts denormalized values to zeros on output. Special values (NaN,\nInf) are not handled.\n@param src input array.\n@param dst output array of the same size and type as src.\n@sa log , cartToPolar , polarToCart , phase , pow , sqrt , magnitude'''
    package = "Math"

    def get_attributes(self):
        return [Input('src', 'src')], \
               [Output('dst', 'dst')], \
               []

    def process_inputs(self, inputs, outputs, parameters):
        src = inputs['src'].value
        dst = cv2.exp(src=src)
        outputs['dst'] = Data(dst)

# cv2.extractChannel
class OpenCVAuto2_ExtractChannel(NormalElement):
    name = 'Extract Channel'
    comment = '''extractChannel(src, coi[, dst]) -> dst\n@brief Extracts a single channel from src (coi is 0-based index)\n@param src input array\n@param dst output array\n@param coi index of channel to extract\n@sa mixChannels, split'''
    package = "Channels"

    def get_attributes(self):
        return [Input('src', 'src')], \
               [Output('dst', 'dst')], \
               [IntParameter('coi', 'coi')]

    def process_inputs(self, inputs, outputs, parameters):
        src = inputs['src'].value
        coi = parameters['coi']
        dst = cv2.extractChannel(src=src, coi=coi)
        outputs['dst'] = Data(dst)

# cv2.fastNlMeansDenoising
class OpenCVAuto2_FastNlMeansDenoising(NormalElement):
    name = 'Fast Nl Means Denoising'
    comment = '''fastNlMeansDenoising(src[, dst[, h[, templateWindowSize[, searchWindowSize]]]]) -> dst\n@brief Perform image denoising using Non-local Means Denoising algorithm\n<http://www.ipol.im/pub/algo/bcm_non_local_means_denoising/> with several computational\noptimizations. Noise expected to be a gaussian white noise\n\n@param src Input 8-bit 1-channel, 2-channel, 3-channel or 4-channel image.\n@param dst Output image with the same size and type as src .\n@param templateWindowSize Size in pixels of the template patch that is used to compute weights.\nShould be odd. Recommended value 7 pixels\n@param searchWindowSize Size in pixels of the window that is used to compute weighted average for\ngiven pixel. Should be odd. Affect performance linearly: greater searchWindowsSize - greater\ndenoising time. Recommended value 21 pixels\n@param h Parameter regulating filter strength. Big h value perfectly removes noise but also\nremoves image details, smaller h value preserves details but also preserves some noise\n\nThis function expected to be applied to grayscale images. For colored images look at\nfastNlMeansDenoisingColored. Advanced usage of this functions can be manual denoising of colored\nimage in different colorspaces. Such approach is used in fastNlMeansDenoisingColored by converting\nimage to CIELAB colorspace and then separately denoise L and AB components with different h\nparameter.



fastNlMeansDenoising(src, h[, dst[, templateWindowSize[, searchWindowSize[, normType]]]]) -> dst\n@brief Perform image denoising using Non-local Means Denoising algorithm\n<http://www.ipol.im/pub/algo/bcm_non_local_means_denoising/> with several computational\noptimizations. Noise expected to be a gaussian white noise\n\n@param src Input 8-bit or 16-bit (only with NORM_L1) 1-channel,\n2-channel, 3-channel or 4-channel image.\n@param dst Output image with the same size and type as src .\n@param templateWindowSize Size in pixels of the template patch that is used to compute weights.\nShould be odd. Recommended value 7 pixels\n@param searchWindowSize Size in pixels of the window that is used to compute weighted average for\ngiven pixel. Should be odd. Affect performance linearly: greater searchWindowsSize - greater\ndenoising time. Recommended value 21 pixels\n@param h Array of parameters regulating filter strength, either one\nparameter applied to all channels or one per channel in dst. Big h value\nperfectly removes noise but also removes image details, smaller h\nvalue preserves details but also preserves some noise\n@param normType Type of norm used for weight calculation. Can be either NORM_L2 or NORM_L1\n\nThis function expected to be applied to grayscale images. For colored images look at\nfastNlMeansDenoisingColored. Advanced usage of this functions can be manual denoising of colored\nimage in different colorspaces. Such approach is used in fastNlMeansDenoisingColored by converting\nimage to CIELAB colorspace and then separately denoise L and AB components with different h\nparameter.'''
    package = "Photo"

    def get_attributes(self):
        return [Input('src', 'src'),
                Input('h', 'h', optional=True)], \
               [Output('dst', 'dst')], \
               [SizeParameter('templateWindowSize', 'Template Window Size'),
                SizeParameter('searchWindowSize', 'Search Window Size')]

    def process_inputs(self, inputs, outputs, parameters):
        src = inputs['src'].value
        h = inputs['h'].value
        templateWindowSize = parameters['templateWindowSize']
        searchWindowSize = parameters['searchWindowSize']
        dst = cv2.fastNlMeansDenoising(src=src, h=h, templateWindowSize=templateWindowSize, searchWindowSize=searchWindowSize)
        outputs['dst'] = Data(dst)

# cv2.fastNlMeansDenoisingColored
class OpenCVAuto2_FastNlMeansDenoisingColored(NormalElement):
    name = 'Fast Nl Means Denoising Colored'
    comment = '''fastNlMeansDenoisingColored(src[, dst[, h[, hColor[, templateWindowSize[, searchWindowSize]]]]]) -> dst\n@brief Modification of fastNlMeansDenoising function for colored images\n\n@param src Input 8-bit 3-channel image.\n@param dst Output image with the same size and type as src .\n@param templateWindowSize Size in pixels of the template patch that is used to compute weights.\nShould be odd. Recommended value 7 pixels\n@param searchWindowSize Size in pixels of the window that is used to compute weighted average for\ngiven pixel. Should be odd. Affect performance linearly: greater searchWindowsSize - greater\ndenoising time. Recommended value 21 pixels\n@param h Parameter regulating filter strength for luminance component. Bigger h value perfectly\nremoves noise but also removes image details, smaller h value preserves details but also preserves\nsome noise\n@param hColor The same as h but for color components. For most images value equals 10\nwill be enough to remove colored noise and do not distort colors\n\nThe function converts image to CIELAB colorspace and then separately denoise L and AB components\nwith given h parameters using fastNlMeansDenoising function.'''
    package = "Photo"

    def get_attributes(self):
        return [Input('src', 'src')], \
               [Output('dst', 'dst')], \
               [FloatParameter('h', 'h'),
                FloatParameter('hColor', 'H Color'),
                SizeParameter('templateWindowSize', 'Template Window Size'),
                SizeParameter('searchWindowSize', 'Search Window Size')]

    def process_inputs(self, inputs, outputs, parameters):
        src = inputs['src'].value
        h = parameters['h']
        hColor = parameters['hColor']
        templateWindowSize = parameters['templateWindowSize']
        searchWindowSize = parameters['searchWindowSize']
        dst = cv2.fastNlMeansDenoisingColored(src=src, h=h, hColor=hColor, templateWindowSize=templateWindowSize, searchWindowSize=searchWindowSize)
        outputs['dst'] = Data(dst)

# cv2.fastNlMeansDenoisingColoredMulti
class OpenCVAuto2_FastNlMeansDenoisingColoredMulti(NormalElement):
    name = 'Fast Nl Means Denoising Colored Multi'
    comment = '''fastNlMeansDenoisingColoredMulti(srcImgs, imgToDenoiseIndex, temporalWindowSize[, dst[, h[, hColor[, templateWindowSize[, searchWindowSize]]]]]) -> dst\n@brief Modification of fastNlMeansDenoisingMulti function for colored images sequences\n\n@param srcImgs Input 8-bit 3-channel images sequence. All images should have the same type and\nsize.\n@param imgToDenoiseIndex Target image to denoise index in srcImgs sequence\n@param temporalWindowSize Number of surrounding images to use for target image denoising. Should\nbe odd. Images from imgToDenoiseIndex - temporalWindowSize / 2 to\nimgToDenoiseIndex - temporalWindowSize / 2 from srcImgs will be used to denoise\nsrcImgs[imgToDenoiseIndex] image.\n@param dst Output image with the same size and type as srcImgs images.\n@param templateWindowSize Size in pixels of the template patch that is used to compute weights.\nShould be odd. Recommended value 7 pixels\n@param searchWindowSize Size in pixels of the window that is used to compute weighted average for\ngiven pixel. Should be odd. Affect performance linearly: greater searchWindowsSize - greater\ndenoising time. Recommended value 21 pixels\n@param h Parameter regulating filter strength for luminance component. Bigger h value perfectly\nremoves noise but also removes image details, smaller h value preserves details but also preserves\nsome noise.\n@param hColor The same as h but for color components.\n\nThe function converts images to CIELAB colorspace and then separately denoise L and AB components\nwith given h parameters using fastNlMeansDenoisingMulti function.'''
    package = "Filters"

    def get_attributes(self):
        return [Input('srcImgs', 'Src Imgs')], \
               [Output('dst', 'dst')], \
               [IntParameter('imgToDenoiseIndex', 'Img To Denoise Index'),
                SizeParameter('temporalWindowSize', 'Temporal Window Size'),
                FloatParameter('h', 'h'),
                FloatParameter('hColor', 'H Color'),
                SizeParameter('templateWindowSize', 'Template Window Size'),
                SizeParameter('searchWindowSize', 'Search Window Size')]

    def process_inputs(self, inputs, outputs, parameters):
        srcImgs = inputs['srcImgs'].value
        imgToDenoiseIndex = parameters['imgToDenoiseIndex']
        temporalWindowSize = parameters['temporalWindowSize']
        h = parameters['h']
        hColor = parameters['hColor']
        templateWindowSize = parameters['templateWindowSize']
        searchWindowSize = parameters['searchWindowSize']
        dst = cv2.fastNlMeansDenoisingColoredMulti(srcImgs=srcImgs, imgToDenoiseIndex=imgToDenoiseIndex, temporalWindowSize=temporalWindowSize, h=h, hColor=hColor, templateWindowSize=templateWindowSize, searchWindowSize=searchWindowSize)
        outputs['dst'] = Data(dst)

# cv2.fastNlMeansDenoisingMulti
class OpenCVAuto2_FastNlMeansDenoisingMulti(NormalElement):
    name = 'Fast Nl Means Denoising Multi'
    comment = '''fastNlMeansDenoisingMulti(srcImgs, imgToDenoiseIndex, temporalWindowSize[, dst[, h[, templateWindowSize[, searchWindowSize]]]]) -> dst\n@brief Modification of fastNlMeansDenoising function for images sequence where consecutive images have been\ncaptured in small period of time. For example video. This version of the function is for grayscale\nimages or for manual manipulation with colorspaces. For more details see\n<http://citeseerx.ist.psu.edu/viewdoc/summary?doi=10.1.1.131.6394>\n\n@param srcImgs Input 8-bit 1-channel, 2-channel, 3-channel or\n4-channel images sequence. All images should have the same type and\nsize.\n@param imgToDenoiseIndex Target image to denoise index in srcImgs sequence\n@param temporalWindowSize Number of surrounding images to use for target image denoising. Should\nbe odd. Images from imgToDenoiseIndex - temporalWindowSize / 2 to\nimgToDenoiseIndex - temporalWindowSize / 2 from srcImgs will be used to denoise\nsrcImgs[imgToDenoiseIndex] image.\n@param dst Output image with the same size and type as srcImgs images.\n@param templateWindowSize Size in pixels of the template patch that is used to compute weights.\nShould be odd. Recommended value 7 pixels\n@param searchWindowSize Size in pixels of the window that is used to compute weighted average for\ngiven pixel. Should be odd. Affect performance linearly: greater searchWindowsSize - greater\ndenoising time. Recommended value 21 pixels\n@param h Parameter regulating filter strength. Bigger h value\nperfectly removes noise but also removes image details, smaller h\nvalue preserves details but also preserves some noise



fastNlMeansDenoisingMulti(srcImgs, imgToDenoiseIndex, temporalWindowSize, h[, dst[, templateWindowSize[, searchWindowSize[, normType]]]]) -> dst\n@brief Modification of fastNlMeansDenoising function for images sequence where consecutive images have been\ncaptured in small period of time. For example video. This version of the function is for grayscale\nimages or for manual manipulation with colorspaces. For more details see\n<http://citeseerx.ist.psu.edu/viewdoc/summary?doi=10.1.1.131.6394>\n\n@param srcImgs Input 8-bit or 16-bit (only with NORM_L1) 1-channel,\n2-channel, 3-channel or 4-channel images sequence. All images should\nhave the same type and size.\n@param imgToDenoiseIndex Target image to denoise index in srcImgs sequence\n@param temporalWindowSize Number of surrounding images to use for target image denoising. Should\nbe odd. Images from imgToDenoiseIndex - temporalWindowSize / 2 to\nimgToDenoiseIndex - temporalWindowSize / 2 from srcImgs will be used to denoise\nsrcImgs[imgToDenoiseIndex] image.\n@param dst Output image with the same size and type as srcImgs images.\n@param templateWindowSize Size in pixels of the template patch that is used to compute weights.\nShould be odd. Recommended value 7 pixels\n@param searchWindowSize Size in pixels of the window that is used to compute weighted average for\ngiven pixel. Should be odd. Affect performance linearly: greater searchWindowsSize - greater\ndenoising time. Recommended value 21 pixels\n@param h Array of parameters regulating filter strength, either one\nparameter applied to all channels or one per channel in dst. Big h value\nperfectly removes noise but also removes image details, smaller h\nvalue preserves details but also preserves some noise\n@param normType Type of norm used for weight calculation. Can be either NORM_L2 or NORM_L1'''
    package = "Filters"

    def get_attributes(self):
        return [Input('srcImgs', 'Src Imgs'),
                Input('h', 'h', optional=True)], \
               [Output('dst', 'dst')], \
               [IntParameter('imgToDenoiseIndex', 'Img To Denoise Index'),
                SizeParameter('temporalWindowSize', 'Temporal Window Size'),
                SizeParameter('templateWindowSize', 'Template Window Size'),
                SizeParameter('searchWindowSize', 'Search Window Size')]

    def process_inputs(self, inputs, outputs, parameters):
        srcImgs = inputs['srcImgs'].value
        h = inputs['h'].value
        imgToDenoiseIndex = parameters['imgToDenoiseIndex']
        temporalWindowSize = parameters['temporalWindowSize']
        templateWindowSize = parameters['templateWindowSize']
        searchWindowSize = parameters['searchWindowSize']
        dst = cv2.fastNlMeansDenoisingMulti(srcImgs=srcImgs, h=h, imgToDenoiseIndex=imgToDenoiseIndex, temporalWindowSize=temporalWindowSize, templateWindowSize=templateWindowSize, searchWindowSize=searchWindowSize)
        outputs['dst'] = Data(dst)

# cv2.fillConvexPoly
class OpenCVAuto2_FillConvexPoly(NormalElement):
    name = 'Fill Convex Poly'
    comment = '''fillConvexPoly(img, points, color[, lineType[, shift]]) -> img\n@brief Fills a convex polygon.\n\nThe function cv::fillConvexPoly draws a filled convex polygon. This function is much faster than the\nfunction #fillPoly . It can fill not only convex polygons but any monotonic polygon without\nself-intersections, that is, a polygon whose contour intersects every horizontal line (scan line)\ntwice at the most (though, its top-most and/or the bottom edge could be horizontal).\n\n@param img Image.\n@param points Polygon vertices.\n@param color Polygon color.\n@param lineType Type of the polygon boundaries. See #LineTypes\n@param shift Number of fractional bits in the vertex coordinates.'''
    package = "Drawing"

    def get_attributes(self):
        return [Input('img', 'img'),
                Input('points', 'points')], \
               [Output('img', 'img')], \
               [ScalarParameter('color', 'color'),
                ComboboxParameter('lineType', name='Line Type', values=[('LINE_4',4),('LINE_8',8),('LINE_AA',16)]),
                IntParameter('shift', 'shift')]

    def process_inputs(self, inputs, outputs, parameters):
        img = inputs['img'].value.copy()
        points = inputs['points'].value
        color = parameters['color']
        lineType = parameters['lineType']
        shift = parameters['shift']
        img = cv2.fillConvexPoly(img=img, points=points, color=color, lineType=lineType, shift=shift)
        outputs['img'] = Data(img)

# cv2.fillPoly
class OpenCVAuto2_FillPoly(NormalElement):
    name = 'Fill Poly'
    comment = '''fillPoly(img, pts, color[, lineType[, shift[, offset]]]) -> img\n@brief Fills the area bounded by one or more polygons.\n\nThe function cv::fillPoly fills an area bounded by several polygonal contours. The function can fill\ncomplex areas, for example, areas with holes, contours with self-intersections (some of their\nparts), and so forth.\n\n@param img Image.\n@param pts Array of polygons where each polygon is represented as an array of points.\n@param color Polygon color.\n@param lineType Type of the polygon boundaries. See #LineTypes\n@param shift Number of fractional bits in the vertex coordinates.\n@param offset Optional offset of all points of the contours.'''
    package = "Drawing"

    def get_attributes(self):
        return [Input('img', 'img'),
                Input('pts', 'pts')], \
               [Output('img', 'img')], \
               [ScalarParameter('color', 'color'),
                ComboboxParameter('lineType', name='Line Type', values=[('LINE_4',4),('LINE_8',8),('LINE_AA',16)]),
                IntParameter('shift', 'shift'),
                PointParameter('offset', 'offset')]

    def process_inputs(self, inputs, outputs, parameters):
        img = inputs['img'].value.copy()
        pts = inputs['pts'].value
        color = parameters['color']
        lineType = parameters['lineType']
        shift = parameters['shift']
        offset = parameters['offset']
        img = cv2.fillPoly(img=img, pts=pts, color=color, lineType=lineType, shift=shift, offset=offset)
        outputs['img'] = Data(img)

# cv2.filter2D
class OpenCVAuto2_Filter2D(NormalElement):
    name = 'Filter 2D'
    comment = '''filter2D(src, ddepth, kernel[, dst[, anchor[, delta[, borderType]]]]) -> dst\n@brief Convolves an image with the kernel.\n\nThe function applies an arbitrary linear filter to an image. In-place operation is supported. When\nthe aperture is partially outside the image, the function interpolates outlier pixel values\naccording to the specified border mode.\n\nThe function does actually compute correlation, not the convolution:\n\n\f[\texttt{dst} (x,y) =  \sum _{ \stackrel{0\leq x' < \texttt{kernel.cols},}{0\leq y' < \texttt{kernel.rows}} }  \texttt{kernel} (x',y')* \texttt{src} (x+x'- \texttt{anchor.x} ,y+y'- \texttt{anchor.y} )\f]\n\nThat is, the kernel is not mirrored around the anchor point. If you need a real convolution, flip\nthe kernel using #flip and set the new anchor to `(kernel.cols - anchor.x - 1, kernel.rows -\nanchor.y - 1)`.\n\nThe function uses the DFT-based algorithm in case of sufficiently large kernels (~`11 x 11` or\nlarger) and the direct algorithm for small kernels.\n\n@param src input image.\n@param dst output image of the same size and the same number of channels as src.\n@param ddepth desired depth of the destination image, see @ref filter_depths "combinations"\n@param kernel convolution kernel (or rather a correlation kernel), a single-channel floating point\nmatrix; if you want to apply different kernels to different channels, split the image into\nseparate color planes using split and process them individually.\n@param anchor anchor of the kernel that indicates the relative position of a filtered point within\nthe kernel; the anchor should lie within the kernel; default value (-1,-1) means that the anchor\nis at the kernel center.\n@param delta optional value added to the filtered pixels before storing them in dst.\n@param borderType pixel extrapolation method, see #BorderTypes\n@sa  sepFilter2D, dft, matchTemplate'''
    package = "Filters"

    def get_attributes(self):
        return [Input('src', 'src'),
                Input('kernel', 'kernel')], \
               [Output('dst', 'dst')], \
               [ComboboxParameter('ddepth', name='ddepth', values=[('NONE',-1),('CV_8U',0),('CV_8UC1',0),('CV_8S',1),('CV_8SC1',1),('CV_16U',2),('CV_16UC1',2),('CV_16S',3),('CV_16SC1',3),('CV_32S',4),('CV_32SC1',4),('CV_32F',5),('CV_32FC1',5),('CV_64F',6),('CV_64FC1',6),('CV_8UC2',8),('CV_8SC2',9),('CV_16UC2',10),('CV_16SC2',11),('CV_32SC2',12),('CV_32FC2',13),('CV_64FC2',14),('CV_8UC3',16),('CV_8SC3',17),('CV_16UC3',18),('CV_16SC3',19),('CV_32SC3',20),('CV_32FC3',21),('CV_64FC3',22),('CV_8UC4',24),('CV_8SC4',25),('CV_16UC4',26),('CV_16SC4',27),('CV_32SC4',28),('CV_32FC4',29),('CV_64FC4',30)]),
                PointParameter('anchor', 'anchor'),
                FloatParameter('delta', 'delta'),
                ComboboxParameter('borderType', name='Border Type', values=[('BORDER_CONSTANT',0),('BORDER_REPLICATE',1),('BORDER_REFLECT',2),('BORDER_WRAP',3),('BORDER_DEFAULT',4),('BORDER_REFLECT101',4),('BORDER_REFLECT_101',4),('BORDER_TRANSPARENT',5),('BORDER_ISOLATED',16)])]

    def process_inputs(self, inputs, outputs, parameters):
        src = inputs['src'].value
        kernel = inputs['kernel'].value
        ddepth = parameters['ddepth']
        anchor = parameters['anchor']
        delta = parameters['delta']
        borderType = parameters['borderType']
        dst = cv2.filter2D(src=src, kernel=kernel, ddepth=ddepth, anchor=anchor, delta=delta, borderType=borderType)
        outputs['dst'] = Data(dst)

# cv2.filterHomographyDecompByVisibleRefpoints
class OpenCVAuto2_FilterHomographyDecompByVisibleRefpoints(NormalElement):
    name = 'Filter Homography Decomp By Visible Refpoints'
    comment = '''filterHomographyDecompByVisibleRefpoints(rotations, normals, beforePoints, afterPoints[, possibleSolutions[, pointsMask]]) -> possibleSolutions\n@brief Filters homography decompositions based on additional information.\n\n@param rotations Vector of rotation matrices.\n@param normals Vector of plane normal matrices.\n@param beforePoints Vector of (rectified) visible reference points before the homography is applied\n@param afterPoints Vector of (rectified) visible reference points after the homography is applied\n@param possibleSolutions Vector of int indices representing the viable solution set after filtering\n@param pointsMask optional Mat/Vector of 8u type representing the mask for the inliers as given by the findHomography function\n\nThis function is intended to filter the output of the decomposeHomographyMat based on additional\ninformation as described in @cite Malis . The summary of the method: the decomposeHomographyMat function\nreturns 2 unique solutions and their "opposites" for a total of 4 solutions. If we have access to the\nsets of points visible in the camera frame before and after the homography transformation is applied,\nwe can determine which are the true potential solutions and which are the opposites by verifying which\nhomographies are consistent with all visible reference points being in front of the camera. The inputs\nare left unchanged; the filtered solution set is returned as indices into the existing one.'''
    package = "3D calibration & reconstruction"

    def get_attributes(self):
        return [Input('rotations', 'rotations'),
                Input('normals', 'normals'),
                Input('beforePoints', 'Before Points'),
                Input('afterPoints', 'After Points'),
                Input('pointsMask', 'Points Mask', optional=True)], \
               [Output('possibleSolutions', 'Possible Solutions')], \
               []

    def process_inputs(self, inputs, outputs, parameters):
        rotations = inputs['rotations'].value
        normals = inputs['normals'].value
        beforePoints = inputs['beforePoints'].value
        afterPoints = inputs['afterPoints'].value
        pointsMask = inputs['pointsMask'].value
        possibleSolutions = cv2.filterHomographyDecompByVisibleRefpoints(rotations=rotations, normals=normals, beforePoints=beforePoints, afterPoints=afterPoints, pointsMask=pointsMask)
        outputs['possibleSolutions'] = Data(possibleSolutions)

# cv2.filterSpeckles
class OpenCVAuto2_FilterSpeckles(NormalElement):
    name = 'Filter Speckles'
    comment = '''filterSpeckles(img, newVal, maxSpeckleSize, maxDiff[, buf]) -> img, buf\n@brief Filters off small noise blobs (speckles) in the disparity map\n\n@param img The input 16-bit signed disparity image\n@param newVal The disparity value used to paint-off the speckles\n@param maxSpeckleSize The maximum speckle size to consider it a speckle. Larger blobs are not\naffected by the algorithm\n@param maxDiff Maximum difference between neighbor disparity pixels to put them into the same\nblob. Note that since StereoBM, StereoSGBM and may be other algorithms return a fixed-point\ndisparity map, where disparity values are multiplied by 16, this scale factor should be taken into\naccount when specifying this parameter value.\n@param buf The optional temporary buffer to avoid memory allocation within the function.'''
    package = "3D calibration & reconstruction"

    def get_attributes(self):
        return [Input('img', 'img')], \
               [Output('img', 'img'),
                Output('buf', 'buf')], \
               [FloatParameter('newVal', 'New Val'),
                SizeParameter('maxSpeckleSize', 'Max Speckle Size'),
                FloatParameter('maxDiff', 'Max Diff')]

    def process_inputs(self, inputs, outputs, parameters):
        img = inputs['img'].value.copy()
        newVal = parameters['newVal']
        maxSpeckleSize = parameters['maxSpeckleSize']
        maxDiff = parameters['maxDiff']
        img, buf = cv2.filterSpeckles(img=img, newVal=newVal, maxSpeckleSize=maxSpeckleSize, maxDiff=maxDiff)
        outputs['img'] = Data(img)
        outputs['buf'] = Data(buf)

# cv2.findChessboardCorners
class OpenCVAuto2_FindChessboardCorners(NormalElement):
    name = 'Find Chessboard Corners'
    comment = '''findChessboardCorners(image, patternSize[, corners[, flags]]) -> retval, corners\n@brief Finds the positions of internal corners of the chessboard.\n\n@param image Source chessboard view. It must be an 8-bit grayscale or color image.\n@param patternSize Number of inner corners per a chessboard row and column\n( patternSize = cvSize(points_per_row,points_per_colum) = cvSize(columns,rows) ).\n@param corners Output array of detected corners.\n@param flags Various operation flags that can be zero or a combination of the following values:\n-   **CALIB_CB_ADAPTIVE_THRESH** Use adaptive thresholding to convert the image to black\nand white, rather than a fixed threshold level (computed from the average image brightness).\n-   **CALIB_CB_NORMALIZE_IMAGE** Normalize the image gamma with equalizeHist before\napplying fixed or adaptive thresholding.\n-   **CALIB_CB_FILTER_QUADS** Use additional criteria (like contour area, perimeter,\nsquare-like shape) to filter out false quads extracted at the contour retrieval stage.\n-   **CALIB_CB_FAST_CHECK** Run a fast check on the image that looks for chessboard corners,\nand shortcut the call if none is found. This can drastically speed up the call in the\ndegenerate condition when no chessboard is observed.\n\nThe function attempts to determine whether the input image is a view of the chessboard pattern and\nlocate the internal chessboard corners. The function returns a non-zero value if all of the corners\nare found and they are placed in a certain order (row by row, left to right in every row).\nOtherwise, if the function fails to find all the corners or reorder them, it returns 0. For example,\na regular chessboard has 8 x 8 squares and 7 x 7 internal corners, that is, points where the black\nsquares touch each other. The detected coordinates are approximate, and to determine their positions\nmore accurately, the function calls cornerSubPix. You also may use the function cornerSubPix with\ndifferent parameters if returned coordinates are not accurate enough.\n\nSample usage of detecting and drawing chessboard corners: :\n@code\nSize patternsize(8,6); //interior number of corners\nMat gray = ....; //source image\nvector<Point2f> corners; //this will be filled by the detected corners\n\n//CALIB_CB_FAST_CHECK saves a lot of time on images\n//that do not contain any chessboard corners\nbool patternfound = findChessboardCorners(gray, patternsize, corners,\nCALIB_CB_ADAPTIVE_THRESH + CALIB_CB_NORMALIZE_IMAGE\n+ CALIB_CB_FAST_CHECK);\n\nif(patternfound)\ncornerSubPix(gray, corners, Size(11, 11), Size(-1, -1),\nTermCriteria(CV_TERMCRIT_EPS + CV_TERMCRIT_ITER, 30, 0.1));\n\ndrawChessboardCorners(img, patternsize, Mat(corners), patternfound);\n@endcode\n@note The function requires white space (like a square-thick border, the wider the better) around\nthe board to make the detection more robust in various environments. Otherwise, if there is no\nborder and the background is dark, the outer black squares cannot be segmented properly and so the\nsquare grouping and ordering algorithm fails.'''
    package = "3D calibration & reconstruction"

    def get_attributes(self):
        return [Input('image', 'image')], \
               [Output('corners', 'corners')], \
               [SizeParameter('patternSize', 'Pattern Size'),
                ComboboxParameter('flags', name='flags', values=[('CALIB_CB_ADAPTIVE_THRESH',1),('CALIB_CB_SYMMETRIC_GRID',1),('CALIB_USE_INTRINSIC_GUESS',1),('CALIB_CB_NORMALIZE_IMAGE',2),('CALIB_CB_ASYMMETRIC_GRID',2),('CALIB_FIX_ASPECT_RATIO',2),('CALIB_CB_FILTER_QUADS',4),('CALIB_CB_CLUSTERING',4),('CALIB_FIX_PRINCIPAL_POINT',4),('CALIB_CB_FAST_CHECK',8),('CALIB_ZERO_TANGENT_DIST',8),('CALIB_FIX_FOCAL_LENGTH',16),('CALIB_FIX_K1',32),('CALIB_FIX_K2',64),('CALIB_FIX_K3',128),('CALIB_FIX_INTRINSIC',256),('CALIB_SAME_FOCAL_LENGTH',512),('CALIB_ZERO_DISPARITY',1024),('CALIB_FIX_K4',2048),('CALIB_FIX_K5',4096),('CALIB_FIX_K6',8192),('CALIB_RATIONAL_MODEL',16384),('CALIB_THIN_PRISM_MODEL',32768),('CALIB_FIX_S1_S2_S3_S4',65536),('CALIB_USE_LU',131072),('CALIB_TILTED_MODEL',262144),('CALIB_FIX_TAUX_TAUY',524288),('CALIB_USE_QR',1048576),('CALIB_FIX_TANGENT_DIST',2097152),('CALIB_USE_EXTRINSIC_GUESS',4194304)])]

    def process_inputs(self, inputs, outputs, parameters):
        image = inputs['image'].value
        patternSize = parameters['patternSize']
        flags = parameters['flags']
        retval, corners = cv2.findChessboardCorners(image=image, patternSize=patternSize, flags=flags)
        outputs['corners'] = Data(corners)

# cv2.findContours
class OpenCVAuto2_FindContours(NormalElement):
    name = 'Find Contours'
    comment = '''findContours(image, mode, method[, contours[, hierarchy[, offset]]]) -> image, contours, hierarchy\n@brief Finds contours in a binary image.\n\nThe function retrieves contours from the binary image using the algorithm @cite Suzuki85 . The contours\nare a useful tool for shape analysis and object detection and recognition. See squares.cpp in the\nOpenCV sample directory.\n@note Since opencv 3.2 source image is not modified by this function.\n\n@param image Source, an 8-bit single-channel image. Non-zero pixels are treated as 1's. Zero\npixels remain 0's, so the image is treated as binary . You can use #compare, #inRange, #threshold ,\n#adaptiveThreshold, #Canny, and others to create a binary image out of a grayscale or color one.\nIf mode equals to #RETR_CCOMP or #RETR_FLOODFILL, the input can also be a 32-bit integer image of labels (CV_32SC1).\n@param contours Detected contours. Each contour is stored as a vector of points (e.g.\nstd::vector<std::vector<cv::Point> >).\n@param hierarchy Optional output vector (e.g. std::vector<cv::Vec4i>), containing information about the image topology. It has\nas many elements as the number of contours. For each i-th contour contours[i], the elements\nhierarchy[i][0] , hierarchy[i][1] , hierarchy[i][2] , and hierarchy[i][3] are set to 0-based indices\nin contours of the next and previous contours at the same hierarchical level, the first child\ncontour and the parent contour, respectively. If for the contour i there are no next, previous,\nparent, or nested contours, the corresponding elements of hierarchy[i] will be negative.\n@param mode Contour retrieval mode, see #RetrievalModes\n@param method Contour approximation method, see #ContourApproximationModes\n@param offset Optional offset by which every contour point is shifted. This is useful if the\ncontours are extracted from the image ROI and then they should be analyzed in the whole image\ncontext.'''
    package = "Shape"

    def get_attributes(self):
        return [Input('image', 'image')], \
               [Output('image', 'image'),
                Output('contours', 'contours'),
                Output('hierarchy', 'hierarchy')], \
               [IntParameter('mode', 'mode'),
                IntParameter('method', 'method'),
                PointParameter('offset', 'offset')]

    def process_inputs(self, inputs, outputs, parameters):
        image = inputs['image'].value.copy()
        mode = parameters['mode']
        method = parameters['method']
        offset = parameters['offset']
        image, contours, hierarchy = cv2.findContours(image=image, mode=mode, method=method, offset=offset)
        outputs['image'] = Data(image)
        outputs['contours'] = Data(contours)
        outputs['hierarchy'] = Data(hierarchy)

# cv2.findEssentialMat
class OpenCVAuto2_FindEssentialMat(NormalElement):
    name = 'Find Essential Mat'
    comment = '''findEssentialMat(points1, points2, cameraMatrix[, method[, prob[, threshold[, mask]]]]) -> retval, mask\n@brief Calculates an essential matrix from the corresponding points in two images.\n\n@param points1 Array of N (N \>= 5) 2D points from the first image. The point coordinates should\nbe floating-point (single or double precision).\n@param points2 Array of the second image points of the same size and format as points1 .\n@param cameraMatrix Camera matrix \f$K = \vecthreethree{f_x}{0}{c_x}{0}{f_y}{c_y}{0}{0}{1}\f$ .\nNote that this function assumes that points1 and points2 are feature points from cameras with the\nsame camera matrix.\n@param method Method for computing an essential matrix.\n-   **RANSAC** for the RANSAC algorithm.\n-   **LMEDS** for the LMedS algorithm.\n@param prob Parameter used for the RANSAC or LMedS methods only. It specifies a desirable level of\nconfidence (probability) that the estimated matrix is correct.\n@param threshold Parameter used for RANSAC. It is the maximum distance from a point to an epipolar\nline in pixels, beyond which the point is considered an outlier and is not used for computing the\nfinal fundamental matrix. It can be set to something like 1-3, depending on the accuracy of the\npoint localization, image resolution, and the image noise.\n@param mask Output array of N elements, every element of which is set to 0 for outliers and to 1\nfor the other points. The array is computed only in the RANSAC and LMedS methods.\n\nThis function estimates essential matrix based on the five-point algorithm solver in @cite Nister03 .\n@cite SteweniusCFS is also a related. The epipolar geometry is described by the following equation:\n\n\f[[p_2; 1]^T K^{-T} E K^{-1} [p_1; 1] = 0\f]\n\nwhere \f$E\f$ is an essential matrix, \f$p_1\f$ and \f$p_2\f$ are corresponding points in the first and the\nsecond images, respectively. The result of this function may be passed further to\ndecomposeEssentialMat or recoverPose to recover the relative pose between cameras.



findEssentialMat(points1, points2[, focal[, pp[, method[, prob[, threshold[, mask]]]]]]) -> retval, mask\n@overload\n@param points1 Array of N (N \>= 5) 2D points from the first image. The point coordinates should\nbe floating-point (single or double precision).\n@param points2 Array of the second image points of the same size and format as points1 .\n@param focal focal length of the camera. Note that this function assumes that points1 and points2\nare feature points from cameras with same focal length and principal point.\n@param pp principal point of the camera.\n@param method Method for computing a fundamental matrix.\n-   **RANSAC** for the RANSAC algorithm.\n-   **LMEDS** for the LMedS algorithm.\n@param threshold Parameter used for RANSAC. It is the maximum distance from a point to an epipolar\nline in pixels, beyond which the point is considered an outlier and is not used for computing the\nfinal fundamental matrix. It can be set to something like 1-3, depending on the accuracy of the\npoint localization, image resolution, and the image noise.\n@param prob Parameter used for the RANSAC or LMedS methods only. It specifies a desirable level of\nconfidence (probability) that the estimated matrix is correct.\n@param mask Output array of N elements, every element of which is set to 0 for outliers and to 1\nfor the other points. The array is computed only in the RANSAC and LMedS methods.\n\nThis function differs from the one above that it computes camera matrix from focal length and\nprincipal point:\n\n\f[K =\n\begin{bmatrix}\nf & 0 & x_{pp}  \\\n0 & f & y_{pp}  \\\n0 & 0 & 1\n\end{bmatrix}\f]'''
    package = "3D calibration & reconstruction"

    def get_attributes(self):
        return [Input('points1', 'points 1'),
                Input('points2', 'points 2'),
                Input('cameraMatrix', 'Camera Matrix')], \
               [Output('mask', 'mask')], \
               [IntParameter('method', 'method'),
                FloatParameter('prob', 'prob'),
                FloatParameter('threshold', 'threshold')]

    def process_inputs(self, inputs, outputs, parameters):
        points1 = inputs['points1'].value
        points2 = inputs['points2'].value
        cameraMatrix = inputs['cameraMatrix'].value
        method = parameters['method']
        prob = parameters['prob']
        threshold = parameters['threshold']
        retval, mask = cv2.findEssentialMat(points1=points1, points2=points2, cameraMatrix=cameraMatrix, method=method, prob=prob, threshold=threshold)
        outputs['mask'] = Data(mask)

# cv2.findHomography
class OpenCVAuto2_FindHomography(NormalElement):
    name = 'Find Homography'
    comment = '''findHomography(srcPoints, dstPoints[, method[, ransacReprojThreshold[, mask[, maxIters[, confidence]]]]]) -> retval, mask\n@brief Finds a perspective transformation between two planes.\n\n@param srcPoints Coordinates of the points in the original plane, a matrix of the type CV_32FC2\nor vector\<Point2f\> .\n@param dstPoints Coordinates of the points in the target plane, a matrix of the type CV_32FC2 or\na vector\<Point2f\> .\n@param method Method used to compute a homography matrix. The following methods are possible:\n-   **0** - a regular method using all the points, i.e., the least squares method\n-   **RANSAC** - RANSAC-based robust method\n-   **LMEDS** - Least-Median robust method\n-   **RHO** - PROSAC-based robust method\n@param ransacReprojThreshold Maximum allowed reprojection error to treat a point pair as an inlier\n(used in the RANSAC and RHO methods only). That is, if\n\f[\| \texttt{dstPoints} _i -  \texttt{convertPointsHomogeneous} ( \texttt{H} * \texttt{srcPoints} _i) \|_2  >  \texttt{ransacReprojThreshold}\f]\nthen the point \f$i\f$ is considered as an outlier. If srcPoints and dstPoints are measured in pixels,\nit usually makes sense to set this parameter somewhere in the range of 1 to 10.\n@param mask Optional output mask set by a robust method ( RANSAC or LMEDS ). Note that the input\nmask values are ignored.\n@param maxIters The maximum number of RANSAC iterations.\n@param confidence Confidence level, between 0 and 1.\n\nThe function finds and returns the perspective transformation \f$H\f$ between the source and the\ndestination planes:\n\n\f[s_i  \vecthree{x'_i}{y'_i}{1} \sim H  \vecthree{x_i}{y_i}{1}\f]\n\nso that the back-projection error\n\n\f[\sum _i \left ( x'_i- \frac{h_{11} x_i + h_{12} y_i + h_{13}}{h_{31} x_i + h_{32} y_i + h_{33}} \right )^2+ \left ( y'_i- \frac{h_{21} x_i + h_{22} y_i + h_{23}}{h_{31} x_i + h_{32} y_i + h_{33}} \right )^2\f]\n\nis minimized. If the parameter method is set to the default value 0, the function uses all the point\npairs to compute an initial homography estimate with a simple least-squares scheme.\n\nHowever, if not all of the point pairs ( \f$srcPoints_i\f$, \f$dstPoints_i\f$ ) fit the rigid perspective\ntransformation (that is, there are some outliers), this initial estimate will be poor. In this case,\nyou can use one of the three robust methods. The methods RANSAC, LMeDS and RHO try many different\nrandom subsets of the corresponding point pairs (of four pairs each, collinear pairs are discarded), estimate the homography matrix\nusing this subset and a simple least-squares algorithm, and then compute the quality/goodness of the\ncomputed homography (which is the number of inliers for RANSAC or the least median re-projection error for\nLMeDS). The best subset is then used to produce the initial estimate of the homography matrix and\nthe mask of inliers/outliers.\n\nRegardless of the method, robust or not, the computed homography matrix is refined further (using\ninliers only in case of a robust method) with the Levenberg-Marquardt method to reduce the\nre-projection error even more.\n\nThe methods RANSAC and RHO can handle practically any ratio of outliers but need a threshold to\ndistinguish inliers from outliers. The method LMeDS does not need any threshold but it works\ncorrectly only when there are more than 50% of inliers. Finally, if there are no outliers and the\nnoise is rather small, use the default method (method=0).\n\nThe function is used to find initial intrinsic and extrinsic matrices. Homography matrix is\ndetermined up to a scale. Thus, it is normalized so that \f$h_{33}=1\f$. Note that whenever an \f$H\f$ matrix\ncannot be estimated, an empty one will be returned.\n\n@sa\ngetAffineTransform, estimateAffine2D, estimateAffinePartial2D, getPerspectiveTransform, warpPerspective,\nperspectiveTransform'''
    package = "3D calibration & reconstruction"

    def get_attributes(self):
        return [Input('srcPoints', 'Src Points'),
                Input('dstPoints', 'Dst Points')], \
               [Output('mask', 'mask')], \
               [IntParameter('method', 'method'),
                FloatParameter('ransacReprojThreshold', 'Ransac Reproj Threshold'),
                IntParameter('maxIters', 'Max Iters', min_=0),
                FloatParameter('confidence', 'confidence')]

    def process_inputs(self, inputs, outputs, parameters):
        srcPoints = inputs['srcPoints'].value
        dstPoints = inputs['dstPoints'].value
        method = parameters['method']
        ransacReprojThreshold = parameters['ransacReprojThreshold']
        maxIters = parameters['maxIters']
        confidence = parameters['confidence']
        retval, mask = cv2.findHomography(srcPoints=srcPoints, dstPoints=dstPoints, method=method, ransacReprojThreshold=ransacReprojThreshold, maxIters=maxIters, confidence=confidence)
        outputs['mask'] = Data(mask)

# cv2.findNonZero
class OpenCVAuto2_FindNonZero(NormalElement):
    name = 'Find Non Zero'
    comment = '''findNonZero(src[, idx]) -> idx\n@brief Returns the list of locations of non-zero pixels\n\nGiven a binary matrix (likely returned from an operation such\nas threshold(), compare(), >, ==, etc, return all of\nthe non-zero indices as a cv::Mat or std::vector<cv::Point> (x,y)\nFor example:\n@code{.cpp}\ncv::Mat binaryImage; // input, binary image\ncv::Mat locations;   // output, locations of non-zero pixels\ncv::findNonZero(binaryImage, locations);\n\n// access pixel coordinates\nPoint pnt = locations.at<Point>(i);\n@endcode\nor\n@code{.cpp}\ncv::Mat binaryImage; // input, binary image\nvector<Point> locations;   // output, locations of non-zero pixels\ncv::findNonZero(binaryImage, locations);\n\n// access pixel coordinates\nPoint pnt = locations[i];\n@endcode\n@param src single-channel array (type CV_8UC1)\n@param idx the output array, type of cv::Mat or std::vector<Point>, corresponding to non-zero indices in the input'''
    package = "Matrix miscellaneous"

    def get_attributes(self):
        return [Input('src', 'src')], \
               [Output('idx', 'idx')], \
               []

    def process_inputs(self, inputs, outputs, parameters):
        src = inputs['src'].value
        idx = cv2.findNonZero(src=src)
        outputs['idx'] = Data(idx)

# cv2.fitLine
class OpenCVAuto2_FitLine(NormalElement):
    name = 'Fit Line'
    comment = '''fitLine(points, distType, param, reps, aeps[, line]) -> line\n@brief Fits a line to a 2D or 3D point set.\n\nThe function fitLine fits a line to a 2D or 3D point set by minimizing \f$\sum_i \rho(r_i)\f$ where\n\f$r_i\f$ is a distance between the \f$i^{th}\f$ point, the line and \f$\rho(r)\f$ is a distance function, one\nof the following:\n-  DIST_L2\n\f[\rho (r) = r^2/2  \quad \text{(the simplest and the fastest least-squares method)}\f]\n- DIST_L1\n\f[\rho (r) = r\f]\n- DIST_L12\n\f[\rho (r) = 2  \cdot ( \sqrt{1 + \frac{r^2}{2}} - 1)\f]\n- DIST_FAIR\n\f[\rho \left (r \right ) = C^2  \cdot \left (  \frac{r}{C} -  \log{\left(1 + \frac{r}{C}\right)} \right )  \quad \text{where} \quad C=1.3998\f]\n- DIST_WELSCH\n\f[\rho \left (r \right ) =  \frac{C^2}{2} \cdot \left ( 1 -  \exp{\left(-\left(\frac{r}{C}\right)^2\right)} \right )  \quad \text{where} \quad C=2.9846\f]\n- DIST_HUBER\n\f[\rho (r) =  \fork{r^2/2}{if \(r < C\)}{C \cdot (r-C/2)}{otherwise} \quad \text{where} \quad C=1.345\f]\n\nThe algorithm is based on the M-estimator ( <http://en.wikipedia.org/wiki/M-estimator> ) technique\nthat iteratively fits the line using the weighted least-squares algorithm. After each iteration the\nweights \f$w_i\f$ are adjusted to be inversely proportional to \f$\rho(r_i)\f$ .\n\n@param points Input vector of 2D or 3D points, stored in std::vector\<\> or Mat.\n@param line Output line parameters. In case of 2D fitting, it should be a vector of 4 elements\n(like Vec4f) - (vx, vy, x0, y0), where (vx, vy) is a normalized vector collinear to the line and\n(x0, y0) is a point on the line. In case of 3D fitting, it should be a vector of 6 elements (like\nVec6f) - (vx, vy, vz, x0, y0, z0), where (vx, vy, vz) is a normalized vector collinear to the line\nand (x0, y0, z0) is a point on the line.\n@param distType Distance used by the M-estimator, see #DistanceTypes\n@param param Numerical parameter ( C ) for some types of distances. If it is 0, an optimal value\nis chosen.\n@param reps Sufficient accuracy for the radius (distance between the coordinate origin and the line).\n@param aeps Sufficient accuracy for the angle. 0.01 would be a good default value for reps and aeps.'''
    package = "Optimization"

    def get_attributes(self):
        return [Input('points', 'points')], \
               [Output('line', 'line')], \
               [IntParameter('distType', 'Dist Type'),
                FloatParameter('param', 'param'),
                FloatParameter('reps', 'reps'),
                FloatParameter('aeps', 'aeps')]

    def process_inputs(self, inputs, outputs, parameters):
        points = inputs['points'].value
        distType = parameters['distType']
        param = parameters['param']
        reps = parameters['reps']
        aeps = parameters['aeps']
        line = cv2.fitLine(points=points, distType=distType, param=param, reps=reps, aeps=aeps)
        outputs['line'] = Data(line)

# cv2.flip
class OpenCVAuto2_Flip(NormalElement):
    name = 'Flip'
    comment = '''flip(src, flipCode[, dst]) -> dst\n@brief Flips a 2D array around vertical, horizontal, or both axes.\n\nThe function cv::flip flips the array in one of three different ways (row\nand column indices are 0-based):\n\f[\texttt{dst} _{ij} =\n\left\{\n\begin{array}{l l}\n\texttt{src} _{\texttt{src.rows}-i-1,j} & if\;  \texttt{flipCode} = 0 \\\n\texttt{src} _{i, \texttt{src.cols} -j-1} & if\;  \texttt{flipCode} > 0 \\\n\texttt{src} _{ \texttt{src.rows} -i-1, \texttt{src.cols} -j-1} & if\; \texttt{flipCode} < 0 \\\n\end{array}\n\right.\f]\nThe example scenarios of using the function are the following:\n*   Vertical flipping of the image (flipCode == 0) to switch between\ntop-left and bottom-left image origin. This is a typical operation\nin video processing on Microsoft Windows\* OS.\n*   Horizontal flipping of the image with the subsequent horizontal\nshift and absolute difference calculation to check for a\nvertical-axis symmetry (flipCode \> 0).\n*   Simultaneous horizontal and vertical flipping of the image with\nthe subsequent shift and absolute difference calculation to check\nfor a central symmetry (flipCode \< 0).\n*   Reversing the order of point arrays (flipCode \> 0 or\nflipCode == 0).\n@param src input array.\n@param dst output array of the same size and type as src.\n@param flipCode a flag to specify how to flip the array; 0 means\nflipping around the x-axis and positive value (for example, 1) means\nflipping around y-axis. Negative value (for example, -1) means flipping\naround both axes.\n@sa transpose , repeat , completeSymm'''
    package = "Transforms"

    def get_attributes(self):
        return [Input('src', 'src')], \
               [Output('dst', 'dst')], \
               [IntParameter('flipCode', 'Flip Code')]

    def process_inputs(self, inputs, outputs, parameters):
        src = inputs['src'].value
        flipCode = parameters['flipCode']
        dst = cv2.flip(src=src, flipCode=flipCode)
        outputs['dst'] = Data(dst)

# cv2.getRectSubPix
class OpenCVAuto2_GetRectSubPix(NormalElement):
    name = 'Get Rect Sub Pix'
    comment = '''getRectSubPix(image, patchSize, center[, patch[, patchType]]) -> patch\n@brief Retrieves a pixel rectangle from an image with sub-pixel accuracy.\n\nThe function getRectSubPix extracts pixels from src:\n\n\f[patch(x, y) = src(x +  \texttt{center.x} - ( \texttt{dst.cols} -1)*0.5, y +  \texttt{center.y} - ( \texttt{dst.rows} -1)*0.5)\f]\n\nwhere the values of the pixels at non-integer coordinates are retrieved using bilinear\ninterpolation. Every channel of multi-channel images is processed independently. Also\nthe image should be a single channel or three channel image. While the center of the\nrectangle must be inside the image, parts of the rectangle may be outside.\n\n@param image Source image.\n@param patchSize Size of the extracted patch.\n@param center Floating point coordinates of the center of the extracted rectangle within the\nsource image. The center must be inside the image.\n@param patch Extracted patch that has the size patchSize and the same number of channels as src .\n@param patchType Depth of the extracted pixels. By default, they have the same depth as src .\n\n@sa  warpAffine, warpPerspective'''
    package = "Transforms"

    def get_attributes(self):
        return [Input('image', 'image')], \
               [Output('patch', 'patch')], \
               [SizeParameter('patchSize', 'Patch Size'),
                PointParameter('center', 'center'),
                IntParameter('patchType', 'Patch Type')]

    def process_inputs(self, inputs, outputs, parameters):
        image = inputs['image'].value
        patchSize = parameters['patchSize']
        center = parameters['center']
        patchType = parameters['patchType']
        patch = cv2.getRectSubPix(image=image, patchSize=patchSize, center=center, patchType=patchType)
        outputs['patch'] = Data(patch)

# cv2.groupRectangles
class OpenCVAuto2_GroupRectangles(NormalElement):
    name = 'Group Rectangles'
    comment = '''groupRectangles(rectList, groupThreshold[, eps]) -> rectList, weights\n@overload'''
    package = "Object detection"

    def get_attributes(self):
        return [Input('rectList', 'Rect List')], \
               [Output('rectList', 'Rect List'),
                Output('weights', 'weights')], \
               [FloatParameter('groupThreshold', 'Group Threshold'),
                FloatParameter('eps', 'eps')]

    def process_inputs(self, inputs, outputs, parameters):
        rectList = inputs['rectList'].value.copy()
        groupThreshold = parameters['groupThreshold']
        eps = parameters['eps']
        rectList, weights = cv2.groupRectangles(rectList=rectList, groupThreshold=groupThreshold, eps=eps)
        outputs['rectList'] = Data(rectList)
        outputs['weights'] = Data(weights)

# cv2.hconcat
class OpenCVAuto2_Hconcat(NormalElement):
    name = 'Hconcat'
    comment = '''hconcat(src[, dst]) -> dst\n@overload\n@code{.cpp}\nstd::vector<cv::Mat> matrices = { cv::Mat(4, 1, CV_8UC1, cv::Scalar(1)),\ncv::Mat(4, 1, CV_8UC1, cv::Scalar(2)),\ncv::Mat(4, 1, CV_8UC1, cv::Scalar(3)),};\n\ncv::Mat out;\ncv::hconcat( matrices, out );\n//out:\n//[1, 2, 3;\n// 1, 2, 3;\n// 1, 2, 3;\n// 1, 2, 3]\n@endcode\n@param src input array or vector of matrices. all of the matrices must have the same number of rows and the same depth.\n@param dst output array. It has the same number of rows and depth as the src, and the sum of cols of the src.\nsame depth.'''
    package = "Matrix miscellaneous"

    def get_attributes(self):
        return [Input('src', 'src')], \
               [Output('dst', 'dst')], \
               []

    def process_inputs(self, inputs, outputs, parameters):
        src = inputs['src'].value
        dst = cv2.hconcat(src=src)
        outputs['dst'] = Data(dst)

# cv2.illuminationChange
class OpenCVAuto2_IlluminationChange(NormalElement):
    name = 'Illumination Change'
    comment = '''illuminationChange(src, mask[, dst[, alpha[, beta]]]) -> dst\n@brief Applying an appropriate non-linear transformation to the gradient field inside the selection and\nthen integrating back with a Poisson solver, modifies locally the apparent illumination of an image.\n\n@param src Input 8-bit 3-channel image.\n@param mask Input 8-bit 1 or 3-channel image.\n@param dst Output image with the same size and type as src.\n@param alpha Value ranges between 0-2.\n@param beta Value ranges between 0-2.\n\nThis is useful to highlight under-exposed foreground objects or to reduce specular reflections.'''
    package = "Photo"

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

# cv2.imencode
class OpenCVAuto2_Imencode(NormalElement):
    name = 'Imencode'
    comment = '''imencode(ext, img[, params]) -> retval, buf\n@brief Encodes an image into a memory buffer.\n\nThe function imencode compresses the image and stores it in the memory buffer that is resized to fit the\nresult. See cv::imwrite for the list of supported formats and flags description.\n\n@param ext File extension that defines the output format.\n@param img Image to be written.\n@param buf Output buffer resized to fit the compressed image.\n@param params Format-specific parameters. See cv::imwrite and cv::ImwriteFlags.'''
    package = "Image IO"

    def get_attributes(self):
        return [Input('img', 'img'),
                Input('params', 'params', optional=True)], \
               [Output('buf', 'buf')], \
               [TextParameter('ext', 'ext')]

    def process_inputs(self, inputs, outputs, parameters):
        img = inputs['img'].value
        params = inputs['params'].value
        ext = parameters['ext']
        retval, buf = cv2.imencode(img=img, params=params, ext=ext)
        outputs['buf'] = Data(buf)

# cv2.imreadmulti
class OpenCVAuto2_Imreadmulti(NormalElement):
    name = 'Imreadmulti'
    comment = '''imreadmulti(filename[, mats[, flags]]) -> retval, mats\n@brief Loads a multi-page image from a file.\n\nThe function imreadmulti loads a multi-page image from the specified file into a vector of Mat objects.\n@param filename Name of file to be loaded.\n@param flags Flag that can take values of cv::ImreadModes, default with cv::IMREAD_ANYCOLOR.\n@param mats A vector of Mat objects holding each page, if more than one.\n@sa cv::imread'''
    package = "Image IO"

    def get_attributes(self):
        return [], \
               [Output('mats', 'mats')], \
               [TextParameter('filename', 'filename'),
                ComboboxParameter('flags', name='flags', values=[('IMREAD_UNCHANGED',-1),('IMREAD_GRAYSCALE',0),('IMREAD_COLOR',1),('IMREAD_ANYDEPTH',2),('IMREAD_ANYCOLOR',4),('IMREAD_LOAD_GDAL',8),('IMREAD_REDUCED_GRAYSCALE_2',16),('IMREAD_REDUCED_COLOR_2',17),('IMREAD_REDUCED_GRAYSCALE_4',32),('IMREAD_REDUCED_COLOR_4',33),('IMREAD_REDUCED_GRAYSCALE_8',64),('IMREAD_REDUCED_COLOR_8',65),('IMREAD_IGNORE_ORIENTATION',128)])]

    def process_inputs(self, inputs, outputs, parameters):
        filename = parameters['filename']
        flags = parameters['flags']
        retval, mats = cv2.imreadmulti(filename=filename, flags=flags)
        outputs['mats'] = Data(mats)

# cv2.inRange
class OpenCVAuto2_InRange(NormalElement):
    name = 'In Range'
    comment = '''inRange(src, lowerb, upperb[, dst]) -> dst\n@brief  Checks if array elements lie between the elements of two other arrays.\n\nThe function checks the range as follows:\n-   For every element of a single-channel input array:\n\f[\texttt{dst} (I)= \texttt{lowerb} (I)_0  \leq \texttt{src} (I)_0 \leq  \texttt{upperb} (I)_0\f]\n-   For two-channel arrays:\n\f[\texttt{dst} (I)= \texttt{lowerb} (I)_0  \leq \texttt{src} (I)_0 \leq  \texttt{upperb} (I)_0  \land \texttt{lowerb} (I)_1  \leq \texttt{src} (I)_1 \leq  \texttt{upperb} (I)_1\f]\n-   and so forth.\n\nThat is, dst (I) is set to 255 (all 1 -bits) if src (I) is within the\nspecified 1D, 2D, 3D, ... box and 0 otherwise.\n\nWhen the lower and/or upper boundary parameters are scalars, the indexes\n(I) at lowerb and upperb in the above formulas should be omitted.\n@param src first input array.\n@param lowerb inclusive lower boundary array or a scalar.\n@param upperb inclusive upper boundary array or a scalar.\n@param dst output array of the same size as src and CV_8U type.'''
    package = "Matrix miscellaneous"

    def get_attributes(self):
        return [Input('src', 'src'),
                Input('lowerb', 'lowerb'),
                Input('upperb', 'upperb')], \
               [Output('dst', 'dst')], \
               []

    def process_inputs(self, inputs, outputs, parameters):
        src = inputs['src'].value
        lowerb = inputs['lowerb'].value
        upperb = inputs['upperb'].value
        dst = cv2.inRange(src=src, lowerb=lowerb, upperb=upperb)
        outputs['dst'] = Data(dst)

# cv2.initUndistortRectifyMap
class OpenCVAuto2_InitUndistortRectifyMap(NormalElement):
    name = 'Init Undistort Rectify Map'
    comment = '''initUndistortRectifyMap(cameraMatrix, distCoeffs, R, newCameraMatrix, size, m1type[, map1[, map2]]) -> map1, map2\n@brief Computes the undistortion and rectification transformation map.\n\nThe function computes the joint undistortion and rectification transformation and represents the\nresult in the form of maps for remap. The undistorted image looks like original, as if it is\ncaptured with a camera using the camera matrix =newCameraMatrix and zero distortion. In case of a\nmonocular camera, newCameraMatrix is usually equal to cameraMatrix, or it can be computed by\n#getOptimalNewCameraMatrix for a better control over scaling. In case of a stereo camera,\nnewCameraMatrix is normally set to P1 or P2 computed by #stereoRectify .\n\nAlso, this new camera is oriented differently in the coordinate space, according to R. That, for\nexample, helps to align two heads of a stereo camera so that the epipolar lines on both images\nbecome horizontal and have the same y- coordinate (in case of a horizontally aligned stereo camera).\n\nThe function actually builds the maps for the inverse mapping algorithm that is used by remap. That\nis, for each pixel \f$(u, v)\f$ in the destination (corrected and rectified) image, the function\ncomputes the corresponding coordinates in the source image (that is, in the original image from\ncamera). The following process is applied:\n\f[\n\begin{array}{l}\nx  \leftarrow (u - {c'}_x)/{f'}_x  \\\ny  \leftarrow (v - {c'}_y)/{f'}_y  \\\n{[X\,Y\,W]} ^T  \leftarrow R^{-1}*[x \, y \, 1]^T  \\\nx'  \leftarrow X/W  \\\ny'  \leftarrow Y/W  \\\nr^2  \leftarrow x'^2 + y'^2 \\\nx''  \leftarrow x' \frac{1 + k_1 r^2 + k_2 r^4 + k_3 r^6}{1 + k_4 r^2 + k_5 r^4 + k_6 r^6}\n+ 2p_1 x' y' + p_2(r^2 + 2 x'^2)  + s_1 r^2 + s_2 r^4\\\ny''  \leftarrow y' \frac{1 + k_1 r^2 + k_2 r^4 + k_3 r^6}{1 + k_4 r^2 + k_5 r^4 + k_6 r^6}\n+ p_1 (r^2 + 2 y'^2) + 2 p_2 x' y' + s_3 r^2 + s_4 r^4 \\\ns\vecthree{x'}{y'}{1} =\n\vecthreethree{R_{33}(\tau_x, \tau_y)}{0}{-R_{13}((\tau_x, \tau_y)}\n{0}{R_{33}(\tau_x, \tau_y)}{-R_{23}(\tau_x, \tau_y)}\n{0}{0}{1} R(\tau_x, \tau_y) \vecthree{x''}{y''}{1}\\\nmap_x(u,v)  \leftarrow x' f_x + c_x  \\\nmap_y(u,v)  \leftarrow y' f_y + c_y\n\end{array}\n\f]\nwhere \f$(k_1, k_2, p_1, p_2[, k_3[, k_4, k_5, k_6[, s_1, s_2, s_3, s_4[, \tau_x, \tau_y]]]])\f$\nare the distortion coefficients.\n\nIn case of a stereo camera, this function is called twice: once for each camera head, after\nstereoRectify, which in its turn is called after #stereoCalibrate. But if the stereo camera\nwas not calibrated, it is still possible to compute the rectification transformations directly from\nthe fundamental matrix using #stereoRectifyUncalibrated. For each camera, the function computes\nhomography H as the rectification transformation in a pixel domain, not a rotation matrix R in 3D\nspace. R can be computed from H as\n\f[\texttt{R} = \texttt{cameraMatrix} ^{-1} \cdot \texttt{H} \cdot \texttt{cameraMatrix}\f]\nwhere cameraMatrix can be chosen arbitrarily.\n\n@param cameraMatrix Input camera matrix \f$A=\vecthreethree{f_x}{0}{c_x}{0}{f_y}{c_y}{0}{0}{1}\f$ .\n@param distCoeffs Input vector of distortion coefficients\n\f$(k_1, k_2, p_1, p_2[, k_3[, k_4, k_5, k_6[, s_1, s_2, s_3, s_4[, \tau_x, \tau_y]]]])\f$\nof 4, 5, 8, 12 or 14 elements. If the vector is NULL/empty, the zero distortion coefficients are assumed.\n@param R Optional rectification transformation in the object space (3x3 matrix). R1 or R2 ,\ncomputed by #stereoRectify can be passed here. If the matrix is empty, the identity transformation\nis assumed. In cvInitUndistortMap R assumed to be an identity matrix.\n@param newCameraMatrix New camera matrix \f$A'=\vecthreethree{f_x'}{0}{c_x'}{0}{f_y'}{c_y'}{0}{0}{1}\f$.\n@param size Undistorted image size.\n@param m1type Type of the first output map that can be CV_32FC1, CV_32FC2 or CV_16SC2, see #convertMaps\n@param map1 The first output map.\n@param map2 The second output map.'''
    package = "Transforms"

    def get_attributes(self):
        return [Input('cameraMatrix', 'Camera Matrix'),
                Input('distCoeffs', 'Dist Coeffs'),
                Input('R', 'R'),
                Input('newCameraMatrix', 'New Camera Matrix')], \
               [Output('map1', 'map 1'),
                Output('map2', 'map 2')], \
               [SizeParameter('size', 'size'),
                IntParameter('m1type', 'm 1 type')]

    def process_inputs(self, inputs, outputs, parameters):
        cameraMatrix = inputs['cameraMatrix'].value
        distCoeffs = inputs['distCoeffs'].value
        R = inputs['R'].value
        newCameraMatrix = inputs['newCameraMatrix'].value
        size = parameters['size']
        m1type = parameters['m1type']
        map1, map2 = cv2.initUndistortRectifyMap(cameraMatrix=cameraMatrix, distCoeffs=distCoeffs, R=R, newCameraMatrix=newCameraMatrix, size=size, m1type=m1type)
        outputs['map1'] = Data(map1)
        outputs['map2'] = Data(map2)

# cv2.initWideAngleProjMap
class OpenCVAuto2_InitWideAngleProjMap(NormalElement):
    name = 'Init Wide Angle Proj Map'
    comment = '''initWideAngleProjMap(cameraMatrix, distCoeffs, imageSize, destImageWidth, m1type[, map1[, map2[, projType[, alpha]]]]) -> retval, map1, map2
.'''
    package = "Transforms"

    def get_attributes(self):
        return [Input('cameraMatrix', 'Camera Matrix'),
                Input('distCoeffs', 'Dist Coeffs'),
                Input('imageSize', 'Image Size'),
                Input('destImageWidth', 'Dest Image Width')], \
               [Output('map1', 'map 1'),
                Output('map2', 'map 2')], \
               [IntParameter('m1type', 'm 1 type'),
                IntParameter('projType', 'Proj Type'),
                FloatParameter('alpha', 'alpha')]

    def process_inputs(self, inputs, outputs, parameters):
        cameraMatrix = inputs['cameraMatrix'].value
        distCoeffs = inputs['distCoeffs'].value
        imageSize = inputs['imageSize'].value
        destImageWidth = inputs['destImageWidth'].value
        m1type = parameters['m1type']
        projType = parameters['projType']
        alpha = parameters['alpha']
        retval, map1, map2 = cv2.initWideAngleProjMap(cameraMatrix=cameraMatrix, distCoeffs=distCoeffs, imageSize=imageSize, destImageWidth=destImageWidth, m1type=m1type, projType=projType, alpha=alpha)
        outputs['map1'] = Data(map1)
        outputs['map2'] = Data(map2)

# cv2.inpaint
class OpenCVAuto2_Inpaint(NormalElement):
    name = 'Inpaint'
    comment = '''inpaint(src, inpaintMask, inpaintRadius, flags[, dst]) -> dst\n@brief Restores the selected region in an image using the region neighborhood.\n\n@param src Input 8-bit, 16-bit unsigned or 32-bit float 1-channel or 8-bit 3-channel image.\n@param inpaintMask Inpainting mask, 8-bit 1-channel image. Non-zero pixels indicate the area that\nneeds to be inpainted.\n@param dst Output image with the same size and type as src .\n@param inpaintRadius Radius of a circular neighborhood of each point inpainted that is considered\nby the algorithm.\n@param flags Inpainting method that could be one of the following:\n-   **INPAINT_NS** Navier-Stokes based method [Navier01]\n-   **INPAINT_TELEA** Method by Alexandru Telea @cite Telea04 .\n\nThe function reconstructs the selected image area from the pixel near the area boundary. The\nfunction may be used to remove dust and scratches from a scanned photo, or to remove undesirable\nobjects from still images or video. See <http://en.wikipedia.org/wiki/Inpainting> for more details.\n\n@note\n-   An example using the inpainting technique can be found at\nopencv_source_code/samples/cpp/inpaint.cpp\n-   (Python) An example using the inpainting technique can be found at\nopencv_source_code/samples/python/inpaint.py'''
    package = "Photo"

    def get_attributes(self):
        return [Input('src', 'src'),
                Input('inpaintMask', 'Inpaint Mask')], \
               [Output('dst', 'dst')], \
               [IntParameter('inpaintRadius', 'Inpaint Radius', min_=0),
                ComboboxParameter('flags', name='flags', values=[('INPAINT_NS',0),('INPAINT_TELEA',1)])]

    def process_inputs(self, inputs, outputs, parameters):
        src = inputs['src'].value
        inpaintMask = inputs['inpaintMask'].value
        inpaintRadius = parameters['inpaintRadius']
        flags = parameters['flags']
        dst = cv2.inpaint(src=src, inpaintMask=inpaintMask, inpaintRadius=inpaintRadius, flags=flags)
        outputs['dst'] = Data(dst)

# cv2.insertChannel
class OpenCVAuto2_InsertChannel(NormalElement):
    name = 'Insert Channel'
    comment = '''insertChannel(src, dst, coi) -> dst\n@brief Inserts a single channel to dst (coi is 0-based index)\n@param src input array\n@param dst output array\n@param coi index of channel for insertion\n@sa mixChannels, merge'''
    package = "Channels"

    def get_attributes(self):
        return [Input('src', 'src'),
                Input('dst', 'dst')], \
               [Output('dst', 'dst')], \
               [IntParameter('coi', 'coi')]

    def process_inputs(self, inputs, outputs, parameters):
        src = inputs['src'].value
        dst = inputs['dst'].value.copy()
        coi = parameters['coi']
        dst = cv2.insertChannel(src=src, dst=dst, coi=coi)
        outputs['dst'] = Data(dst)

# cv2.integral
class OpenCVAuto2_Integral(NormalElement):
    name = 'Integral'
    comment = '''integral(src[, sum[, sdepth]]) -> sum\n@overload'''
    package = "Math"

    def get_attributes(self):
        return [Input('src', 'src')], \
               [Output('sum', 'sum')], \
               [IntParameter('sdepth', 'sdepth')]

    def process_inputs(self, inputs, outputs, parameters):
        src = inputs['src'].value
        sdepth = parameters['sdepth']
        sum = cv2.integral(src=src, sdepth=sdepth)
        outputs['sum'] = Data(sum)

# cv2.integral2
class OpenCVAuto2_Integral2(NormalElement):
    name = 'Integral 2'
    comment = '''integral2(src[, sum[, sqsum[, sdepth[, sqdepth]]]]) -> sum, sqsum\n@overload'''
    package = "Math"

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

# cv2.integral3
class OpenCVAuto2_Integral3(NormalElement):
    name = 'Integral 3'
    comment = '''integral3(src[, sum[, sqsum[, tilted[, sdepth[, sqdepth]]]]]) -> sum, sqsum, tilted\n@brief Calculates the integral of an image.\n\nThe function calculates one or more integral images for the source image as follows:\n\n\f[\texttt{sum} (X,Y) =  \sum _{x<X,y<Y}  \texttt{image} (x,y)\f]\n\n\f[\texttt{sqsum} (X,Y) =  \sum _{x<X,y<Y}  \texttt{image} (x,y)^2\f]\n\n\f[\texttt{tilted} (X,Y) =  \sum _{y<Y,abs(x-X+1) \leq Y-y-1}  \texttt{image} (x,y)\f]\n\nUsing these integral images, you can calculate sum, mean, and standard deviation over a specific\nup-right or rotated rectangular region of the image in a constant time, for example:\n\n\f[\sum _{x_1 \leq x < x_2,  \, y_1  \leq y < y_2}  \texttt{image} (x,y) =  \texttt{sum} (x_2,y_2)- \texttt{sum} (x_1,y_2)- \texttt{sum} (x_2,y_1)+ \texttt{sum} (x_1,y_1)\f]\n\nIt makes possible to do a fast blurring or fast block correlation with a variable window size, for\nexample. In case of multi-channel images, sums for each channel are accumulated independently.\n\nAs a practical example, the next figure shows the calculation of the integral of a straight\nrectangle Rect(3,3,3,2) and of a tilted rectangle Rect(5,1,2,3) . The selected pixels in the\noriginal image are shown, as well as the relative pixels in the integral images sum and tilted .\n\n![integral calculation example](pics/integral.png)\n\n@param src input image as \f$W \times H\f$, 8-bit or floating-point (32f or 64f).\n@param sum integral image as \f$(W+1)\times (H+1)\f$ , 32-bit integer or floating-point (32f or 64f).\n@param sqsum integral image for squared pixel values; it is \f$(W+1)\times (H+1)\f$, double-precision\nfloating-point (64f) array.\n@param tilted integral for the image rotated by 45 degrees; it is \f$(W+1)\times (H+1)\f$ array with\nthe same data type as sum.\n@param sdepth desired depth of the integral and the tilted integral images, CV_32S, CV_32F, or\nCV_64F.\n@param sqdepth desired depth of the integral image of squared pixel values, CV_32F or CV_64F.'''
    package = "Math"

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

# cv2.invertAffineTransform
class OpenCVAuto2_InvertAffineTransform(NormalElement):
    name = 'Invert Affine Transform'
    comment = '''invertAffineTransform(M[, iM]) -> iM\n@brief Inverts an affine transformation.\n\nThe function computes an inverse affine transformation represented by \f$2 \times 3\f$ matrix M:\n\n\f[\begin{bmatrix} a_{11} & a_{12} & b_1  \\ a_{21} & a_{22} & b_2 \end{bmatrix}\f]\n\nThe result is also a \f$2 \times 3\f$ matrix of the same type as M.\n\n@param M Original affine transformation.\n@param iM Output reverse affine transformation.'''
    package = "Transforms"

    def get_attributes(self):
        return [Input('M', 'M')], \
               [Output('iM', 'I M')], \
               []

    def process_inputs(self, inputs, outputs, parameters):
        M = inputs['M'].value
        iM = cv2.invertAffineTransform(M=M)
        outputs['iM'] = Data(iM)

# cv2.line
class OpenCVAuto2_Line(NormalElement):
    name = 'Line'
    comment = '''line(img, pt1, pt2, color[, thickness[, lineType[, shift]]]) -> img\n@brief Draws a line segment connecting two points.\n\nThe function line draws the line segment between pt1 and pt2 points in the image. The line is\nclipped by the image boundaries. For non-antialiased lines with integer coordinates, the 8-connected\nor 4-connected Bresenham algorithm is used. Thick lines are drawn with rounding endings. Antialiased\nlines are drawn using Gaussian filtering.\n\n@param img Image.\n@param pt1 First point of the line segment.\n@param pt2 Second point of the line segment.\n@param color Line color.\n@param thickness Line thickness.\n@param lineType Type of the line. See #LineTypes.\n@param shift Number of fractional bits in the point coordinates.'''
    package = "Drawing"

    def get_attributes(self):
        return [Input('img', 'img')], \
               [Output('img', 'img')], \
               [PointParameter('pt1', 'pt 1'),
                PointParameter('pt2', 'pt 2'),
                ScalarParameter('color', 'color'),
                IntParameter('thickness', 'thickness', min_=-1, max_=100),
                ComboboxParameter('lineType', name='Line Type', values=[('LINE_4',4),('LINE_8',8),('LINE_AA',16)]),
                IntParameter('shift', 'shift')]

    def process_inputs(self, inputs, outputs, parameters):
        img = inputs['img'].value.copy()
        pt1 = parameters['pt1']
        pt2 = parameters['pt2']
        color = parameters['color']
        thickness = parameters['thickness']
        lineType = parameters['lineType']
        shift = parameters['shift']
        img = cv2.line(img=img, pt1=pt1, pt2=pt2, color=color, thickness=thickness, lineType=lineType, shift=shift)
        outputs['img'] = Data(img)

# cv2.linearPolar
class OpenCVAuto2_LinearPolar(NormalElement):
    name = 'Linear Polar'
    comment = '''linearPolar(src, center, maxRadius, flags[, dst]) -> dst\n@brief Remaps an image to polar coordinates space.\n\n@deprecated This function produces same result as cv::warpPolar(src, dst, src.size(), center, maxRadius, flags)\n\n@internal\nTransform the source image using the following transformation (See @ref polar_remaps_reference_image "Polar remaps reference image c)"):\n\f[\begin{array}{l}\ndst( \rho , \phi ) = src(x,y) \\\ndst.size() \leftarrow src.size()\n\end{array}\f]\n\nwhere\n\f[\begin{array}{l}\nI = (dx,dy) = (x - center.x,y - center.y) \\\n\rho = Kmag \cdot \texttt{magnitude} (I) ,\\\n\phi = angle \cdot \texttt{angle} (I)\n\end{array}\f]\n\nand\n\f[\begin{array}{l}\nKx = src.cols / maxRadius \\\nKy = src.rows / 2\Pi\n\end{array}\f]\n\n\n@param src Source image\n@param dst Destination image. It will have same size and type as src.\n@param center The transformation center;\n@param maxRadius The radius of the bounding circle to transform. It determines the inverse magnitude scale parameter too.\n@param flags A combination of interpolation methods, see #InterpolationFlags\n\n@note\n-   The function can not operate in-place.\n-   To calculate magnitude and angle in degrees #cartToPolar is used internally thus angles are measured from 0 to 360 with accuracy about 0.3 degrees.\n\n@sa cv::logPolar\n@endinternal'''
    package = "Math"

    def get_attributes(self):
        return [Input('src', 'src')], \
               [Output('dst', 'dst')], \
               [PointParameter('center', 'center'),
                IntParameter('maxRadius', 'Max Radius', min_=0),
                ComboboxParameter('flags', name='flags', values=[('INTER_NEAREST',0),('INTER_LINEAR',1),('INTER_CUBIC',2),('INTER_AREA',3),('INTER_LANCZOS4',4),('INTER_BITS',5),('INTER_LINEAR_EXACT',5),('INTER_MAX',7),('INTER_BITS2',10),('INTER_TAB_SIZE',32),('INTER_TAB_SIZE2',1024)])]

    def process_inputs(self, inputs, outputs, parameters):
        src = inputs['src'].value
        center = parameters['center']
        maxRadius = parameters['maxRadius']
        flags = parameters['flags']
        dst = cv2.linearPolar(src=src, center=center, maxRadius=maxRadius, flags=flags)
        outputs['dst'] = Data(dst)

# cv2.log
class OpenCVAuto2_Log(NormalElement):
    name = 'Log'
    comment = '''log(src[, dst]) -> dst\n@brief Calculates the natural logarithm of every array element.\n\nThe function cv::log calculates the natural logarithm of every element of the input array:\n\f[\texttt{dst} (I) =  \log (\texttt{src}(I)) \f]\n\nOutput on zero, negative and special (NaN, Inf) values is undefined.\n\n@param src input array.\n@param dst output array of the same size and type as src .\n@sa exp, cartToPolar, polarToCart, phase, pow, sqrt, magnitude'''
    package = "Math"

    def get_attributes(self):
        return [Input('src', 'src')], \
               [Output('dst', 'dst')], \
               []

    def process_inputs(self, inputs, outputs, parameters):
        src = inputs['src'].value
        dst = cv2.log(src=src)
        outputs['dst'] = Data(dst)

# cv2.logPolar
class OpenCVAuto2_LogPolar(NormalElement):
    name = 'Log Polar'
    comment = '''logPolar(src, center, M, flags[, dst]) -> dst\n@brief Remaps an image to semilog-polar coordinates space.\n\n@deprecated This function produces same result as cv::warpPolar(src, dst, src.size(), center, maxRadius, flags+WARP_POLAR_LOG);\n\n@internal\nTransform the source image using the following transformation (See @ref polar_remaps_reference_image "Polar remaps reference image d)"):\n\f[\begin{array}{l}\ndst( \rho , \phi ) = src(x,y) \\\ndst.size() \leftarrow src.size()\n\end{array}\f]\n\nwhere\n\f[\begin{array}{l}\nI = (dx,dy) = (x - center.x,y - center.y) \\\n\rho = M \cdot log_e(\texttt{magnitude} (I)) ,\\\n\phi = Kangle \cdot \texttt{angle} (I) \\\n\end{array}\f]\n\nand\n\f[\begin{array}{l}\nM = src.cols / log_e(maxRadius) \\\nKangle = src.rows / 2\Pi \\\n\end{array}\f]\n\nThe function emulates the human "foveal" vision and can be used for fast scale and\nrotation-invariant template matching, for object tracking and so forth.\n@param src Source image\n@param dst Destination image. It will have same size and type as src.\n@param center The transformation center; where the output precision is maximal\n@param M Magnitude scale parameter. It determines the radius of the bounding circle to transform too.\n@param flags A combination of interpolation methods, see #InterpolationFlags\n\n@note\n-   The function can not operate in-place.\n-   To calculate magnitude and angle in degrees #cartToPolar is used internally thus angles are measured from 0 to 360 with accuracy about 0.3 degrees.\n\n@sa cv::linearPolar\n@endinternal'''
    package = "Math"

    def get_attributes(self):
        return [Input('src', 'src'),
                Input('M', 'M')], \
               [Output('dst', 'dst')], \
               [PointParameter('center', 'center'),
                ComboboxParameter('flags', name='flags', values=[('INTER_NEAREST',0),('INTER_LINEAR',1),('INTER_CUBIC',2),('INTER_AREA',3),('INTER_LANCZOS4',4),('INTER_BITS',5),('INTER_LINEAR_EXACT',5),('INTER_MAX',7),('INTER_BITS2',10),('INTER_TAB_SIZE',32),('INTER_TAB_SIZE2',1024)])]

    def process_inputs(self, inputs, outputs, parameters):
        src = inputs['src'].value
        M = inputs['M'].value
        center = parameters['center']
        flags = parameters['flags']
        dst = cv2.logPolar(src=src, M=M, center=center, flags=flags)
        outputs['dst'] = Data(dst)

# cv2.magnitude
class OpenCVAuto2_Magnitude(NormalElement):
    name = 'Magnitude'
    comment = '''magnitude(x, y[, magnitude]) -> magnitude\n@brief Calculates the magnitude of 2D vectors.\n\nThe function cv::magnitude calculates the magnitude of 2D vectors formed\nfrom the corresponding elements of x and y arrays:\n\f[\texttt{dst} (I) =  \sqrt{\texttt{x}(I)^2 + \texttt{y}(I)^2}\f]\n@param x floating-point array of x-coordinates of the vectors.\n@param y floating-point array of y-coordinates of the vectors; it must\nhave the same size as x.\n@param magnitude output array of the same size and type as x.\n@sa cartToPolar, polarToCart, phase, sqrt'''
    package = "Miscellaneous"

    def get_attributes(self):
        return [Input('x', 'x'),
                Input('y', 'y')], \
               [Output('magnitude', 'magnitude')], \
               []

    def process_inputs(self, inputs, outputs, parameters):
        x = inputs['x'].value
        y = inputs['y'].value
        magnitude = cv2.magnitude(x=x, y=y)
        outputs['magnitude'] = Data(magnitude)

# cv2.matMulDeriv
class OpenCVAuto2_MatMulDeriv(NormalElement):
    name = 'Mat Mul Deriv'
    comment = '''matMulDeriv(A, B[, dABdA[, dABdB]]) -> dABdA, dABdB\n@brief Computes partial derivatives of the matrix product for each multiplied matrix.\n\n@param A First multiplied matrix.\n@param B Second multiplied matrix.\n@param dABdA First output derivative matrix d(A\*B)/dA of size\n\f$\texttt{A.rows*B.cols} \times {A.rows*A.cols}\f$ .\n@param dABdB Second output derivative matrix d(A\*B)/dB of size\n\f$\texttt{A.rows*B.cols} \times {B.rows*B.cols}\f$ .\n\nThe function computes partial derivatives of the elements of the matrix product \f$A*B\f$ with regard to\nthe elements of each of the two input matrices. The function is used to compute the Jacobian\nmatrices in stereoCalibrate but can also be used in any other similar optimization function.'''
    package = "Math"

    def get_attributes(self):
        return [Input('A', 'A'),
                Input('B', 'B')], \
               [Output('dABdA', 'D A Bd A'),
                Output('dABdB', 'D A Bd B')], \
               []

    def process_inputs(self, inputs, outputs, parameters):
        A = inputs['A'].value
        B = inputs['B'].value
        dABdA, dABdB = cv2.matMulDeriv(A=A, B=B)
        outputs['dABdA'] = Data(dABdA)
        outputs['dABdB'] = Data(dABdB)

# cv2.matchTemplate
class OpenCVAuto2_MatchTemplate(NormalElement):
    name = 'Match Template'
    comment = '''matchTemplate(image, templ, method[, result[, mask]]) -> result\n@brief Compares a template against overlapped image regions.\n\nThe function slides through image , compares the overlapped patches of size \f$w \times h\f$ against\ntempl using the specified method and stores the comparison results in result . Here are the formulae\nfor the available comparison methods ( \f$I\f$ denotes image, \f$T\f$ template, \f$R\f$ result ). The summation\nis done over template and/or the image patch: \f$x' = 0...w-1, y' = 0...h-1\f$\n\nAfter the function finishes the comparison, the best matches can be found as global minimums (when\n#TM_SQDIFF was used) or maximums (when #TM_CCORR or #TM_CCOEFF was used) using the\n#minMaxLoc function. In case of a color image, template summation in the numerator and each sum in\nthe denominator is done over all of the channels and separate mean values are used for each channel.\nThat is, the function can take a color template and a color image. The result will still be a\nsingle-channel image, which is easier to analyze.\n\n@param image Image where the search is running. It must be 8-bit or 32-bit floating-point.\n@param templ Searched template. It must be not greater than the source image and have the same\ndata type.\n@param result Map of comparison results. It must be single-channel 32-bit floating-point. If image\nis \f$W \times H\f$ and templ is \f$w \times h\f$ , then result is \f$(W-w+1) \times (H-h+1)\f$ .\n@param method Parameter specifying the comparison method, see #TemplateMatchModes\n@param mask Mask of searched template. It must have the same datatype and size with templ. It is\nnot set by default. Currently, only the #TM_SQDIFF and #TM_CCORR_NORMED methods are supported.'''
    package = "Object detection"

    def get_attributes(self):
        return [Input('image', 'image'),
                Input('templ', 'templ'),
                Input('mask', 'mask', optional=True)], \
               [Output('result', 'result')], \
               [IntParameter('method', 'method')]

    def process_inputs(self, inputs, outputs, parameters):
        image = inputs['image'].value
        templ = inputs['templ'].value
        mask = inputs['mask'].value
        method = parameters['method']
        result = cv2.matchTemplate(image=image, templ=templ, mask=mask, method=method)
        outputs['result'] = Data(result)

# cv2.max
class OpenCVAuto2_Max(NormalElement):
    name = 'Max'
    comment = '''max(src1, src2[, dst]) -> dst\n@brief Calculates per-element maximum of two arrays or an array and a scalar.\n\nThe function cv::max calculates the per-element maximum of two arrays:\n\f[\texttt{dst} (I)= \max ( \texttt{src1} (I), \texttt{src2} (I))\f]\nor array and a scalar:\n\f[\texttt{dst} (I)= \max ( \texttt{src1} (I), \texttt{value} )\f]\n@param src1 first input array.\n@param src2 second input array of the same size and type as src1 .\n@param dst output array of the same size and type as src1.\n@sa  min, compare, inRange, minMaxLoc, @ref MatrixExpressions'''
    package = "Operators"

    def get_attributes(self):
        return [Input('src1', 'src 1'),
                Input('src2', 'src 2')], \
               [Output('dst', 'dst')], \
               []

    def process_inputs(self, inputs, outputs, parameters):
        src1 = inputs['src1'].value
        src2 = inputs['src2'].value
        dst = cv2.max(src1=src1, src2=src2)
        outputs['dst'] = Data(dst)

# cv2.meanStdDev
class OpenCVAuto2_MeanStdDev(NormalElement):
    name = 'Mean Std Dev'
    comment = '''meanStdDev(src[, mean[, stddev[, mask]]]) -> mean, stddev\nCalculates a mean and standard deviation of array elements.\n\nThe function cv::meanStdDev calculates the mean and the standard deviation M\nof array elements independently for each channel and returns it via the\noutput parameters:\n\f[\begin{array}{l} N =  \sum _{I, \texttt{mask} (I)  \ne 0} 1 \\ \texttt{mean} _c =  \frac{\sum_{ I: \; \texttt{mask}(I) \ne 0} \texttt{src} (I)_c}{N} \\ \texttt{stddev} _c =  \sqrt{\frac{\sum_{ I: \; \texttt{mask}(I) \ne 0} \left ( \texttt{src} (I)_c -  \texttt{mean} _c \right )^2}{N}} \end{array}\f]\nWhen all the mask elements are 0's, the function returns\nmean=stddev=Scalar::all(0).\n@note The calculated standard deviation is only the diagonal of the\ncomplete normalized covariance matrix. If the full matrix is needed, you\ncan reshape the multi-channel array M x N to the single-channel array\nM\*N x mtx.channels() (only possible when the matrix is continuous) and\nthen pass the matrix to calcCovarMatrix .\n@param src input array that should have from 1 to 4 channels so that the results can be stored in\nScalar_ 's.\n@param mean output parameter: calculated mean value.\n@param stddev output parameter: calculated standard deviation.\n@param mask optional operation mask.\n@sa  countNonZero, mean, norm, minMaxLoc, calcCovarMatrix'''
    package = "Math"

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

# cv2.medianBlur
class OpenCVAuto2_MedianBlur(NormalElement):
    name = 'Median Blur'
    comment = '''medianBlur(src, ksize[, dst]) -> dst\n@brief Blurs an image using the median filter.\n\nThe function smoothes an image using the median filter with the \f$\texttt{ksize} \times\n\texttt{ksize}\f$ aperture. Each channel of a multi-channel image is processed independently.\nIn-place operation is supported.\n\n@note The median filter uses #BORDER_REPLICATE internally to cope with border pixels, see #BorderTypes\n\n@param src input 1-, 3-, or 4-channel image; when ksize is 3 or 5, the image depth should be\nCV_8U, CV_16U, or CV_32F, for larger aperture sizes, it can only be CV_8U.\n@param dst destination array of the same size and type as src.\n@param ksize aperture linear size; it must be odd and greater than 1, for example: 3, 5, 7 ...\n@sa  bilateralFilter, blur, boxFilter, GaussianBlur'''
    package = "Filters"

    def get_attributes(self):
        return [Input('src', 'src')], \
               [Output('dst', 'dst')], \
               [SizeParameter('ksize', 'ksize')]

    def process_inputs(self, inputs, outputs, parameters):
        src = inputs['src'].value
        ksize = parameters['ksize']
        dst = cv2.medianBlur(src=src, ksize=ksize)
        outputs['dst'] = Data(dst)

# cv2.merge
class OpenCVAuto2_Merge(NormalElement):
    name = 'Merge'
    comment = '''merge(mv[, dst]) -> dst\n@overload\n@param mv input vector of matrices to be merged; all the matrices in mv must have the same\nsize and the same depth.\n@param dst output array of the same size and the same depth as mv[0]; The number of channels will\nbe the total number of channels in the matrix array.'''
    package = "Channels"

    def get_attributes(self):
        return [Input('mv', 'mv')], \
               [Output('dst', 'dst')], \
               []

    def process_inputs(self, inputs, outputs, parameters):
        mv = inputs['mv'].value
        dst = cv2.merge(mv=mv)
        outputs['dst'] = Data(dst)

# cv2.min
class OpenCVAuto2_Min(NormalElement):
    name = 'Min'
    comment = '''min(src1, src2[, dst]) -> dst\n@brief Calculates per-element minimum of two arrays or an array and a scalar.\n\nThe function cv::min calculates the per-element minimum of two arrays:\n\f[\texttt{dst} (I)= \min ( \texttt{src1} (I), \texttt{src2} (I))\f]\nor array and a scalar:\n\f[\texttt{dst} (I)= \min ( \texttt{src1} (I), \texttt{value} )\f]\n@param src1 first input array.\n@param src2 second input array of the same size and type as src1.\n@param dst output array of the same size and type as src1.\n@sa max, compare, inRange, minMaxLoc'''
    package = "Operators"

    def get_attributes(self):
        return [Input('src1', 'src 1'),
                Input('src2', 'src 2')], \
               [Output('dst', 'dst')], \
               []

    def process_inputs(self, inputs, outputs, parameters):
        src1 = inputs['src1'].value
        src2 = inputs['src2'].value
        dst = cv2.min(src1=src1, src2=src2)
        outputs['dst'] = Data(dst)

# cv2.minEnclosingCircle
class OpenCVAuto2_MinEnclosingCircle(NormalElement):
    name = 'Min Enclosing Circle'
    comment = '''minEnclosingCircle(points) -> center, radius\n@brief Finds a circle of the minimum area enclosing a 2D point set.\n\nThe function finds the minimal enclosing circle of a 2D point set using an iterative algorithm.\n\n@param points Input vector of 2D points, stored in std::vector\<\> or Mat\n@param center Output center of the circle.\n@param radius Output radius of the circle.'''
    package = "Contours"

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

# cv2.minEnclosingTriangle
class OpenCVAuto2_MinEnclosingTriangle(NormalElement):
    name = 'Min Enclosing Triangle'
    comment = '''minEnclosingTriangle(points[, triangle]) -> retval, triangle\n@brief Finds a triangle of minimum area enclosing a 2D point set and returns its area.\n\nThe function finds a triangle of minimum area enclosing the given set of 2D points and returns its\narea. The output for a given 2D point set is shown in the image below. 2D points are depicted in\n*red* and the enclosing triangle in *yellow*.\n\n![Sample output of the minimum enclosing triangle function](pics/minenclosingtriangle.png)\n\nThe implementation of the algorithm is based on O'Rourke's @cite ORourke86 and Klee and Laskowski's\n@cite KleeLaskowski85 papers. O'Rourke provides a \f$\theta(n)\f$ algorithm for finding the minimal\nenclosing triangle of a 2D convex polygon with n vertices. Since the #minEnclosingTriangle function\ntakes a 2D point set as input an additional preprocessing step of computing the convex hull of the\n2D point set is required. The complexity of the #convexHull function is \f$O(n log(n))\f$ which is higher\nthan \f$\theta(n)\f$. Thus the overall complexity of the function is \f$O(n log(n))\f$.\n\n@param points Input vector of 2D points with depth CV_32S or CV_32F, stored in std::vector\<\> or Mat\n@param triangle Output vector of three 2D points defining the vertices of the triangle. The depth\nof the OutputArray must be CV_32F.'''
    package = "Contours"

    def get_attributes(self):
        return [Input('points', 'points')], \
               [Output('triangle', 'triangle')], \
               []

    def process_inputs(self, inputs, outputs, parameters):
        points = inputs['points'].value
        retval, triangle = cv2.minEnclosingTriangle(points=points)
        outputs['triangle'] = Data(triangle)

# cv2.minMaxLoc
class OpenCVAuto2_MinMaxLoc(NormalElement):
    name = 'Min Max Loc'
    comment = '''minMaxLoc(src[, mask]) -> minVal, maxVal, minLoc, maxLoc\n@brief Finds the global minimum and maximum in an array.\n\nThe function cv::minMaxLoc finds the minimum and maximum element values and their positions. The\nextremums are searched across the whole array or, if mask is not an empty array, in the specified\narray region.\n\nThe function do not work with multi-channel arrays. If you need to find minimum or maximum\nelements across all the channels, use Mat::reshape first to reinterpret the array as\nsingle-channel. Or you may extract the particular channel using either extractImageCOI , or\nmixChannels , or split .\n@param src input single-channel array.\n@param minVal pointer to the returned minimum value; NULL is used if not required.\n@param maxVal pointer to the returned maximum value; NULL is used if not required.\n@param minLoc pointer to the returned minimum location (in 2D case); NULL is used if not required.\n@param maxLoc pointer to the returned maximum location (in 2D case); NULL is used if not required.\n@param mask optional mask used to select a sub-array.\n@sa max, min, compare, inRange, extractImageCOI, mixChannels, split, Mat::reshape'''
    package = "Matrix miscellaneous"

    def get_attributes(self):
        return [Input('src', 'src'),
                Input('mask', 'mask', optional=True)], \
               [Output('minVal', 'Min Val'),
                Output('maxVal', 'Max Val'),
                Output('minLoc', 'Min Loc'),
                Output('maxLoc', 'Max Loc')], \
               []

    def process_inputs(self, inputs, outputs, parameters):
        src = inputs['src'].value
        mask = inputs['mask'].value
        minVal, maxVal, minLoc, maxLoc = cv2.minMaxLoc(src=src, mask=mask)
        outputs['minVal'] = Data(minVal)
        outputs['maxVal'] = Data(maxVal)
        outputs['minLoc'] = Data(minLoc)
        outputs['maxLoc'] = Data(maxLoc)

# cv2.mixChannels
class OpenCVAuto2_MixChannels(NormalElement):
    name = 'Mix Channels'
    comment = '''mixChannels(src, dst, fromTo) -> dst\n@overload\n@param src input array or vector of matrices; all of the matrices must have the same size and the\nsame depth.\n@param dst output array or vector of matrices; all the matrices **must be allocated**; their size and\ndepth must be the same as in src[0].\n@param fromTo array of index pairs specifying which channels are copied and where; fromTo[k\*2] is\na 0-based index of the input channel in src, fromTo[k\*2+1] is an index of the output channel in\ndst; the continuous channel numbering is used: the first input image channels are indexed from 0 to\nsrc[0].channels()-1, the second input image channels are indexed from src[0].channels() to\nsrc[0].channels() + src[1].channels()-1, and so on, the same scheme is used for the output image\nchannels; as a special case, when fromTo[k\*2] is negative, the corresponding output channel is\nfilled with zero .'''
    package = "Channels"

    def get_attributes(self):
        return [Input('src', 'src'),
                Input('dst', 'dst'),
                Input('fromTo', 'From To')], \
               [Output('dst', 'dst')], \
               []

    def process_inputs(self, inputs, outputs, parameters):
        src = inputs['src'].value
        dst = inputs['dst'].value.copy()
        fromTo = inputs['fromTo'].value
        dst = cv2.mixChannels(src=src, dst=dst, fromTo=fromTo)
        outputs['dst'] = Data(dst)

# cv2.morphologyEx
class OpenCVAuto2_MorphologyEx(NormalElement):
    name = 'Morphology Ex'
    comment = '''morphologyEx(src, op, kernel[, dst[, anchor[, iterations[, borderType[, borderValue]]]]]) -> dst\n@brief Performs advanced morphological transformations.\n\nThe function cv::morphologyEx can perform advanced morphological transformations using an erosion and dilation as\nbasic operations.\n\nAny of the operations can be done in-place. In case of multi-channel images, each channel is\nprocessed independently.\n\n@param src Source image. The number of channels can be arbitrary. The depth should be one of\nCV_8U, CV_16U, CV_16S, CV_32F or CV_64F.\n@param dst Destination image of the same size and type as source image.\n@param op Type of a morphological operation, see #MorphTypes\n@param kernel Structuring element. It can be created using #getStructuringElement.\n@param anchor Anchor position with the kernel. Negative values mean that the anchor is at the\nkernel center.\n@param iterations Number of times erosion and dilation are applied.\n@param borderType Pixel extrapolation method, see #BorderTypes\n@param borderValue Border value in case of a constant border. The default value has a special\nmeaning.\n@sa  dilate, erode, getStructuringElement\n@note The number of iterations is the number of times erosion or dilatation operation will be applied.\nFor instance, an opening operation (#MORPH_OPEN) with two iterations is equivalent to apply\nsuccessively: erode -> erode -> dilate -> dilate (and not erode -> dilate -> erode -> dilate).'''
    package = "Filters"

    def get_attributes(self):
        return [Input('src', 'src'),
                Input('kernel', 'kernel')], \
               [Output('dst', 'dst')], \
               [IntParameter('op', 'op'),
                PointParameter('anchor', 'anchor'),
                IntParameter('iterations', 'iterations', min_=0),
                ComboboxParameter('borderType', name='Border Type', values=[('BORDER_CONSTANT',0),('BORDER_REPLICATE',1),('BORDER_REFLECT',2),('BORDER_WRAP',3),('BORDER_DEFAULT',4),('BORDER_REFLECT101',4),('BORDER_REFLECT_101',4),('BORDER_TRANSPARENT',5),('BORDER_ISOLATED',16)]),
                ScalarParameter('borderValue', 'Border Value')]

    def process_inputs(self, inputs, outputs, parameters):
        src = inputs['src'].value
        kernel = inputs['kernel'].value
        op = parameters['op']
        anchor = parameters['anchor']
        iterations = parameters['iterations']
        borderType = parameters['borderType']
        borderValue = parameters['borderValue']
        dst = cv2.morphologyEx(src=src, kernel=kernel, op=op, anchor=anchor, iterations=iterations, borderType=borderType, borderValue=borderValue)
        outputs['dst'] = Data(dst)

# cv2.multiply
class OpenCVAuto2_Multiply(NormalElement):
    name = 'Multiply'
    comment = '''multiply(src1, src2[, dst[, scale[, dtype]]]) -> dst\n@brief Calculates the per-element scaled product of two arrays.\n\nThe function multiply calculates the per-element product of two arrays:\n\n\f[\texttt{dst} (I)= \texttt{saturate} ( \texttt{scale} \cdot \texttt{src1} (I)  \cdot \texttt{src2} (I))\f]\n\nThere is also a @ref MatrixExpressions -friendly variant of the first function. See Mat::mul .\n\nFor a not-per-element matrix product, see gemm .\n\n@note Saturation is not applied when the output array has the depth\nCV_32S. You may even get result of an incorrect sign in the case of\noverflow.\n@param src1 first input array.\n@param src2 second input array of the same size and the same type as src1.\n@param dst output array of the same size and type as src1.\n@param scale optional scale factor.\n@param dtype optional depth of the output array\n@sa add, subtract, divide, scaleAdd, addWeighted, accumulate, accumulateProduct, accumulateSquare,\nMat::convertTo'''
    package = "Operators"

    def get_attributes(self):
        return [Input('src1', 'src 1'),
                Input('src2', 'src 2')], \
               [Output('dst', 'dst')], \
               [FloatParameter('scale', 'scale'),
                ComboboxParameter('dtype', name='dtype', values=[('NONE',-1),('CV_8U',0),('CV_8UC1',0),('CV_8S',1),('CV_8SC1',1),('CV_16U',2),('CV_16UC1',2),('CV_16S',3),('CV_16SC1',3),('CV_32S',4),('CV_32SC1',4),('CV_32F',5),('CV_32FC1',5),('CV_64F',6),('CV_64FC1',6),('CV_8UC2',8),('CV_8SC2',9),('CV_16UC2',10),('CV_16SC2',11),('CV_32SC2',12),('CV_32FC2',13),('CV_64FC2',14),('CV_8UC3',16),('CV_8SC3',17),('CV_16UC3',18),('CV_16SC3',19),('CV_32SC3',20),('CV_32FC3',21),('CV_64FC3',22),('CV_8UC4',24),('CV_8SC4',25),('CV_16UC4',26),('CV_16SC4',27),('CV_32SC4',28),('CV_32FC4',29),('CV_64FC4',30)])]

    def process_inputs(self, inputs, outputs, parameters):
        src1 = inputs['src1'].value
        src2 = inputs['src2'].value
        scale = parameters['scale']
        dtype = parameters['dtype']
        dst = cv2.multiply(src1=src1, src2=src2, scale=scale, dtype=dtype)
        outputs['dst'] = Data(dst)

# cv2.normalize
class OpenCVAuto2_Normalize(NormalElement):
    name = 'Normalize'
    comment = '''normalize(src, dst[, alpha[, beta[, norm_type[, dtype[, mask]]]]]) -> dst\n@brief Normalizes the norm or value range of an array.\n\nThe function cv::normalize normalizes scale and shift the input array elements so that\n\f[\| \texttt{dst} \| _{L_p}= \texttt{alpha}\f]\n(where p=Inf, 1 or 2) when normType=NORM_INF, NORM_L1, or NORM_L2, respectively; or so that\n\f[\min _I  \texttt{dst} (I)= \texttt{alpha} , \, \, \max _I  \texttt{dst} (I)= \texttt{beta}\f]\n\nwhen normType=NORM_MINMAX (for dense arrays only). The optional mask specifies a sub-array to be\nnormalized. This means that the norm or min-n-max are calculated over the sub-array, and then this\nsub-array is modified to be normalized. If you want to only use the mask to calculate the norm or\nmin-max but modify the whole array, you can use norm and Mat::convertTo.\n\nIn case of sparse matrices, only the non-zero values are analyzed and transformed. Because of this,\nthe range transformation for sparse matrices is not allowed since it can shift the zero level.\n\nPossible usage with some positive example data:\n@code{.cpp}\nvector<double> positiveData = { 2.0, 8.0, 10.0 };\nvector<double> normalizedData_l1, normalizedData_l2, normalizedData_inf, normalizedData_minmax;\n\n// Norm to probability (total count)\n// sum(numbers) = 20.0\n// 2.0      0.1     (2.0/20.0)\n// 8.0      0.4     (8.0/20.0)\n// 10.0     0.5     (10.0/20.0)\nnormalize(positiveData, normalizedData_l1, 1.0, 0.0, NORM_L1);\n\n// Norm to unit vector: ||positiveData|| = 1.0\n// 2.0      0.15\n// 8.0      0.62\n// 10.0     0.77\nnormalize(positiveData, normalizedData_l2, 1.0, 0.0, NORM_L2);\n\n// Norm to max element\n// 2.0      0.2     (2.0/10.0)\n// 8.0      0.8     (8.0/10.0)\n// 10.0     1.0     (10.0/10.0)\nnormalize(positiveData, normalizedData_inf, 1.0, 0.0, NORM_INF);\n\n// Norm to range [0.0;1.0]\n// 2.0      0.0     (shift to left border)\n// 8.0      0.75    (6.0/8.0)\n// 10.0     1.0     (shift to right border)\nnormalize(positiveData, normalizedData_minmax, 1.0, 0.0, NORM_MINMAX);\n@endcode\n\n@param src input array.\n@param dst output array of the same size as src .\n@param alpha norm value to normalize to or the lower range boundary in case of the range\nnormalization.\n@param beta upper range boundary in case of the range normalization; it is not used for the norm\nnormalization.\n@param norm_type normalization type (see cv::NormTypes).\n@param dtype when negative, the output array has the same type as src; otherwise, it has the same\nnumber of channels as src and the depth =CV_MAT_DEPTH(dtype).\n@param mask optional operation mask.\n@sa norm, Mat::convertTo, SparseMat::convertTo'''
    package = "Color"

    def get_attributes(self):
        return [Input('src', 'src'),
                Input('dst', 'dst'),
                Input('mask', 'mask', optional=True)], \
               [Output('dst', 'dst')], \
               [FloatParameter('alpha', 'alpha'),
                FloatParameter('beta', 'beta'),
                ComboboxParameter('norm_type', name='norm type', values=[('NORM_INF',1),('NORM_L1',2),('NORM_L2',4),('NORM_L2SQR',5),('NORM_HAMMING',6),('NORM_HAMMING2',7),('NORM_TYPE_MASK',7),('NORM_RELATIVE',8),('NORM_MINMAX',32)]),
                ComboboxParameter('dtype', name='dtype', values=[('NONE',-1),('CV_8U',0),('CV_8UC1',0),('CV_8S',1),('CV_8SC1',1),('CV_16U',2),('CV_16UC1',2),('CV_16S',3),('CV_16SC1',3),('CV_32S',4),('CV_32SC1',4),('CV_32F',5),('CV_32FC1',5),('CV_64F',6),('CV_64FC1',6),('CV_8UC2',8),('CV_8SC2',9),('CV_16UC2',10),('CV_16SC2',11),('CV_32SC2',12),('CV_32FC2',13),('CV_64FC2',14),('CV_8UC3',16),('CV_8SC3',17),('CV_16UC3',18),('CV_16SC3',19),('CV_32SC3',20),('CV_32FC3',21),('CV_64FC3',22),('CV_8UC4',24),('CV_8SC4',25),('CV_16UC4',26),('CV_16SC4',27),('CV_32SC4',28),('CV_32FC4',29),('CV_64FC4',30)])]

    def process_inputs(self, inputs, outputs, parameters):
        src = inputs['src'].value
        dst = inputs['dst'].value.copy()
        mask = inputs['mask'].value
        alpha = parameters['alpha']
        beta = parameters['beta']
        norm_type = parameters['norm_type']
        dtype = parameters['dtype']
        dst = cv2.normalize(src=src, dst=dst, mask=mask, alpha=alpha, beta=beta, norm_type=norm_type, dtype=dtype)
        outputs['dst'] = Data(dst)

# cv2.patchNaNs
class OpenCVAuto2_PatchNaNs(NormalElement):
    name = 'Patch Na Ns'
    comment = '''patchNaNs(a[, val]) -> a\n@brief converts NaN's to the given number'''
    package = "Miscellaneous"

    def get_attributes(self):
        return [Input('a', 'a')], \
               [Output('a', 'a')], \
               [FloatParameter('val', 'val')]

    def process_inputs(self, inputs, outputs, parameters):
        a = inputs['a'].value.copy()
        val = parameters['val']
        a = cv2.patchNaNs(a=a, val=val)
        outputs['a'] = Data(a)

# cv2.pencilSketch
class OpenCVAuto2_PencilSketch(NormalElement):
    name = 'Pencil Sketch'
    comment = '''pencilSketch(src[, dst1[, dst2[, sigma_s[, sigma_r[, shade_factor]]]]]) -> dst1, dst2\n@brief Pencil-like non-photorealistic line drawing\n\n@param src Input 8-bit 3-channel image.\n@param dst1 Output 8-bit 1-channel image.\n@param dst2 Output image with the same size and type as src.\n@param sigma_s Range between 0 to 200.\n@param sigma_r Range between 0 to 1.\n@param shade_factor Range between 0 to 0.1.'''
    package = "Photo"

    def get_attributes(self):
        return [Input('src', 'src')], \
               [Output('dst1', 'dst 1'),
                Output('dst2', 'dst 2')], \
               [FloatParameter('sigma_s', 'sigma s'),
                FloatParameter('sigma_r', 'sigma r'),
                FloatParameter('shade_factor', 'shade factor')]

    def process_inputs(self, inputs, outputs, parameters):
        src = inputs['src'].value
        sigma_s = parameters['sigma_s']
        sigma_r = parameters['sigma_r']
        shade_factor = parameters['shade_factor']
        dst1, dst2 = cv2.pencilSketch(src=src, sigma_s=sigma_s, sigma_r=sigma_r, shade_factor=shade_factor)
        outputs['dst1'] = Data(dst1)
        outputs['dst2'] = Data(dst2)

# cv2.perspectiveTransform
class OpenCVAuto2_PerspectiveTransform(NormalElement):
    name = 'Perspective Transform'
    comment = '''perspectiveTransform(src, m[, dst]) -> dst\n@brief Performs the perspective matrix transformation of vectors.\n\nThe function cv::perspectiveTransform transforms every element of src by\ntreating it as a 2D or 3D vector, in the following way:\n\f[(x, y, z)  \rightarrow (x'/w, y'/w, z'/w)\f]\nwhere\n\f[(x', y', z', w') =  \texttt{mat} \cdot \begin{bmatrix} x & y & z & 1  \end{bmatrix}\f]\nand\n\f[w =  \fork{w'}{if \(w' \ne 0\)}{\infty}{otherwise}\f]\n\nHere a 3D vector transformation is shown. In case of a 2D vector\ntransformation, the z component is omitted.\n\n@note The function transforms a sparse set of 2D or 3D vectors. If you\nwant to transform an image using perspective transformation, use\nwarpPerspective . If you have an inverse problem, that is, you want to\ncompute the most probable perspective transformation out of several\npairs of corresponding points, you can use getPerspectiveTransform or\nfindHomography .\n@param src input two-channel or three-channel floating-point array; each\nelement is a 2D/3D vector to be transformed.\n@param dst output array of the same size and type as src.\n@param m 3x3 or 4x4 floating-point transformation matrix.\n@sa  transform, warpPerspective, getPerspectiveTransform, findHomography'''
    package = "Transforms"

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

# cv2.phaseCorrelate
class OpenCVAuto2_PhaseCorrelate(NormalElement):
    name = 'Phase Correlate'
    comment = '''phaseCorrelate(src1, src2[, window]) -> retval, response\n@brief The function is used to detect translational shifts that occur between two images.\n\nThe operation takes advantage of the Fourier shift theorem for detecting the translational shift in\nthe frequency domain. It can be used for fast image registration as well as motion estimation. For\nmore information please see <http://en.wikipedia.org/wiki/Phase_correlation>\n\nCalculates the cross-power spectrum of two supplied source arrays. The arrays are padded if needed\nwith getOptimalDFTSize.\n\nThe function performs the following equations:\n- First it applies a Hanning window (see <http://en.wikipedia.org/wiki/Hann_function>) to each\nimage to remove possible edge effects. This window is cached until the array size changes to speed\nup processing time.\n- Next it computes the forward DFTs of each source array:\n\f[\mathbf{G}_a = \mathcal{F}\{src_1\}, \; \mathbf{G}_b = \mathcal{F}\{src_2\}\f]\nwhere \f$\mathcal{F}\f$ is the forward DFT.\n- It then computes the cross-power spectrum of each frequency domain array:\n\f[R = \frac{ \mathbf{G}_a \mathbf{G}_b^*}{|\mathbf{G}_a \mathbf{G}_b^*|}\f]\n- Next the cross-correlation is converted back into the time domain via the inverse DFT:\n\f[r = \mathcal{F}^{-1}\{R\}\f]\n- Finally, it computes the peak location and computes a 5x5 weighted centroid around the peak to\nachieve sub-pixel accuracy.\n\f[(\Delta x, \Delta y) = \texttt{weightedCentroid} \{\arg \max_{(x, y)}\{r\}\}\f]\n- If non-zero, the response parameter is computed as the sum of the elements of r within the 5x5\ncentroid around the peak location. It is normalized to a maximum of 1 (meaning there is a single\npeak) and will be smaller when there are multiple peaks.\n\n@param src1 Source floating point array (CV_32FC1 or CV_64FC1)\n@param src2 Source floating point array (CV_32FC1 or CV_64FC1)\n@param window Floating point array with windowing coefficients to reduce edge effects (optional).\n@param response Signal power within the 5x5 centroid around the peak, between 0 and 1 (optional).\n@returns detected phase shift (sub-pixel) between the two arrays.\n\n@sa dft, getOptimalDFTSize, idft, mulSpectrums createHanningWindow'''
    package = "Motion detection"

    def get_attributes(self):
        return [Input('src1', 'src 1'),
                Input('src2', 'src 2'),
                Input('window', 'window', optional=True)], \
               [Output('response', 'response')], \
               []

    def process_inputs(self, inputs, outputs, parameters):
        src1 = inputs['src1'].value
        src2 = inputs['src2'].value
        window = inputs['window'].value
        retval, response = cv2.phaseCorrelate(src1=src1, src2=src2, window=window)
        outputs['response'] = Data(response)

# cv2.pow
class OpenCVAuto2_Pow(NormalElement):
    name = 'Pow'
    comment = '''pow(src, power[, dst]) -> dst\n@brief Raises every array element to a power.\n\nThe function cv::pow raises every element of the input array to power :\n\f[\texttt{dst} (I) =  \fork{\texttt{src}(I)^{power}}{if \(\texttt{power}\) is integer}{|\texttt{src}(I)|^{power}}{otherwise}\f]\n\nSo, for a non-integer power exponent, the absolute values of input array\nelements are used. However, it is possible to get true values for\nnegative values using some extra operations. In the example below,\ncomputing the 5th root of array src shows:\n@code{.cpp}\nMat mask = src < 0;\npow(src, 1./5, dst);\nsubtract(Scalar::all(0), dst, dst, mask);\n@endcode\nFor some values of power, such as integer values, 0.5 and -0.5,\nspecialized faster algorithms are used.\n\nSpecial values (NaN, Inf) are not handled.\n@param src input array.\n@param power exponent of power.\n@param dst output array of the same size and type as src.\n@sa sqrt, exp, log, cartToPolar, polarToCart'''
    package = "Math"

    def get_attributes(self):
        return [Input('src', 'src')], \
               [Output('dst', 'dst')], \
               [FloatParameter('power', 'power')]

    def process_inputs(self, inputs, outputs, parameters):
        src = inputs['src'].value
        power = parameters['power']
        dst = cv2.pow(src=src, power=power)
        outputs['dst'] = Data(dst)

# cv2.preCornerDetect
class OpenCVAuto2_PreCornerDetect(NormalElement):
    name = 'Pre Corner Detect'
    comment = '''preCornerDetect(src, ksize[, dst[, borderType]]) -> dst\n@brief Calculates a feature map for corner detection.\n\nThe function calculates the complex spatial derivative-based function of the source image\n\n\f[\texttt{dst} = (D_x  \texttt{src} )^2  \cdot D_{yy}  \texttt{src} + (D_y  \texttt{src} )^2  \cdot D_{xx}  \texttt{src} - 2 D_x  \texttt{src} \cdot D_y  \texttt{src} \cdot D_{xy}  \texttt{src}\f]\n\nwhere \f$D_x\f$,\f$D_y\f$ are the first image derivatives, \f$D_{xx}\f$,\f$D_{yy}\f$ are the second image\nderivatives, and \f$D_{xy}\f$ is the mixed derivative.\n\nThe corners can be found as local maximums of the functions, as shown below:\n@code\nMat corners, dilated_corners;\npreCornerDetect(image, corners, 3);\n// dilation with 3x3 rectangular structuring element\ndilate(corners, dilated_corners, Mat(), 1);\nMat corner_mask = corners == dilated_corners;\n@endcode\n\n@param src Source single-channel 8-bit of floating-point image.\n@param dst Output image that has the type CV_32F and the same size as src .\n@param ksize %Aperture size of the Sobel .\n@param borderType Pixel extrapolation method. See #BorderTypes.'''
    package = "Features 2D"

    def get_attributes(self):
        return [Input('src', 'src')], \
               [Output('dst', 'dst')], \
               [SizeParameter('ksize', 'ksize'),
                ComboboxParameter('borderType', name='Border Type', values=[('BORDER_CONSTANT',0),('BORDER_REPLICATE',1),('BORDER_REFLECT',2),('BORDER_WRAP',3),('BORDER_DEFAULT',4),('BORDER_REFLECT101',4),('BORDER_REFLECT_101',4),('BORDER_TRANSPARENT',5),('BORDER_ISOLATED',16)])]

    def process_inputs(self, inputs, outputs, parameters):
        src = inputs['src'].value
        ksize = parameters['ksize']
        borderType = parameters['borderType']
        dst = cv2.preCornerDetect(src=src, ksize=ksize, borderType=borderType)
        outputs['dst'] = Data(dst)

# cv2.projectPoints
class OpenCVAuto2_ProjectPoints(NormalElement):
    name = 'Project Points'
    comment = '''projectPoints(objectPoints, rvec, tvec, cameraMatrix, distCoeffs[, imagePoints[, jacobian[, aspectRatio]]]) -> imagePoints, jacobian\n@brief Projects 3D points to an image plane.\n\n@param objectPoints Array of object points, 3xN/Nx3 1-channel or 1xN/Nx1 3-channel (or\nvector\<Point3f\> ), where N is the number of points in the view.\n@param rvec Rotation vector. See Rodrigues for details.\n@param tvec Translation vector.\n@param cameraMatrix Camera matrix \f$A = \vecthreethree{f_x}{0}{c_x}{0}{f_y}{c_y}{0}{0}{_1}\f$ .\n@param distCoeffs Input vector of distortion coefficients\n\f$(k_1, k_2, p_1, p_2[, k_3[, k_4, k_5, k_6 [, s_1, s_2, s_3, s_4[, \tau_x, \tau_y]]]])\f$ of\n4, 5, 8, 12 or 14 elements. If the vector is empty, the zero distortion coefficients are assumed.\n@param imagePoints Output array of image points, 2xN/Nx2 1-channel or 1xN/Nx1 2-channel, or\nvector\<Point2f\> .\n@param jacobian Optional output 2Nx(10+\<numDistCoeffs\>) jacobian matrix of derivatives of image\npoints with respect to components of the rotation vector, translation vector, focal lengths,\ncoordinates of the principal point and the distortion coefficients. In the old interface different\ncomponents of the jacobian are returned via different output parameters.\n@param aspectRatio Optional "fixed aspect ratio" parameter. If the parameter is not 0, the\nfunction assumes that the aspect ratio (*fx/fy*) is fixed and correspondingly adjusts the jacobian\nmatrix.\n\nThe function computes projections of 3D points to the image plane given intrinsic and extrinsic\ncamera parameters. Optionally, the function computes Jacobians - matrices of partial derivatives of\nimage points coordinates (as functions of all the input parameters) with respect to the particular\nparameters, intrinsic and/or extrinsic. The Jacobians are used during the global optimization in\ncalibrateCamera, solvePnP, and stereoCalibrate . The function itself can also be used to compute a\nre-projection error given the current intrinsic and extrinsic parameters.\n\n@note By setting rvec=tvec=(0,0,0) or by setting cameraMatrix to a 3x3 identity matrix, or by\npassing zero distortion coefficients, you can get various useful partial cases of the function. This\nmeans that you can compute the distorted coordinates for a sparse set of points or apply a\nperspective transformation (and also compute the derivatives) in the ideal zero-distortion setup.'''
    package = "3D calibration & reconstruction"

    def get_attributes(self):
        return [Input('objectPoints', 'Object Points'),
                Input('rvec', 'rvec'),
                Input('tvec', 'tvec'),
                Input('cameraMatrix', 'Camera Matrix'),
                Input('distCoeffs', 'Dist Coeffs')], \
               [Output('imagePoints', 'Image Points'),
                Output('jacobian', 'jacobian')], \
               [FloatParameter('aspectRatio', 'Aspect Ratio')]

    def process_inputs(self, inputs, outputs, parameters):
        objectPoints = inputs['objectPoints'].value
        rvec = inputs['rvec'].value
        tvec = inputs['tvec'].value
        cameraMatrix = inputs['cameraMatrix'].value
        distCoeffs = inputs['distCoeffs'].value
        aspectRatio = parameters['aspectRatio']
        imagePoints, jacobian = cv2.projectPoints(objectPoints=objectPoints, rvec=rvec, tvec=tvec, cameraMatrix=cameraMatrix, distCoeffs=distCoeffs, aspectRatio=aspectRatio)
        outputs['imagePoints'] = Data(imagePoints)
        outputs['jacobian'] = Data(jacobian)

# cv2.pyrDown
class OpenCVAuto2_PyrDown(NormalElement):
    name = 'Pyr Down'
    comment = '''pyrDown(src[, dst[, dstsize[, borderType]]]) -> dst\n@brief Blurs an image and downsamples it.\n\nBy default, size of the output image is computed as `Size((src.cols+1)/2, (src.rows+1)/2)`, but in\nany case, the following conditions should be satisfied:\n\n\f[\begin{array}{l} | \texttt{dstsize.width} *2-src.cols| \leq 2 \\ | \texttt{dstsize.height} *2-src.rows| \leq 2 \end{array}\f]\n\nThe function performs the downsampling step of the Gaussian pyramid construction. First, it\nconvolves the source image with the kernel:\n\n\f[\frac{1}{256} \begin{bmatrix} 1 & 4 & 6 & 4 & 1  \\ 4 & 16 & 24 & 16 & 4  \\ 6 & 24 & 36 & 24 & 6  \\ 4 & 16 & 24 & 16 & 4  \\ 1 & 4 & 6 & 4 & 1 \end{bmatrix}\f]\n\nThen, it downsamples the image by rejecting even rows and columns.\n\n@param src input image.\n@param dst output image; it has the specified size and the same type as src.\n@param dstsize size of the output image.\n@param borderType Pixel extrapolation method, see #BorderTypes (#BORDER_CONSTANT isn't supported)'''
    package = "Filters"

    def get_attributes(self):
        return [Input('src', 'src')], \
               [Output('dst', 'dst')], \
               [SizeParameter('dstsize', 'dstsize'),
                ComboboxParameter('borderType', name='Border Type', values=[('BORDER_CONSTANT',0),('BORDER_REPLICATE',1),('BORDER_REFLECT',2),('BORDER_WRAP',3),('BORDER_DEFAULT',4),('BORDER_REFLECT101',4),('BORDER_REFLECT_101',4),('BORDER_TRANSPARENT',5),('BORDER_ISOLATED',16)])]

    def process_inputs(self, inputs, outputs, parameters):
        src = inputs['src'].value
        dstsize = parameters['dstsize']
        borderType = parameters['borderType']
        dst = cv2.pyrDown(src=src, dstsize=dstsize, borderType=borderType)
        outputs['dst'] = Data(dst)

# cv2.pyrUp
class OpenCVAuto2_PyrUp(NormalElement):
    name = 'Pyr Up'
    comment = '''pyrUp(src[, dst[, dstsize[, borderType]]]) -> dst\n@brief Upsamples an image and then blurs it.\n\nBy default, size of the output image is computed as `Size(src.cols\*2, (src.rows\*2)`, but in any\ncase, the following conditions should be satisfied:\n\n\f[\begin{array}{l} | \texttt{dstsize.width} -src.cols*2| \leq  ( \texttt{dstsize.width}   \mod  2)  \\ | \texttt{dstsize.height} -src.rows*2| \leq  ( \texttt{dstsize.height}   \mod  2) \end{array}\f]\n\nThe function performs the upsampling step of the Gaussian pyramid construction, though it can\nactually be used to construct the Laplacian pyramid. First, it upsamples the source image by\ninjecting even zero rows and columns and then convolves the result with the same kernel as in\npyrDown multiplied by 4.\n\n@param src input image.\n@param dst output image. It has the specified size and the same type as src .\n@param dstsize size of the output image.\n@param borderType Pixel extrapolation method, see #BorderTypes (only #BORDER_DEFAULT is supported)'''
    package = "Filters"

    def get_attributes(self):
        return [Input('src', 'src')], \
               [Output('dst', 'dst')], \
               [SizeParameter('dstsize', 'dstsize'),
                ComboboxParameter('borderType', name='Border Type', values=[('BORDER_CONSTANT',0),('BORDER_REPLICATE',1),('BORDER_REFLECT',2),('BORDER_WRAP',3),('BORDER_DEFAULT',4),('BORDER_REFLECT101',4),('BORDER_REFLECT_101',4),('BORDER_TRANSPARENT',5),('BORDER_ISOLATED',16)])]

    def process_inputs(self, inputs, outputs, parameters):
        src = inputs['src'].value
        dstsize = parameters['dstsize']
        borderType = parameters['borderType']
        dst = cv2.pyrUp(src=src, dstsize=dstsize, borderType=borderType)
        outputs['dst'] = Data(dst)

# cv2.randShuffle
class OpenCVAuto2_RandShuffle(NormalElement):
    name = 'Rand Shuffle'
    comment = '''randShuffle(dst[, iterFactor]) -> dst\n@brief Shuffles the array elements randomly.\n\nThe function cv::randShuffle shuffles the specified 1D array by randomly choosing pairs of elements and\nswapping them. The number of such swap operations will be dst.rows\*dst.cols\*iterFactor .\n@param dst input/output numerical 1D array.\n@param iterFactor scale factor that determines the number of random swap operations (see the details\nbelow).\n@param rng optional random number generator used for shuffling; if it is zero, theRNG () is used\ninstead.\n@sa RNG, sort'''
    package = "Random"

    def get_attributes(self):
        return [Input('dst', 'dst')], \
               [Output('dst', 'dst')], \
               [FloatParameter('iterFactor', 'Iter Factor')]

    def process_inputs(self, inputs, outputs, parameters):
        dst = inputs['dst'].value.copy()
        iterFactor = parameters['iterFactor']
        dst = cv2.randShuffle(dst=dst, iterFactor=iterFactor)
        outputs['dst'] = Data(dst)

# cv2.randn
class OpenCVAuto2_Randn(NormalElement):
    name = 'Randn'
    comment = '''randn(dst, mean, stddev) -> dst\n@brief Fills the array with normally distributed random numbers.\n\nThe function cv::randn fills the matrix dst with normally distributed random numbers with the specified\nmean vector and the standard deviation matrix. The generated random numbers are clipped to fit the\nvalue range of the output array data type.\n@param dst output array of random numbers; the array must be pre-allocated and have 1 to 4 channels.\n@param mean mean value (expectation) of the generated random numbers.\n@param stddev standard deviation of the generated random numbers; it can be either a vector (in\nwhich case a diagonal standard deviation matrix is assumed) or a square matrix.\n@sa RNG, randu'''
    package = "Random"

    def get_attributes(self):
        return [Input('dst', 'dst'),
                Input('mean', 'mean'),
                Input('stddev', 'stddev')], \
               [Output('dst', 'dst')], \
               []

    def process_inputs(self, inputs, outputs, parameters):
        dst = inputs['dst'].value.copy()
        mean = inputs['mean'].value
        stddev = inputs['stddev'].value
        dst = cv2.randn(dst=dst, mean=mean, stddev=stddev)
        outputs['dst'] = Data(dst)

# cv2.randu
class OpenCVAuto2_Randu(NormalElement):
    name = 'Randu'
    comment = '''randu(dst, low, high) -> dst\n@brief Generates a single uniformly-distributed random number or an array of random numbers.\n\nNon-template variant of the function fills the matrix dst with uniformly-distributed\nrandom numbers from the specified range:\n\f[\texttt{low} _c  \leq \texttt{dst} (I)_c <  \texttt{high} _c\f]\n@param dst output array of random numbers; the array must be pre-allocated.\n@param low inclusive lower boundary of the generated random numbers.\n@param high exclusive upper boundary of the generated random numbers.\n@sa RNG, randn, theRNG'''
    package = "Random"

    def get_attributes(self):
        return [Input('dst', 'dst'),
                Input('low', 'low'),
                Input('high', 'high')], \
               [Output('dst', 'dst')], \
               []

    def process_inputs(self, inputs, outputs, parameters):
        dst = inputs['dst'].value.copy()
        low = inputs['low'].value
        high = inputs['high'].value
        dst = cv2.randu(dst=dst, low=low, high=high)
        outputs['dst'] = Data(dst)

# cv2.recoverPose
class OpenCVAuto2_RecoverPose(NormalElement):
    name = 'Recover Pose'
    comment = '''recoverPose(E, points1, points2, cameraMatrix[, R[, t[, mask]]]) -> retval, R, t, mask\n@brief Recover relative camera rotation and translation from an estimated essential matrix and the\ncorresponding points in two images, using cheirality check. Returns the number of inliers which pass\nthe check.\n\n@param E The input essential matrix.\n@param points1 Array of N 2D points from the first image. The point coordinates should be\nfloating-point (single or double precision).\n@param points2 Array of the second image points of the same size and format as points1 .\n@param cameraMatrix Camera matrix \f$K = \vecthreethree{f_x}{0}{c_x}{0}{f_y}{c_y}{0}{0}{1}\f$ .\nNote that this function assumes that points1 and points2 are feature points from cameras with the\nsame camera matrix.\n@param R Recovered relative rotation.\n@param t Recovered relative translation.\n@param mask Input/output mask for inliers in points1 and points2.\n:   If it is not empty, then it marks inliers in points1 and points2 for then given essential\nmatrix E. Only these inliers will be used to recover pose. In the output mask only inliers\nwhich pass the cheirality check.\nThis function decomposes an essential matrix using decomposeEssentialMat and then verifies possible\npose hypotheses by doing cheirality check. The cheirality check basically means that the\ntriangulated 3D points should have positive depth. Some details can be found in @cite Nister03 .\n\nThis function can be used to process output E and mask from findEssentialMat. In this scenario,\npoints1 and points2 are the same input for findEssentialMat. :\n@code\n// Example. Estimation of fundamental matrix using the RANSAC algorithm\nint point_count = 100;\nvector<Point2f> points1(point_count);\nvector<Point2f> points2(point_count);\n\n// initialize the points here ...\nfor( int i = 0; i < point_count; i++ )\n{\npoints1[i] = ...;\npoints2[i] = ...;\n}\n\n// cametra matrix with both focal lengths = 1, and principal point = (0, 0)\nMat cameraMatrix = Mat::eye(3, 3, CV_64F);\n\nMat E, R, t, mask;\n\nE = findEssentialMat(points1, points2, cameraMatrix, RANSAC, 0.999, 1.0, mask);\nrecoverPose(E, points1, points2, cameraMatrix, R, t, mask);\n@endcode



recoverPose(E, points1, points2[, R[, t[, focal[, pp[, mask]]]]]) -> retval, R, t, mask\n@overload\n@param E The input essential matrix.\n@param points1 Array of N 2D points from the first image. The point coordinates should be\nfloating-point (single or double precision).\n@param points2 Array of the second image points of the same size and format as points1 .\n@param R Recovered relative rotation.\n@param t Recovered relative translation.\n@param focal Focal length of the camera. Note that this function assumes that points1 and points2\nare feature points from cameras with same focal length and principal point.\n@param pp principal point of the camera.\n@param mask Input/output mask for inliers in points1 and points2.\n:   If it is not empty, then it marks inliers in points1 and points2 for then given essential\nmatrix E. Only these inliers will be used to recover pose. In the output mask only inliers\nwhich pass the cheirality check.\n\nThis function differs from the one above that it computes camera matrix from focal length and\nprincipal point:\n\n\f[K =\n\begin{bmatrix}\nf & 0 & x_{pp}  \\\n0 & f & y_{pp}  \\\n0 & 0 & 1\n\end{bmatrix}\f]



recoverPose(E, points1, points2, cameraMatrix, distanceThresh[, R[, t[, mask[, triangulatedPoints]]]]) -> retval, R, t, mask, triangulatedPoints\n@overload\n@param E The input essential matrix.\n@param points1 Array of N 2D points from the first image. The point coordinates should be\nfloating-point (single or double precision).\n@param points2 Array of the second image points of the same size and format as points1.\n@param cameraMatrix Camera matrix \f$K = \vecthreethree{f_x}{0}{c_x}{0}{f_y}{c_y}{0}{0}{1}\f$ .\nNote that this function assumes that points1 and points2 are feature points from cameras with the\nsame camera matrix.\n@param R Recovered relative rotation.\n@param t Recovered relative translation.\n@param distanceThresh threshold distance which is used to filter out far away points (i.e. infinite points).\n@param mask Input/output mask for inliers in points1 and points2.\n:   If it is not empty, then it marks inliers in points1 and points2 for then given essential\nmatrix E. Only these inliers will be used to recover pose. In the output mask only inliers\nwhich pass the cheirality check.\n@param triangulatedPoints 3d points which were reconstructed by triangulation.'''
    package = "3D calibration & reconstruction"

    def get_attributes(self):
        return [Input('E', 'E'),
                Input('points1', 'points 1'),
                Input('points2', 'points 2'),
                Input('cameraMatrix', 'Camera Matrix')], \
               [Output('R', 'R'),
                Output('t', 't'),
                Output('mask', 'mask')], \
               []

    def process_inputs(self, inputs, outputs, parameters):
        E = inputs['E'].value
        points1 = inputs['points1'].value
        points2 = inputs['points2'].value
        cameraMatrix = inputs['cameraMatrix'].value
        retval, R, t, mask = cv2.recoverPose(E=E, points1=points1, points2=points2, cameraMatrix=cameraMatrix)
        outputs['R'] = Data(R)
        outputs['t'] = Data(t)
        outputs['mask'] = Data(mask)

# cv2.rectangle
class OpenCVAuto2_Rectangle(NormalElement):
    name = 'Rectangle'
    comment = '''rectangle(img, pt1, pt2, color[, thickness[, lineType[, shift]]]) -> img\n@brief Draws a simple, thick, or filled up-right rectangle.\n\nThe function cv::rectangle draws a rectangle outline or a filled rectangle whose two opposite corners\nare pt1 and pt2.\n\n@param img Image.\n@param pt1 Vertex of the rectangle.\n@param pt2 Vertex of the rectangle opposite to pt1 .\n@param color Rectangle color or brightness (grayscale image).\n@param thickness Thickness of lines that make up the rectangle. Negative values, like #FILLED,\nmean that the function has to draw a filled rectangle.\n@param lineType Type of the line. See #LineTypes\n@param shift Number of fractional bits in the point coordinates.'''
    package = "Drawing"

    def get_attributes(self):
        return [Input('img', 'img')], \
               [Output('img', 'img')], \
               [PointParameter('pt1', 'pt 1'),
                PointParameter('pt2', 'pt 2'),
                ScalarParameter('color', 'color'),
                IntParameter('thickness', 'thickness', min_=-1, max_=100),
                ComboboxParameter('lineType', name='Line Type', values=[('LINE_4',4),('LINE_8',8),('LINE_AA',16)]),
                IntParameter('shift', 'shift')]

    def process_inputs(self, inputs, outputs, parameters):
        img = inputs['img'].value.copy()
        pt1 = parameters['pt1']
        pt2 = parameters['pt2']
        color = parameters['color']
        thickness = parameters['thickness']
        lineType = parameters['lineType']
        shift = parameters['shift']
        img = cv2.rectangle(img=img, pt1=pt1, pt2=pt2, color=color, thickness=thickness, lineType=lineType, shift=shift)
        outputs['img'] = Data(img)

# cv2.reduce
class OpenCVAuto2_Reduce(NormalElement):
    name = 'Reduce'
    comment = '''reduce(src, dim, rtype[, dst[, dtype]]) -> dst\n@brief Reduces a matrix to a vector.\n\nThe function #reduce reduces the matrix to a vector by treating the matrix rows/columns as a set of\n1D vectors and performing the specified operation on the vectors until a single row/column is\nobtained. For example, the function can be used to compute horizontal and vertical projections of a\nraster image. In case of #REDUCE_MAX and #REDUCE_MIN , the output image should have the same type as the source one.\nIn case of #REDUCE_SUM and #REDUCE_AVG , the output may have a larger element bit-depth to preserve accuracy.\nAnd multi-channel arrays are also supported in these two reduction modes.\n\nThe following code demonstrates its usage for a single channel matrix.\n@snippet snippets/core_reduce.cpp example\n\nAnd the following code demonstrates its usage for a two-channel matrix.\n@snippet snippets/core_reduce.cpp example2\n\n@param src input 2D matrix.\n@param dst output vector. Its size and type is defined by dim and dtype parameters.\n@param dim dimension index along which the matrix is reduced. 0 means that the matrix is reduced to\na single row. 1 means that the matrix is reduced to a single column.\n@param rtype reduction operation that could be one of #ReduceTypes\n@param dtype when negative, the output vector will have the same type as the input matrix,\notherwise, its type will be CV_MAKE_TYPE(CV_MAT_DEPTH(dtype), src.channels()).\n@sa repeat'''
    package = "Miscellaneous"

    def get_attributes(self):
        return [Input('src', 'src')], \
               [Output('dst', 'dst')], \
               [IntParameter('dim', 'dim'),
                IntParameter('rtype', 'rtype'),
                ComboboxParameter('dtype', name='dtype', values=[('NONE',-1),('CV_8U',0),('CV_8UC1',0),('CV_8S',1),('CV_8SC1',1),('CV_16U',2),('CV_16UC1',2),('CV_16S',3),('CV_16SC1',3),('CV_32S',4),('CV_32SC1',4),('CV_32F',5),('CV_32FC1',5),('CV_64F',6),('CV_64FC1',6),('CV_8UC2',8),('CV_8SC2',9),('CV_16UC2',10),('CV_16SC2',11),('CV_32SC2',12),('CV_32FC2',13),('CV_64FC2',14),('CV_8UC3',16),('CV_8SC3',17),('CV_16UC3',18),('CV_16SC3',19),('CV_32SC3',20),('CV_32FC3',21),('CV_64FC3',22),('CV_8UC4',24),('CV_8SC4',25),('CV_16UC4',26),('CV_16SC4',27),('CV_32SC4',28),('CV_32FC4',29),('CV_64FC4',30)])]

    def process_inputs(self, inputs, outputs, parameters):
        src = inputs['src'].value
        dim = parameters['dim']
        rtype = parameters['rtype']
        dtype = parameters['dtype']
        dst = cv2.reduce(src=src, dim=dim, rtype=rtype, dtype=dtype)
        outputs['dst'] = Data(dst)

# cv2.remap
class OpenCVAuto2_Remap(NormalElement):
    name = 'Remap'
    comment = '''remap(src, map1, map2, interpolation[, dst[, borderMode[, borderValue]]]) -> dst\n@brief Applies a generic geometrical transformation to an image.\n\nThe function remap transforms the source image using the specified map:\n\n\f[\texttt{dst} (x,y) =  \texttt{src} (map_x(x,y),map_y(x,y))\f]\n\nwhere values of pixels with non-integer coordinates are computed using one of available\ninterpolation methods. \f$map_x\f$ and \f$map_y\f$ can be encoded as separate floating-point maps\nin \f$map_1\f$ and \f$map_2\f$ respectively, or interleaved floating-point maps of \f$(x,y)\f$ in\n\f$map_1\f$, or fixed-point maps created by using convertMaps. The reason you might want to\nconvert from floating to fixed-point representations of a map is that they can yield much faster\n(\~2x) remapping operations. In the converted case, \f$map_1\f$ contains pairs (cvFloor(x),\ncvFloor(y)) and \f$map_2\f$ contains indices in a table of interpolation coefficients.\n\nThis function cannot operate in-place.\n\n@param src Source image.\n@param dst Destination image. It has the same size as map1 and the same type as src .\n@param map1 The first map of either (x,y) points or just x values having the type CV_16SC2 ,\nCV_32FC1, or CV_32FC2. See convertMaps for details on converting a floating point\nrepresentation to fixed-point for speed.\n@param map2 The second map of y values having the type CV_16UC1, CV_32FC1, or none (empty map\nif map1 is (x,y) points), respectively.\n@param interpolation Interpolation method (see #InterpolationFlags). The method #INTER_AREA is\nnot supported by this function.\n@param borderMode Pixel extrapolation method (see #BorderTypes). When\nborderMode=#BORDER_TRANSPARENT, it means that the pixels in the destination image that\ncorresponds to the "outliers" in the source image are not modified by the function.\n@param borderValue Value used in case of a constant border. By default, it is 0.\n@note\nDue to current implementation limitations the size of an input and output images should be less than 32767x32767.'''
    package = "Transforms"

    def get_attributes(self):
        return [Input('src', 'src'),
                Input('map1', 'map 1'),
                Input('map2', 'map 2')], \
               [Output('dst', 'dst')], \
               [ComboboxParameter('interpolation', name='interpolation', values=[('INTER_NEAREST',0),('INTER_LINEAR',1),('INTER_CUBIC',2),('INTER_AREA',3),('INTER_LANCZOS4',4),('INTER_BITS',5),('INTER_LINEAR_EXACT',5),('INTER_MAX',7),('INTER_BITS2',10),('INTER_TAB_SIZE',32),('INTER_TAB_SIZE2',1024)]),
                ComboboxParameter('borderMode', name='Border Mode', values=[('BORDER_CONSTANT',0),('BORDER_REPLICATE',1),('BORDER_REFLECT',2),('BORDER_WRAP',3),('BORDER_DEFAULT',4),('BORDER_REFLECT101',4),('BORDER_REFLECT_101',4),('BORDER_TRANSPARENT',5),('BORDER_ISOLATED',16)]),
                ScalarParameter('borderValue', 'Border Value')]

    def process_inputs(self, inputs, outputs, parameters):
        src = inputs['src'].value
        map1 = inputs['map1'].value
        map2 = inputs['map2'].value
        interpolation = parameters['interpolation']
        borderMode = parameters['borderMode']
        borderValue = parameters['borderValue']
        dst = cv2.remap(src=src, map1=map1, map2=map2, interpolation=interpolation, borderMode=borderMode, borderValue=borderValue)
        outputs['dst'] = Data(dst)

# cv2.repeat
class OpenCVAuto2_Repeat(NormalElement):
    name = 'Repeat'
    comment = '''repeat(src, ny, nx[, dst]) -> dst\n@brief Fills the output array with repeated copies of the input array.\n\nThe function cv::repeat duplicates the input array one or more times along each of the two axes:\n\f[\texttt{dst} _{ij}= \texttt{src} _{i\mod src.rows, \; j\mod src.cols }\f]\nThe second variant of the function is more convenient to use with @ref MatrixExpressions.\n@param src input array to replicate.\n@param ny Flag to specify how many times the `src` is repeated along the\nvertical axis.\n@param nx Flag to specify how many times the `src` is repeated along the\nhorizontal axis.\n@param dst output array of the same type as `src`.\n@sa cv::reduce'''
    package = "Miscellaneous"

    def get_attributes(self):
        return [Input('src', 'src')], \
               [Output('dst', 'dst')], \
               [IntParameter('ny', 'ny'),
                IntParameter('nx', 'nx')]

    def process_inputs(self, inputs, outputs, parameters):
        src = inputs['src'].value
        ny = parameters['ny']
        nx = parameters['nx']
        dst = cv2.repeat(src=src, ny=ny, nx=nx)
        outputs['dst'] = Data(dst)

# cv2.resize
class OpenCVAuto2_Resize(NormalElement):
    name = 'Resize'
    comment = '''resize(src, dsize[, dst[, fx[, fy[, interpolation]]]]) -> dst\n@brief Resizes an image.\n\nThe function resize resizes the image src down to or up to the specified size. Note that the\ninitial dst type or size are not taken into account. Instead, the size and type are derived from\nthe `src`,`dsize`,`fx`, and `fy`. If you want to resize src so that it fits the pre-created dst,\nyou may call the function as follows:\n@code\n// explicitly specify dsize=dst.size(); fx and fy will be computed from that.\nresize(src, dst, dst.size(), 0, 0, interpolation);\n@endcode\nIf you want to decimate the image by factor of 2 in each direction, you can call the function this\nway:\n@code\n// specify fx and fy and let the function compute the destination image size.\nresize(src, dst, Size(), 0.5, 0.5, interpolation);\n@endcode\nTo shrink an image, it will generally look best with #INTER_AREA interpolation, whereas to\nenlarge an image, it will generally look best with c#INTER_CUBIC (slow) or #INTER_LINEAR\n(faster but still looks OK).\n\n@param src input image.\n@param dst output image; it has the size dsize (when it is non-zero) or the size computed from\nsrc.size(), fx, and fy; the type of dst is the same as of src.\n@param dsize output image size; if it equals zero, it is computed as:\n\f[\texttt{dsize = Size(round(fx*src.cols), round(fy*src.rows))}\f]\nEither dsize or both fx and fy must be non-zero.\n@param fx scale factor along the horizontal axis; when it equals 0, it is computed as\n\f[\texttt{(double)dsize.width/src.cols}\f]\n@param fy scale factor along the vertical axis; when it equals 0, it is computed as\n\f[\texttt{(double)dsize.height/src.rows}\f]\n@param interpolation interpolation method, see #InterpolationFlags\n\n@sa  warpAffine, warpPerspective, remap'''
    package = "Transforms"

    def get_attributes(self):
        return [Input('src', 'src')], \
               [Output('dst', 'dst')], \
               [SizeParameter('dsize', 'dsize'),
                FloatParameter('fx', 'fx'),
                FloatParameter('fy', 'fy'),
                ComboboxParameter('interpolation', name='interpolation', values=[('INTER_NEAREST',0),('INTER_LINEAR',1),('INTER_CUBIC',2),('INTER_AREA',3),('INTER_LANCZOS4',4),('INTER_BITS',5),('INTER_LINEAR_EXACT',5),('INTER_MAX',7),('INTER_BITS2',10),('INTER_TAB_SIZE',32),('INTER_TAB_SIZE2',1024)])]

    def process_inputs(self, inputs, outputs, parameters):
        src = inputs['src'].value
        dsize = parameters['dsize']
        fx = parameters['fx']
        fy = parameters['fy']
        interpolation = parameters['interpolation']
        dst = cv2.resize(src=src, dsize=dsize, fx=fx, fy=fy, interpolation=interpolation)
        outputs['dst'] = Data(dst)

# cv2.rotate
class OpenCVAuto2_Rotate(NormalElement):
    name = 'Rotate'
    comment = '''rotate(src, rotateCode[, dst]) -> dst\n@brief Rotates a 2D array in multiples of 90 degrees.\nThe function cv::rotate rotates the array in one of three different ways:\n*   Rotate by 90 degrees clockwise (rotateCode = ROTATE_90_CLOCKWISE).\n*   Rotate by 180 degrees clockwise (rotateCode = ROTATE_180).\n*   Rotate by 270 degrees clockwise (rotateCode = ROTATE_90_COUNTERCLOCKWISE).\n@param src input array.\n@param dst output array of the same type as src.  The size is the same with ROTATE_180,\nand the rows and cols are switched for ROTATE_90_CLOCKWISE and ROTATE_90_COUNTERCLOCKWISE.\n@param rotateCode an enum to specify how to rotate the array; see the enum #RotateFlags\n@sa transpose , repeat , completeSymm, flip, RotateFlags'''
    package = "Transforms"

    def get_attributes(self):
        return [Input('src', 'src')], \
               [Output('dst', 'dst')], \
               [IntParameter('rotateCode', 'Rotate Code')]

    def process_inputs(self, inputs, outputs, parameters):
        src = inputs['src'].value
        rotateCode = parameters['rotateCode']
        dst = cv2.rotate(src=src, rotateCode=rotateCode)
        outputs['dst'] = Data(dst)

# cv2.scaleAdd
class OpenCVAuto2_ScaleAdd(NormalElement):
    name = 'Scale Add'
    comment = '''scaleAdd(src1, alpha, src2[, dst]) -> dst\n@brief Calculates the sum of a scaled array and another array.\n\nThe function scaleAdd is one of the classical primitive linear algebra operations, known as DAXPY\nor SAXPY in [BLAS](http://en.wikipedia.org/wiki/Basic_Linear_Algebra_Subprograms). It calculates\nthe sum of a scaled array and another array:\n\f[\texttt{dst} (I)= \texttt{scale} \cdot \texttt{src1} (I) +  \texttt{src2} (I)\f]\nThe function can also be emulated with a matrix expression, for example:\n@code{.cpp}\nMat A(3, 3, CV_64F);\n...\nA.row(0) = A.row(1)*2 + A.row(2);\n@endcode\n@param src1 first input array.\n@param alpha scale factor for the first array.\n@param src2 second input array of the same size and type as src1.\n@param dst output array of the same size and type as src1.\n@sa add, addWeighted, subtract, Mat::dot, Mat::convertTo'''
    package = "Operators"

    def get_attributes(self):
        return [Input('src1', 'src 1'),
                Input('src2', 'src 2')], \
               [Output('dst', 'dst')], \
               [FloatParameter('alpha', 'alpha')]

    def process_inputs(self, inputs, outputs, parameters):
        src1 = inputs['src1'].value
        src2 = inputs['src2'].value
        alpha = parameters['alpha']
        dst = cv2.scaleAdd(src1=src1, src2=src2, alpha=alpha)
        outputs['dst'] = Data(dst)

# cv2.seamlessClone
class OpenCVAuto2_SeamlessClone(NormalElement):
    name = 'Seamless Clone'
    comment = '''seamlessClone(src, dst, mask, p, flags[, blend]) -> blend\n@brief Image editing tasks concern either global changes (color/intensity corrections, filters,\ndeformations) or local changes concerned to a selection. Here we are interested in achieving local\nchanges, ones that are restricted to a region manually selected (ROI), in a seamless and effortless\nmanner. The extent of the changes ranges from slight distortions to complete replacement by novel\ncontent @cite PM03 .\n\n@param src Input 8-bit 3-channel image.\n@param dst Input 8-bit 3-channel image.\n@param mask Input 8-bit 1 or 3-channel image.\n@param p Point in dst image where object is placed.\n@param blend Output image with the same size and type as dst.\n@param flags Cloning method that could be one of the following:\n-   **NORMAL_CLONE** The power of the method is fully expressed when inserting objects with\ncomplex outlines into a new background\n-   **MIXED_CLONE** The classic method, color-based selection and alpha masking might be time\nconsuming and often leaves an undesirable halo. Seamless cloning, even averaged with the\noriginal image, is not effective. Mixed seamless cloning based on a loose selection proves\neffective.\n-   **MONOCHROME_TRANSFER** Monochrome transfer allows the user to easily replace certain features of\none object by alternative features.'''
    package = "Photo"

    def get_attributes(self):
        return [Input('src', 'src'),
                Input('dst', 'dst'),
                Input('mask', 'mask')], \
               [Output('blend', 'blend')], \
               [PointParameter('p', 'p'),
                ComboboxParameter('flags', name='flags', values=[('NORMAL_CLONE',1),('MIXED_CLONE',2),('MONOCHROME_TRANSFER',3)])]

    def process_inputs(self, inputs, outputs, parameters):
        src = inputs['src'].value
        dst = inputs['dst'].value
        mask = inputs['mask'].value
        p = parameters['p']
        flags = parameters['flags']
        blend = cv2.seamlessClone(src=src, dst=dst, mask=mask, p=p, flags=flags)
        outputs['blend'] = Data(blend)

# cv2.sepFilter2D
class OpenCVAuto2_SepFilter2D(NormalElement):
    name = 'Sep Filter 2D'
    comment = '''sepFilter2D(src, ddepth, kernelX, kernelY[, dst[, anchor[, delta[, borderType]]]]) -> dst\n@brief Applies a separable linear filter to an image.\n\nThe function applies a separable linear filter to the image. That is, first, every row of src is\nfiltered with the 1D kernel kernelX. Then, every column of the result is filtered with the 1D\nkernel kernelY. The final result shifted by delta is stored in dst .\n\n@param src Source image.\n@param dst Destination image of the same size and the same number of channels as src .\n@param ddepth Destination image depth, see @ref filter_depths "combinations"\n@param kernelX Coefficients for filtering each row.\n@param kernelY Coefficients for filtering each column.\n@param anchor Anchor position within the kernel. The default value \f$(-1,-1)\f$ means that the anchor\nis at the kernel center.\n@param delta Value added to the filtered results before storing them.\n@param borderType Pixel extrapolation method, see #BorderTypes\n@sa  filter2D, Sobel, GaussianBlur, boxFilter, blur'''
    package = "Filters"

    def get_attributes(self):
        return [Input('src', 'src'),
                Input('kernelX', 'Kernel X'),
                Input('kernelY', 'Kernel Y')], \
               [Output('dst', 'dst')], \
               [ComboboxParameter('ddepth', name='ddepth', values=[('NONE',-1),('CV_8U',0),('CV_8UC1',0),('CV_8S',1),('CV_8SC1',1),('CV_16U',2),('CV_16UC1',2),('CV_16S',3),('CV_16SC1',3),('CV_32S',4),('CV_32SC1',4),('CV_32F',5),('CV_32FC1',5),('CV_64F',6),('CV_64FC1',6),('CV_8UC2',8),('CV_8SC2',9),('CV_16UC2',10),('CV_16SC2',11),('CV_32SC2',12),('CV_32FC2',13),('CV_64FC2',14),('CV_8UC3',16),('CV_8SC3',17),('CV_16UC3',18),('CV_16SC3',19),('CV_32SC3',20),('CV_32FC3',21),('CV_64FC3',22),('CV_8UC4',24),('CV_8SC4',25),('CV_16UC4',26),('CV_16SC4',27),('CV_32SC4',28),('CV_32FC4',29),('CV_64FC4',30)]),
                PointParameter('anchor', 'anchor'),
                FloatParameter('delta', 'delta'),
                ComboboxParameter('borderType', name='Border Type', values=[('BORDER_CONSTANT',0),('BORDER_REPLICATE',1),('BORDER_REFLECT',2),('BORDER_WRAP',3),('BORDER_DEFAULT',4),('BORDER_REFLECT101',4),('BORDER_REFLECT_101',4),('BORDER_TRANSPARENT',5),('BORDER_ISOLATED',16)])]

    def process_inputs(self, inputs, outputs, parameters):
        src = inputs['src'].value
        kernelX = inputs['kernelX'].value
        kernelY = inputs['kernelY'].value
        ddepth = parameters['ddepth']
        anchor = parameters['anchor']
        delta = parameters['delta']
        borderType = parameters['borderType']
        dst = cv2.sepFilter2D(src=src, kernelX=kernelX, kernelY=kernelY, ddepth=ddepth, anchor=anchor, delta=delta, borderType=borderType)
        outputs['dst'] = Data(dst)

# cv2.setIdentity
class OpenCVAuto2_SetIdentity(NormalElement):
    name = 'Set Identity'
    comment = '''setIdentity(mtx[, s]) -> mtx\n@brief Initializes a scaled identity matrix.\n\nThe function cv::setIdentity initializes a scaled identity matrix:\n\f[\texttt{mtx} (i,j)= \fork{\texttt{value}}{ if \(i=j\)}{0}{otherwise}\f]\n\nThe function can also be emulated using the matrix initializers and the\nmatrix expressions:\n@code\nMat A = Mat::eye(4, 3, CV_32F)*5;\n// A will be set to [[5, 0, 0], [0, 5, 0], [0, 0, 5], [0, 0, 0]]\n@endcode\n@param mtx matrix to initialize (not necessarily square).\n@param s value to assign to diagonal elements.\n@sa Mat::zeros, Mat::ones, Mat::setTo, Mat::operator='''
    package = "Miscellaneous"

    def get_attributes(self):
        return [Input('mtx', 'mtx')], \
               [Output('mtx', 'mtx')], \
               [ScalarParameter('s', 's')]

    def process_inputs(self, inputs, outputs, parameters):
        mtx = inputs['mtx'].value.copy()
        s = parameters['s']
        mtx = cv2.setIdentity(mtx=mtx, s=s)
        outputs['mtx'] = Data(mtx)

# cv2.solveCubic
class OpenCVAuto2_SolveCubic(NormalElement):
    name = 'Solve Cubic'
    comment = '''solveCubic(coeffs[, roots]) -> retval, roots\n@brief Finds the real roots of a cubic equation.\n\nThe function solveCubic finds the real roots of a cubic equation:\n-   if coeffs is a 4-element vector:\n\f[\texttt{coeffs} [0] x^3 +  \texttt{coeffs} [1] x^2 +  \texttt{coeffs} [2] x +  \texttt{coeffs} [3] = 0\f]\n-   if coeffs is a 3-element vector:\n\f[x^3 +  \texttt{coeffs} [0] x^2 +  \texttt{coeffs} [1] x +  \texttt{coeffs} [2] = 0\f]\n\nThe roots are stored in the roots array.\n@param coeffs equation coefficients, an array of 3 or 4 elements.\n@param roots output array of real roots that has 1 or 3 elements.\n@return number of real roots. It can be 0, 1 or 2.'''
    package = "Optimization"

    def get_attributes(self):
        return [Input('coeffs', 'coeffs')], \
               [Output('roots', 'roots')], \
               []

    def process_inputs(self, inputs, outputs, parameters):
        coeffs = inputs['coeffs'].value
        retval, roots = cv2.solveCubic(coeffs=coeffs)
        outputs['roots'] = Data(roots)

# cv2.solveP3P
class OpenCVAuto2_SolveP3P(NormalElement):
    name = 'Solve P3P'
    comment = '''solveP3P(objectPoints, imagePoints, cameraMatrix, distCoeffs, flags[, rvecs[, tvecs]]) -> retval, rvecs, tvecs\n@brief Finds an object pose from 3 3D-2D point correspondences.\n\n@param objectPoints Array of object points in the object coordinate space, 3x3 1-channel or\n1x3/3x1 3-channel. vector\<Point3f\> can be also passed here.\n@param imagePoints Array of corresponding image points, 3x2 1-channel or 1x3/3x1 2-channel.\nvector\<Point2f\> can be also passed here.\n@param cameraMatrix Input camera matrix \f$A = \vecthreethree{fx}{0}{cx}{0}{fy}{cy}{0}{0}{1}\f$ .\n@param distCoeffs Input vector of distortion coefficients\n\f$(k_1, k_2, p_1, p_2[, k_3[, k_4, k_5, k_6 [, s_1, s_2, s_3, s_4[, \tau_x, \tau_y]]]])\f$ of\n4, 5, 8, 12 or 14 elements. If the vector is NULL/empty, the zero distortion coefficients are\nassumed.\n@param rvecs Output rotation vectors (see Rodrigues ) that, together with tvecs , brings points from\nthe model coordinate system to the camera coordinate system. A P3P problem has up to 4 solutions.\n@param tvecs Output translation vectors.\n@param flags Method for solving a P3P problem:\n-   **SOLVEPNP_P3P** Method is based on the paper of X.S. Gao, X.-R. Hou, J. Tang, H.-F. Chang\n"Complete Solution Classification for the Perspective-Three-Point Problem" (@cite gao2003complete).\n-   **SOLVEPNP_AP3P** Method is based on the paper of Tong Ke and Stergios I. Roumeliotis.\n"An Efficient Algebraic Solution to the Perspective-Three-Point Problem" (@cite Ke17).\n\nThe function estimates the object pose given 3 object points, their corresponding image\nprojections, as well as the camera matrix and the distortion coefficients.'''
    package = "3D calibration & reconstruction"

    def get_attributes(self):
        return [Input('objectPoints', 'Object Points'),
                Input('imagePoints', 'Image Points'),
                Input('cameraMatrix', 'Camera Matrix'),
                Input('distCoeffs', 'Dist Coeffs')], \
               [Output('rvecs', 'rvecs'),
                Output('tvecs', 'tvecs')], \
               [ComboboxParameter('flags', name='flags', values=[('SOLVEPNP_ITERATIVE',0),('SOLVEPNP_EPNP',1),('SOLVEPNP_P3P',2),('SOLVEPNP_DLS',3),('SOLVEPNP_UPNP',4),('SOLVEPNP_AP3P',5),('SOLVEPNP_MAX_COUNT',6)])]

    def process_inputs(self, inputs, outputs, parameters):
        objectPoints = inputs['objectPoints'].value
        imagePoints = inputs['imagePoints'].value
        cameraMatrix = inputs['cameraMatrix'].value
        distCoeffs = inputs['distCoeffs'].value
        flags = parameters['flags']
        retval, rvecs, tvecs = cv2.solveP3P(objectPoints=objectPoints, imagePoints=imagePoints, cameraMatrix=cameraMatrix, distCoeffs=distCoeffs, flags=flags)
        outputs['rvecs'] = Data(rvecs)
        outputs['tvecs'] = Data(tvecs)

# cv2.solvePoly
class OpenCVAuto2_SolvePoly(NormalElement):
    name = 'Solve Poly'
    comment = '''solvePoly(coeffs[, roots[, maxIters]]) -> retval, roots\n@brief Finds the real or complex roots of a polynomial equation.\n\nThe function cv::solvePoly finds real and complex roots of a polynomial equation:\n\f[\texttt{coeffs} [n] x^{n} +  \texttt{coeffs} [n-1] x^{n-1} + ... +  \texttt{coeffs} [1] x +  \texttt{coeffs} [0] = 0\f]\n@param coeffs array of polynomial coefficients.\n@param roots output (complex) array of roots.\n@param maxIters maximum number of iterations the algorithm does.'''
    package = "3D calibration & reconstruction"

    def get_attributes(self):
        return [Input('coeffs', 'coeffs')], \
               [Output('roots', 'roots')], \
               [IntParameter('maxIters', 'Max Iters', min_=0)]

    def process_inputs(self, inputs, outputs, parameters):
        coeffs = inputs['coeffs'].value
        maxIters = parameters['maxIters']
        retval, roots = cv2.solvePoly(coeffs=coeffs, maxIters=maxIters)
        outputs['roots'] = Data(roots)

# cv2.spatialGradient
class OpenCVAuto2_SpatialGradient(NormalElement):
    name = 'Spatial Gradient'
    comment = '''spatialGradient(src[, dx[, dy[, ksize[, borderType]]]]) -> dx, dy\n@brief Calculates the first order image derivative in both x and y using a Sobel operator\n\nEquivalent to calling:\n\n@code\nSobel( src, dx, CV_16SC1, 1, 0, 3 );\nSobel( src, dy, CV_16SC1, 0, 1, 3 );\n@endcode\n\n@param src input image.\n@param dx output image with first-order derivative in x.\n@param dy output image with first-order derivative in y.\n@param ksize size of Sobel kernel. It must be 3.\n@param borderType pixel extrapolation method, see #BorderTypes\n\n@sa Sobel'''
    package = "Math"

    def get_attributes(self):
        return [Input('src', 'src')], \
               [Output('dx', 'dx'),
                Output('dy', 'dy')], \
               [SizeParameter('ksize', 'ksize'),
                ComboboxParameter('borderType', name='Border Type', values=[('BORDER_CONSTANT',0),('BORDER_REPLICATE',1),('BORDER_REFLECT',2),('BORDER_WRAP',3),('BORDER_DEFAULT',4),('BORDER_REFLECT101',4),('BORDER_REFLECT_101',4),('BORDER_TRANSPARENT',5),('BORDER_ISOLATED',16)])]

    def process_inputs(self, inputs, outputs, parameters):
        src = inputs['src'].value
        ksize = parameters['ksize']
        borderType = parameters['borderType']
        dx, dy = cv2.spatialGradient(src=src, ksize=ksize, borderType=borderType)
        outputs['dx'] = Data(dx)
        outputs['dy'] = Data(dy)

# cv2.split
class OpenCVAuto2_Split(NormalElement):
    name = 'Split'
    comment = '''split(m[, mv]) -> mv\n@overload\n@param m input multi-channel array.\n@param mv output vector of arrays; the arrays themselves are reallocated, if needed.'''
    package = "Channels"

    def get_attributes(self):
        return [Input('m', 'm')], \
               [Output('mv', 'mv')], \
               []

    def process_inputs(self, inputs, outputs, parameters):
        m = inputs['m'].value
        mv = cv2.split(m=m)
        outputs['mv'] = Data(mv)

# cv2.sqrt
class OpenCVAuto2_Sqrt(NormalElement):
    name = 'Sqrt'
    comment = '''sqrt(src[, dst]) -> dst\n@brief Calculates a square root of array elements.\n\nThe function cv::sqrt calculates a square root of each input array element.\nIn case of multi-channel arrays, each channel is processed\nindependently. The accuracy is approximately the same as of the built-in\nstd::sqrt .\n@param src input floating-point array.\n@param dst output array of the same size and type as src.'''
    package = "Math"

    def get_attributes(self):
        return [Input('src', 'src')], \
               [Output('dst', 'dst')], \
               []

    def process_inputs(self, inputs, outputs, parameters):
        src = inputs['src'].value
        dst = cv2.sqrt(src=src)
        outputs['dst'] = Data(dst)

# cv2.stereoRectify
class OpenCVAuto2_StereoRectify(NormalElement):
    name = 'Stereo Rectify'
    comment = '''stereoRectify(cameraMatrix1, distCoeffs1, cameraMatrix2, distCoeffs2, imageSize, R, T[, R1[, R2[, P1[, P2[, Q[, flags[, alpha[, newImageSize]]]]]]]]) -> R1, R2, P1, P2, Q, validPixROI1, validPixROI2\n@brief Computes rectification transforms for each head of a calibrated stereo camera.\n\n@param cameraMatrix1 First camera matrix.\n@param distCoeffs1 First camera distortion parameters.\n@param cameraMatrix2 Second camera matrix.\n@param distCoeffs2 Second camera distortion parameters.\n@param imageSize Size of the image used for stereo calibration.\n@param R Rotation matrix between the coordinate systems of the first and the second cameras.\n@param T Translation vector between coordinate systems of the cameras.\n@param R1 Output 3x3 rectification transform (rotation matrix) for the first camera.\n@param R2 Output 3x3 rectification transform (rotation matrix) for the second camera.\n@param P1 Output 3x4 projection matrix in the new (rectified) coordinate systems for the first\ncamera.\n@param P2 Output 3x4 projection matrix in the new (rectified) coordinate systems for the second\ncamera.\n@param Q Output \f$4 \times 4\f$ disparity-to-depth mapping matrix (see reprojectImageTo3D ).\n@param flags Operation flags that may be zero or CALIB_ZERO_DISPARITY . If the flag is set,\nthe function makes the principal points of each camera have the same pixel coordinates in the\nrectified views. And if the flag is not set, the function may still shift the images in the\nhorizontal or vertical direction (depending on the orientation of epipolar lines) to maximize the\nuseful image area.\n@param alpha Free scaling parameter. If it is -1 or absent, the function performs the default\nscaling. Otherwise, the parameter should be between 0 and 1. alpha=0 means that the rectified\nimages are zoomed and shifted so that only valid pixels are visible (no black areas after\nrectification). alpha=1 means that the rectified image is decimated and shifted so that all the\npixels from the original images from the cameras are retained in the rectified images (no source\nimage pixels are lost). Obviously, any intermediate value yields an intermediate result between\nthose two extreme cases.\n@param newImageSize New image resolution after rectification. The same size should be passed to\ninitUndistortRectifyMap (see the stereo_calib.cpp sample in OpenCV samples directory). When (0,0)\nis passed (default), it is set to the original imageSize . Setting it to larger value can help you\npreserve details in the original image, especially when there is a big radial distortion.\n@param validPixROI1 Optional output rectangles inside the rectified images where all the pixels\nare valid. If alpha=0 , the ROIs cover the whole images. Otherwise, they are likely to be smaller\n(see the picture below).\n@param validPixROI2 Optional output rectangles inside the rectified images where all the pixels\nare valid. If alpha=0 , the ROIs cover the whole images. Otherwise, they are likely to be smaller\n(see the picture below).\n\nThe function computes the rotation matrices for each camera that (virtually) make both camera image\nplanes the same plane. Consequently, this makes all the epipolar lines parallel and thus simplifies\nthe dense stereo correspondence problem. The function takes the matrices computed by stereoCalibrate\nas input. As output, it provides two rotation matrices and also two projection matrices in the new\ncoordinates. The function distinguishes the following two cases:\n\n-   **Horizontal stereo**: the first and the second camera views are shifted relative to each other\nmainly along the x axis (with possible small vertical shift). In the rectified images, the\ncorresponding epipolar lines in the left and right cameras are horizontal and have the same\ny-coordinate. P1 and P2 look like:\n\n\f[\texttt{P1} = \begin{bmatrix} f & 0 & cx_1 & 0 \\ 0 & f & cy & 0 \\ 0 & 0 & 1 & 0 \end{bmatrix}\f]\n\n\f[\texttt{P2} = \begin{bmatrix} f & 0 & cx_2 & T_x*f \\ 0 & f & cy & 0 \\ 0 & 0 & 1 & 0 \end{bmatrix} ,\f]\n\nwhere \f$T_x\f$ is a horizontal shift between the cameras and \f$cx_1=cx_2\f$ if\nCALIB_ZERO_DISPARITY is set.\n\n-   **Vertical stereo**: the first and the second camera views are shifted relative to each other\nmainly in vertical direction (and probably a bit in the horizontal direction too). The epipolar\nlines in the rectified images are vertical and have the same x-coordinate. P1 and P2 look like:\n\n\f[\texttt{P1} = \begin{bmatrix} f & 0 & cx & 0 \\ 0 & f & cy_1 & 0 \\ 0 & 0 & 1 & 0 \end{bmatrix}\f]\n\n\f[\texttt{P2} = \begin{bmatrix} f & 0 & cx & 0 \\ 0 & f & cy_2 & T_y*f \\ 0 & 0 & 1 & 0 \end{bmatrix} ,\f]\n\nwhere \f$T_y\f$ is a vertical shift between the cameras and \f$cy_1=cy_2\f$ if CALIB_ZERO_DISPARITY is\nset.\n\nAs you can see, the first three columns of P1 and P2 will effectively be the new "rectified" camera\nmatrices. The matrices, together with R1 and R2 , can then be passed to initUndistortRectifyMap to\ninitialize the rectification map for each camera.\n\nSee below the screenshot from the stereo_calib.cpp sample. Some red horizontal lines pass through\nthe corresponding image regions. This means that the images are well rectified, which is what most\nstereo correspondence algorithms rely on. The green rectangles are roi1 and roi2 . You see that\ntheir interiors are all valid pixels.\n\n![image](pics/stereo_undistort.jpg)'''
    package = "3D calibration & reconstruction"

    def get_attributes(self):
        return [Input('cameraMatrix1', 'Camera Matrix 1'),
                Input('distCoeffs1', 'Dist Coeffs 1'),
                Input('cameraMatrix2', 'Camera Matrix 2'),
                Input('distCoeffs2', 'Dist Coeffs 2'),
                Input('imageSize', 'Image Size'),
                Input('R', 'R'),
                Input('T', 'T'),
                Input('newImageSize', 'New Image Size', optional=True)], \
               [Output('R1', 'R1'),
                Output('R2', 'R2'),
                Output('P1', 'P1'),
                Output('P2', 'P2'),
                Output('Q', 'Q'),
                Output('validPixROI1', 'Valid Pix ROI1'),
                Output('validPixROI2', 'Valid Pix ROI2')], \
               [ComboboxParameter('flags', name='flags', values=[('CALIB_CB_ADAPTIVE_THRESH',1),('CALIB_CB_SYMMETRIC_GRID',1),('CALIB_USE_INTRINSIC_GUESS',1),('CALIB_CB_ASYMMETRIC_GRID',2),('CALIB_CB_NORMALIZE_IMAGE',2),('CALIB_FIX_ASPECT_RATIO',2),('CALIB_CB_CLUSTERING',4),('CALIB_CB_FILTER_QUADS',4),('CALIB_FIX_PRINCIPAL_POINT',4),('CALIB_CB_FAST_CHECK',8),('CALIB_ZERO_TANGENT_DIST',8),('CALIB_FIX_FOCAL_LENGTH',16),('CALIB_FIX_K1',32),('CALIB_FIX_K2',64),('CALIB_FIX_K3',128),('CALIB_FIX_INTRINSIC',256),('CALIB_SAME_FOCAL_LENGTH',512),('CALIB_ZERO_DISPARITY',1024),('CALIB_FIX_K4',2048),('CALIB_FIX_K5',4096),('CALIB_FIX_K6',8192),('CALIB_RATIONAL_MODEL',16384),('CALIB_THIN_PRISM_MODEL',32768),('CALIB_FIX_S1_S2_S3_S4',65536),('CALIB_USE_LU',131072),('CALIB_TILTED_MODEL',262144),('CALIB_FIX_TAUX_TAUY',524288),('CALIB_USE_QR',1048576),('CALIB_FIX_TANGENT_DIST',2097152),('CALIB_USE_EXTRINSIC_GUESS',4194304)]),
                FloatParameter('alpha', 'alpha')]

    def process_inputs(self, inputs, outputs, parameters):
        cameraMatrix1 = inputs['cameraMatrix1'].value
        distCoeffs1 = inputs['distCoeffs1'].value
        cameraMatrix2 = inputs['cameraMatrix2'].value
        distCoeffs2 = inputs['distCoeffs2'].value
        imageSize = inputs['imageSize'].value
        R = inputs['R'].value
        T = inputs['T'].value
        newImageSize = inputs['newImageSize'].value
        flags = parameters['flags']
        alpha = parameters['alpha']
        R1, R2, P1, P2, Q, validPixROI1, validPixROI2 = cv2.stereoRectify(cameraMatrix1=cameraMatrix1, distCoeffs1=distCoeffs1, cameraMatrix2=cameraMatrix2, distCoeffs2=distCoeffs2, imageSize=imageSize, R=R, T=T, newImageSize=newImageSize, flags=flags, alpha=alpha)
        outputs['R1'] = Data(R1)
        outputs['R2'] = Data(R2)
        outputs['P1'] = Data(P1)
        outputs['P2'] = Data(P2)
        outputs['Q'] = Data(Q)
        outputs['validPixROI1'] = Data(validPixROI1)
        outputs['validPixROI2'] = Data(validPixROI2)

# cv2.stereoRectifyUncalibrated
class OpenCVAuto2_StereoRectifyUncalibrated(NormalElement):
    name = 'Stereo Rectify Uncalibrated'
    comment = '''stereoRectifyUncalibrated(points1, points2, F, imgSize[, H1[, H2[, threshold]]]) -> retval, H1, H2\n@brief Computes a rectification transform for an uncalibrated stereo camera.\n\n@param points1 Array of feature points in the first image.\n@param points2 The corresponding points in the second image. The same formats as in\nfindFundamentalMat are supported.\n@param F Input fundamental matrix. It can be computed from the same set of point pairs using\nfindFundamentalMat .\n@param imgSize Size of the image.\n@param H1 Output rectification homography matrix for the first image.\n@param H2 Output rectification homography matrix for the second image.\n@param threshold Optional threshold used to filter out the outliers. If the parameter is greater\nthan zero, all the point pairs that do not comply with the epipolar geometry (that is, the points\nfor which \f$|\texttt{points2[i]}^T*\texttt{F}*\texttt{points1[i]}|>\texttt{threshold}\f$ ) are\nrejected prior to computing the homographies. Otherwise, all the points are considered inliers.\n\nThe function computes the rectification transformations without knowing intrinsic parameters of the\ncameras and their relative position in the space, which explains the suffix "uncalibrated". Another\nrelated difference from stereoRectify is that the function outputs not the rectification\ntransformations in the object (3D) space, but the planar perspective transformations encoded by the\nhomography matrices H1 and H2 . The function implements the algorithm @cite Hartley99 .\n\n@note\nWhile the algorithm does not need to know the intrinsic parameters of the cameras, it heavily\ndepends on the epipolar geometry. Therefore, if the camera lenses have a significant distortion,\nit would be better to correct it before computing the fundamental matrix and calling this\nfunction. For example, distortion coefficients can be estimated for each head of stereo camera\nseparately by using calibrateCamera . Then, the images can be corrected using undistort , or\njust the point coordinates can be corrected with undistortPoints .'''
    package = "3D calibration & reconstruction"

    def get_attributes(self):
        return [Input('points1', 'points 1'),
                Input('points2', 'points 2'),
                Input('F', 'F')], \
               [Output('H1', 'H1'),
                Output('H2', 'H2')], \
               [SizeParameter('imgSize', 'Img Size'),
                FloatParameter('threshold', 'threshold')]

    def process_inputs(self, inputs, outputs, parameters):
        points1 = inputs['points1'].value
        points2 = inputs['points2'].value
        F = inputs['F'].value
        imgSize = parameters['imgSize']
        threshold = parameters['threshold']
        retval, H1, H2 = cv2.stereoRectifyUncalibrated(points1=points1, points2=points2, F=F, imgSize=imgSize, threshold=threshold)
        outputs['H1'] = Data(H1)
        outputs['H2'] = Data(H2)

# cv2.stylization
class OpenCVAuto2_Stylization(NormalElement):
    name = 'Stylization'
    comment = '''stylization(src[, dst[, sigma_s[, sigma_r]]]) -> dst\n@brief Stylization aims to produce digital imagery with a wide variety of effects not focused on\nphotorealism. Edge-aware filters are ideal for stylization, as they can abstract regions of low\ncontrast while preserving, or enhancing, high-contrast features.\n\n@param src Input 8-bit 3-channel image.\n@param dst Output image with the same size and type as src.\n@param sigma_s Range between 0 to 200.\n@param sigma_r Range between 0 to 1.'''
    package = "Photo"

    def get_attributes(self):
        return [Input('src', 'src')], \
               [Output('dst', 'dst')], \
               [FloatParameter('sigma_s', 'sigma s'),
                FloatParameter('sigma_r', 'sigma r')]

    def process_inputs(self, inputs, outputs, parameters):
        src = inputs['src'].value
        sigma_s = parameters['sigma_s']
        sigma_r = parameters['sigma_r']
        dst = cv2.stylization(src=src, sigma_s=sigma_s, sigma_r=sigma_r)
        outputs['dst'] = Data(dst)

# cv2.subtract
class OpenCVAuto2_Subtract(NormalElement):
    name = 'Subtract'
    comment = '''subtract(src1, src2[, dst[, mask[, dtype]]]) -> dst\n@brief Calculates the per-element difference between two arrays or array and a scalar.\n\nThe function subtract calculates:\n- Difference between two arrays, when both input arrays have the same size and the same number of\nchannels:\n\f[\texttt{dst}(I) =  \texttt{saturate} ( \texttt{src1}(I) -  \texttt{src2}(I)) \quad \texttt{if mask}(I) \ne0\f]\n- Difference between an array and a scalar, when src2 is constructed from Scalar or has the same\nnumber of elements as `src1.channels()`:\n\f[\texttt{dst}(I) =  \texttt{saturate} ( \texttt{src1}(I) -  \texttt{src2} ) \quad \texttt{if mask}(I) \ne0\f]\n- Difference between a scalar and an array, when src1 is constructed from Scalar or has the same\nnumber of elements as `src2.channels()`:\n\f[\texttt{dst}(I) =  \texttt{saturate} ( \texttt{src1} -  \texttt{src2}(I) ) \quad \texttt{if mask}(I) \ne0\f]\n- The reverse difference between a scalar and an array in the case of `SubRS`:\n\f[\texttt{dst}(I) =  \texttt{saturate} ( \texttt{src2} -  \texttt{src1}(I) ) \quad \texttt{if mask}(I) \ne0\f]\nwhere I is a multi-dimensional index of array elements. In case of multi-channel arrays, each\nchannel is processed independently.\n\nThe first function in the list above can be replaced with matrix expressions:\n@code{.cpp}\ndst = src1 - src2;\ndst -= src1; // equivalent to subtract(dst, src1, dst);\n@endcode\nThe input arrays and the output array can all have the same or different depths. For example, you\ncan subtract to 8-bit unsigned arrays and store the difference in a 16-bit signed array. Depth of\nthe output array is determined by dtype parameter. In the second and third cases above, as well as\nin the first case, when src1.depth() == src2.depth(), dtype can be set to the default -1. In this\ncase the output array will have the same depth as the input array, be it src1, src2 or both.\n@note Saturation is not applied when the output array has the depth CV_32S. You may even get\nresult of an incorrect sign in the case of overflow.\n@param src1 first input array or a scalar.\n@param src2 second input array or a scalar.\n@param dst output array of the same size and the same number of channels as the input array.\n@param mask optional operation mask; this is an 8-bit single channel array that specifies elements\nof the output array to be changed.\n@param dtype optional depth of the output array\n@sa  add, addWeighted, scaleAdd, Mat::convertTo'''
    package = "Operators"

    def get_attributes(self):
        return [Input('src1', 'src 1'),
                Input('src2', 'src 2'),
                Input('mask', 'mask', optional=True)], \
               [Output('dst', 'dst')], \
               [ComboboxParameter('dtype', name='dtype', values=[('NONE',-1),('CV_8U',0),('CV_8UC1',0),('CV_8S',1),('CV_8SC1',1),('CV_16U',2),('CV_16UC1',2),('CV_16S',3),('CV_16SC1',3),('CV_32S',4),('CV_32SC1',4),('CV_32F',5),('CV_32FC1',5),('CV_64F',6),('CV_64FC1',6),('CV_8UC2',8),('CV_8SC2',9),('CV_16UC2',10),('CV_16SC2',11),('CV_32SC2',12),('CV_32FC2',13),('CV_64FC2',14),('CV_8UC3',16),('CV_8SC3',17),('CV_16UC3',18),('CV_16SC3',19),('CV_32SC3',20),('CV_32FC3',21),('CV_64FC3',22),('CV_8UC4',24),('CV_8SC4',25),('CV_16UC4',26),('CV_16SC4',27),('CV_32SC4',28),('CV_32FC4',29),('CV_64FC4',30)])]

    def process_inputs(self, inputs, outputs, parameters):
        src1 = inputs['src1'].value
        src2 = inputs['src2'].value
        mask = inputs['mask'].value
        dtype = parameters['dtype']
        dst = cv2.subtract(src1=src1, src2=src2, mask=mask, dtype=dtype)
        outputs['dst'] = Data(dst)

# cv2.textureFlattening
class OpenCVAuto2_TextureFlattening(NormalElement):
    name = 'Texture Flattening'
    comment = '''textureFlattening(src, mask[, dst[, low_threshold[, high_threshold[, kernel_size]]]]) -> dst\n@brief By retaining only the gradients at edge locations, before integrating with the Poisson solver, one\nwashes out the texture of the selected region, giving its contents a flat aspect. Here Canny Edge\nDetector is used.\n\n@param src Input 8-bit 3-channel image.\n@param mask Input 8-bit 1 or 3-channel image.\n@param dst Output image with the same size and type as src.\n@param low_threshold Range from 0 to 100.\n@param high_threshold Value \> 100.\n@param kernel_size The size of the Sobel kernel to be used.\n\n**NOTE:**\n\nThe algorithm assumes that the color of the source image is close to that of the destination. This\nassumption means that when the colors don't match, the source image color gets tinted toward the\ncolor of the destination image.'''
    package = "Filters"

    def get_attributes(self):
        return [Input('src', 'src'),
                Input('mask', 'mask')], \
               [Output('dst', 'dst')], \
               [FloatParameter('low_threshold', 'low threshold'),
                FloatParameter('high_threshold', 'high threshold'),
                SizeParameter('kernel_size', 'kernel size')]

    def process_inputs(self, inputs, outputs, parameters):
        src = inputs['src'].value
        mask = inputs['mask'].value
        low_threshold = parameters['low_threshold']
        high_threshold = parameters['high_threshold']
        kernel_size = parameters['kernel_size']
        dst = cv2.textureFlattening(src=src, mask=mask, low_threshold=low_threshold, high_threshold=high_threshold, kernel_size=kernel_size)
        outputs['dst'] = Data(dst)

# cv2.threshold
class OpenCVAuto2_Threshold(NormalElement):
    name = 'Threshold'
    comment = '''threshold(src, thresh, maxval, type[, dst]) -> retval, dst\n@brief Applies a fixed-level threshold to each array element.\n\nThe function applies fixed-level thresholding to a multiple-channel array. The function is typically\nused to get a bi-level (binary) image out of a grayscale image ( #compare could be also used for\nthis purpose) or for removing a noise, that is, filtering out pixels with too small or too large\nvalues. There are several types of thresholding supported by the function. They are determined by\ntype parameter.\n\nAlso, the special values #THRESH_OTSU or #THRESH_TRIANGLE may be combined with one of the\nabove values. In these cases, the function determines the optimal threshold value using the Otsu's\nor Triangle algorithm and uses it instead of the specified thresh.\n\n@note Currently, the Otsu's and Triangle methods are implemented only for 8-bit single-channel images.\n\n@param src input array (multiple-channel, 8-bit or 32-bit floating point).\n@param dst output array of the same size  and type and the same number of channels as src.\n@param thresh threshold value.\n@param maxval maximum value to use with the #THRESH_BINARY and #THRESH_BINARY_INV thresholding\ntypes.\n@param type thresholding type (see #ThresholdTypes).\n@return the computed threshold value if Otsu's or Triangle methods used.\n\n@sa  adaptiveThreshold, findContours, compare, min, max'''
    package = "Threshold"

    def get_attributes(self):
        return [Input('src', 'src')], \
               [Output('dst', 'dst')], \
               [FloatParameter('thresh', 'thresh'),
                FloatParameter('maxval', 'maxval'),
                ComboboxParameter('type', name='type', values=[('THRESH_BINARY',0),('THRESH_BINARY_INV',1),('THRESH_TRUNC',2),('THRESH_TOZERO',3),('THRESH_TOZERO_INV',4),('THRESH_MASK',7),('THRESH_OTSU',8),('THRESH_TRIANGLE',16)])]

    def process_inputs(self, inputs, outputs, parameters):
        src = inputs['src'].value
        thresh = parameters['thresh']
        maxval = parameters['maxval']
        type = parameters['type']
        retval, dst = cv2.threshold(src=src, thresh=thresh, maxval=maxval, type=type)
        outputs['dst'] = Data(dst)

# cv2.transform
class OpenCVAuto2_Transform(NormalElement):
    name = 'Transform'
    comment = '''transform(src, m[, dst]) -> dst\n@brief Performs the matrix transformation of every array element.\n\nThe function cv::transform performs the matrix transformation of every\nelement of the array src and stores the results in dst :\n\f[\texttt{dst} (I) =  \texttt{m} \cdot \texttt{src} (I)\f]\n(when m.cols=src.channels() ), or\n\f[\texttt{dst} (I) =  \texttt{m} \cdot [ \texttt{src} (I); 1]\f]\n(when m.cols=src.channels()+1 )\n\nEvery element of the N -channel array src is interpreted as N -element\nvector that is transformed using the M x N or M x (N+1) matrix m to\nM-element vector - the corresponding element of the output array dst .\n\nThe function may be used for geometrical transformation of\nN -dimensional points, arbitrary linear color space transformation (such\nas various kinds of RGB to YUV transforms), shuffling the image\nchannels, and so forth.\n@param src input array that must have as many channels (1 to 4) as\nm.cols or m.cols-1.\n@param dst output array of the same size and depth as src; it has as\nmany channels as m.rows.\n@param m transformation 2x2 or 2x3 floating-point matrix.\n@sa perspectiveTransform, getAffineTransform, estimateAffine2D, warpAffine, warpPerspective'''
    package = "Transforms"

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

# cv2.transpose
class OpenCVAuto2_Transpose(NormalElement):
    name = 'Transpose'
    comment = '''transpose(src[, dst]) -> dst\n@brief Transposes a matrix.\n\nThe function cv::transpose transposes the matrix src :\n\f[\texttt{dst} (i,j) =  \texttt{src} (j,i)\f]\n@note No complex conjugation is done in case of a complex matrix. It\nshould be done separately if needed.\n@param src input array.\n@param dst output array of the same type as src.'''
    package = "Transforms"

    def get_attributes(self):
        return [Input('src', 'src')], \
               [Output('dst', 'dst')], \
               []

    def process_inputs(self, inputs, outputs, parameters):
        src = inputs['src'].value
        dst = cv2.transpose(src=src)
        outputs['dst'] = Data(dst)

# cv2.triangulatePoints
class OpenCVAuto2_TriangulatePoints(NormalElement):
    name = 'Triangulate Points'
    comment = '''triangulatePoints(projMatr1, projMatr2, projPoints1, projPoints2[, points4D]) -> points4D\n@brief Reconstructs points by triangulation.\n\n@param projMatr1 3x4 projection matrix of the first camera.\n@param projMatr2 3x4 projection matrix of the second camera.\n@param projPoints1 2xN array of feature points in the first image. In case of c++ version it can\nbe also a vector of feature points or two-channel matrix of size 1xN or Nx1.\n@param projPoints2 2xN array of corresponding points in the second image. In case of c++ version\nit can be also a vector of feature points or two-channel matrix of size 1xN or Nx1.\n@param points4D 4xN array of reconstructed points in homogeneous coordinates.\n\nThe function reconstructs 3-dimensional points (in homogeneous coordinates) by using their\nobservations with a stereo camera. Projections matrices can be obtained from stereoRectify.\n\n@note\nKeep in mind that all input data should be of float type in order for this function to work.\n\n@sa\nreprojectImageTo3D'''
    package = "3D calibration & reconstruction"

    def get_attributes(self):
        return [Input('projMatr1', 'Proj Matr 1'),
                Input('projMatr2', 'Proj Matr 2'),
                Input('projPoints1', 'Proj Points 1'),
                Input('projPoints2', 'Proj Points 2')], \
               [Output('points4D', 'Points 4D')], \
               []

    def process_inputs(self, inputs, outputs, parameters):
        projMatr1 = inputs['projMatr1'].value
        projMatr2 = inputs['projMatr2'].value
        projPoints1 = inputs['projPoints1'].value
        projPoints2 = inputs['projPoints2'].value
        points4D = cv2.triangulatePoints(projMatr1=projMatr1, projMatr2=projMatr2, projPoints1=projPoints1, projPoints2=projPoints2)
        outputs['points4D'] = Data(points4D)

# cv2.undistort
class OpenCVAuto2_Undistort(NormalElement):
    name = 'Undistort'
    comment = '''undistort(src, cameraMatrix, distCoeffs[, dst[, newCameraMatrix]]) -> dst\n@brief Transforms an image to compensate for lens distortion.\n\nThe function transforms an image to compensate radial and tangential lens distortion.\n\nThe function is simply a combination of #initUndistortRectifyMap (with unity R ) and #remap\n(with bilinear interpolation). See the former function for details of the transformation being\nperformed.\n\nThose pixels in the destination image, for which there is no correspondent pixels in the source\nimage, are filled with zeros (black color).\n\nA particular subset of the source image that will be visible in the corrected image can be regulated\nby newCameraMatrix. You can use #getOptimalNewCameraMatrix to compute the appropriate\nnewCameraMatrix depending on your requirements.\n\nThe camera matrix and the distortion parameters can be determined using #calibrateCamera. If\nthe resolution of images is different from the resolution used at the calibration stage, \f$f_x,\nf_y, c_x\f$ and \f$c_y\f$ need to be scaled accordingly, while the distortion coefficients remain\nthe same.\n\n@param src Input (distorted) image.\n@param dst Output (corrected) image that has the same size and type as src .\n@param cameraMatrix Input camera matrix \f$A = \vecthreethree{f_x}{0}{c_x}{0}{f_y}{c_y}{0}{0}{1}\f$ .\n@param distCoeffs Input vector of distortion coefficients\n\f$(k_1, k_2, p_1, p_2[, k_3[, k_4, k_5, k_6[, s_1, s_2, s_3, s_4[, \tau_x, \tau_y]]]])\f$\nof 4, 5, 8, 12 or 14 elements. If the vector is NULL/empty, the zero distortion coefficients are assumed.\n@param newCameraMatrix Camera matrix of the distorted image. By default, it is the same as\ncameraMatrix but you may additionally scale and shift the result by using a different matrix.'''
    package = "Filters"

    def get_attributes(self):
        return [Input('src', 'src'),
                Input('cameraMatrix', 'Camera Matrix'),
                Input('distCoeffs', 'Dist Coeffs'),
                Input('newCameraMatrix', 'New Camera Matrix', optional=True)], \
               [Output('dst', 'dst')], \
               []

    def process_inputs(self, inputs, outputs, parameters):
        src = inputs['src'].value
        cameraMatrix = inputs['cameraMatrix'].value
        distCoeffs = inputs['distCoeffs'].value
        newCameraMatrix = inputs['newCameraMatrix'].value
        dst = cv2.undistort(src=src, cameraMatrix=cameraMatrix, distCoeffs=distCoeffs, newCameraMatrix=newCameraMatrix)
        outputs['dst'] = Data(dst)

# cv2.undistortPoints
class OpenCVAuto2_UndistortPoints(NormalElement):
    name = 'Undistort Points'
    comment = '''undistortPoints(src, cameraMatrix, distCoeffs[, dst[, R[, P]]]) -> dst\n@brief Computes the ideal point coordinates from the observed point coordinates.\n\nThe function is similar to #undistort and #initUndistortRectifyMap but it operates on a\nsparse set of points instead of a raster image. Also the function performs a reverse transformation\nto projectPoints. In case of a 3D object, it does not reconstruct its 3D coordinates, but for a\nplanar object, it does, up to a translation vector, if the proper R is specified.\n\nFor each observed point coordinate \f$(u, v)\f$ the function computes:\n\f[\n\begin{array}{l}\nx^{"}  \leftarrow (u - c_x)/f_x  \\\ny^{"}  \leftarrow (v - c_y)/f_y  \\\n(x',y') = undistort(x^{"},y^{"}, \texttt{distCoeffs}) \\\n{[X\,Y\,W]} ^T  \leftarrow R*[x' \, y' \, 1]^T  \\\nx  \leftarrow X/W  \\\ny  \leftarrow Y/W  \\\n\text{only performed if P is specified:} \\\nu'  \leftarrow x {f'}_x + {c'}_x  \\\nv'  \leftarrow y {f'}_y + {c'}_y\n\end{array}\n\f]\n\nwhere *undistort* is an approximate iterative algorithm that estimates the normalized original\npoint coordinates out of the normalized distorted point coordinates ("normalized" means that the\ncoordinates do not depend on the camera matrix).\n\nThe function can be used for both a stereo camera head or a monocular camera (when R is empty).\n\n@param src Observed point coordinates, 1xN or Nx1 2-channel (CV_32FC2 or CV_64FC2).\n@param dst Output ideal point coordinates after undistortion and reverse perspective\ntransformation. If matrix P is identity or omitted, dst will contain normalized point coordinates.\n@param cameraMatrix Camera matrix \f$\vecthreethree{f_x}{0}{c_x}{0}{f_y}{c_y}{0}{0}{1}\f$ .\n@param distCoeffs Input vector of distortion coefficients\n\f$(k_1, k_2, p_1, p_2[, k_3[, k_4, k_5, k_6[, s_1, s_2, s_3, s_4[, \tau_x, \tau_y]]]])\f$\nof 4, 5, 8, 12 or 14 elements. If the vector is NULL/empty, the zero distortion coefficients are assumed.\n@param R Rectification transformation in the object space (3x3 matrix). R1 or R2 computed by\n#stereoRectify can be passed here. If the matrix is empty, the identity transformation is used.\n@param P New camera matrix (3x3) or new projection matrix (3x4) \f$\begin{bmatrix} {f'}_x & 0 & {c'}_x & t_x \\ 0 & {f'}_y & {c'}_y & t_y \\ 0 & 0 & 1 & t_z \end{bmatrix}\f$. P1 or P2 computed by\n#stereoRectify can be passed here. If the matrix is empty, the identity new camera matrix is used.'''
    package = "Contours"

    def get_attributes(self):
        return [Input('src', 'src'),
                Input('cameraMatrix', 'Camera Matrix'),
                Input('distCoeffs', 'Dist Coeffs'),
                Input('R', 'R', optional=True),
                Input('P', 'P', optional=True)], \
               [Output('dst', 'dst')], \
               []

    def process_inputs(self, inputs, outputs, parameters):
        src = inputs['src'].value
        cameraMatrix = inputs['cameraMatrix'].value
        distCoeffs = inputs['distCoeffs'].value
        R = inputs['R'].value
        P = inputs['P'].value
        dst = cv2.undistortPoints(src=src, cameraMatrix=cameraMatrix, distCoeffs=distCoeffs, R=R, P=P)
        outputs['dst'] = Data(dst)

# cv2.validateDisparity
class OpenCVAuto2_ValidateDisparity(NormalElement):
    name = 'Validate Disparity'
    comment = '''validateDisparity(disparity, cost, minDisparity, numberOfDisparities[, disp12MaxDisp]) -> disparity
.'''
    package = "3D calibration & reconstruction"

    def get_attributes(self):
        return [Input('disparity', 'disparity'),
                Input('cost', 'cost')], \
               [Output('disparity', 'disparity')], \
               [IntParameter('minDisparity', 'Min Disparity'),
                IntParameter('numberOfDisparities', 'Number Of Disparities'),
                IntParameter('disp12MaxDisp', 'Disp 12 Max Disp')]

    def process_inputs(self, inputs, outputs, parameters):
        disparity = inputs['disparity'].value.copy()
        cost = inputs['cost'].value
        minDisparity = parameters['minDisparity']
        numberOfDisparities = parameters['numberOfDisparities']
        disp12MaxDisp = parameters['disp12MaxDisp']
        disparity = cv2.validateDisparity(disparity=disparity, cost=cost, minDisparity=minDisparity, numberOfDisparities=numberOfDisparities, disp12MaxDisp=disp12MaxDisp)
        outputs['disparity'] = Data(disparity)

# cv2.vconcat
class OpenCVAuto2_Vconcat(NormalElement):
    name = 'Vconcat'
    comment = '''vconcat(src[, dst]) -> dst\n@overload\n@code{.cpp}\nstd::vector<cv::Mat> matrices = { cv::Mat(1, 4, CV_8UC1, cv::Scalar(1)),\ncv::Mat(1, 4, CV_8UC1, cv::Scalar(2)),\ncv::Mat(1, 4, CV_8UC1, cv::Scalar(3)),};\n\ncv::Mat out;\ncv::vconcat( matrices, out );\n//out:\n//[1,   1,   1,   1;\n// 2,   2,   2,   2;\n// 3,   3,   3,   3]\n@endcode\n@param src input array or vector of matrices. all of the matrices must have the same number of cols and the same depth\n@param dst output array. It has the same number of cols and depth as the src, and the sum of rows of the src.\nsame depth.'''
    package = "Matrix miscellaneous"

    def get_attributes(self):
        return [Input('src', 'src')], \
               [Output('dst', 'dst')], \
               []

    def process_inputs(self, inputs, outputs, parameters):
        src = inputs['src'].value
        dst = cv2.vconcat(src=src)
        outputs['dst'] = Data(dst)

# cv2.warpAffine
class OpenCVAuto2_WarpAffine(NormalElement):
    name = 'Warp Affine'
    comment = '''warpAffine(src, M, dsize[, dst[, flags[, borderMode[, borderValue]]]]) -> dst\n@brief Applies an affine transformation to an image.\n\nThe function warpAffine transforms the source image using the specified matrix:\n\n\f[\texttt{dst} (x,y) =  \texttt{src} ( \texttt{M} _{11} x +  \texttt{M} _{12} y +  \texttt{M} _{13}, \texttt{M} _{21} x +  \texttt{M} _{22} y +  \texttt{M} _{23})\f]\n\nwhen the flag #WARP_INVERSE_MAP is set. Otherwise, the transformation is first inverted\nwith #invertAffineTransform and then put in the formula above instead of M. The function cannot\noperate in-place.\n\n@param src input image.\n@param dst output image that has the size dsize and the same type as src .\n@param M \f$2\times 3\f$ transformation matrix.\n@param dsize size of the output image.\n@param flags combination of interpolation methods (see #InterpolationFlags) and the optional\nflag #WARP_INVERSE_MAP that means that M is the inverse transformation (\n\f$\texttt{dst}\rightarrow\texttt{src}\f$ ).\n@param borderMode pixel extrapolation method (see #BorderTypes); when\nborderMode=#BORDER_TRANSPARENT, it means that the pixels in the destination image corresponding to\nthe "outliers" in the source image are not modified by the function.\n@param borderValue value used in case of a constant border; by default, it is 0.\n\n@sa  warpPerspective, resize, remap, getRectSubPix, transform'''
    package = "Transforms"

    def get_attributes(self):
        return [Input('src', 'src'),
                Input('M', 'M')], \
               [Output('dst', 'dst')], \
               [SizeParameter('dsize', 'dsize'),
                ComboboxParameter('flags', name='flags', values=[('WARP_POLAR_LINEAR',0),('INTER_NEAREST',0),('INTER_LINEAR',1),('INTER_CUBIC',2),('INTER_AREA',3),('INTER_LANCZOS4',4),('INTER_BITS',5),('INTER_LINEAR_EXACT',5),('INTER_MAX',7),('WARP_FILL_OUTLIERS',8),('INTER_BITS2',10),('WARP_INVERSE_MAP',16),('INTER_TAB_SIZE',32),('WARP_POLAR_LOG',256),('INTER_TAB_SIZE2',1024)]),
                ComboboxParameter('borderMode', name='Border Mode', values=[('BORDER_CONSTANT',0),('BORDER_REPLICATE',1),('BORDER_REFLECT',2),('BORDER_WRAP',3),('BORDER_DEFAULT',4),('BORDER_REFLECT101',4),('BORDER_REFLECT_101',4),('BORDER_TRANSPARENT',5),('BORDER_ISOLATED',16)]),
                ScalarParameter('borderValue', 'Border Value')]

    def process_inputs(self, inputs, outputs, parameters):
        src = inputs['src'].value
        M = inputs['M'].value
        dsize = parameters['dsize']
        flags = parameters['flags']
        borderMode = parameters['borderMode']
        borderValue = parameters['borderValue']
        dst = cv2.warpAffine(src=src, M=M, dsize=dsize, flags=flags, borderMode=borderMode, borderValue=borderValue)
        outputs['dst'] = Data(dst)

# cv2.warpPerspective
class OpenCVAuto2_WarpPerspective(NormalElement):
    name = 'Warp Perspective'
    comment = '''warpPerspective(src, M, dsize[, dst[, flags[, borderMode[, borderValue]]]]) -> dst\n@brief Applies a perspective transformation to an image.\n\nThe function warpPerspective transforms the source image using the specified matrix:\n\n\f[\texttt{dst} (x,y) =  \texttt{src} \left ( \frac{M_{11} x + M_{12} y + M_{13}}{M_{31} x + M_{32} y + M_{33}} ,\n\frac{M_{21} x + M_{22} y + M_{23}}{M_{31} x + M_{32} y + M_{33}} \right )\f]\n\nwhen the flag #WARP_INVERSE_MAP is set. Otherwise, the transformation is first inverted with invert\nand then put in the formula above instead of M. The function cannot operate in-place.\n\n@param src input image.\n@param dst output image that has the size dsize and the same type as src .\n@param M \f$3\times 3\f$ transformation matrix.\n@param dsize size of the output image.\n@param flags combination of interpolation methods (#INTER_LINEAR or #INTER_NEAREST) and the\noptional flag #WARP_INVERSE_MAP, that sets M as the inverse transformation (\n\f$\texttt{dst}\rightarrow\texttt{src}\f$ ).\n@param borderMode pixel extrapolation method (#BORDER_CONSTANT or #BORDER_REPLICATE).\n@param borderValue value used in case of a constant border; by default, it equals 0.\n\n@sa  warpAffine, resize, remap, getRectSubPix, perspectiveTransform'''
    package = "Transforms"

    def get_attributes(self):
        return [Input('src', 'src'),
                Input('M', 'M')], \
               [Output('dst', 'dst')], \
               [SizeParameter('dsize', 'dsize'),
                ComboboxParameter('flags', name='flags', values=[('INTER_NEAREST',0),('WARP_POLAR_LINEAR',0),('INTER_LINEAR',1),('INTER_CUBIC',2),('INTER_AREA',3),('INTER_LANCZOS4',4),('INTER_BITS',5),('INTER_LINEAR_EXACT',5),('INTER_MAX',7),('WARP_FILL_OUTLIERS',8),('INTER_BITS2',10),('WARP_INVERSE_MAP',16),('INTER_TAB_SIZE',32),('WARP_POLAR_LOG',256),('INTER_TAB_SIZE2',1024)]),
                ComboboxParameter('borderMode', name='Border Mode', values=[('BORDER_CONSTANT',0),('BORDER_REPLICATE',1),('BORDER_REFLECT',2),('BORDER_WRAP',3),('BORDER_DEFAULT',4),('BORDER_REFLECT101',4),('BORDER_REFLECT_101',4),('BORDER_TRANSPARENT',5),('BORDER_ISOLATED',16)]),
                ScalarParameter('borderValue', 'Border Value')]

    def process_inputs(self, inputs, outputs, parameters):
        src = inputs['src'].value
        M = inputs['M'].value
        dsize = parameters['dsize']
        flags = parameters['flags']
        borderMode = parameters['borderMode']
        borderValue = parameters['borderValue']
        dst = cv2.warpPerspective(src=src, M=M, dsize=dsize, flags=flags, borderMode=borderMode, borderValue=borderValue)
        outputs['dst'] = Data(dst)

# cv2.warpPolar
class OpenCVAuto2_WarpPolar(NormalElement):
    name = 'Warp Polar'
    comment = '''warpPolar(src, dsize, center, maxRadius, flags[, dst]) -> dst\n\brief Remaps an image to polar or semilog-polar coordinates space\n\n@anchor polar_remaps_reference_image\n![Polar remaps reference](pics/polar_remap_doc.png)\n\nTransform the source image using the following transformation:\n\f[\ndst(\rho , \phi ) = src(x,y)\n\f]\n\nwhere\n\f[\n\begin{array}{l}\n\vec{I} = (x - center.x, \;y - center.y) \\\n\phi = Kangle \cdot \texttt{angle} (\vec{I}) \\\n\rho = \left\{\begin{matrix}\nKlin \cdot \texttt{magnitude} (\vec{I}) & default \\\nKlog \cdot log_e(\texttt{magnitude} (\vec{I})) & if \; semilog \\\n\end{matrix}\right.\n\end{array}\n\f]\n\nand\n\f[\n\begin{array}{l}\nKangle = dsize.height / 2\Pi \\\nKlin = dsize.width / maxRadius \\\nKlog = dsize.width / log_e(maxRadius) \\\n\end{array}\n\f]\n\n\n\par Linear vs semilog mapping\n\nPolar mapping can be linear or semi-log. Add one of #WarpPolarMode to `flags` to specify the polar mapping mode.\n\nLinear is the default mode.\n\nThe semilog mapping emulates the human "foveal" vision that permit very high acuity on the line of sight (central vision)\nin contrast to peripheral vision where acuity is minor.\n\n\par Option on `dsize`:\n\n- if both values in `dsize <=0 ` (default),\nthe destination image will have (almost) same area of source bounding circle:\n\f[\begin{array}{l}\ndsize.area  \leftarrow (maxRadius^2 \cdot \Pi) \\\ndsize.width = \texttt{cvRound}(maxRadius) \\\ndsize.height = \texttt{cvRound}(maxRadius \cdot \Pi) \\\n\end{array}\f]\n\n\n- if only `dsize.height <= 0`,\nthe destination image area will be proportional to the bounding circle area but scaled by `Kx * Kx`:\n\f[\begin{array}{l}\ndsize.height = \texttt{cvRound}(dsize.width \cdot \Pi) \\\n\end{array}\n\f]\n\n- if both values in `dsize > 0 `,\nthe destination image will have the given size therefore the area of the bounding circle will be scaled to `dsize`.\n\n\n\par Reverse mapping\n\nYou can get reverse mapping adding #WARP_INVERSE_MAP to `flags`\n\snippet polar_transforms.cpp InverseMap\n\nIn addiction, to calculate the original coordinate from a polar mapped coordinate \f$(rho, phi)->(x, y)\f$:\n\snippet polar_transforms.cpp InverseCoordinate\n\n@param src Source image.\n@param dst Destination image. It will have same type as src.\n@param dsize The destination image size (see description for valid options).\n@param center The transformation center.\n@param maxRadius The radius of the bounding circle to transform. It determines the inverse magnitude scale parameter too.\n@param flags A combination of interpolation methods, #InterpolationFlags + #WarpPolarMode.\n- Add #WARP_POLAR_LINEAR to select linear polar mapping (default)\n- Add #WARP_POLAR_LOG to select semilog polar mapping\n- Add #WARP_INVERSE_MAP for reverse mapping.\n@note\n-  The function can not operate in-place.\n-  To calculate magnitude and angle in degrees #cartToPolar is used internally thus angles are measured from 0 to 360 with accuracy about 0.3 degrees.\n-  This function uses #remap. Due to current implementation limitations the size of an input and output images should be less than 32767x32767.\n\n@sa cv::remap'''
    package = "Transforms"

    def get_attributes(self):
        return [Input('src', 'src')], \
               [Output('dst', 'dst')], \
               [SizeParameter('dsize', 'dsize'),
                PointParameter('center', 'center'),
                IntParameter('maxRadius', 'Max Radius', min_=0),
                ComboboxParameter('flags', name='flags', values=[('WARP_POLAR_LINEAR',0),('INTER_NEAREST',0),('INTER_LINEAR',1),('INTER_CUBIC',2),('INTER_AREA',3),('INTER_LANCZOS4',4),('INTER_BITS',5),('INTER_LINEAR_EXACT',5),('INTER_MAX',7),('WARP_FILL_OUTLIERS',8),('INTER_BITS2',10),('WARP_INVERSE_MAP',16),('INTER_TAB_SIZE',32),('WARP_POLAR_LOG',256),('INTER_TAB_SIZE2',1024)])]

    def process_inputs(self, inputs, outputs, parameters):
        src = inputs['src'].value
        dsize = parameters['dsize']
        center = parameters['center']
        maxRadius = parameters['maxRadius']
        flags = parameters['flags']
        dst = cv2.warpPolar(src=src, dsize=dsize, center=center, maxRadius=maxRadius, flags=flags)
        outputs['dst'] = Data(dst)

# cv2.watershed
class OpenCVAuto2_Watershed(NormalElement):
    name = 'Watershed'
    comment = '''watershed(image, markers) -> markers\n@brief Performs a marker-based image segmentation using the watershed algorithm.\n\nThe function implements one of the variants of watershed, non-parametric marker-based segmentation\nalgorithm, described in @cite Meyer92 .\n\nBefore passing the image to the function, you have to roughly outline the desired regions in the\nimage markers with positive (\>0) indices. So, every region is represented as one or more connected\ncomponents with the pixel values 1, 2, 3, and so on. Such markers can be retrieved from a binary\nmask using #findContours and #drawContours (see the watershed.cpp demo). The markers are "seeds" of\nthe future image regions. All the other pixels in markers , whose relation to the outlined regions\nis not known and should be defined by the algorithm, should be set to 0's. In the function output,\neach pixel in markers is set to a value of the "seed" components or to -1 at boundaries between the\nregions.\n\n@note Any two neighbor connected components are not necessarily separated by a watershed boundary\n(-1's pixels); for example, they can touch each other in the initial marker image passed to the\nfunction.\n\n@param image Input 8-bit 3-channel image.\n@param markers Input/output 32-bit single-channel image (map) of markers. It should have the same\nsize as image .\n\n@sa findContours\n\n@ingroup imgproc_misc'''
    package = "Segmentation"

    def get_attributes(self):
        return [Input('image', 'image'),
                Input('markers', 'markers')], \
               [Output('markers', 'markers')], \
               []

    def process_inputs(self, inputs, outputs, parameters):
        image = inputs['image'].value
        markers = inputs['markers'].value.copy()
        markers = cv2.watershed(image=image, markers=markers)
        outputs['markers'] = Data(markers)


register_elements_auto(__name__, locals(), "OpenCV autogenerated 2", 15)
