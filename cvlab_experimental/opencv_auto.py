import cv2

from cvlab.diagram.elements.base import *

################# GENERATED CODE (gen2_gui.py) #################

class OpenCVAuto_Canny(NormalElement):
    name = 'Canny'
    comment = 'Canny(image, threshold1, threshold2[, edges[, apertureSize[, L2gradient]]]) -> edges'

    def get_attributes(self):
        return [Input('image', 'image')], \
               [Output('edges', 'edges')], \
               [FloatParameter('threshold1', 'threshold1', max_=100000), FloatParameter('threshold2', 'threshold2', max_=100000),
                IntParameter('apertureSize', 'apertureSize', min_=1, max_=7, step=2), IntParameter('L2gradient', 'L2gradient', min_=0, max_=1)]

    def process_inputs(self, inputs, outputs, parameters):
        image = inputs['image'].value
        threshold1 = parameters['threshold1']
        threshold2 = parameters['threshold2']
        apertureSize = parameters['apertureSize']
        L2gradient = parameters['L2gradient']
        edges = cv2.Canny(image=image, threshold1=threshold1, threshold2=threshold2, apertureSize=apertureSize,
                          L2gradient=L2gradient)
        outputs['edges'] = Data(edges)


class OpenCVAuto_GaussianBlur(NormalElement):
    name = 'GaussianBlur'
    comment = 'GaussianBlur(src, ksize, sigmaX[, dst[, sigmaY[, borderType]]]) -> dst'

    def get_attributes(self):
        return [Input('src', 'src')], \
               [Output('dst', 'dst')], \
               [SizeParameter('ksize', 'ksize'), FloatParameter('sigmaX', 'sigmaX', min_=0),
                FloatParameter('sigmaY', 'sigmaY', min_=0),
                IntParameter('borderType', 'borderType')]

    def process_inputs(self, inputs, outputs, parameters):
        src = inputs['src'].value
        ksize = parameters['ksize']
        sigmaX = parameters['sigmaX']
        sigmaY = parameters['sigmaY']
        borderType = parameters['borderType']
        dst = cv2.GaussianBlur(src=src, ksize=ksize, sigmaX=sigmaX, sigmaY=sigmaY, borderType=borderType)
        outputs['dst'] = Data(dst)


class OpenCVAuto_HoughCircles(NormalElement):
    name = 'HoughCircles'
    comment = 'HoughCircles(image, method, dp, minDist[, circles[, param1[, param2[, minRadius[, maxRadius]]]]]) -> ' \
              'circles'

    def get_attributes(self):
        return [Input('image', 'image')], \
               [Output('circles', 'circles')], \
               [IntParameter('method', 'method'), FloatParameter('dp', 'dp'), FloatParameter('minDist', 'minDist'),
                FloatParameter('param1', 'param1'), FloatParameter('param2', 'param2'),
                IntParameter('minRadius', 'minRadius'), IntParameter('maxRadius', 'maxRadius')]

    def process_inputs(self, inputs, outputs, parameters):
        image = inputs['image'].value
        method = parameters['method']
        dp = parameters['dp']
        minDist = parameters['minDist']
        param1 = parameters['param1']
        param2 = parameters['param2']
        minRadius = parameters['minRadius']
        maxRadius = parameters['maxRadius']
        circles = cv2.HoughCircles(image=image, method=method, dp=dp, minDist=minDist, param1=param1, param2=param2,
                                   minRadius=minRadius, maxRadius=maxRadius)
        outputs['circles'] = Data(circles)


class OpenCVAuto_HoughLines(NormalElement):
    name = 'HoughLines'
    comment = 'HoughLines(image, rho, theta, threshold[, lines[, srn[, stn]]]) -> lines'

    def get_attributes(self):
        return [Input('image', 'image')], \
               [Output('lines', 'lines')], \
               [FloatParameter('rho', 'rho'), FloatParameter('theta', 'theta'), IntParameter('threshold', 'threshold'),
                FloatParameter('srn', 'srn'), FloatParameter('stn', 'stn')]

    def process_inputs(self, inputs, outputs, parameters):
        image = inputs['image'].value
        rho = parameters['rho']
        theta = parameters['theta']
        threshold = parameters['threshold']
        srn = parameters['srn']
        stn = parameters['stn']
        lines = cv2.HoughLines(image=image, rho=rho, theta=theta, threshold=threshold, srn=srn, stn=stn)
        outputs['lines'] = Data(lines)


class OpenCVAuto_HoughLinesP(NormalElement):
    name = 'HoughLinesP'
    comment = 'HoughLinesP(image, rho, theta, threshold[, lines[, minLineLength[, maxLineGap]]]) -> lines'

    def get_attributes(self):
        return [Input('image', 'image')], \
               [Output('lines', 'lines')], \
               [FloatParameter('rho', 'rho'), FloatParameter('theta', 'theta'), IntParameter('threshold', 'threshold'),
                FloatParameter('minLineLength', 'minLineLength'), FloatParameter('maxLineGap', 'maxLineGap')]

    def process_inputs(self, inputs, outputs, parameters):
        image = inputs['image'].value
        rho = parameters['rho']
        theta = parameters['theta']
        threshold = parameters['threshold']
        minLineLength = parameters['minLineLength']
        maxLineGap = parameters['maxLineGap']
        lines = cv2.HoughLinesP(image=image, rho=rho, theta=theta, threshold=threshold, minLineLength=minLineLength,
                                maxLineGap=maxLineGap)
        outputs['lines'] = Data(lines)


class OpenCVAuto_HuMoments(NormalElement):
    name = 'HuMoments'
    comment = 'HuMoments(m[, hu]) -> hu'

    def get_attributes(self):
        return [], \
            [Output('hu', 'hu')], \
            []

    def process_inputs(self, inputs, outputs, parameters):
        hu = cv2.HuMoments(m=None)
        outputs['hu'] = Data(hu)


class OpenCVAuto_LUT(NormalElement):
    name = 'LUT'
    comment = 'LUT(src, lut[, dst[, interpolation]]) -> dst'

    def get_attributes(self):
        return [Input('src', 'src'), Input('lut', 'lut')], \
               [Output('dst', 'dst')], \
               [IntParameter('interpolation', 'interpolation')]

    def process_inputs(self, inputs, outputs, parameters):
        src = inputs['src'].value
        lut = inputs['lut'].value
        interpolation = parameters['interpolation']
        dst = cv2.LUT(src=src, lut=lut, interpolation=interpolation)
        outputs['dst'] = Data(dst)


class OpenCVAuto_Laplacian(NormalElement):
    name = 'Laplacian'
    comment = 'Laplacian(src, ddepth[, dst[, ksize[, scale[, delta[, borderType]]]]]) -> dst'

    def get_attributes(self):
        return [Input('src', 'src')], \
               [Output('dst', 'dst')], \
               [IntParameter('ddepth', 'ddepth'), IntParameter('ksize', 'ksize'), FloatParameter('scale', 'scale'),
                FloatParameter('delta', 'delta'), IntParameter('borderType', 'borderType')]

    def process_inputs(self, inputs, outputs, parameters):
        src = inputs['src'].value
        ddepth = parameters['ddepth']
        ksize = parameters['ksize']
        scale = parameters['scale']
        delta = parameters['delta']
        borderType = parameters['borderType']
        dst = cv2.Laplacian(src=src, ddepth=ddepth, ksize=ksize, scale=scale, delta=delta, borderType=borderType)
        outputs['dst'] = Data(dst)


class OpenCVAuto_PCABackProject(NormalElement):
    name = 'PCABackProject'
    comment = 'PCABackProject(data, mean, eigenvectors[, result]) -> result'

    def get_attributes(self):
        return [Input('data', 'data'), Input('mean', 'mean'), Input('eigenvectors', 'eigenvectors')], \
               [Output('result', 'result')], \
            []

    def process_inputs(self, inputs, outputs, parameters):
        data = inputs['data'].value
        mean = inputs['mean'].value
        eigenvectors = inputs['eigenvectors'].value
        result = cv2.PCABackProject(data=data, mean=mean, eigenvectors=eigenvectors)
        outputs['result'] = Data(result)


class OpenCVAuto_PCACompute(NormalElement):
    name = 'PCACompute'
    comment = 'PCACompute(data[, mean[, eigenvectors[, maxComponents]]]) -> mean, eigenvectors'

    def get_attributes(self):
        return [Input('data', 'data')], \
               [Output('eigenvectors', 'eigenvectors')], \
               [IntParameter('maxComponents', 'maxComponents')]

    def process_inputs(self, inputs, outputs, parameters):
        data = inputs['data'].value
        maxComponents = parameters['maxComponents']
        eigenvectors = cv2.PCACompute(data=data, maxComponents=maxComponents)
        outputs['eigenvectors'] = Data(eigenvectors)


class OpenCVAuto_PCAComputeVar(NormalElement):
    name = 'PCAComputeVar'
    comment = 'PCAComputeVar(data, retainedVariance[, mean[, eigenvectors]]) -> mean, eigenvectors'

    def get_attributes(self):
        return [Input('data', 'data')], \
               [Output('eigenvectors', 'eigenvectors')], \
               [FloatParameter('retainedVariance', 'retainedVariance')]

    def process_inputs(self, inputs, outputs, parameters):
        data = inputs['data'].value
        retainedVariance = parameters['retainedVariance']
        eigenvectors = cv2.PCAComputeVar(data=data, retainedVariance=retainedVariance)
        outputs['eigenvectors'] = Data(eigenvectors)


class OpenCVAuto_PCAProject(NormalElement):
    name = 'PCAProject'
    comment = 'PCAProject(data, mean, eigenvectors[, result]) -> result'

    def get_attributes(self):
        return [Input('data', 'data'), Input('mean', 'mean'), Input('eigenvectors', 'eigenvectors')], \
               [Output('result', 'result')], \
            []

    def process_inputs(self, inputs, outputs, parameters):
        data = inputs['data'].value
        mean = inputs['mean'].value
        eigenvectors = inputs['eigenvectors'].value
        result = cv2.PCAProject(data=data, mean=mean, eigenvectors=eigenvectors)
        outputs['result'] = Data(result)


class OpenCVAuto_RQDecomp3x3(NormalElement):
    name = 'RQDecomp3x3'
    comment = 'RQDecomp3x3(src[, mtxR[, mtxQ[, Qx[, Qy[, Qz]]]]]) -> retval, mtxR, mtxQ, Qx, Qy, Qz'

    def get_attributes(self):
        return [Input('src', 'src')], \
               [Output('mtxR', 'mtxR'), Output('mtxQ', 'mtxQ'), Output('Qx', 'Qx'), Output('Qy', 'Qy'),
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


class OpenCVAuto_Rodrigues(NormalElement):
    name = 'Rodrigues'
    comment = 'Rodrigues(src[, dst[, jacobian]]) -> dst, jacobian'

    def get_attributes(self):
        return [Input('src', 'src')], \
               [Output('dst', 'dst'), Output('jacobian', 'jacobian')], \
            []

    def process_inputs(self, inputs, outputs, parameters):
        src = inputs['src'].value
        dst, jacobian = cv2.Rodrigues(src=src)
        outputs['dst'] = Data(dst)
        outputs['jacobian'] = Data(jacobian)


class OpenCVAuto_SVBackSubst(NormalElement):
    name = 'SVBackSubst'
    comment = 'SVBackSubst(w, u, vt, rhs[, dst]) -> dst'

    def get_attributes(self):
        return [Input('w', 'w'), Input('u', 'u'), Input('vt', 'vt'), Input('rhs', 'rhs')], \
               [Output('dst', 'dst')], \
            []

    def process_inputs(self, inputs, outputs, parameters):
        w = inputs['w'].value
        u = inputs['u'].value
        vt = inputs['vt'].value
        rhs = inputs['rhs'].value
        dst = cv2.SVBackSubst(w=w, u=u, vt=vt, rhs=rhs)
        outputs['dst'] = Data(dst)


class OpenCVAuto_SVDecomp(NormalElement):
    name = 'SVDecomp'
    comment = 'SVDecomp(src[, w[, u[, vt[, flags]]]]) -> w, u, vt'

    def get_attributes(self):
        return [Input('src', 'src')], \
               [Output('w', 'w'), Output('u', 'u'), Output('vt', 'vt')], \
               [IntParameter('flags', 'flags')]

    def process_inputs(self, inputs, outputs, parameters):
        src = inputs['src'].value
        flags = parameters['flags']
        w, u, vt = cv2.SVDecomp(src=src, flags=flags)
        outputs['w'] = Data(w)
        outputs['u'] = Data(u)
        outputs['vt'] = Data(vt)


class OpenCVAuto_Scharr(NormalElement):
    name = 'Scharr'
    comment = 'Scharr(src, ddepth, dx, dy[, dst[, scale[, delta[, borderType]]]]) -> dst'

    def get_attributes(self):
        return [Input('src', 'src')], \
               [Output('dst', 'dst')], \
               [IntParameter('ddepth', 'ddepth'), IntParameter('dx', 'dx'), IntParameter('dy', 'dy'),
                FloatParameter('scale', 'scale'), FloatParameter('delta', 'delta'),
                IntParameter('borderType', 'borderType')]

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


class OpenCVAuto_Sobel(NormalElement):
    name = 'Sobel'
    comment = 'Sobel(src, ddepth, dx, dy[, dst[, ksize[, scale[, delta[, borderType]]]]]) -> dst'

    def get_attributes(self):
        return [Input('src', 'src')], \
               [Output('dst', 'dst')], \
               [IntParameter('ddepth', 'ddepth'), IntParameter('dx', 'dx'), IntParameter('dy', 'dy'),
                IntParameter('ksize', 'ksize'), FloatParameter('scale', 'scale'), FloatParameter('delta', 'delta'),
                IntParameter('borderType', 'borderType')]

    def process_inputs(self, inputs, outputs, parameters):
        src = inputs['src'].value
        ddepth = parameters['ddepth']
        dx = parameters['dx']
        dy = parameters['dy']
        ksize = parameters['ksize']
        scale = parameters['scale']
        delta = parameters['delta']
        borderType = parameters['borderType']
        dst = cv2.Sobel(src=src, ddepth=ddepth, dx=dx, dy=dy, ksize=ksize, scale=scale, delta=delta,
                        borderType=borderType)
        outputs['dst'] = Data(dst)


class OpenCVAuto_absdiff(NormalElement):
    name = 'absdiff'
    comment = 'absdiff(src1, src2[, dst]) -> dst'

    def get_attributes(self):
        return [Input('src1', 'src1'), Input('src2', 'src2')], \
               [Output('dst', 'dst')], \
            []

    def process_inputs(self, inputs, outputs, parameters):
        src1 = inputs['src1'].value
        src2 = inputs['src2'].value
        dst = cv2.absdiff(src1=src1, src2=src2)
        outputs['dst'] = Data(dst)


class OpenCVAuto_adaptiveThreshold(NormalElement):
    name = 'adaptiveThreshold'
    comment = 'adaptiveThreshold(src, maxValue, adaptiveMethod, thresholdType, blockSize, C[, dst]) -> dst'

    def get_attributes(self):
        return [Input('src', 'src')], \
               [Output('dst', 'dst')], \
               [FloatParameter('maxValue', 'maxValue'), IntParameter('adaptiveMethod', 'adaptiveMethod'),
                IntParameter('thresholdType', 'thresholdType'), IntParameter('blockSize', 'blockSize'),
                FloatParameter('C', 'C')]

    def process_inputs(self, inputs, outputs, parameters):
        src = inputs['src'].value
        maxValue = parameters['maxValue']
        adaptiveMethod = parameters['adaptiveMethod']
        thresholdType = parameters['thresholdType']
        blockSize = parameters['blockSize']
        C = parameters['C']
        dst = cv2.adaptiveThreshold(src=src, maxValue=maxValue, adaptiveMethod=adaptiveMethod,
                                    thresholdType=thresholdType, blockSize=blockSize, C=C)
        outputs['dst'] = Data(dst)


class OpenCVAuto_add(NormalElement):
    name = 'add'
    comment = 'add(src1, src2[, dst[, mask[, dtype]]]) -> dst'

    def get_attributes(self):
        return [Input('src1', 'src1'), Input('src2', 'src2'), Input('mask', 'mask', optional=True)], \
               [Output('dst', 'dst')], \
               [IntParameter('dtype', 'dtype')]

    def process_inputs(self, inputs, outputs, parameters):
        src1 = inputs['src1'].value
        src2 = inputs['src2'].value
        mask = inputs['mask'].value
        dtype = parameters['dtype']
        dst = cv2.add(src1=src1, src2=src2, mask=mask, dtype=dtype)
        outputs['dst'] = Data(dst)


class OpenCVAuto_addWeighted(NormalElement):
    name = 'addWeighted'
    comment = 'addWeighted(src1, alpha, src2, beta, gamma[, dst[, dtype]]) -> dst'

    def get_attributes(self):
        return [Input('src1', 'src1'), Input('src2', 'src2')], \
               [Output('dst', 'dst')], \
               [FloatParameter('alpha', 'alpha'), FloatParameter('beta', 'beta'), FloatParameter('gamma', 'gamma'),
                IntParameter('dtype', 'dtype')]

    def process_inputs(self, inputs, outputs, parameters):
        src1 = inputs['src1'].value
        alpha = parameters['alpha']
        src2 = inputs['src2'].value
        beta = parameters['beta']
        gamma = parameters['gamma']
        dtype = parameters['dtype']
        dst = cv2.addWeighted(src1=src1, alpha=alpha, src2=src2, beta=beta, gamma=gamma, dtype=dtype)
        outputs['dst'] = Data(dst)


class OpenCVAuto_applyColorMap(NormalElement):
    name = 'applyColorMap'
    comment = 'applyColorMap(src, colormap[, dst]) -> dst'

    def get_attributes(self):
        return [Input('src', 'src')], \
               [Output('dst', 'dst')], \
               [IntParameter('colormap', 'colormap')]

    def process_inputs(self, inputs, outputs, parameters):
        src = inputs['src'].value
        colormap = parameters['colormap']
        dst = cv2.applyColorMap(src=src, colormap=colormap)
        outputs['dst'] = Data(dst)


class OpenCVAuto_approxPolyDP(NormalElement):
    name = 'approxPolyDP'
    comment = 'approxPolyDP(curve, epsilon, closed[, approxCurve]) -> approxCurve'

    def get_attributes(self):
        return [Input('curve', 'curve')], \
               [Output('approxCurve', 'approxCurve')], \
               [FloatParameter('epsilon', 'epsilon'), IntParameter('closed', 'closed')]

    def process_inputs(self, inputs, outputs, parameters):
        curve = inputs['curve'].value
        epsilon = parameters['epsilon']
        closed = parameters['closed']
        approxCurve = cv2.approxPolyDP(curve=curve, epsilon=epsilon, closed=closed)
        outputs['approxCurve'] = Data(approxCurve)


class OpenCVAuto_batchDistance(NormalElement):
    name = 'batchDistance'
    comment = 'batchDistance(src1, src2, dtype[, dist[, nidx[, normType[, K[, mask[, update[, crosscheck]]]]]]]) -> ' \
              'dist, nidx'

    def get_attributes(self):
        return [Input('src1', 'src1'), Input('src2', 'src2'), Input('mask', 'mask', optional=True)], \
               [Output('dist', 'dist'), Output('nidx', 'nidx')], \
               [IntParameter('dtype', 'dtype'), IntParameter('normType', 'normType'), IntParameter('K', 'K'),
                IntParameter('update', 'update'), IntParameter('crosscheck', 'crosscheck')]

    def process_inputs(self, inputs, outputs, parameters):
        src1 = inputs['src1'].value
        src2 = inputs['src2'].value
        dtype = parameters['dtype']
        normType = parameters['normType']
        K = parameters['K']
        mask = inputs['mask'].value
        update = parameters['update']
        crosscheck = parameters['crosscheck']
        dist, nidx = cv2.batchDistance(src1=src1, src2=src2, dtype=dtype, normType=normType, K=K, mask=mask,
                                       update=update, crosscheck=crosscheck)
        outputs['dist'] = Data(dist)
        outputs['nidx'] = Data(nidx)


class OpenCVAuto_bilateralFilter(NormalElement):
    name = 'bilateralFilter'
    comment = 'bilateralFilter(src, d, sigmaColor, sigmaSpace[, dst[, borderType]]) -> dst'

    def get_attributes(self):
        return [Input('src', 'src')], \
               [Output('dst', 'dst')], \
               [IntParameter('d', 'd'), FloatParameter('sigmaColor', 'sigmaColor'),
                FloatParameter('sigmaSpace', 'sigmaSpace'), IntParameter('borderType', 'borderType')]

    def process_inputs(self, inputs, outputs, parameters):
        src = inputs['src'].value
        d = parameters['d']
        sigmaColor = parameters['sigmaColor']
        sigmaSpace = parameters['sigmaSpace']
        borderType = parameters['borderType']
        dst = cv2.bilateralFilter(src=src, d=d, sigmaColor=sigmaColor, sigmaSpace=sigmaSpace, borderType=borderType)
        outputs['dst'] = Data(dst)


class OpenCVAuto_bitwise_and(NormalElement):
    name = 'bitwise_and'
    comment = 'bitwise_and(src1, src2[, dst[, mask]]) -> dst'

    def get_attributes(self):
        return [Input('src1', 'src1'), Input('src2', 'src2'), Input('mask', 'mask', optional=True)], \
               [Output('dst', 'dst')], \
            []

    def process_inputs(self, inputs, outputs, parameters):
        src1 = inputs['src1'].value
        src2 = inputs['src2'].value
        mask = inputs['mask'].value
        dst = cv2.bitwise_and(src1=src1, src2=src2, mask=mask)
        outputs['dst'] = Data(dst)


class OpenCVAuto_bitwise_not(NormalElement):
    name = 'bitwise_not'
    comment = 'bitwise_not(src[, dst[, mask]]) -> dst'

    def get_attributes(self):
        return [Input('src', 'src'), Input('mask', 'mask', optional=True)], \
               [Output('dst', 'dst')], \
            []

    def process_inputs(self, inputs, outputs, parameters):
        src = inputs['src'].value
        mask = inputs['mask'].value
        dst = cv2.bitwise_not(src=src, mask=mask)
        outputs['dst'] = Data(dst)


class OpenCVAuto_bitwise_or(NormalElement):
    name = 'bitwise_or'
    comment = 'bitwise_or(src1, src2[, dst[, mask]]) -> dst'

    def get_attributes(self):
        return [Input('src1', 'src1'), Input('src2', 'src2'), Input('mask', 'mask', optional=True)], \
               [Output('dst', 'dst')], \
            []

    def process_inputs(self, inputs, outputs, parameters):
        src1 = inputs['src1'].value
        src2 = inputs['src2'].value
        mask = inputs['mask'].value
        dst = cv2.bitwise_or(src1=src1, src2=src2, mask=mask)
        outputs['dst'] = Data(dst)


class OpenCVAuto_bitwise_xor(NormalElement):
    name = 'bitwise_xor'
    comment = 'bitwise_xor(src1, src2[, dst[, mask]]) -> dst'

    def get_attributes(self):
        return [Input('src1', 'src1'), Input('src2', 'src2'), Input('mask', 'mask', optional=True)], \
               [Output('dst', 'dst')], \
            []

    def process_inputs(self, inputs, outputs, parameters):
        src1 = inputs['src1'].value
        src2 = inputs['src2'].value
        mask = inputs['mask'].value
        dst = cv2.bitwise_xor(src1=src1, src2=src2, mask=mask)
        outputs['dst'] = Data(dst)


class OpenCVAuto_blur(NormalElement):
    name = 'blur'
    comment = 'blur(src, ksize[, dst[, anchor[, borderType]]]) -> dst'

    def get_attributes(self):
        return [Input('src', 'src')], \
               [Output('dst', 'dst')], \
               [SizeParameter('ksize', 'ksize'), PointParameter('anchor', 'anchor'),
                IntParameter('borderType', 'borderType')]

    def process_inputs(self, inputs, outputs, parameters):
        src = inputs['src'].value
        ksize = parameters['ksize']
        anchor = parameters['anchor']
        borderType = parameters['borderType']
        dst = cv2.blur(src=src, ksize=ksize, anchor=anchor, borderType=borderType)
        outputs['dst'] = Data(dst)


class OpenCVAuto_boxFilter(NormalElement):
    name = 'boxFilter'
    comment = 'boxFilter(src, ddepth, ksize[, dst[, anchor[, normalize[, borderType]]]]) -> dst'

    def get_attributes(self):
        return [Input('src', 'src')], \
               [Output('dst', 'dst')], \
               [IntParameter('ddepth', 'ddepth'), SizeParameter('ksize', 'ksize'), PointParameter('anchor', 'anchor'),
                IntParameter('normalize', 'normalize'), IntParameter('borderType', 'borderType')]

    def process_inputs(self, inputs, outputs, parameters):
        src = inputs['src'].value
        ddepth = parameters['ddepth']
        ksize = parameters['ksize']
        anchor = parameters['anchor']
        normalize = parameters['normalize']
        borderType = parameters['borderType']
        dst = cv2.boxFilter(src=src, ddepth=ddepth, ksize=ksize, anchor=anchor, normalize=normalize,
                            borderType=borderType)
        outputs['dst'] = Data(dst)


class OpenCVAuto_calcBackProject(NormalElement):
    name = 'calcBackProject'
    comment = 'calcBackProject(images, channels, hist, ranges, scale[, dst]) -> dst'

    def get_attributes(self):
        return [Input('hist', 'hist')], \
               [Output('dst', 'dst')], \
               [FloatParameter('scale', 'scale')]

    def process_inputs(self, inputs, outputs, parameters):
        hist = inputs['hist'].value
        scale = parameters['scale']
        dst = cv2.calcBackProject(images=None, channels=None, hist=hist, ranges=None, scale=scale)
        outputs['dst'] = Data(dst)


class OpenCVAuto_calcCovarMatrix(NormalElement):
    name = 'calcCovarMatrix'
    comment = 'calcCovarMatrix(samples, flags[, covar[, mean[, ctype]]]) -> covar, mean'

    def get_attributes(self):
        return [Input('samples', 'samples')], \
               [Output('covar', 'covar'), Output('mean', 'mean')], \
               [IntParameter('flags', 'flags'), IntParameter('ctype', 'ctype')]

    def process_inputs(self, inputs, outputs, parameters):
        samples = inputs['samples'].value
        flags = parameters['flags']
        ctype = parameters['ctype']
        covar, mean = cv2.calcCovarMatrix(samples=samples, flags=flags, ctype=ctype)
        outputs['covar'] = Data(covar)
        outputs['mean'] = Data(mean)


class OpenCVAuto_calcHist(NormalElement):
    name = 'calcHist'
    comment = 'calcHist(images, channels, mask, histSize, ranges[, hist[, accumulate]]) -> hist'

    def get_attributes(self):
        return [Input('mask', 'mask')], \
               [Output('hist', 'hist')], \
               [IntParameter('accumulate', 'accumulate')]

    def process_inputs(self, inputs, outputs, parameters):
        mask = inputs['mask'].value
        accumulate = parameters['accumulate']
        hist = cv2.calcHist(images=None, channels=None, mask=mask, histSize=None, ranges=None, accumulate=accumulate)
        outputs['hist'] = Data(hist)


class OpenCVAuto_calcMotionGradient(NormalElement):
    name = 'calcMotionGradient'
    comment = 'calcMotionGradient(mhi, delta1, delta2[, mask[, orientation[, apertureSize]]]) -> mask, orientation'

    def get_attributes(self):
        return [Input('mhi', 'mhi')], \
               [Output('mask', 'mask'), Output('orientation', 'orientation')], \
               [FloatParameter('delta1', 'delta1'), FloatParameter('delta2', 'delta2'),
                IntParameter('apertureSize', 'apertureSize')]

    def process_inputs(self, inputs, outputs, parameters):
        mhi = inputs['mhi'].value
        delta1 = parameters['delta1']
        delta2 = parameters['delta2']
        apertureSize = parameters['apertureSize']
        mask, orientation = cv2.calcMotionGradient(mhi=mhi, delta1=delta1, delta2=delta2, apertureSize=apertureSize)
        outputs['mask'] = Data(mask)
        outputs['orientation'] = Data(orientation)


class OpenCVAuto_calcOpticalFlowPyrLK(NormalElement):
    name = 'calcOpticalFlowPyrLK'
    comment = 'calcOpticalFlowPyrLK(prevImg, nextImg, prevPts[, nextPts[, status[, err[, winSize[, maxLevel[, ' \
              'criteria[, flags[, minEigThreshold]]]]]]]]) -> nextPts, status, err'

    def get_attributes(self):
        return [Input('prevImg', 'prevImg'), Input('nextImg', 'nextImg'), Input('prevPts', 'prevPts')], \
               [Output('status', 'status'), Output('err', 'err')], \
               [SizeParameter('winSize', 'winSize'), IntParameter('maxLevel', 'maxLevel'),
                IntParameter('flags', 'flags'), FloatParameter('minEigThreshold', 'minEigThreshold')]

    def process_inputs(self, inputs, outputs, parameters):
        prevImg = inputs['prevImg'].value
        nextImg = inputs['nextImg'].value
        prevPts = inputs['prevPts'].value
        winSize = parameters['winSize']
        maxLevel = parameters['maxLevel']
        flags = parameters['flags']
        minEigThreshold = parameters['minEigThreshold']
        status, err = cv2.calcOpticalFlowPyrLK(prevImg=prevImg, nextImg=nextImg, prevPts=prevPts, winSize=winSize,
                                               maxLevel=maxLevel, flags=flags, minEigThreshold=minEigThreshold)
        outputs['status'] = Data(status)
        outputs['err'] = Data(err)


class OpenCVAuto_cartToPolar(NormalElement):
    name = 'cartToPolar'
    comment = 'cartToPolar(x, y[, magnitude[, angle[, angleInDegrees]]]) -> magnitude, angle'

    def get_attributes(self):
        return [Input('x', 'x'), Input('y', 'y')], \
               [Output('magnitude', 'magnitude'), Output('angle', 'angle')], \
               [IntParameter('angleInDegrees', 'angleInDegrees')]

    def process_inputs(self, inputs, outputs, parameters):
        x = inputs['x'].value
        y = inputs['y'].value
        angleInDegrees = parameters['angleInDegrees']
        magnitude, angle = cv2.cartToPolar(x=x, y=y, angleInDegrees=angleInDegrees)
        outputs['magnitude'] = Data(magnitude)
        outputs['angle'] = Data(angle)


class OpenCVAuto_compare(NormalElement):
    name = 'compare'
    comment = 'compare(src1, src2, cmpop[, dst]) -> dst'

    def get_attributes(self):
        return [Input('src1', 'src1'), Input('src2', 'src2')], \
               [Output('dst', 'dst')], \
               [IntParameter('cmpop', 'cmpop')]

    def process_inputs(self, inputs, outputs, parameters):
        src1 = inputs['src1'].value
        src2 = inputs['src2'].value
        cmpop = parameters['cmpop']
        dst = cv2.compare(src1=src1, src2=src2, cmpop=cmpop)
        outputs['dst'] = Data(dst)


class OpenCVAuto_composeRT(NormalElement):
    name = 'composeRT'
    comment = 'composeRT(rvec1, tvec1, rvec2, tvec2[, rvec3[, tvec3[, dr3dr1[, dr3dt1[, dr3dr2[, dr3dt2[, dt3dr1[, ' \
              'dt3dt1[, dt3dr2[, dt3dt2]]]]]]]]]]) -> rvec3, tvec3, dr3dr1, dr3dt1, dr3dr2, dr3dt2, dt3dr1, dt3dt1, ' \
              'dt3dr2, dt3dt2'

    def get_attributes(self):
        return [Input('rvec1', 'rvec1'), Input('tvec1', 'tvec1'), Input('rvec2', 'rvec2'), Input('tvec2', 'tvec2')], \
               [Output('rvec3', 'rvec3'), Output('tvec3', 'tvec3'), Output('dr3dr1', 'dr3dr1'),
                Output('dr3dt1', 'dr3dt1'), Output('dr3dr2', 'dr3dr2'), Output('dr3dt2', 'dr3dt2'),
                Output('dt3dr1', 'dt3dr1'), Output('dt3dt1', 'dt3dt1'), Output('dt3dr2', 'dt3dr2'),
                Output('dt3dt2', 'dt3dt2')], \
            []

    def process_inputs(self, inputs, outputs, parameters):
        rvec1 = inputs['rvec1'].value
        tvec1 = inputs['tvec1'].value
        rvec2 = inputs['rvec2'].value
        tvec2 = inputs['tvec2'].value
        rvec3, tvec3, dr3dr1, dr3dt1, dr3dr2, dr3dt2, dt3dr1, dt3dt1, dt3dr2, dt3dt2 = cv2.composeRT(rvec1=rvec1,
                                                                                                     tvec1=tvec1,
                                                                                                     rvec2=rvec2,
                                                                                                     tvec2=tvec2)
        outputs['rvec3'] = Data(rvec3)
        outputs['tvec3'] = Data(tvec3)
        outputs['dr3dr1'] = Data(dr3dr1)
        outputs['dr3dt1'] = Data(dr3dt1)
        outputs['dr3dr2'] = Data(dr3dr2)
        outputs['dr3dt2'] = Data(dr3dt2)
        outputs['dt3dr1'] = Data(dt3dr1)
        outputs['dt3dt1'] = Data(dt3dt1)
        outputs['dt3dr2'] = Data(dt3dr2)
        outputs['dt3dt2'] = Data(dt3dt2)


class OpenCVAuto_computeCorrespondEpilines(NormalElement):
    name = 'computeCorrespondEpilines'
    comment = 'computeCorrespondEpilines(points, whichImage, F[, lines]) -> lines'

    def get_attributes(self):
        return [Input('points', 'points'), Input('F', 'F')], \
               [Output('lines', 'lines')], \
               [IntParameter('whichImage', 'whichImage')]

    def process_inputs(self, inputs, outputs, parameters):
        points = inputs['points'].value
        whichImage = parameters['whichImage']
        F = inputs['F'].value
        lines = cv2.computeCorrespondEpilines(points=points, whichImage=whichImage, F=F)
        outputs['lines'] = Data(lines)


class OpenCVAuto_convertMaps(NormalElement):
    name = 'convertMaps'
    comment = 'convertMaps(map1, map2, dstmap1type[, dstmap1[, dstmap2[, nninterpolation]]]) -> dstmap1, dstmap2'

    def get_attributes(self):
        return [Input('map1', 'map1'), Input('map2', 'map2')], \
               [Output('dstmap1', 'dstmap1'), Output('dstmap2', 'dstmap2')], \
               [IntParameter('dstmap1type', 'dstmap1type'), IntParameter('nninterpolation', 'nninterpolation')]

    def process_inputs(self, inputs, outputs, parameters):
        map1 = inputs['map1'].value
        map2 = inputs['map2'].value
        dstmap1type = parameters['dstmap1type']
        nninterpolation = parameters['nninterpolation']
        dstmap1, dstmap2 = cv2.convertMaps(map1=map1, map2=map2, dstmap1type=dstmap1type,
                                           nninterpolation=nninterpolation)
        outputs['dstmap1'] = Data(dstmap1)
        outputs['dstmap2'] = Data(dstmap2)


class OpenCVAuto_convertPointsFromHomogeneous(NormalElement):
    name = 'convertPointsFromHomogeneous'
    comment = 'convertPointsFromHomogeneous(src[, dst]) -> dst'

    def get_attributes(self):
        return [Input('src', 'src')], \
               [Output('dst', 'dst')], \
            []

    def process_inputs(self, inputs, outputs, parameters):
        src = inputs['src'].value
        dst = cv2.convertPointsFromHomogeneous(src=src)
        outputs['dst'] = Data(dst)


class OpenCVAuto_convertPointsToHomogeneous(NormalElement):
    name = 'convertPointsToHomogeneous'
    comment = 'convertPointsToHomogeneous(src[, dst]) -> dst'

    def get_attributes(self):
        return [Input('src', 'src')], \
               [Output('dst', 'dst')], \
            []

    def process_inputs(self, inputs, outputs, parameters):
        src = inputs['src'].value
        dst = cv2.convertPointsToHomogeneous(src=src)
        outputs['dst'] = Data(dst)


class OpenCVAuto_convertScaleAbs(NormalElement):
    name = 'convertScaleAbs'
    comment = 'convertScaleAbs(src[, dst[, alpha[, beta]]]) -> dst'

    def get_attributes(self):
        return [Input('src', 'src')], \
               [Output('dst', 'dst')], \
               [FloatParameter('alpha', 'alpha'), FloatParameter('beta', 'beta')]

    def process_inputs(self, inputs, outputs, parameters):
        src = inputs['src'].value
        alpha = parameters['alpha']
        beta = parameters['beta']
        dst = cv2.convertScaleAbs(src=src, alpha=alpha, beta=beta)
        outputs['dst'] = Data(dst)


class OpenCVAuto_convexHull(NormalElement):
    name = 'convexHull'
    comment = 'convexHull(points[, hull[, clockwise[, returnPoints]]]) -> hull'

    def get_attributes(self):
        return [Input('points', 'points')], \
               [Output('hull', 'hull')], \
               [IntParameter('clockwise', 'clockwise'), IntParameter('returnPoints', 'returnPoints')]

    def process_inputs(self, inputs, outputs, parameters):
        points = inputs['points'].value
        clockwise = parameters['clockwise']
        returnPoints = parameters['returnPoints']
        hull = cv2.convexHull(points=points, clockwise=clockwise, returnPoints=returnPoints)
        outputs['hull'] = Data(hull)


class OpenCVAuto_convexityDefects(NormalElement):
    name = 'convexityDefects'
    comment = 'convexityDefects(contour, convexhull[, convexityDefects]) -> convexityDefects'

    def get_attributes(self):
        return [Input('contour', 'contour'), Input('convexhull', 'convexhull')], \
               [Output('convexityDefects', 'convexityDefects')], \
            []

    def process_inputs(self, inputs, outputs, parameters):
        contour = inputs['contour'].value
        convexhull = inputs['convexhull'].value
        convexityDefects = cv2.convexityDefects(contour=contour, convexhull=convexhull)
        outputs['convexityDefects'] = Data(convexityDefects)


class OpenCVAuto_copyMakeBorder(NormalElement):
    name = 'copyMakeBorder'
    comment = 'copyMakeBorder(src, top, bottom, left, right, borderType[, dst[, value]]) -> dst'

    def get_attributes(self):
        return [Input('src', 'src')], \
               [Output('dst', 'dst')], \
               [IntParameter('top', 'top'), IntParameter('bottom', 'bottom'), IntParameter('left', 'left'),
                IntParameter('right', 'right'), IntParameter('borderType', 'borderType')]

    def process_inputs(self, inputs, outputs, parameters):
        src = inputs['src'].value
        top = parameters['top']
        bottom = parameters['bottom']
        left = parameters['left']
        right = parameters['right']
        borderType = parameters['borderType']
        dst = cv2.copyMakeBorder(src=src, top=top, bottom=bottom, left=left, right=right, borderType=borderType)
        outputs['dst'] = Data(dst)


class OpenCVAuto_cornerEigenValsAndVecs(NormalElement):
    name = 'cornerEigenValsAndVecs'
    comment = 'cornerEigenValsAndVecs(src, blockSize, ksize[, dst[, borderType]]) -> dst'

    def get_attributes(self):
        return [Input('src', 'src')], \
               [Output('dst', 'dst')], \
               [IntParameter('blockSize', 'blockSize'), IntParameter('ksize', 'ksize'),
                IntParameter('borderType', 'borderType')]

    def process_inputs(self, inputs, outputs, parameters):
        src = inputs['src'].value
        blockSize = parameters['blockSize']
        ksize = parameters['ksize']
        borderType = parameters['borderType']
        dst = cv2.cornerEigenValsAndVecs(src=src, blockSize=blockSize, ksize=ksize, borderType=borderType)
        outputs['dst'] = Data(dst)


class OpenCVAuto_cornerHarris(NormalElement):
    name = 'cornerHarris'
    comment = 'cornerHarris(src, blockSize, ksize, k[, dst[, borderType]]) -> dst'

    def get_attributes(self):
        return [Input('src', 'src')], \
               [Output('dst', 'dst')], \
               [IntParameter('blockSize', 'blockSize'), IntParameter('ksize', 'ksize'), FloatParameter('k', 'k'),
                IntParameter('borderType', 'borderType')]

    def process_inputs(self, inputs, outputs, parameters):
        src = inputs['src'].value
        blockSize = parameters['blockSize']
        ksize = parameters['ksize']
        k = parameters['k']
        borderType = parameters['borderType']
        dst = cv2.cornerHarris(src=src, blockSize=blockSize, ksize=ksize, k=k, borderType=borderType)
        outputs['dst'] = Data(dst)


class OpenCVAuto_cornerMinEigenVal(NormalElement):
    name = 'cornerMinEigenVal'
    comment = 'cornerMinEigenVal(src, blockSize[, dst[, ksize[, borderType]]]) -> dst'

    def get_attributes(self):
        return [Input('src', 'src')], \
               [Output('dst', 'dst')], \
               [IntParameter('blockSize', 'blockSize'), IntParameter('ksize', 'ksize'),
                IntParameter('borderType', 'borderType')]

    def process_inputs(self, inputs, outputs, parameters):
        src = inputs['src'].value
        blockSize = parameters['blockSize']
        ksize = parameters['ksize']
        borderType = parameters['borderType']
        dst = cv2.cornerMinEigenVal(src=src, blockSize=blockSize, ksize=ksize, borderType=borderType)
        outputs['dst'] = Data(dst)


class OpenCVAuto_correctMatches(NormalElement):
    name = 'correctMatches'
    comment = 'correctMatches(F, points1, points2[, newPoints1[, newPoints2]]) -> newPoints1, newPoints2'

    def get_attributes(self):
        return [Input('F', 'F'), Input('points1', 'points1'), Input('points2', 'points2')], \
               [Output('newPoints1', 'newPoints1'), Output('newPoints2', 'newPoints2')], \
            []

    def process_inputs(self, inputs, outputs, parameters):
        F = inputs['F'].value
        points1 = inputs['points1'].value
        points2 = inputs['points2'].value
        newPoints1, newPoints2 = cv2.correctMatches(F=F, points1=points1, points2=points2)
        outputs['newPoints1'] = Data(newPoints1)
        outputs['newPoints2'] = Data(newPoints2)


class OpenCVAuto_createHanningWindow(NormalElement):
    name = 'createHanningWindow'
    comment = 'createHanningWindow(winSize, type[, dst]) -> dst'

    def get_attributes(self):
        return [], \
            [Output('dst', 'dst')], \
            [SizeParameter('winSize', 'winSize'), IntParameter('type', 'type')]

    def process_inputs(self, inputs, outputs, parameters):
        winSize = parameters['winSize']
        type = parameters['type']
        dst = cv2.createHanningWindow(winSize=winSize, type=type)
        outputs['dst'] = Data(dst)


class OpenCVAuto_cvtColor(NormalElement):
    name = 'cvtColor'
    comment = 'cvtColor(src, code[, dst[, dstCn]]) -> dst'

    def get_attributes(self):
        return [Input('src', 'src')], \
               [Output('dst', 'dst')], \
               [IntParameter('code', 'code'), IntParameter('dstCn', 'dstCn')]

    def process_inputs(self, inputs, outputs, parameters):
        src = inputs['src'].value
        code = parameters['code']
        dstCn = parameters['dstCn']
        dst = cv2.cvtColor(src=src, code=code, dstCn=dstCn)
        outputs['dst'] = Data(dst)


class OpenCVAuto_dct(NormalElement):
    name = 'dct'
    comment = 'dct(src[, dst[, flags]]) -> dst'

    def get_attributes(self):
        return [Input('src', 'src')], \
               [Output('dst', 'dst')], \
               [IntParameter('flags', 'flags')]

    def process_inputs(self, inputs, outputs, parameters):
        src = inputs['src'].value
        flags = parameters['flags']
        dst = cv2.dct(src=src, flags=flags)
        outputs['dst'] = Data(dst)


class OpenCVAuto_decomposeProjectionMatrix(NormalElement):
    name = 'decomposeProjectionMatrix'
    comment = 'decomposeProjectionMatrix(projMatrix[, cameraMatrix[, rotMatrix[, transVect[, rotMatrixX[, rotMatrixY[' \
              ', rotMatrixZ[, eulerAngles]]]]]]]) -> cameraMatrix, rotMatrix, transVect, rotMatrixX, rotMatrixY, ' \
              'rotMatrixZ, eulerAngles'

    def get_attributes(self):
        return [Input('projMatrix', 'projMatrix')], \
               [Output('cameraMatrix', 'cameraMatrix'), Output('rotMatrix', 'rotMatrix'),
                Output('transVect', 'transVect'), Output('rotMatrixX', 'rotMatrixX'),
                Output('rotMatrixY', 'rotMatrixY'), Output('rotMatrixZ', 'rotMatrixZ'),
                Output('eulerAngles', 'eulerAngles')], \
            []

    def process_inputs(self, inputs, outputs, parameters):
        projMatrix = inputs['projMatrix'].value
        cameraMatrix, rotMatrix, transVect, rotMatrixX, rotMatrixY, rotMatrixZ, eulerAngles = cv2 \
            .decomposeProjectionMatrix(
            projMatrix=projMatrix)
        outputs['cameraMatrix'] = Data(cameraMatrix)
        outputs['rotMatrix'] = Data(rotMatrix)
        outputs['transVect'] = Data(transVect)
        outputs['rotMatrixX'] = Data(rotMatrixX)
        outputs['rotMatrixY'] = Data(rotMatrixY)
        outputs['rotMatrixZ'] = Data(rotMatrixZ)
        outputs['eulerAngles'] = Data(eulerAngles)


class OpenCVAuto_dft(NormalElement):
    name = 'dft'
    comment = 'dft(src[, dst[, flags[, nonzeroRows]]]) -> dst'

    def get_attributes(self):
        return [Input('src', 'src')], \
               [Output('dst', 'dst')], \
               [IntParameter('flags', 'flags'), IntParameter('nonzeroRows', 'nonzeroRows')]

    def process_inputs(self, inputs, outputs, parameters):
        src = inputs['src'].value
        flags = parameters['flags']
        nonzeroRows = parameters['nonzeroRows']
        dst = cv2.dft(src=src, flags=flags, nonzeroRows=nonzeroRows)
        outputs['dst'] = Data(dst)


class OpenCVAuto_dilate(NormalElement):
    name = 'dilate'
    comment = 'dilate(src, kernel[, dst[, anchor[, iterations[, borderType[, borderValue]]]]]) -> dst'

    def get_attributes(self):
        return [Input('src', 'src'), Input('kernel', 'kernel')], \
               [Output('dst', 'dst')], \
               [PointParameter('anchor', 'anchor'), IntParameter('iterations', 'iterations'),
                IntParameter('borderType', 'borderType')]

    def process_inputs(self, inputs, outputs, parameters):
        src = inputs['src'].value
        kernel = inputs['kernel'].value
        anchor = parameters['anchor']
        iterations = parameters['iterations']
        borderType = parameters['borderType']
        dst = cv2.dilate(src=src, kernel=kernel, anchor=anchor, iterations=iterations, borderType=borderType)
        outputs['dst'] = Data(dst)


class OpenCVAuto_distanceTransform(NormalElement):
    name = 'distanceTransform'
    comment = 'distanceTransform(src, distanceType, maskSize[, dst]) -> dst'

    def get_attributes(self):
        return [Input('src', 'src')], \
               [Output('dst', 'dst')], \
               [IntParameter('distanceType', 'distanceType'), IntParameter('maskSize', 'maskSize')]

    def process_inputs(self, inputs, outputs, parameters):
        src = inputs['src'].value
        distanceType = parameters['distanceType']
        maskSize = parameters['maskSize']
        dst = cv2.distanceTransform(src=src, distanceType=distanceType, maskSize=maskSize)
        outputs['dst'] = Data(dst)


class OpenCVAuto_distanceTransformWithLabels(NormalElement):
    name = 'distanceTransformWithLabels'
    comment = 'distanceTransformWithLabels(src, distanceType, maskSize[, dst[, labels[, labelType]]]) -> dst, labels'

    def get_attributes(self):
        return [Input('src', 'src')], \
               [Output('dst', 'dst'), Output('labels', 'labels')], \
               [IntParameter('distanceType', 'distanceType'), IntParameter('maskSize', 'maskSize'),
                IntParameter('labelType', 'labelType')]

    def process_inputs(self, inputs, outputs, parameters):
        src = inputs['src'].value
        distanceType = parameters['distanceType']
        maskSize = parameters['maskSize']
        labelType = parameters['labelType']
        dst, labels = cv2.distanceTransformWithLabels(src=src, distanceType=distanceType, maskSize=maskSize,
                                                      labelType=labelType)
        outputs['dst'] = Data(dst)
        outputs['labels'] = Data(labels)


class OpenCVAuto_divide(NormalElement):
    name = 'divide'
    comment = 'divide(src1, src2[, dst[, scale[, dtype]]]) -> dst'

    def get_attributes(self):
        return [Input('src1', 'src1'), Input('src2', 'src2')], \
               [Output('dst', 'dst')], \
               [FloatParameter('scale', 'scale'), IntParameter('dtype', 'dtype')]

    def process_inputs(self, inputs, outputs, parameters):
        src1 = inputs['src1'].value
        src2 = inputs['src2'].value
        scale = parameters['scale']
        dtype = parameters['dtype']
        dst = cv2.divide(src1=src1, src2=src2, scale=scale, dtype=dtype)
        outputs['dst'] = Data(dst)


class OpenCVAuto_divide(NormalElement):
    name = 'divide'
    comment = 'divide(scale, src2[, dst[, dtype]]) -> dst'

    def get_attributes(self):
        return [Input('src2', 'src2')], \
               [Output('dst', 'dst')], \
               [FloatParameter('scale', 'scale'), IntParameter('dtype', 'dtype')]

    def process_inputs(self, inputs, outputs, parameters):
        scale = parameters['scale']
        src2 = inputs['src2'].value
        dtype = parameters['dtype']
        dst = cv2.divide(scale=scale, src2=src2, dtype=dtype)
        outputs['dst'] = Data(dst)


class OpenCVAuto_drawKeypoints(NormalElement):
    name = 'drawKeypoints'
    comment = 'drawKeypoints(image, keypoints[, outImage[, color[, flags]]]) -> outImage'

    def get_attributes(self):
        return [Input('image', 'image')], \
               [Output('outImage', 'outImage')], \
               [IntParameter('flags', 'flags')]

    def process_inputs(self, inputs, outputs, parameters):
        image = inputs['image'].value
        flags = parameters['flags']
        outImage = cv2.drawKeypoints(image=image, keypoints=None, flags=flags)
        outputs['outImage'] = Data(outImage)


class OpenCVAuto_eigen(NormalElement):
    name = 'eigen'
    comment = 'eigen(src, computeEigenvectors[, eigenvalues[, eigenvectors]]) -> retval, eigenvalues, eigenvectors'

    def get_attributes(self):
        return [Input('src', 'src')], \
               [Output('eigenvalues', 'eigenvalues'), Output('eigenvectors', 'eigenvectors')], \
               [IntParameter('computeEigenvectors', 'computeEigenvectors')]

    def process_inputs(self, inputs, outputs, parameters):
        src = inputs['src'].value
        computeEigenvectors = parameters['computeEigenvectors']
        retval, eigenvalues, eigenvectors = cv2.eigen(src=src, computeEigenvectors=computeEigenvectors)
        outputs['eigenvalues'] = Data(eigenvalues)
        outputs['eigenvectors'] = Data(eigenvectors)


class OpenCVAuto_equalizeHist(NormalElement):
    name = 'equalizeHist'
    comment = 'equalizeHist(src[, dst]) -> dst'

    def get_attributes(self):
        return [Input('src', 'src')], \
               [Output('dst', 'dst')], \
            []

    def process_inputs(self, inputs, outputs, parameters):
        src = inputs['src'].value
        dst = cv2.equalizeHist(src=src)
        outputs['dst'] = Data(dst)


class OpenCVAuto_erode(NormalElement):
    name = 'erode'
    comment = 'erode(src, kernel[, dst[, anchor[, iterations[, borderType[, borderValue]]]]]) -> dst'

    def get_attributes(self):
        return [Input('src', 'src'), Input('kernel', 'kernel')], \
               [Output('dst', 'dst')], \
               [PointParameter('anchor', 'anchor'), IntParameter('iterations', 'iterations'),
                IntParameter('borderType', 'borderType')]

    def process_inputs(self, inputs, outputs, parameters):
        src = inputs['src'].value
        kernel = inputs['kernel'].value
        anchor = parameters['anchor']
        iterations = parameters['iterations']
        borderType = parameters['borderType']
        dst = cv2.erode(src=src, kernel=kernel, anchor=anchor, iterations=iterations, borderType=borderType)
        outputs['dst'] = Data(dst)


class OpenCVAuto_estimateAffine3D(NormalElement):
    name = 'estimateAffine3D'
    comment = 'estimateAffine3D(src, dst[, out[, inliers[, ransacThreshold[, confidence]]]]) -> retval, out, inliers'

    def get_attributes(self):
        return [Input('src', 'src'), Input('dst', 'dst')], \
               [Output('out', 'out'), Output('inliers', 'inliers')], \
               [FloatParameter('ransacThreshold', 'ransacThreshold'), FloatParameter('confidence', 'confidence')]

    def process_inputs(self, inputs, outputs, parameters):
        src = inputs['src'].value
        dst = inputs['dst'].value
        ransacThreshold = parameters['ransacThreshold']
        confidence = parameters['confidence']
        retval, out, inliers = cv2.estimateAffine3D(src=src, dst=dst, ransacThreshold=ransacThreshold,
                                                    confidence=confidence)
        outputs['out'] = Data(out)
        outputs['inliers'] = Data(inliers)


class OpenCVAuto_exp(NormalElement):
    name = 'exp'
    comment = 'exp(src[, dst]) -> dst'

    def get_attributes(self):
        return [Input('src', 'src')], \
               [Output('dst', 'dst')], \
            []

    def process_inputs(self, inputs, outputs, parameters):
        src = inputs['src'].value
        dst = cv2.exp(src=src)
        outputs['dst'] = Data(dst)


class OpenCVAuto_extractChannel(NormalElement):
    name = 'extractChannel'
    comment = 'extractChannel(src, coi[, dst]) -> dst'

    def get_attributes(self):
        return [Input('src', 'src')], \
               [Output('dst', 'dst')], \
               [IntParameter('coi', 'coi')]

    def process_inputs(self, inputs, outputs, parameters):
        src = inputs['src'].value
        coi = parameters['coi']
        dst = cv2.extractChannel(src=src, coi=coi)
        outputs['dst'] = Data(dst)


class OpenCVAuto_filter2D(NormalElement):
    name = 'filter2D'
    comment = 'filter2D(src, ddepth, kernel[, dst[, anchor[, delta[, borderType]]]]) -> dst'

    def get_attributes(self):
        return [Input('src', 'src'), Input('kernel', 'kernel')], \
               [Output('dst', 'dst')], \
               [IntParameter('ddepth', 'ddepth'), PointParameter('anchor', 'anchor'), FloatParameter('delta', 'delta'),
                IntParameter('borderType', 'borderType')]

    def process_inputs(self, inputs, outputs, parameters):
        src = inputs['src'].value
        ddepth = parameters['ddepth']
        kernel = inputs['kernel'].value
        anchor = parameters['anchor']
        delta = parameters['delta']
        borderType = parameters['borderType']
        dst = cv2.filter2D(src=src, ddepth=ddepth, kernel=kernel, anchor=anchor, delta=delta, borderType=borderType)
        outputs['dst'] = Data(dst)


class OpenCVAuto_findChessboardCorners(NormalElement):
    name = 'findChessboardCorners'
    comment = 'findChessboardCorners(image, patternSize[, corners[, flags]]) -> retval, corners'

    def get_attributes(self):
        return [Input('image', 'image')], \
               [Output('corners', 'corners')], \
               [SizeParameter('patternSize', 'patternSize'), IntParameter('flags', 'flags')]

    def process_inputs(self, inputs, outputs, parameters):
        image = inputs['image'].value
        patternSize = parameters['patternSize']
        flags = parameters['flags']
        retval, corners = cv2.findChessboardCorners(image=image, patternSize=patternSize, flags=flags)
        outputs['corners'] = Data(corners)


class OpenCVAuto_findCirclesGrid(NormalElement):
    name = 'findCirclesGrid'
    comment = 'findCirclesGrid(image, patternSize[, centers[, flags[, blobDetector]]]) -> retval, centers'

    def get_attributes(self):
        return [Input('image', 'image')], \
               [Output('centers', 'centers')], \
               [SizeParameter('patternSize', 'patternSize'), IntParameter('flags', 'flags')]

    def process_inputs(self, inputs, outputs, parameters):
        image = inputs['image'].value
        patternSize = parameters['patternSize']
        flags = parameters['flags']
        retval, centers = cv2.findCirclesGrid(image=image, patternSize=patternSize, flags=flags)
        outputs['centers'] = Data(centers)


class OpenCVAuto_findCirclesGridDefault(NormalElement):
    name = 'findCirclesGridDefault'
    comment = 'findCirclesGridDefault(image, patternSize[, centers[, flags]]) -> retval, centers'

    def get_attributes(self):
        return [Input('image', 'image')], \
               [Output('centers', 'centers')], \
               [SizeParameter('patternSize', 'patternSize'), IntParameter('flags', 'flags')]

    def process_inputs(self, inputs, outputs, parameters):
        image = inputs['image'].value
        patternSize = parameters['patternSize']
        flags = parameters['flags']
        retval, centers = cv2.findCirclesGridDefault(image=image, patternSize=patternSize, flags=flags)
        outputs['centers'] = Data(centers)


class OpenCVAuto_findContours(NormalElement):
    name = 'findContours'
    comment = 'findContours(image, mode, method[, contours[, hierarchy[, offset]]]) -> contours, hierarchy'

    def get_attributes(self):
        return [], \
            [Output('hierarchy', 'hierarchy')], \
            [IntParameter('mode', 'mode'), IntParameter('method', 'method'), PointParameter('offset', 'offset')]

    def process_inputs(self, inputs, outputs, parameters):
        mode = parameters['mode']
        method = parameters['method']
        offset = parameters['offset']
        hierarchy = cv2.findContours(image=None, mode=mode, method=method, offset=offset)
        outputs['hierarchy'] = Data(hierarchy)


class OpenCVAuto_findDataMatrix(NormalElement):
    name = 'findDataMatrix'
    comment = 'findDataMatrix(image[, corners[, dmtx]]) -> codes, corners, dmtx'

    def get_attributes(self):
        return [Input('image', 'image')], \
               [Output('corners', 'corners')], \
            []

    def process_inputs(self, inputs, outputs, parameters):
        image = inputs['image'].value
        corners = cv2.findDataMatrix(image=image)
        outputs['corners'] = Data(corners)


class OpenCVAuto_findFundamentalMat(NormalElement):
    name = 'findFundamentalMat'
    comment = 'findFundamentalMat(points1, points2[, method[, param1[, param2[, mask]]]]) -> retval, mask'

    def get_attributes(self):
        return [Input('points1', 'points1'), Input('points2', 'points2')], \
               [Output('mask', 'mask')], \
               [IntParameter('method', 'method'), FloatParameter('param1', 'param1'),
                FloatParameter('param2', 'param2')]

    def process_inputs(self, inputs, outputs, parameters):
        points1 = inputs['points1'].value
        points2 = inputs['points2'].value
        method = parameters['method']
        param1 = parameters['param1']
        param2 = parameters['param2']
        retval, mask = cv2.findFundamentalMat(points1=points1, points2=points2, method=method, param1=param1,
                                              param2=param2)
        outputs['mask'] = Data(mask)


class OpenCVAuto_findHomography(NormalElement):
    name = 'findHomography'
    comment = 'findHomography(srcPoints, dstPoints[, method[, ransacReprojThreshold[, mask]]]) -> retval, mask'

    def get_attributes(self):
        return [Input('srcPoints', 'srcPoints'), Input('dstPoints', 'dstPoints')], \
               [Output('mask', 'mask')], \
               [IntParameter('method', 'method'), FloatParameter('ransacReprojThreshold', 'ransacReprojThreshold')]

    def process_inputs(self, inputs, outputs, parameters):
        srcPoints = inputs['srcPoints'].value
        dstPoints = inputs['dstPoints'].value
        method = parameters['method']
        ransacReprojThreshold = parameters['ransacReprojThreshold']
        retval, mask = cv2.findHomography(srcPoints=srcPoints, dstPoints=dstPoints, method=method,
                                          ransacReprojThreshold=ransacReprojThreshold)
        outputs['mask'] = Data(mask)


class OpenCVAuto_findNonZero(NormalElement):
    name = 'findNonZero'
    comment = 'findNonZero(src[, idx]) -> idx'

    def get_attributes(self):
        return [Input('src', 'src')], \
               [Output('idx', 'idx')], \
            []

    def process_inputs(self, inputs, outputs, parameters):
        src = inputs['src'].value
        idx = cv2.findNonZero(src=src)
        outputs['idx'] = Data(idx)


class OpenCVAuto_fitLine(NormalElement):
    name = 'fitLine'
    comment = 'fitLine(points, distType, param, reps, aeps[, line]) -> line'

    def get_attributes(self):
        return [Input('points', 'points')], \
               [Output('line', 'line')], \
               [IntParameter('distType', 'distType'), FloatParameter('param', 'param'), FloatParameter('reps', 'reps'),
                FloatParameter('aeps', 'aeps')]

    def process_inputs(self, inputs, outputs, parameters):
        points = inputs['points'].value
        distType = parameters['distType']
        param = parameters['param']
        reps = parameters['reps']
        aeps = parameters['aeps']
        line = cv2.fitLine(points=points, distType=distType, param=param, reps=reps, aeps=aeps)
        outputs['line'] = Data(line)


class OpenCVAuto_flip(NormalElement):
    name = 'flip'
    comment = 'flip(src, flipCode[, dst]) -> dst'

    def get_attributes(self):
        return [Input('src', 'src')], \
               [Output('dst', 'dst')], \
               [IntParameter('flipCode', 'flipCode')]

    def process_inputs(self, inputs, outputs, parameters):
        src = inputs['src'].value
        flipCode = parameters['flipCode']
        dst = cv2.flip(src=src, flipCode=flipCode)
        outputs['dst'] = Data(dst)


class OpenCVAuto_gemm(NormalElement):
    name = 'gemm'
    comment = 'gemm(src1, src2, alpha, src3, gamma[, dst[, flags]]) -> dst'

    def get_attributes(self):
        return [Input('src1', 'src1'), Input('src2', 'src2'), Input('src3', 'src3')], \
               [Output('dst', 'dst')], \
               [FloatParameter('alpha', 'alpha'), FloatParameter('gamma', 'gamma'), IntParameter('flags', 'flags')]

    def process_inputs(self, inputs, outputs, parameters):
        src1 = inputs['src1'].value
        src2 = inputs['src2'].value
        alpha = parameters['alpha']
        src3 = inputs['src3'].value
        gamma = parameters['gamma']
        flags = parameters['flags']
        dst = cv2.gemm(src1=src1, src2=src2, alpha=alpha, src3=src3, gamma=gamma, flags=flags)
        outputs['dst'] = Data(dst)


class OpenCVAuto_getDerivKernels(NormalElement):
    name = 'getDerivKernels'
    comment = 'getDerivKernels(dx, dy, ksize[, kx[, ky[, normalize[, ktype]]]]) -> kx, ky'

    def get_attributes(self):
        return [], \
            [Output('kx', 'kx'), Output('ky', 'ky')], \
            [IntParameter('dx', 'dx'), IntParameter('dy', 'dy'), IntParameter('ksize', 'ksize'),
             IntParameter('normalize', 'normalize'), IntParameter('ktype', 'ktype')]

    def process_inputs(self, inputs, outputs, parameters):
        dx = parameters['dx']
        dy = parameters['dy']
        ksize = parameters['ksize']
        normalize = parameters['normalize']
        ktype = parameters['ktype']
        kx, ky = cv2.getDerivKernels(dx=dx, dy=dy, ksize=ksize, normalize=normalize, ktype=ktype)
        outputs['kx'] = Data(kx)
        outputs['ky'] = Data(ky)


class OpenCVAuto_getRectSubPix(NormalElement):
    name = 'getRectSubPix'
    comment = 'getRectSubPix(image, patchSize, center[, patch[, patchType]]) -> patch'

    def get_attributes(self):
        return [Input('image', 'image')], \
               [Output('patch', 'patch')], \
               [SizeParameter('patchSize', 'patchSize'), IntParameter('patchType', 'patchType')]

    def process_inputs(self, inputs, outputs, parameters):
        image = inputs['image'].value
        patchSize = parameters['patchSize']
        patchType = parameters['patchType']
        patch = cv2.getRectSubPix(image=image, patchSize=patchSize, center=None, patchType=patchType)
        outputs['patch'] = Data(patch)


class OpenCVAuto_goodFeaturesToTrack(NormalElement):
    name = 'goodFeaturesToTrack'
    comment = 'goodFeaturesToTrack(image, maxCorners, qualityLevel, minDistance[, corners[, mask[, blockSize[, ' \
              'useHarrisDetector[, k]]]]]) -> corners'

    def get_attributes(self):
        return [Input('image', 'image'), Input('mask', 'mask', optional=True)], \
               [Output('corners', 'corners')], \
               [IntParameter('maxCorners', 'maxCorners'), FloatParameter('qualityLevel', 'qualityLevel'),
                FloatParameter('minDistance', 'minDistance'), IntParameter('blockSize', 'blockSize'),
                IntParameter('useHarrisDetector', 'useHarrisDetector'), FloatParameter('k', 'k')]

    def process_inputs(self, inputs, outputs, parameters):
        image = inputs['image'].value
        maxCorners = parameters['maxCorners']
        qualityLevel = parameters['qualityLevel']
        minDistance = parameters['minDistance']
        mask = inputs['mask'].value
        blockSize = parameters['blockSize']
        useHarrisDetector = parameters['useHarrisDetector']
        k = parameters['k']
        corners = cv2.goodFeaturesToTrack(image=image, maxCorners=maxCorners, qualityLevel=qualityLevel,
                                          minDistance=minDistance, mask=mask, blockSize=blockSize,
                                          useHarrisDetector=useHarrisDetector, k=k)
        outputs['corners'] = Data(corners)


class OpenCVAuto_hconcat(NormalElement):
    name = 'hconcat'
    comment = 'hconcat(src[, dst]) -> dst'

    def get_attributes(self):
        return [], \
            [Output('dst', 'dst')], \
            []

    def process_inputs(self, inputs, outputs, parameters):
        dst = cv2.hconcat(src=None)
        outputs['dst'] = Data(dst)


class OpenCVAuto_idct(NormalElement):
    name = 'idct'
    comment = 'idct(src[, dst[, flags]]) -> dst'

    def get_attributes(self):
        return [Input('src', 'src')], \
               [Output('dst', 'dst')], \
               [IntParameter('flags', 'flags')]

    def process_inputs(self, inputs, outputs, parameters):
        src = inputs['src'].value
        flags = parameters['flags']
        dst = cv2.idct(src=src, flags=flags)
        outputs['dst'] = Data(dst)


class OpenCVAuto_idft(NormalElement):
    name = 'idft'
    comment = 'idft(src[, dst[, flags[, nonzeroRows]]]) -> dst'

    def get_attributes(self):
        return [Input('src', 'src')], \
               [Output('dst', 'dst')], \
               [IntParameter('flags', 'flags'), IntParameter('nonzeroRows', 'nonzeroRows')]

    def process_inputs(self, inputs, outputs, parameters):
        src = inputs['src'].value
        flags = parameters['flags']
        nonzeroRows = parameters['nonzeroRows']
        dst = cv2.idft(src=src, flags=flags, nonzeroRows=nonzeroRows)
        outputs['dst'] = Data(dst)


class OpenCVAuto_inRange(NormalElement):
    name = 'inRange'
    comment = 'inRange(src, lowerb, upperb[, dst]) -> dst'

    def get_attributes(self):
        return [Input('src', 'src'), Input('lowerb', 'lowerb'), Input('upperb', 'upperb')], \
               [Output('dst', 'dst')], \
            []

    def process_inputs(self, inputs, outputs, parameters):
        src = inputs['src'].value
        lowerb = inputs['lowerb'].value
        upperb = inputs['upperb'].value
        dst = cv2.inRange(src=src, lowerb=lowerb, upperb=upperb)
        outputs['dst'] = Data(dst)


class OpenCVAuto_initUndistortRectifyMap(NormalElement):
    name = 'initUndistortRectifyMap'
    comment = 'initUndistortRectifyMap(cameraMatrix, distCoeffs, R, newCameraMatrix, size, m1type[, map1[, ' \
              'map2]]) -> map1, map2'

    def get_attributes(self):
        return [Input('cameraMatrix', 'cameraMatrix'), Input('distCoeffs', 'distCoeffs'), Input('R', 'R'),
                Input('newCameraMatrix', 'newCameraMatrix')], \
               [Output('map1', 'map1'), Output('map2', 'map2')], \
               [SizeParameter('size', 'size'), IntParameter('m1type', 'm1type')]

    def process_inputs(self, inputs, outputs, parameters):
        cameraMatrix = inputs['cameraMatrix'].value
        distCoeffs = inputs['distCoeffs'].value
        R = inputs['R'].value
        newCameraMatrix = inputs['newCameraMatrix'].value
        size = parameters['size']
        m1type = parameters['m1type']
        map1, map2 = cv2.initUndistortRectifyMap(cameraMatrix=cameraMatrix, distCoeffs=distCoeffs, R=R,
                                                 newCameraMatrix=newCameraMatrix, size=size, m1type=m1type)
        outputs['map1'] = Data(map1)
        outputs['map2'] = Data(map2)


class OpenCVAuto_initWideAngleProjMap(NormalElement):
    name = 'initWideAngleProjMap'
    comment = 'initWideAngleProjMap(cameraMatrix, distCoeffs, imageSize, destImageWidth, m1type[, map1[, map2[, ' \
              'projType[, alpha]]]]) -> retval, map1, map2'

    def get_attributes(self):
        return [Input('cameraMatrix', 'cameraMatrix'), Input('distCoeffs', 'distCoeffs')], \
               [Output('map1', 'map1'), Output('map2', 'map2')], \
               [SizeParameter('imageSize', 'imageSize'), IntParameter('destImageWidth', 'destImageWidth'),
                IntParameter('m1type', 'm1type'), IntParameter('projType', 'projType'),
                FloatParameter('alpha', 'alpha')]

    def process_inputs(self, inputs, outputs, parameters):
        cameraMatrix = inputs['cameraMatrix'].value
        distCoeffs = inputs['distCoeffs'].value
        imageSize = parameters['imageSize']
        destImageWidth = parameters['destImageWidth']
        m1type = parameters['m1type']
        projType = parameters['projType']
        alpha = parameters['alpha']
        retval, map1, map2 = cv2.initWideAngleProjMap(cameraMatrix=cameraMatrix, distCoeffs=distCoeffs,
                                                      imageSize=imageSize, destImageWidth=destImageWidth, m1type=m1type,
                                                      projType=projType, alpha=alpha)
        outputs['map1'] = Data(map1)
        outputs['map2'] = Data(map2)


class OpenCVAuto_integral(NormalElement):
    name = 'integral'
    comment = 'integral(src[, sum[, sdepth]]) -> sum'

    def get_attributes(self):
        return [Input('src', 'src')], \
               [Output('sum', 'sum')], \
               [IntParameter('sdepth', 'sdepth')]

    def process_inputs(self, inputs, outputs, parameters):
        src = inputs['src'].value
        sdepth = parameters['sdepth']
        sum = cv2.integral(src=src, sdepth=sdepth)
        outputs['sum'] = Data(sum)


class OpenCVAuto_integral2(NormalElement):
    name = 'integral2'
    comment = 'integral2(src[, sum[, sqsum[, sdepth]]]) -> sum, sqsum'

    def get_attributes(self):
        return [Input('src', 'src')], \
               [Output('sum', 'sum'), Output('sqsum', 'sqsum')], \
               [IntParameter('sdepth', 'sdepth')]

    def process_inputs(self, inputs, outputs, parameters):
        src = inputs['src'].value
        sdepth = parameters['sdepth']
        sum, sqsum = cv2.integral2(src=src, sdepth=sdepth)
        outputs['sum'] = Data(sum)
        outputs['sqsum'] = Data(sqsum)


class OpenCVAuto_integral3(NormalElement):
    name = 'integral3'
    comment = 'integral3(src[, sum[, sqsum[, tilted[, sdepth]]]]) -> sum, sqsum, tilted'

    def get_attributes(self):
        return [Input('src', 'src')], \
               [Output('sum', 'sum'), Output('sqsum', 'sqsum'), Output('tilted', 'tilted')], \
               [IntParameter('sdepth', 'sdepth')]

    def process_inputs(self, inputs, outputs, parameters):
        src = inputs['src'].value
        sdepth = parameters['sdepth']
        sum, sqsum, tilted = cv2.integral3(src=src, sdepth=sdepth)
        outputs['sum'] = Data(sum)
        outputs['sqsum'] = Data(sqsum)
        outputs['tilted'] = Data(tilted)


class OpenCVAuto_intersectConvexConvex(NormalElement):
    name = 'intersectConvexConvex'
    comment = 'intersectConvexConvex(_p1, _p2[, _p12[, handleNested]]) -> retval, _p12'

    def get_attributes(self):
        return [Input('_p1', '_p1'), Input('_p2', '_p2')], \
               [Output('_p12', '_p12')], \
               [IntParameter('handleNested', 'handleNested')]

    def process_inputs(self, inputs, outputs, parameters):
        _p1 = inputs['_p1'].value
        _p2 = inputs['_p2'].value
        handleNested = parameters['handleNested']
        retval, _p12 = cv2.intersectConvexConvex(_p1=_p1, _p2=_p2, handleNested=handleNested)
        outputs['_p12'] = Data(_p12)


class OpenCVAuto_invert(NormalElement):
    name = 'invert'
    comment = 'invert(src[, dst[, flags]]) -> retval, dst'

    def get_attributes(self):
        return [Input('src', 'src')], \
               [Output('dst', 'dst')], \
               [IntParameter('flags', 'flags')]

    def process_inputs(self, inputs, outputs, parameters):
        src = inputs['src'].value
        flags = parameters['flags']
        retval, dst = cv2.invert(src=src, flags=flags)
        outputs['dst'] = Data(dst)


class OpenCVAuto_invertAffineTransform(NormalElement):
    name = 'invertAffineTransform'
    comment = 'invertAffineTransform(M[, iM]) -> iM'

    def get_attributes(self):
        return [Input('M', 'M')], \
               [Output('iM', 'iM')], \
            []

    def process_inputs(self, inputs, outputs, parameters):
        M = inputs['M'].value
        iM = cv2.invertAffineTransform(M=M)
        outputs['iM'] = Data(iM)


class OpenCVAuto_kmeans(NormalElement):
    name = 'kmeans'
    comment = 'kmeans(data, K, criteria, attempts, flags[, bestLabels[, centers]]) -> retval, bestLabels, centers'

    def get_attributes(self):
        return [Input('data', 'data')], \
               [Output('centers', 'centers')], \
               [IntParameter('K', 'K'), IntParameter('attempts', 'attempts'), IntParameter('flags', 'flags')]

    def process_inputs(self, inputs, outputs, parameters):
        data = inputs['data'].value
        K = parameters['K']
        attempts = parameters['attempts']
        flags = parameters['flags']
        retval, centers = cv2.kmeans(data=data, K=K, criteria=None, attempts=attempts, flags=flags)
        outputs['centers'] = Data(centers)


class OpenCVAuto_log(NormalElement):
    name = 'log'
    comment = 'log(src[, dst]) -> dst'

    def get_attributes(self):
        return [Input('src', 'src')], \
               [Output('dst', 'dst')], \
            []

    def process_inputs(self, inputs, outputs, parameters):
        src = inputs['src'].value
        dst = cv2.log(src=src)
        outputs['dst'] = Data(dst)


class OpenCVAuto_magnitude(NormalElement):
    name = 'magnitude'
    comment = 'magnitude(x, y[, magnitude]) -> magnitude'

    def get_attributes(self):
        return [Input('x', 'x'), Input('y', 'y')], \
               [Output('magnitude', 'magnitude')], \
            []

    def process_inputs(self, inputs, outputs, parameters):
        x = inputs['x'].value
        y = inputs['y'].value
        magnitude = cv2.magnitude(x=x, y=y)
        outputs['magnitude'] = Data(magnitude)


class OpenCVAuto_matMulDeriv(NormalElement):
    name = 'matMulDeriv'
    comment = 'matMulDeriv(A, B[, dABdA[, dABdB]]) -> dABdA, dABdB'

    def get_attributes(self):
        return [Input('A', 'A'), Input('B', 'B')], \
               [Output('dABdA', 'dABdA'), Output('dABdB', 'dABdB')], \
            []

    def process_inputs(self, inputs, outputs, parameters):
        A = inputs['A'].value
        B = inputs['B'].value
        dABdA, dABdB = cv2.matMulDeriv(A=A, B=B)
        outputs['dABdA'] = Data(dABdA)
        outputs['dABdB'] = Data(dABdB)


class OpenCVAuto_matchTemplate(NormalElement):
    name = 'matchTemplate'
    comment = 'matchTemplate(image, templ, method[, result]) -> result'

    def get_attributes(self):
        return [Input('image', 'image'), Input('templ', 'templ')], \
               [Output('result', 'result')], \
               [IntParameter('method', 'method')]

    def process_inputs(self, inputs, outputs, parameters):
        image = inputs['image'].value
        templ = inputs['templ'].value
        method = parameters['method']
        result = cv2.matchTemplate(image=image, templ=templ, method=method)
        outputs['result'] = Data(result)


class OpenCVAuto_max(NormalElement):
    name = 'max'
    comment = 'max(src1, src2[, dst]) -> dst'

    def get_attributes(self):
        return [Input('src1', 'src1'), Input('src2', 'src2')], \
               [Output('dst', 'dst')], \
            []

    def process_inputs(self, inputs, outputs, parameters):
        src1 = inputs['src1'].value
        src2 = inputs['src2'].value
        dst = cv2.max(src1=src1, src2=src2)
        outputs['dst'] = Data(dst)


class OpenCVAuto_meanStdDev(NormalElement):
    name = 'meanStdDev'
    comment = 'meanStdDev(src[, mean[, stddev[, mask]]]) -> mean, stddev'

    def get_attributes(self):
        return [Input('src', 'src'), Input('mask', 'mask', optional=True)], \
               [Output('mean', 'mean'), Output('stddev', 'stddev')], \
            []

    def process_inputs(self, inputs, outputs, parameters):
        src = inputs['src'].value
        mask = inputs['mask'].value
        mean, stddev = cv2.meanStdDev(src=src, mask=mask)
        outputs['mean'] = Data(mean)
        outputs['stddev'] = Data(stddev)


class OpenCVAuto_medianBlur(NormalElement):
    name = 'medianBlur'
    comment = 'medianBlur(src, ksize[, dst]) -> dst'

    def get_attributes(self):
        return [Input('src', 'src')], \
               [Output('dst', 'dst')], \
               [IntParameter('ksize', 'ksize')]

    def process_inputs(self, inputs, outputs, parameters):
        src = inputs['src'].value
        ksize = parameters['ksize']
        dst = cv2.medianBlur(src=src, ksize=ksize)
        outputs['dst'] = Data(dst)


class OpenCVAuto_merge(NormalElement):
    name = 'merge'
    comment = 'merge(mv[, dst]) -> dst'

    def get_attributes(self):
        return [], \
            [Output('dst', 'dst')], \
            []

    def process_inputs(self, inputs, outputs, parameters):
        dst = cv2.merge(mv=None)
        outputs['dst'] = Data(dst)


class OpenCVAuto_min(NormalElement):
    name = 'min'
    comment = 'min(src1, src2[, dst]) -> dst'

    def get_attributes(self):
        return [Input('src1', 'src1'), Input('src2', 'src2')], \
               [Output('dst', 'dst')], \
            []

    def process_inputs(self, inputs, outputs, parameters):
        src1 = inputs['src1'].value
        src2 = inputs['src2'].value
        dst = cv2.min(src1=src1, src2=src2)
        outputs['dst'] = Data(dst)


class OpenCVAuto_morphologyEx(NormalElement):
    name = 'morphologyEx'
    comment = 'morphologyEx(src, op, kernel[, dst[, anchor[, iterations[, borderType[, borderValue]]]]]) -> dst'

    def get_attributes(self):
        return [Input('src', 'src'), Input('kernel', 'kernel')], \
               [Output('dst', 'dst')], \
               [IntParameter('op', 'op'), PointParameter('anchor', 'anchor'), IntParameter('iterations', 'iterations'),
                IntParameter('borderType', 'borderType')]

    def process_inputs(self, inputs, outputs, parameters):
        src = inputs['src'].value
        op = parameters['op']
        kernel = inputs['kernel'].value
        anchor = parameters['anchor']
        iterations = parameters['iterations']
        borderType = parameters['borderType']
        dst = cv2.morphologyEx(src=src, op=op, kernel=kernel, anchor=anchor, iterations=iterations,
                               borderType=borderType)
        outputs['dst'] = Data(dst)


class OpenCVAuto_mulSpectrums(NormalElement):
    name = 'mulSpectrums'
    comment = 'mulSpectrums(a, b, flags[, c[, conjB]]) -> c'

    def get_attributes(self):
        return [Input('a', 'a'), Input('b', 'b')], \
               [Output('c', 'c')], \
               [IntParameter('flags', 'flags'), IntParameter('conjB', 'conjB')]

    def process_inputs(self, inputs, outputs, parameters):
        a = inputs['a'].value
        b = inputs['b'].value
        flags = parameters['flags']
        conjB = parameters['conjB']
        c = cv2.mulSpectrums(a=a, b=b, flags=flags, conjB=conjB)
        outputs['c'] = Data(c)


class OpenCVAuto_mulTransposed(NormalElement):
    name = 'mulTransposed'
    comment = 'mulTransposed(src, aTa[, dst[, delta[, scale[, dtype]]]]) -> dst'

    def get_attributes(self):
        return [Input('src', 'src'), Input('delta', 'delta', optional=True)], \
               [Output('dst', 'dst')], \
               [IntParameter('aTa', 'aTa'), FloatParameter('scale', 'scale'), IntParameter('dtype', 'dtype')]

    def process_inputs(self, inputs, outputs, parameters):
        src = inputs['src'].value
        aTa = parameters['aTa']
        delta = inputs['delta'].value
        scale = parameters['scale']
        dtype = parameters['dtype']
        dst = cv2.mulTransposed(src=src, aTa=aTa, delta=delta, scale=scale, dtype=dtype)
        outputs['dst'] = Data(dst)


class OpenCVAuto_multiply(NormalElement):
    name = 'multiply'
    comment = 'multiply(src1, src2[, dst[, scale[, dtype]]]) -> dst'

    def get_attributes(self):
        return [Input('src1', 'src1'), Input('src2', 'src2')], \
               [Output('dst', 'dst')], \
               [FloatParameter('scale', 'scale'), IntParameter('dtype', 'dtype')]

    def process_inputs(self, inputs, outputs, parameters):
        src1 = inputs['src1'].value
        src2 = inputs['src2'].value
        scale = parameters['scale']
        dtype = parameters['dtype']
        dst = cv2.multiply(src1=src1, src2=src2, scale=scale, dtype=dtype)
        outputs['dst'] = Data(dst)


class OpenCVAuto_normalize(NormalElement):
    name = 'normalize'
    comment = 'normalize(src[, dst[, alpha[, beta[, norm_type[, dtype[, mask]]]]]]) -> dst'

    def get_attributes(self):
        return [Input('src', 'src'), Input('mask', 'mask', optional=True)], \
               [Output('dst', 'dst')], \
               [FloatParameter('alpha', 'alpha'), FloatParameter('beta', 'beta'),
                IntParameter('norm_type', 'norm_type'), IntParameter('dtype', 'dtype')]

    def process_inputs(self, inputs, outputs, parameters):
        src = inputs['src'].value
        alpha = parameters['alpha']
        beta = parameters['beta']
        norm_type = parameters['norm_type']
        dtype = parameters['dtype']
        mask = inputs['mask'].value
        dst = cv2.normalize(src=src, alpha=alpha, beta=beta, norm_type=norm_type, dtype=dtype, mask=mask, dst=None)
        outputs['dst'] = Data(dst)


class OpenCVAuto_perspectiveTransform(NormalElement):
    name = 'perspectiveTransform'
    comment = 'perspectiveTransform(src, m[, dst]) -> dst'

    def get_attributes(self):
        return [Input('src', 'src'), Input('m', 'm')], \
               [Output('dst', 'dst')], \
            []

    def process_inputs(self, inputs, outputs, parameters):
        src = inputs['src'].value
        m = inputs['m'].value
        dst = cv2.perspectiveTransform(src=src, m=m)
        outputs['dst'] = Data(dst)


class OpenCVAuto_phase(NormalElement):
    name = 'phase'
    comment = 'phase(x, y[, angle[, angleInDegrees]]) -> angle'

    def get_attributes(self):
        return [Input('x', 'x'), Input('y', 'y')], \
               [Output('angle', 'angle')], \
               [IntParameter('angleInDegrees', 'angleInDegrees')]

    def process_inputs(self, inputs, outputs, parameters):
        x = inputs['x'].value
        y = inputs['y'].value
        angleInDegrees = parameters['angleInDegrees']
        angle = cv2.phase(x=x, y=y, angleInDegrees=angleInDegrees)
        outputs['angle'] = Data(angle)


class OpenCVAuto_polarToCart(NormalElement):
    name = 'polarToCart'
    comment = 'polarToCart(magnitude, angle[, x[, y[, angleInDegrees]]]) -> x, y'

    def get_attributes(self):
        return [Input('magnitude', 'magnitude'), Input('angle', 'angle')], \
               [Output('x', 'x'), Output('y', 'y')], \
               [IntParameter('angleInDegrees', 'angleInDegrees')]

    def process_inputs(self, inputs, outputs, parameters):
        magnitude = inputs['magnitude'].value
        angle = inputs['angle'].value
        angleInDegrees = parameters['angleInDegrees']
        x, y = cv2.polarToCart(magnitude=magnitude, angle=angle, angleInDegrees=angleInDegrees)
        outputs['x'] = Data(x)
        outputs['y'] = Data(y)


class OpenCVAuto_pow(NormalElement):
    name = 'pow'
    comment = 'pow(src, power[, dst]) -> dst'

    def get_attributes(self):
        return [Input('src', 'src')], \
               [Output('dst', 'dst')], \
               [FloatParameter('power', 'power')]

    def process_inputs(self, inputs, outputs, parameters):
        src = inputs['src'].value
        power = parameters['power']
        dst = cv2.pow(src=src, power=power)
        outputs['dst'] = Data(dst)


class OpenCVAuto_preCornerDetect(NormalElement):
    name = 'preCornerDetect'
    comment = 'preCornerDetect(src, ksize[, dst[, borderType]]) -> dst'

    def get_attributes(self):
        return [Input('src', 'src')], \
               [Output('dst', 'dst')], \
               [IntParameter('ksize', 'ksize'), IntParameter('borderType', 'borderType')]

    def process_inputs(self, inputs, outputs, parameters):
        src = inputs['src'].value
        ksize = parameters['ksize']
        borderType = parameters['borderType']
        dst = cv2.preCornerDetect(src=src, ksize=ksize, borderType=borderType)
        outputs['dst'] = Data(dst)


class OpenCVAuto_projectPoints(NormalElement):
    name = 'projectPoints'
    comment = 'projectPoints(objectPoints, rvec, tvec, cameraMatrix, distCoeffs[, imagePoints[, jacobian[, ' \
              'aspectRatio]]]) -> imagePoints, jacobian'

    def get_attributes(self):
        return [Input('objectPoints', 'objectPoints'), Input('rvec', 'rvec'), Input('tvec', 'tvec'),
                Input('cameraMatrix', 'cameraMatrix'), Input('distCoeffs', 'distCoeffs')], \
               [Output('imagePoints', 'imagePoints'), Output('jacobian', 'jacobian')], \
               [FloatParameter('aspectRatio', 'aspectRatio')]

    def process_inputs(self, inputs, outputs, parameters):
        objectPoints = inputs['objectPoints'].value
        rvec = inputs['rvec'].value
        tvec = inputs['tvec'].value
        cameraMatrix = inputs['cameraMatrix'].value
        distCoeffs = inputs['distCoeffs'].value
        aspectRatio = parameters['aspectRatio']
        imagePoints, jacobian = cv2.projectPoints(objectPoints=objectPoints, rvec=rvec, tvec=tvec,
                                                  cameraMatrix=cameraMatrix, distCoeffs=distCoeffs,
                                                  aspectRatio=aspectRatio)
        outputs['imagePoints'] = Data(imagePoints)
        outputs['jacobian'] = Data(jacobian)


class OpenCVAuto_pyrDown(NormalElement):
    name = 'pyrDown'
    comment = 'pyrDown(src[, dst[, dstsize[, borderType]]]) -> dst'

    def get_attributes(self):
        return [Input('src', 'src')], \
               [Output('dst', 'dst')], \
               [SizeParameter('dstsize', 'dstsize'), IntParameter('borderType', 'borderType')]

    def process_inputs(self, inputs, outputs, parameters):
        src = inputs['src'].value
        dstsize = parameters['dstsize']
        borderType = parameters['borderType']
        dst = cv2.pyrDown(src=src, dstsize=dstsize, borderType=borderType)
        outputs['dst'] = Data(dst)


class OpenCVAuto_pyrMeanShiftFiltering(NormalElement):
    name = 'pyrMeanShiftFiltering'
    comment = 'pyrMeanShiftFiltering(src, sp, sr[, dst[, maxLevel[, termcrit]]]) -> dst'

    def get_attributes(self):
        return [Input('src', 'src')], \
               [Output('dst', 'dst')], \
               [FloatParameter('sp', 'sp'), FloatParameter('sr', 'sr'), IntParameter('maxLevel', 'maxLevel')]

    def process_inputs(self, inputs, outputs, parameters):
        src = inputs['src'].value
        sp = parameters['sp']
        sr = parameters['sr']
        maxLevel = parameters['maxLevel']
        dst = cv2.pyrMeanShiftFiltering(src=src, sp=sp, sr=sr, maxLevel=maxLevel)
        outputs['dst'] = Data(dst)


class OpenCVAuto_pyrUp(NormalElement):
    name = 'pyrUp'
    comment = 'pyrUp(src[, dst[, dstsize[, borderType]]]) -> dst'

    def get_attributes(self):
        return [Input('src', 'src')], \
               [Output('dst', 'dst')], \
               [SizeParameter('dstsize', 'dstsize'), IntParameter('borderType', 'borderType')]

    def process_inputs(self, inputs, outputs, parameters):
        src = inputs['src'].value
        dstsize = parameters['dstsize']
        borderType = parameters['borderType']
        dst = cv2.pyrUp(src=src, dstsize=dstsize, borderType=borderType)
        outputs['dst'] = Data(dst)


class OpenCVAuto_rectify3Collinear(NormalElement):
    name = 'rectify3Collinear'
    comment = 'rectify3Collinear(cameraMatrix1, distCoeffs1, cameraMatrix2, distCoeffs2, cameraMatrix3, distCoeffs3, ' \
              'imgpt1, imgpt3, imageSize, R12, T12, R13, T13, alpha, newImgSize, flags[, R1[, R2[, R3[, P1[, P2[, ' \
              'P3[, Q]]]]]]]) -> retval, R1, R2, R3, P1, P2, P3, Q, roi1, roi2'

    def get_attributes(self):
        return [Input('cameraMatrix1', 'cameraMatrix1'), Input('distCoeffs1', 'distCoeffs1'),
                Input('cameraMatrix2', 'cameraMatrix2'), Input('distCoeffs2', 'distCoeffs2'),
                Input('cameraMatrix3', 'cameraMatrix3'), Input('distCoeffs3', 'distCoeffs3'), Input('R12', 'R12'),
                Input('T12', 'T12'), Input('R13', 'R13'), Input('T13', 'T13')], \
               [Output('R1', 'R1'), Output('R2', 'R2'), Output('R3', 'R3'), Output('P1', 'P1'), Output('P2', 'P2'),
                Output('P3', 'P3'), Output('Q', 'Q')], \
               [SizeParameter('imageSize', 'imageSize'), FloatParameter('alpha', 'alpha'),
                SizeParameter('newImgSize', 'newImgSize'), IntParameter('flags', 'flags')]

    def process_inputs(self, inputs, outputs, parameters):
        cameraMatrix1 = inputs['cameraMatrix1'].value
        distCoeffs1 = inputs['distCoeffs1'].value
        cameraMatrix2 = inputs['cameraMatrix2'].value
        distCoeffs2 = inputs['distCoeffs2'].value
        cameraMatrix3 = inputs['cameraMatrix3'].value
        distCoeffs3 = inputs['distCoeffs3'].value
        imageSize = parameters['imageSize']
        R12 = inputs['R12'].value
        T12 = inputs['T12'].value
        R13 = inputs['R13'].value
        T13 = inputs['T13'].value
        alpha = parameters['alpha']
        newImgSize = parameters['newImgSize']
        flags = parameters['flags']
        retval, R1, R2, R3, P1, P2, P3, Q = cv2.rectify3Collinear(cameraMatrix1=cameraMatrix1, distCoeffs1=distCoeffs1,
                                                                  cameraMatrix2=cameraMatrix2, distCoeffs2=distCoeffs2,
                                                                  cameraMatrix3=cameraMatrix3, distCoeffs3=distCoeffs3,
                                                                  imgpt1=None, imgpt3=None, imageSize=imageSize,
                                                                  R12=R12, T12=T12, R13=R13, T13=T13, alpha=alpha,
                                                                  newImgSize=newImgSize, flags=flags)
        outputs['R1'] = Data(R1)
        outputs['R2'] = Data(R2)
        outputs['R3'] = Data(R3)
        outputs['P1'] = Data(P1)
        outputs['P2'] = Data(P2)
        outputs['P3'] = Data(P3)
        outputs['Q'] = Data(Q)


class OpenCVAuto_reduce(NormalElement):
    name = 'reduce'
    comment = 'reduce(src, dim, rtype[, dst[, dtype]]) -> dst'

    def get_attributes(self):
        return [Input('src', 'src')], \
               [Output('dst', 'dst')], \
               [IntParameter('dim', 'dim'), IntParameter('rtype', 'rtype'), IntParameter('dtype', 'dtype')]

    def process_inputs(self, inputs, outputs, parameters):
        src = inputs['src'].value
        dim = parameters['dim']
        rtype = parameters['rtype']
        dtype = parameters['dtype']
        dst = cv2.reduce(src=src, dim=dim, rtype=rtype, dtype=dtype)
        outputs['dst'] = Data(dst)


class OpenCVAuto_remap(NormalElement):
    name = 'remap'
    comment = 'remap(src, map1, map2, interpolation[, dst[, borderMode[, borderValue]]]) -> dst'

    def get_attributes(self):
        return [Input('src', 'src'), Input('map1', 'map1'), Input('map2', 'map2')], \
               [Output('dst', 'dst')], \
               [IntParameter('interpolation', 'interpolation'), IntParameter('borderMode', 'borderMode')]

    def process_inputs(self, inputs, outputs, parameters):
        src = inputs['src'].value
        map1 = inputs['map1'].value
        map2 = inputs['map2'].value
        interpolation = parameters['interpolation']
        borderMode = parameters['borderMode']
        dst = cv2.remap(src=src, map1=map1, map2=map2, interpolation=interpolation, borderMode=borderMode)
        outputs['dst'] = Data(dst)


class OpenCVAuto_repeat(NormalElement):
    name = 'repeat'
    comment = 'repeat(src, ny, nx[, dst]) -> dst'

    def get_attributes(self):
        return [Input('src', 'src')], \
               [Output('dst', 'dst')], \
               [IntParameter('ny', 'ny'), IntParameter('nx', 'nx')]

    def process_inputs(self, inputs, outputs, parameters):
        src = inputs['src'].value
        ny = parameters['ny']
        nx = parameters['nx']
        dst = cv2.repeat(src=src, ny=ny, nx=nx)
        outputs['dst'] = Data(dst)


class OpenCVAuto_reprojectImageTo3D(NormalElement):
    name = 'reprojectImageTo3D'
    comment = 'reprojectImageTo3D(disparity, Q[, _3dImage[, handleMissingValues[, ddepth]]]) -> _3dImage'

    def get_attributes(self):
        return [Input('disparity', 'disparity'), Input('Q', 'Q')], \
               [Output('_3dImage', '_3dImage')], \
               [IntParameter('handleMissingValues', 'handleMissingValues'), IntParameter('ddepth', 'ddepth')]

    def process_inputs(self, inputs, outputs, parameters):
        disparity = inputs['disparity'].value
        Q = inputs['Q'].value
        handleMissingValues = parameters['handleMissingValues']
        ddepth = parameters['ddepth']
        _3dImage = cv2.reprojectImageTo3D(disparity=disparity, Q=Q, handleMissingValues=handleMissingValues,
                                          ddepth=ddepth)
        outputs['_3dImage'] = Data(_3dImage)


class OpenCVAuto_resize(NormalElement):
    name = 'resize'
    comment = 'resize(src, dsize[, dst[, fx[, fy[, interpolation]]]]) -> dst'

    def get_attributes(self):
        return [Input('src', 'src')], \
               [Output('dst', 'dst')], \
               [SizeParameter('dsize', 'dsize'), FloatParameter('fx', 'fx'), FloatParameter('fy', 'fy'),
                IntParameter('interpolation', 'interpolation')]

    def process_inputs(self, inputs, outputs, parameters):
        src = inputs['src'].value
        dsize = parameters['dsize']
        fx = parameters['fx']
        fy = parameters['fy']
        interpolation = parameters['interpolation']
        dst = cv2.resize(src=src, dsize=dsize, fx=fx, fy=fy, interpolation=interpolation)
        outputs['dst'] = Data(dst)


class OpenCVAuto_scaleAdd(NormalElement):
    name = 'scaleAdd'
    comment = 'scaleAdd(src1, alpha, src2[, dst]) -> dst'

    def get_attributes(self):
        return [Input('src1', 'src1'), Input('src2', 'src2')], \
               [Output('dst', 'dst')], \
               [FloatParameter('alpha', 'alpha')]

    def process_inputs(self, inputs, outputs, parameters):
        src1 = inputs['src1'].value
        alpha = parameters['alpha']
        src2 = inputs['src2'].value
        dst = cv2.scaleAdd(src1=src1, alpha=alpha, src2=src2)
        outputs['dst'] = Data(dst)


class OpenCVAuto_segmentMotion(NormalElement):
    name = 'segmentMotion'
    comment = 'segmentMotion(mhi, timestamp, segThresh[, segmask]) -> segmask, boundingRects'

    def get_attributes(self):
        return [Input('mhi', 'mhi')], \
               [Output('segmask', 'segmask')], \
               [FloatParameter('timestamp', 'timestamp'), FloatParameter('segThresh', 'segThresh')]

    def process_inputs(self, inputs, outputs, parameters):
        mhi = inputs['mhi'].value
        timestamp = parameters['timestamp']
        segThresh = parameters['segThresh']
        segmask = cv2.segmentMotion(mhi=mhi, timestamp=timestamp, segThresh=segThresh)
        outputs['segmask'] = Data(segmask)


class OpenCVAuto_sepFilter2D(NormalElement):
    name = 'sepFilter2D'
    comment = 'sepFilter2D(src, ddepth, kernelX, kernelY[, dst[, anchor[, delta[, borderType]]]]) -> dst'

    def get_attributes(self):
        return [Input('src', 'src'), Input('kernelX', 'kernelX'), Input('kernelY', 'kernelY')], \
               [Output('dst', 'dst')], \
               [IntParameter('ddepth', 'ddepth'), PointParameter('anchor', 'anchor'), FloatParameter('delta', 'delta'),
                IntParameter('borderType', 'borderType')]

    def process_inputs(self, inputs, outputs, parameters):
        src = inputs['src'].value
        ddepth = parameters['ddepth']
        kernelX = inputs['kernelX'].value
        kernelY = inputs['kernelY'].value
        anchor = parameters['anchor']
        delta = parameters['delta']
        borderType = parameters['borderType']
        dst = cv2.sepFilter2D(src=src, ddepth=ddepth, kernelX=kernelX, kernelY=kernelY, anchor=anchor, delta=delta,
                              borderType=borderType)
        outputs['dst'] = Data(dst)


class OpenCVAuto_solve(NormalElement):
    name = 'solve'
    comment = 'solve(src1, src2[, dst[, flags]]) -> retval, dst'

    def get_attributes(self):
        return [Input('src1', 'src1'), Input('src2', 'src2')], \
               [Output('dst', 'dst')], \
               [IntParameter('flags', 'flags')]

    def process_inputs(self, inputs, outputs, parameters):
        src1 = inputs['src1'].value
        src2 = inputs['src2'].value
        flags = parameters['flags']
        retval, dst = cv2.solve(src1=src1, src2=src2, flags=flags)
        outputs['dst'] = Data(dst)


class OpenCVAuto_solveCubic(NormalElement):
    name = 'solveCubic'
    comment = 'solveCubic(coeffs[, roots]) -> retval, roots'

    def get_attributes(self):
        return [Input('coeffs', 'coeffs')], \
               [Output('roots', 'roots')], \
            []

    def process_inputs(self, inputs, outputs, parameters):
        coeffs = inputs['coeffs'].value
        retval, roots = cv2.solveCubic(coeffs=coeffs)
        outputs['roots'] = Data(roots)


class OpenCVAuto_solvePnP(NormalElement):
    name = 'solvePnP'
    comment = 'solvePnP(objectPoints, imagePoints, cameraMatrix, distCoeffs[, rvec[, tvec[, useExtrinsicGuess[, ' \
              'flags]]]]) -> retval, rvec, tvec'

    def get_attributes(self):
        return [Input('objectPoints', 'objectPoints'), Input('imagePoints', 'imagePoints'),
                Input('cameraMatrix', 'cameraMatrix'), Input('distCoeffs', 'distCoeffs')], \
               [Output('rvec', 'rvec'), Output('tvec', 'tvec')], \
               [IntParameter('useExtrinsicGuess', 'useExtrinsicGuess'), IntParameter('flags', 'flags')]

    def process_inputs(self, inputs, outputs, parameters):
        objectPoints = inputs['objectPoints'].value
        imagePoints = inputs['imagePoints'].value
        cameraMatrix = inputs['cameraMatrix'].value
        distCoeffs = inputs['distCoeffs'].value
        useExtrinsicGuess = parameters['useExtrinsicGuess']
        flags = parameters['flags']
        retval, rvec, tvec = cv2.solvePnP(objectPoints=objectPoints, imagePoints=imagePoints, cameraMatrix=cameraMatrix,
                                          distCoeffs=distCoeffs, useExtrinsicGuess=useExtrinsicGuess, flags=flags)
        outputs['rvec'] = Data(rvec)
        outputs['tvec'] = Data(tvec)


class OpenCVAuto_solvePnPRansac(NormalElement):
    name = 'solvePnPRansac'
    comment = 'solvePnPRansac(objectPoints, imagePoints, cameraMatrix, distCoeffs[, rvec[, tvec[, useExtrinsicGuess[, ' \
              '' \
              '' \
              'iterationsCount[, reprojectionError[, minInliersCount[, inliers[, flags]]]]]]]]) -> rvec, tvec, inliers'

    def get_attributes(self):
        return [Input('objectPoints', 'objectPoints'), Input('imagePoints', 'imagePoints'),
                Input('cameraMatrix', 'cameraMatrix'), Input('distCoeffs', 'distCoeffs')], \
               [Output('rvec', 'rvec'), Output('tvec', 'tvec'), Output('inliers', 'inliers')], \
               [IntParameter('useExtrinsicGuess', 'useExtrinsicGuess'),
                IntParameter('iterationsCount', 'iterationsCount'),
                FloatParameter('reprojectionError', 'reprojectionError'),
                IntParameter('minInliersCount', 'minInliersCount'), IntParameter('flags', 'flags')]

    def process_inputs(self, inputs, outputs, parameters):
        objectPoints = inputs['objectPoints'].value
        imagePoints = inputs['imagePoints'].value
        cameraMatrix = inputs['cameraMatrix'].value
        distCoeffs = inputs['distCoeffs'].value
        useExtrinsicGuess = parameters['useExtrinsicGuess']
        iterationsCount = parameters['iterationsCount']
        reprojectionError = parameters['reprojectionError']
        minInliersCount = parameters['minInliersCount']
        flags = parameters['flags']
        rvec, tvec, inliers = cv2.solvePnPRansac(objectPoints=objectPoints, imagePoints=imagePoints,
                                                 cameraMatrix=cameraMatrix, distCoeffs=distCoeffs,
                                                 useExtrinsicGuess=useExtrinsicGuess, iterationsCount=iterationsCount,
                                                 reprojectionError=reprojectionError, minInliersCount=minInliersCount,
                                                 flags=flags)
        outputs['rvec'] = Data(rvec)
        outputs['tvec'] = Data(tvec)
        outputs['inliers'] = Data(inliers)


class OpenCVAuto_solvePoly(NormalElement):
    name = 'solvePoly'
    comment = 'solvePoly(coeffs[, roots[, maxIters]]) -> retval, roots'

    def get_attributes(self):
        return [Input('coeffs', 'coeffs')], \
               [Output('roots', 'roots')], \
               [IntParameter('maxIters', 'maxIters')]

    def process_inputs(self, inputs, outputs, parameters):
        coeffs = inputs['coeffs'].value
        maxIters = parameters['maxIters']
        retval, roots = cv2.solvePoly(coeffs=coeffs, maxIters=maxIters)
        outputs['roots'] = Data(roots)


class OpenCVAuto_sort(NormalElement):
    name = 'sort'
    comment = 'sort(src, flags[, dst]) -> dst'

    def get_attributes(self):
        return [Input('src', 'src')], \
               [Output('dst', 'dst')], \
               [IntParameter('flags', 'flags')]

    def process_inputs(self, inputs, outputs, parameters):
        src = inputs['src'].value
        flags = parameters['flags']
        dst = cv2.sort(src=src, flags=flags)
        outputs['dst'] = Data(dst)


class OpenCVAuto_sortIdx(NormalElement):
    name = 'sortIdx'
    comment = 'sortIdx(src, flags[, dst]) -> dst'

    def get_attributes(self):
        return [Input('src', 'src')], \
               [Output('dst', 'dst')], \
               [IntParameter('flags', 'flags')]

    def process_inputs(self, inputs, outputs, parameters):
        src = inputs['src'].value
        flags = parameters['flags']
        dst = cv2.sortIdx(src=src, flags=flags)
        outputs['dst'] = Data(dst)


class OpenCVAuto_sqrt(NormalElement):
    name = 'sqrt'
    comment = 'sqrt(src[, dst]) -> dst'

    def get_attributes(self):
        return [Input('src', 'src')], \
               [Output('dst', 'dst')], \
            []

    def process_inputs(self, inputs, outputs, parameters):
        src = inputs['src'].value
        dst = cv2.sqrt(src=src)
        outputs['dst'] = Data(dst)


class OpenCVAuto_stereoCalibrate(NormalElement):
    name = 'stereoCalibrate'
    comment = 'stereoCalibrate(objectPoints, imagePoints1, imagePoints2, imageSize[, cameraMatrix1[, distCoeffs1[, ' \
              'cameraMatrix2[, distCoeffs2[, R[, T[, E[, F[, criteria[, flags]]]]]]]]]]) -> retval, cameraMatrix1, ' \
              'distCoeffs1, cameraMatrix2, distCoeffs2, R, T, E, F'

    def get_attributes(self):
        return [], \
            [Output('R', 'R'), Output('T', 'T'), Output('E', 'E'), Output('F', 'F')], \
            [SizeParameter('imageSize', 'imageSize'), IntParameter('flags', 'flags')]

    def process_inputs(self, inputs, outputs, parameters):
        imageSize = parameters['imageSize']
        flags = parameters['flags']
        retval, R, T, E, F = cv2.stereoCalibrate(objectPoints=None, imagePoints1=None, imagePoints2=None,
                                                 imageSize=imageSize, flags=flags)
        outputs['R'] = Data(R)
        outputs['T'] = Data(T)
        outputs['E'] = Data(E)
        outputs['F'] = Data(F)


class OpenCVAuto_stereoRectify(NormalElement):
    name = 'stereoRectify'
    comment = 'stereoRectify(cameraMatrix1, distCoeffs1, cameraMatrix2, distCoeffs2, imageSize, R, T[, R1[, R2[, P1[, ' \
              '' \
              '' \
              'P2[, Q[, flags[, alpha[, newImageSize]]]]]]]]) -> R1, R2, P1, P2, Q, validPixROI1, validPixROI2'

    def get_attributes(self):
        return [Input('cameraMatrix1', 'cameraMatrix1'), Input('distCoeffs1', 'distCoeffs1'),
                Input('cameraMatrix2', 'cameraMatrix2'), Input('distCoeffs2', 'distCoeffs2'), Input('R', 'R'),
                Input('T', 'T')], \
               [Output('R1', 'R1'), Output('R2', 'R2'), Output('P1', 'P1'), Output('P2', 'P2'), Output('Q', 'Q')], \
               [SizeParameter('imageSize', 'imageSize'), IntParameter('flags', 'flags'),
                FloatParameter('alpha', 'alpha'), SizeParameter('newImageSize', 'newImageSize')]

    def process_inputs(self, inputs, outputs, parameters):
        cameraMatrix1 = inputs['cameraMatrix1'].value
        distCoeffs1 = inputs['distCoeffs1'].value
        cameraMatrix2 = inputs['cameraMatrix2'].value
        distCoeffs2 = inputs['distCoeffs2'].value
        imageSize = parameters['imageSize']
        R = inputs['R'].value
        T = inputs['T'].value
        flags = parameters['flags']
        alpha = parameters['alpha']
        newImageSize = parameters['newImageSize']
        R1, R2, P1, P2, Q = cv2.stereoRectify(cameraMatrix1=cameraMatrix1, distCoeffs1=distCoeffs1,
                                              cameraMatrix2=cameraMatrix2, distCoeffs2=distCoeffs2, imageSize=imageSize,
                                              R=R, T=T, flags=flags, alpha=alpha, newImageSize=newImageSize)
        outputs['R1'] = Data(R1)
        outputs['R2'] = Data(R2)
        outputs['P1'] = Data(P1)
        outputs['P2'] = Data(P2)
        outputs['Q'] = Data(Q)


class OpenCVAuto_stereoRectifyUncalibrated(NormalElement):
    name = 'stereoRectifyUncalibrated'
    comment = 'stereoRectifyUncalibrated(points1, points2, F, imgSize[, H1[, H2[, threshold]]]) -> retval, H1, H2'

    def get_attributes(self):
        return [Input('points1', 'points1'), Input('points2', 'points2'), Input('F', 'F')], \
               [Output('H1', 'H1'), Output('H2', 'H2')], \
               [SizeParameter('imgSize', 'imgSize'), FloatParameter('threshold', 'threshold')]

    def process_inputs(self, inputs, outputs, parameters):
        points1 = inputs['points1'].value
        points2 = inputs['points2'].value
        F = inputs['F'].value
        imgSize = parameters['imgSize']
        threshold = parameters['threshold']
        retval, H1, H2 = cv2.stereoRectifyUncalibrated(points1=points1, points2=points2, F=F, imgSize=imgSize,
                                                       threshold=threshold)
        outputs['H1'] = Data(H1)
        outputs['H2'] = Data(H2)


class OpenCVAuto_subtract(NormalElement):
    name = 'subtract'
    comment = 'subtract(src1, src2[, dst[, mask[, dtype]]]) -> dst'

    def get_attributes(self):
        return [Input('src1', 'src1'), Input('src2', 'src2'), Input('mask', 'mask', optional=True)], \
               [Output('dst', 'dst')], \
               [IntParameter('dtype', 'dtype')]

    def process_inputs(self, inputs, outputs, parameters):
        src1 = inputs['src1'].value
        src2 = inputs['src2'].value
        mask = inputs['mask'].value
        dtype = parameters['dtype']
        dst = cv2.subtract(src1=src1, src2=src2, mask=mask, dtype=dtype)
        outputs['dst'] = Data(dst)


class OpenCVAuto_threshold(NormalElement):
    name = 'threshold'
    comment = 'threshold(src, thresh, maxval, type[, dst]) -> retval, dst'

    def get_attributes(self):
        return [Input('src', 'src')], \
               [Output('dst', 'dst')], \
               [FloatParameter('thresh', 'thresh'), FloatParameter('maxval', 'maxval'), IntParameter('type', 'type')]

    def process_inputs(self, inputs, outputs, parameters):
        src = inputs['src'].value
        thresh = parameters['thresh']
        maxval = parameters['maxval']
        type = parameters['type']
        retval, dst = cv2.threshold(src=src, thresh=thresh, maxval=maxval, type=type)
        outputs['dst'] = Data(dst)


class OpenCVAuto_transform(NormalElement):
    name = 'transform'
    comment = 'transform(src, m[, dst]) -> dst'

    def get_attributes(self):
        return [Input('src', 'src'), Input('m', 'm')], \
               [Output('dst', 'dst')], \
            []

    def process_inputs(self, inputs, outputs, parameters):
        src = inputs['src'].value
        m = inputs['m'].value
        dst = cv2.transform(src=src, m=m)
        outputs['dst'] = Data(dst)


class OpenCVAuto_transpose(NormalElement):
    name = 'transpose'
    comment = 'transpose(src[, dst]) -> dst'

    def get_attributes(self):
        return [Input('src', 'src')], \
               [Output('dst', 'dst')], \
            []

    def process_inputs(self, inputs, outputs, parameters):
        src = inputs['src'].value
        dst = cv2.transpose(src=src)
        outputs['dst'] = Data(dst)


class OpenCVAuto_triangulatePoints(NormalElement):
    name = 'triangulatePoints'
    comment = 'triangulatePoints(projMatr1, projMatr2, projPoints1, projPoints2[, points4D]) -> points4D'

    def get_attributes(self):
        return [Input('projMatr1', 'projMatr1'), Input('projMatr2', 'projMatr2'), Input('projPoints1', 'projPoints1'),
                Input('projPoints2', 'projPoints2')], \
               [Output('points4D', 'points4D')], \
            []

    def process_inputs(self, inputs, outputs, parameters):
        projMatr1 = inputs['projMatr1'].value
        projMatr2 = inputs['projMatr2'].value
        projPoints1 = inputs['projPoints1'].value
        projPoints2 = inputs['projPoints2'].value
        points4D = cv2.triangulatePoints(projMatr1=projMatr1, projMatr2=projMatr2, projPoints1=projPoints1,
                                         projPoints2=projPoints2)
        outputs['points4D'] = Data(points4D)


class OpenCVAuto_undistort(NormalElement):
    name = 'undistort'
    comment = 'undistort(src, cameraMatrix, distCoeffs[, dst[, newCameraMatrix]]) -> dst'

    def get_attributes(self):
        return [Input('src', 'src'), Input('cameraMatrix', 'cameraMatrix'), Input('distCoeffs', 'distCoeffs'),
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


class OpenCVAuto_undistortPoints(NormalElement):
    name = 'undistortPoints'
    comment = 'undistortPoints(src, cameraMatrix, distCoeffs[, dst[, R[, P]]]) -> dst'

    def get_attributes(self):
        return [Input('src', 'src'), Input('cameraMatrix', 'cameraMatrix'), Input('distCoeffs', 'distCoeffs'),
                Input('R', 'R', optional=True), Input('P', 'P', optional=True)], \
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


class OpenCVAuto_vconcat(NormalElement):
    name = 'vconcat'
    comment = 'vconcat(src[, dst]) -> dst'

    def get_attributes(self):
        return [], \
            [Output('dst', 'dst')], \
            []

    def process_inputs(self, inputs, outputs, parameters):
        dst = cv2.vconcat(src=None)
        outputs['dst'] = Data(dst)


class OpenCVAuto_warpAffine(NormalElement):
    name = 'warpAffine'
    comment = 'warpAffine(src, M, dsize[, dst[, flags[, borderMode[, borderValue]]]]) -> dst'

    def get_attributes(self):
        return [Input('src', 'src'), Input('M', 'M')], \
               [Output('dst', 'dst')], \
               [SizeParameter('dsize', 'dsize'), IntParameter('flags', 'flags'),
                IntParameter('borderMode', 'borderMode')]

    def process_inputs(self, inputs, outputs, parameters):
        src = inputs['src'].value
        M = inputs['M'].value
        dsize = parameters['dsize']
        flags = parameters['flags']
        borderMode = parameters['borderMode']
        dst = cv2.warpAffine(src=src, M=M, dsize=dsize, flags=flags, borderMode=borderMode)
        outputs['dst'] = Data(dst)


class OpenCVAuto_warpPerspective(NormalElement):
    name = 'warpPerspective'
    comment = 'warpPerspective(src, M, dsize[, dst[, flags[, borderMode[, borderValue]]]]) -> dst'

    def get_attributes(self):
        return [Input('src', 'src'), Input('M', 'M')], \
               [Output('dst', 'dst')], \
               [SizeParameter('dsize', 'dsize'), IntParameter('flags', 'flags'),
                IntParameter('borderMode', 'borderMode')]

    def process_inputs(self, inputs, outputs, parameters):
        src = inputs['src'].value
        M = inputs['M'].value
        dsize = parameters['dsize']
        flags = parameters['flags']
        borderMode = parameters['borderMode']
        dst = cv2.warpPerspective(src=src, M=M, dsize=dsize, flags=flags, borderMode=borderMode)
        outputs['dst'] = Data(dst)


################# END OF GENERATED CODE #################

register_elements_auto(__name__, locals(), "OpenCV legacy autogenerated", 10000)
