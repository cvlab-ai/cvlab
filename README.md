CV Lab - Computer Vision Laboratory - a rapid prototyping tool for computer vision algorithms

<a href="https://drive.google.com/uc?export=download&id=15G4UPlZWxftl5pN53kN1co1yP02lZBGh">
<img align="right" height="220" src="https://drive.google.com/uc?export=download&id=15G4UPlZWxftl5pN53kN1co1yP02lZBGh">
</a>

- [INSTALLATION](#installation)
- [DESCRIPTION](#description)
- [USAGE](#usage)
- [FAQ](#faq)
- [KNOWN ISSUES](#issues)
- [COPYRIGHT](#copyright)

# INSTALLATION 

Installation using pip:

    pip install --upgrade cvlab
    
Or for Python 3:

    pip3 install --upgrade cvlab

This command will install CV Lab (or update if you have already installed it). See the [PyPI page](https://pypi.python.org/pypi/cvlab) for more information.

Alternatively you can clone entire git repository:


CV Lab requires: `PyQt4`, `OpenCV`, `numpy`, `scipy`, `pygments`, `future`, `tinycss2`.

# DESCRIPTION

CV Lab enables convenient development of computer vision algorithms by means of graphical designing of the processing flow. Writing code with OpenCV might be a time-consuming process. It is often required to compile and run the code multiple times in order to see the results of the modifications of the algorithm. Especially when some parameters are to be tuned for establishing the optimal values. Some code also has to be added to provide presentation of the intermediate or final results of the algorithm.

Instead, CV Lab offers interactive construction of the algorithms. OpenCV functions are available in a form of a palette of image processing blocks. They can be drag'n'dropped into a diagram and connected to each other for defining the data flow. Outputs of the functions in the diagram can be previewed. Parameters are available as convenient widgets like sliders or spinners. Therefore, any change in the diagram or parameter values can be instantly observed in the selected previews.

**Homepage** on GitHub: https://github.com/cvlab-ai/cvlab
**PyPI** package: https://pypi.python.org/pypi/cvlab
    
# USAGE

To run CV Lab just write in console:

    cvlab
    
or:

    python -O cvlab/__main__.py
    
### Creating image processing diagram

1. Drag&drop processing elements from the palette to diagram area
1. Connect elements by drag&dropping their connectors
1. Open output previews by double-clicking elements
1. Adjust parameters and see the outputs

### Moving around the diagram

1. Use middle mouse button or mouse wheel to scroll the diagram
1. Select single element by clicking on it
1. Select multiple elements by clicking and dragging on the diagram area  
1. Move elements with drag&drop

### Displaying output images or data

1. Double-click on the element to open data previews
1. Use mouse wheel on the previews to zoom in/out
1. Double-click on the preview to open external window with additional preview

### Writing python code using CV Lab

1. Put `Code element` on the diagram and connect its inputs/outputs, open previews
1. Open `Edit code` dialog
1. Write whatever python code you like :)
1. See the results in real-time
1. Be careful about infinite loops...
1. In long loops use `intpoint()` - it will allow the code to be interrupted when it's needed
1. To store state of the code element, you can use `memory` (a `dict` which survives recalculations) 

### Generating python code from the diagram

1. Right-click on last element of the diagram
1. Select `generate code`. The code will be copied to system clipboard.
1. Paste the code to empty python file
1. You can use the code as a library or as a script

Note: code generation is experimental. It may not work correctly with diagrams utilizing Sequences or some sophisticated elements.  

### Creating your own elements

Adding elements to CV Lab is really simple. See: `cvlab/diagram/elements/custom/sample.py`

# KNOWN ISSUES

### Random crashes

Due to a bug in old versions of OpenCV Python binding (<3.1), some OpenCV functions may cause random crashes of the entire application. Please use latest version of OpenCV available on the [official OpenCV website](https://opencv.org/releases.html).

Alternatively, you can install latest unofficial build of OpenCV using pip:

    pip install --upgrade opencv-python

Note that most Linux OS packages often use outdated version of OpenCV. Before using above command you should uninstall them.

### Broken python generated code

Automatic code generation is experimental. Only experienced users shall use it.

Some elements cannot be easily translated to python script code. Also, code generated from diagrams utilizing sequences may not work correctly.  

Please, forgive us.

# COPYRIGHT

                                 CV Lab
                      
           Copyright (c) 2013-2017 Adam Brzeski, Jan Cychnerski
                  
          This software is distributed under 'AGPL-3.0+' license,
           excluding cvlab/diagram/elements and cvlab/thirdparty
      
              Files in directory cvlab/diagram/elements are
                     distributed under 'MIT License'.
         
                  Files in directory cvlab/thirdparty are
                 distributed under their specific licenses.
                 
                 
