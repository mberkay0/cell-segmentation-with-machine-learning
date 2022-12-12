# Cell Segmentation Using Machine Learning
It is a study that compares machine learning and deep learning in shallow data for the cell segmentation problem.

<h3>
  <strong>Introduction</strong>
</h3>


> Label-free Cell segmentation is the task of classifying a microscopic image area, pixels representing individual cell instances. It is a fundamental step in many biomedical studies and needs to be meticulously processed.


<h3>
  <strong>Problem Definition</strong>
</h3>

> It is a process that helps the biologist quickly notice the background from the foreground in the cell segmentation task, categorizing pixels into significant regions. Cell segmentation is crucial for biologists to extract cells' morphology, polarity and motility. It increases the accuracy and speed of the diagnosis. It is also more robust and provides reliable results for biologists to use.

Data Source: *TUBITAK 119E578 Cell Motility Phase Contrast Time Lapse Microscopy Data-Set*

<h3>
  <strong>Table of Contents</strong>
</h3>

1.   [Semantic Segmentation with Traditional Machine Learning Methods](#cell-id1)

    1.1. [Data Generator](#cell-id1.1)

    1.2. [Machine Learning Approaches](#cell-id1.2)

2.   [Semantic Segmentation with Deep Learning Methods](#cell-id2) 

    2.1. [Data Generator](#cell-id2.1)
    
    2.2. [Convolutional Neural Networks Approaches](#cell-id2.2)

3.   [Analyzes](#cell-id3)


[References](#cell-id5)



<div class="alert alert-success" markdown="1">

> This is a blockquote
> - And a list
>     - And a nested element with a **bold**

</div>


[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://githubtocolab.com/mberkay0/cell-segmentation-with-machine-learning/blob/main/CellSegmentation.ipynb)


<div align="center">
  <img src="/images/data.jpg" width="50%"/>
</div>


<h3>
  <strong>1.1 Local Binary Patterns</strong>
</h3>


<div align="center">
<img src="https://929687.smushcdn.com/2633864/wp-content/uploads/2015/12/lbp_thresholding.jpg?lossy=1&strip=1&webp=1" width="43%"/>
</div>

**Figure 1.1.1:** The first step in constructing a LBP is to take the 8 pixel neighborhood surrounding a center pixel and threshold it to construct a set of 8 binary digits.


<div align="center">
<img src="https://929687.smushcdn.com/2633864/wp-content/uploads/2015/12/lbp_calculation-1024x299.jpg?lossy=1&strip=1&webp=1" width="45%"/>
</div>


<div align="center">
<img src="https://pyimagesearch.com/wp-content/uploads/2015/12/lbp_to_output.jpg" width="45%"/>
</div>

**Figure 1.1.2:** Taking the 8-bit binary neighborhood of the center pixel and converting it into a decimal representation. 

<div align="center">
<img src="https://929687.smushcdn.com/2633864/wp-content/uploads/2015/12/lbp_num_points_radii.jpg?lossy=1&strip=1&webp=1" width="35%"/>
</div>

**Figure 1.1.3:** Three neighborhood examples with varying p and r used to construct Local Binary Patterns.




<div align="center">
  <img src="/images/unet.png" width="50%"/>

  <img src="/images/linknet.png" width="50%"/>

  <img src="/images/fpn.png" width="50%"/>

</div>


<div align="center">
  <img src="/images/avg_IoU.png" width="50%"/>
  <img src="/images/IoU_framebyframe.png" width="50%"/>
</div>


[1] [Ronneberger, O., Fischer, P., & Brox, T. (2015, October). U-net: Convolutional networks for biomedical image segmentation. In International Conference on Medical image computing and computer-assisted intervention (pp. 234-241). Springer, Cham.](https://https://arxiv.org/pdf/1505.04597.pdf)

[2] [Chaurasia, A., & Culurciello, E. (2017, December). Linknet: Exploiting encoder representations for efficient semantic segmentation. In 2017 IEEE Visual Communications and Image Processing (VCIP) (pp. 1-4). IEEE.](https://arxiv.org/pdf/1707.03718.pdf)

[3] [Lin, T. Y., Dollár, P., Girshick, R., He, K., Hariharan, B., & Belongie, S. (2017). Feature pyramid networks for object detection. In Proceedings of the IEEE conference on computer vision and pattern recognition (pp. 2117-2125).](http://presentations.cocodataset.org/COCO17-Stuff-FAIR.pdf)
