# Cell Segmentation Using Machine Learning
It is a study that compares machine learning and deep learning in shallow data for the cell segmentation problem.


<div class="alert alert-success" markdown="1">

> This is a blockquote
> - And a list
>     - And a nested element with a **bold**

</div>


[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://githubtocolab.com/mberkay0/cell-segmentation-with-machine-learning/blob/main/CellSegmentation.ipynb)



<h3>
  <strong>1.1 Local Binary Patterns</strong>
</h3>


<div>
<img src="https://929687.smushcdn.com/2633864/wp-content/uploads/2015/12/lbp_thresholding.jpg?lossy=1&strip=1&webp=1" width="43%"/>
</div>

**Figure 1.1.1:** The first step in constructing a LBP is to take the 8 pixel neighborhood surrounding a center pixel and threshold it to construct a set of 8 binary digits.


<div>
<img src="https://929687.smushcdn.com/2633864/wp-content/uploads/2015/12/lbp_calculation-1024x299.jpg?lossy=1&strip=1&webp=1" width="45%"/>
</div>


<div>
<img src="https://pyimagesearch.com/wp-content/uploads/2015/12/lbp_to_output.jpg" width="45%"/>
</div>

**Figure 1.1.2:** Taking the 8-bit binary neighborhood of the center pixel and converting it into a decimal representation. 

<div>
<img src="https://929687.smushcdn.com/2633864/wp-content/uploads/2015/12/lbp_num_points_radii.jpg?lossy=1&strip=1&webp=1" width="35%"/>
</div>

**Figure 1.1.3:** Three neighborhood examples with varying p and r used to construct Local Binary Patterns.




<div>
  <img src="https://camo.githubusercontent.com/d55a437337d0e08c6a082714959253d80b81ce4e6c18e94688d9aff16e3bf2f8/68747470733a2f2f6c6d622e696e666f726d6174696b2e756e692d66726569627572672e64652f70656f706c652f726f6e6e656265722f752d6e65742f752d6e65742d6172636869746563747572652e706e67" width="50%"/>

  <img src="https://d3i71xaburhd42.cloudfront.net/7447a957fe1a4922fb7e28cf672d3d84b2963d83/2-Figure1-1.png" width="50%"/>

  <img src="https://chadrick-kwag.net/wp-content/uploads/2021/01/1.png" width="50%"/>

</div>


[1] [Ronneberger, O., Fischer, P., & Brox, T. (2015, October). U-net: Convolutional networks for biomedical image segmentation. In International Conference on Medical image computing and computer-assisted intervention (pp. 234-241). Springer, Cham.](https://https://arxiv.org/pdf/1505.04597.pdf)

[2] [Chaurasia, A., & Culurciello, E. (2017, December). Linknet: Exploiting encoder representations for efficient semantic segmentation. In 2017 IEEE Visual Communications and Image Processing (VCIP) (pp. 1-4). IEEE.](https://arxiv.org/pdf/1707.03718.pdf)

[3] [Lin, T. Y., Dollár, P., Girshick, R., He, K., Hariharan, B., & Belongie, S. (2017). Feature pyramid networks for object detection. In Proceedings of the IEEE conference on computer vision and pattern recognition (pp. 2117-2125).](http://presentations.cocodataset.org/COCO17-Stuff-FAIR.pdf)
