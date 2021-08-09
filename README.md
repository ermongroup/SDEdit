# SDEdit: Image Synthesis and Editing with Stochastic Differential Equations
<br>

<p align="center">
<img src="https://github.com/ermongroup/SDEdit/blob/main/images/sde_animation.gif" width="320"/>
</p>

[**Project**](https://sde-image-editing.github.io/) | [**Paper**](https://arxiv.org/abs/2108.01073) | [**Colab**](https://colab.research.google.com/drive/1KkLS53PndXKQpPlS1iK-k1nRQYmlb4aO?usp=sharing)

PyTorch implementation of SDEdit: Image Synthesis and Editing with Stochastic Differential Equations.

[Chenlin Meng](https://cs.stanford.edu/~chenlin/), [Yang Song](https://yang-song.github.io/), [Jiaming Song](http://tsong.me/),
[Jiajun Wu](https://jiajunwu.com/), [Jun-Yan Zhu](https://www.cs.cmu.edu/~junyanz/), [Stefano Ermon](https://cs.stanford.edu/~ermon/)

Stanford and CMU


<p align="center">
<img src="https://github.com/ermongroup/SDEdit/blob/main/images/teaser.jpg" />
</p>

## Overview
The key intuition of SDEdit is to "hijack" the reverse stochastic process of SDE-based generative models, as illustrated in the figure below. Given an input image for editing, such as a stroke painting or an image with color strokes, we can add a suitable amount of noise to make its artifacts undetectable, while still preserving the overall structure of the image. We then initialize the reverse SDE with this noisy input, and simulate the reverse process to obtain a denoised image of high quality. The final output is realistic while resembling the overall image structure of the input.

<p align="center">
<img src="https://github.com/ermongroup/SDEdit/blob/main/images/sde_stroke_generation.jpg" />
</p>

## Getting Started
The code will automatically download pretrained SDE (VP) PyTorch models on
[CelebA-HQ](https://image-editing-test-12345.s3-us-west-2.amazonaws.com/checkpoints/celeba_hq.ckpt),
[LSUN bedroom](https://image-editing-test-12345.s3-us-west-2.amazonaws.com/checkpoints/bedroom.ckpt),
and [LSUN church outdoor](https://image-editing-test-12345.s3-us-west-2.amazonaws.com/checkpoints/church_outdoor.ckpt).

### Data format
We save the image and the corresponding mask in an array format ``[image, mask]``, where
"image" is the image with range ``[0,1]`` in the PyTorch tensor format, "mask" is the corresponding binary mask (also the PyTorch tensor format) specifying the editing region.
We provide a few examples, and ``functions/process_data.py``  will automatically download the examples to the ``colab_demo`` folder.

### Re-training the model
Here is the [PyTorch implementation](https://github.com/ermongroup/ddim) for training the model.


## Stroke-based image generation
Given an input stroke painting, our goal is to generate a realistic image that shares the same structure as the input painting.
SDEdit can synthesize multiple diverse outputs for each input on LSUN bedroom, LSUN church and CelebA-HQ datasets.



To generate results on LSUN datasets, please run

```
python main.py --exp ./runs/ --config bedroom.yml --sample -i images --npy_name lsun_bedroom1 --sample_step 3 --t 500  --ni
```
```
python main.py --exp ./runs/ --config church.yml --sample -i images --npy_name lsun_church --sample_step 3 --t 500  --ni
```

<p align="center">
<img src="https://github.com/ermongroup/SDEdit/blob/main/images/stroke_based_generation.jpg" width="800">
</p>

## Stroke-based image editing
Given an input image with user strokes, we want to manipulate a natural input image based on the user's edit.
SDEdit can generate image edits that are both realistic and faithful (to the user edit), while avoid introducing undesired changes.
<p align="center">
<img src="https://github.com/ermongroup/SDEdit/blob/main/images/stroke_edit.jpg" width="800">
</p>
To perform stroke-based image editing, run

```
python main.py --exp ./runs/  --config church.yml --sample -i images --npy_name lsun_edit --sample_step 3 --t 500  --ni
```

## Additional results
<p align="center">
<img src="https://github.com/ermongroup/SDEdit/blob/main/images/stroke_generation_extra.jpg" width="800">
</p>

## References
If you find this repository useful for your research, please cite the following work.
```
@article{meng2021sdedit,
      title={SDEdit: Image Synthesis and Editing with Stochastic Differential Equations},
      author={Chenlin Meng and Yang Song and Jiaming Song and Jiajun Wu and Jun-Yan Zhu and Stefano Ermon},
      year={2021},
      journal={arXiv preprint arXiv:2108.01073},
}
```

This implementation is based on / inspired by:

- [DDIM PyTorch repo](https://github.com/ermongroup/ddim).
- [DDPM TensorFlow repo](https://github.com/hojonathanho/diffusion).
- [PyTorch helper that loads the DDPM model](https://github.com/pesser/pytorch_diffusion).
- [code structure](https://github.com/ermongroup/ncsnv2).
