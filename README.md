# SDEdit: Image Synthesis and Editing with Stochastic Differential Equations
<br>

[**Project**](https://chenlin9.github.io/SDEdit/) | [**Paper**](https://arxiv.org/abs/2108.01073) | [**Colab**](https://colab.research.google.com/drive/1KkLS53PndXKQpPlS1iK-k1nRQYmlb4aO?usp=sharing)

PyTorch implementation of SDEdit: Image Synthesis and Editing with Stochastic Differential Equations.

[Chenlin Meng](https://cs.stanford.edu/~chenlin/), [Yang Song](https://yang-song.github.io/), [Jiaming Song](http://tsong.me/),
[Jiajun Wu](https://jiajunwu.com/), [Jun-Yan Zhu](https://www.cs.cmu.edu/~junyanz/) and [Stefano Ermon](https://cs.stanford.edu/~ermon/)

Stanford University and Carnegie Mellon University


<p align="center">
<img src="https://github.com/ermongroup/SDEdit/blob/main/images/teaser.jpg" />
</p>





## Downloading the pretrained checkpoints
The code will automatically download pretrained SDE (VP) PyTorch checkpoints on
[CelebA-HQ](https://image-editing-test-12345.s3-us-west-2.amazonaws.com/checkpoints/celeba_hq.ckpt),
[LSUN bedroom](https://image-editing-test-12345.s3-us-west-2.amazonaws.com/checkpoints/bedroom.ckpt),
and [LSUN church outdoor](https://image-editing-test-12345.s3-us-west-2.amazonaws.com/checkpoints/church_outdoor.ckpt).

## Data format
We save the image and the corresponding mask in an array format ``[image, mask]``, where
"image" is the image with range ``[0,1]`` in the PyTorch tensor format, "mask" is the corresponding binary mask (also the PyTorch tensor format) specifying the pixels that need editing.
We provide a few examples, and ``functions/process_data.py``  will automatically download the examples to the ``colab_demo`` folder.


## Stroke-based image generation
Given an input stroke painting, our goal is to generate a realistic image that shares the same structure as the input when no paired data is available.
We present stroke-based image synthesis with SDEdit on LSUN bedroom, LSUN church and CelebA-HQ datasets.
SDEdit can synthesize multiple diverse images for each input.
<p align="center">
<img src="https://github.com/ermongroup/SDEdit/blob/main/images/sde_stroke_generation.jpg" />

To generate images based on stroke images on LSUN datasets, run

```
python main.py --exp ./runs/ --config bedroom.yml --sample -i images --npy_name lsun_bedroom1 --sample_step 3 --t 500  --ni
```
```
python main.py --exp ./runs/ --config church.yml --sample -i images --npy_name lsun_church --sample_step 3 --t 500  --ni
```
</p>

<p align="center">
<img src="https://github.com/ermongroup/SDEdit/blob/main/images/stroke_based_generation.jpg" width="800">
</p>

## Stroke-based image editing
Given an input image with user strokes, we want to generate a realistic image based on the user's edit.
We observe that our method is able to generate image edits that are both realistic and faithful (to the user edit),
while avoid making undesired modifications.
<p align="center">
<img src="https://github.com/ermongroup/SDEdit/blob/main/images/stroke_edit.jpg" width="800">
</p>
To perform stroke-based image editing, run

```
python main.py --exp ./runs/  --config church.yml --sample -i images --npy_name lsun_edit --sample_step 3 --t 500  --ni
```

## References
```
@misc{meng2021sdedit,
      title={SDEdit: Image Synthesis and Editing with Stochastic Differential Equations}, 
      author={Chenlin Meng and Yang Song and Jiaming Song and Jiajun Wu and Jun-Yan Zhu and Stefano Ermon},
      year={2021},
      eprint={2108.01073},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
```

This implementation is based on / inspired by:

- [DDIM PyTorch repo](https://github.com/ermongroup/ddim).
- [DDPM TensorFlow repo](https://github.com/hojonathanho/diffusion).
- [PyTorch helper that loads the DDPM model](https://github.com/pesser/pytorch_diffusion).
- [code structure](https://github.com/ermongroup/ncsnv2).
