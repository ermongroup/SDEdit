# SDEdit: Image Synthesis and Editing with Stochastic Differential Equations
<br>

[Chenlin Meng](https://cs.stanford.edu/~chenlin/), [Yang Song](https://yang-song.github.io/), [Jiaming Song](http://tsong.me/), 
[Jiajun Wu](https://jiajunwu.com/), [Jun-Yan Zhu](https://www.cs.cmu.edu/~junyanz/) and [Stefano Ermon](https://cs.stanford.edu/~ermon/)

Stanford University and Carnegie Mellon University 

PyTorch implemention of SDEdit: Image Synthesis and Editing with Stochastic Differential Equations.

<p align="center">
<img src="https://github.com/chenlin9/image_editing_ddpm/blob/main/images/figure1.jpg" />
</p>



[**Project**](https://chenlin9.github.io/SDEdit/) | [**Paper**]() | [**Colab**](https://colab.research.google.com/drive/1KkLS53PndXKQpPlS1iK-k1nRQYmlb4aO?usp=sharing)
## Downloading the pretrained checkpoints
The code will automatically download pretrained SDE (VP) PyTorch checkpoints on
[CelebA-HQ](https://image-editing-test-12345.s3-us-west-2.amazonaws.com/checkpoints/celeba_hq.ckpt), 
[LSUN bedroom](https://image-editing-test-12345.s3-us-west-2.amazonaws.com/checkpoints/bedroom.ckpt),
and [LSUN church outdoor](https://image-editing-test-12345.s3-us-west-2.amazonaws.com/checkpoints/church_outdoor.ckpt).

## Data format
We save the image and the corresponding mask in an array format ``[image, mask]``, where 
"image" is the image with range ``[0,1]`` in the PyTorch tensor format, "mask" is the corresponding binary mask (also the PyTorch tensor format) specifying the pixels that need editing.
We provide a few examples, and ``functions/process_data.py``  will automatically download the examples to the ``colab_demo`` folder.


## Scribble-based image generation
Given an input scribble painting, our goal is to generate a realistic image that shares the same structure as the input when no paired data is available. 
We present scribble-based image synthesis with SDEdit on LSUN bedroom, LSUN church and CelebA-HQ datasets. 
We notice that SDEdit is able to generated multiple diverse images for each scribble painting.
<p align="center">
<img src="https://github.com/chenlin9/image_editing_ddpm/blob/main/images/sde_stroke_generation.jpg" />

To generate images based on scribble images on LSUN datasets, run

```
python main.py --exp ./runs/ --config bedroom.yml --sample -i images --npy_name lsun_bedroom1 --sample_step 3 --t 500  --ni
```
```
python main.py --exp ./runs/ --config church.yml --sample -i images --npy_name lsun_church --sample_step 3 --t 500  --ni
```
</p>

<p align="center">
<img src="https://github.com/chenlin9/image_editing_ddpm/blob/main/images/stroke_based_generation.jpg" width="800">
</p>

## Scribble-based image editing
Given an input with user added scribbles, we want to generate a realistic image based on the user's edit. 
We observe that our method is able to generate image edits that are both realistic and faithful (to the user edit), 
while avoid making undesired modifications.
<p align="center">
<img src="https://github.com/chenlin9/image_editing_ddpm/blob/main/images/stroke_edit.jpg" width="800">
</p>
To perform scribble-based image editing, run

```
python main.py --exp ./runs/  --config church.yml --sample -i images --npy_name lsun_edit --sample_step 3 --t 500  --ni
```

## References and Acknowledgements
```
TODO
```

This implementation is based on / inspired by:

- [https://github.com/hojonathanho/diffusion](https://github.com/ermongroup/ddim) (the DDIM PyTorch repo), 
- [https://github.com/hojonathanho/diffusion](https://github.com/hojonathanho/diffusion) (the DDPM TensorFlow repo), 
- [https://github.com/pesser/pytorch_diffusion](https://github.com/pesser/pytorch_diffusion) (PyTorch helper that loads the DDPM model), and
- [https://github.com/ermongroup/ncsnv2](https://github.com/ermongroup/ncsnv2) (code structure).
