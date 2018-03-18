# Font Design GAN (with Extensions)

Code for training/generating GAN, that is for font design.

This is extended from [uchidalab/fontdesign_gan](https://github.com/uchidalab/fontdesign_gan).  
There are added functions:
- Classifier, for multi-fonts character recognition
- Visualize intermediate layers' outputs
- Evaluate generated fonts by using pseudo-Hamming distance

The author use this codes for writing a master's thesis, so you may know how to use them if you read. (but it's Japanese...)

## Samples
Generated fonts, they have various styles and they are readable as a character.
![various](samples/various.png)

Random walk in style-input space.
You can watch gradually transformed fonts.

![A](samples/walk_a.gif)
![B](samples/walk_b.gif)
![C](samples/walk_c.gif)
![D](samples/walk_d.gif)
![E](samples/walk_e.gif)
![F](samples/walk_f.gif)
![G](samples/walk_g.gif)
![H](samples/walk_h.gif)
![I](samples/walk_i.gif)
![J](samples/walk_j.gif)
![K](samples/walk_k.gif)
![L](samples/walk_l.gif)
![M](samples/walk_m.gif)
![N](samples/walk_n.gif)
![O](samples/walk_o.gif)
![P](samples/walk_p.gif)
![Q](samples/walk_q.gif)
![R](samples/walk_r.gif)
![S](samples/walk_s.gif)
![T](samples/walk_t.gif)
![U](samples/walk_u.gif)
![V](samples/walk_v.gif)
![W](samples/walk_w.gif)
![X](samples/walk_x.gif)
![Y](samples/walk_y.gif)
![Z](samples/walk_z.gif)

## How to use

### Environment
The auther checked running codes in only following environment:
- Ubuntu 16.04 LTS
- GeForce GTX 1080 & [Driver](http://www.nvidia.com/Download/index.aspx) (Driver Version: 384.111)
- [CUDA 8.0](https://developer.nvidia.com/cuda-80-ga2-download-archive)
- [cuDNN v6.0](https://developer.nvidia.com/cudnn) (Membership required)

If you'd like to run with latest environment, revise some files as necessary.

Firstly clone this repository.
Add `--recursive` because this repository contains submodule ([font2img](https://github.com/uchidalab/font2img))
```
git clone --recursive https://github.com/uchidalab/fontdesign_gan
cd fontdesign_gan
```

#### with Docker (GPU required)

If you use GPU and Docker, it's easy to set up your environment.
Requirement libraries are written in Dockerfile.  
Install GPU Driver/[NVIDIA Docker](https://github.com/NVIDIA/nvidia-docker) and run following commands:
```
docker build -t fontdesign_gan .
docker run --runtime=nvidia -it --rm -p 6006:6006 --volume `pwd`:/workdir -w /workdir/ fontdesign_gan
```

#### no Docker

with GPU:
Install GPU Driver/CUDA/cuDNN and run following command:
```
pip install -r requirements_gpu.txt
pip install Multicore-TSNE/
```

with CPU:
Run following command:
```
pip install -r requirements_cpu.txt
pip install Multicore-TSNE/
```

### Prepare

Convert font files (.ttf) into image files (.png), and pack them into a HDF5 file (.h5)

Make a directory (ex. `./ttfs`) and copy font files.
Set destination path (ex. `./src/realfonts.h5`) and run.
```
python main.py --ttf2png --png2h5 --font_ttfs ./ttfs --font_h5 ./src/realfonts.h5
```

`--ttf2png` is the option for converting, image files are saved in `./src/pngs/{YYYY-MM-DD_HHmmss}`.
If you use `--font_pngs` option, you can set path yourself.  
`--png2h5` is the option for packing, packed file is saved in the path you set with `--font_h5` option.

### Train

Set packed file's path and run.

```
python main.py --train --font_h5 ./src/realfonts.h5
```

Results are saved in `./result/gan/{YYYY-MM-DD_HHmmss}`.
You can set destination with `--gan_dir`.

In `log` directory, saved same files: flags' log, latest/kept TensorFlow's dumps (.ckpt\*), TensorBoard's log.  
In `sample` directory, fonts that generated temporary are saved.

While training, TensorBoard is also running.
Access URL that will shown in command line.

### Generate

Set a path of trained GAN's output directory and a JSON file.

The JSON file have to be written style/character IDs.
A style ID correspond with a random matrix.
A character ID correspond with a character.

IDs are loaded in order, and you can use some operands.
For example, if JSON file is following:
```
"style_ids": [
  "0", "4", "21", "10..29:5"
],
"char_ids": [
  "0-3", "6*4"
],
"col_n": 4
```
Input IDs will be like this:
```
style_ids = [0, 4, 21, (between 10 and 29 with 5 steps)]
char_ids = [0, 1, 2, 3, 6, 6, 6, 6]
```
and, # of result columns is 4. Also check sample files (`./jsons/sample*.json`).

After preparing JSON file, run like this:
```
python main.py --generate --gan_dir ./result/gan/{trained} --ids ./jsons/sample01.json
```
Generated fonts are saved in `./result/gan/{trained}/generated/{YYYY-MM-DD_HHmmss}.png`.
You can set output file name with `--gen_name`.

If you want random walking fonts, use `--generate_walk`. A JSON file is needless.
```
python main.py --generate_walk --gan_dir ./result/gan/{YYYY-MM-DD_HHmmss} --char_img_n 256
```
256 styles' fonts will be generated, and they are transformed gradually.

### Use Classifier (Extension)

In this project, Classifier is used for character recognition.

When you want to train Classifier:
```
python main.py --train_c --font_h5 ./src/realfonts.h5
```
In default, train:test = 9:1. You can change by using `--train_rate`.
Results are saved in `./result/classifier/{YYYY-MM-DD_HHmmss}`.
You can set destination with `--classifier_dir`.

To test generated fonts, run like this:
```
# Generate 1000 randomly in ./result/gan/{trained}/recognition_test
python main.py --generate_test --gan_dir ./result/gan/{trained} --char_img_n 1000

# Pack generated fonts into ./result/gan/{trained}/recognition_test/generated_1000fonts.h5
python main.py --png2h5 --font_pngs ./result/gan/{trained}/recognition_test/generated_1000fonts.h5

# Test generated fonts
python main.py --test_c --classifier_dir ./result/classifier/{trained_c} --font_h5 ./result/gan/{trained}/recognition_test/generated_1000fonts.h5
```

### Visualization of intermediate layers' outputs (Extension)

Visualize when generate from selected IDs.
Firstly I recommend to use `jsons/sample04.json`.
```
python main.py --intermediate --gan_dir ./result/gan/{trained}/ --ids ./jsons/sample04.json --change_align
```
Results are saved in `./result/gan/{trained}/intermediate`.

In default, the method of plotting is t-SNE.
Also supported MDS, PCA. (See Options)

### Evaluate generated fonts by measuring pseudo-Hamming distance (Extension)

Measure between generated fonts and real fonts.
If you want to know about "pseudo-Hamming distance", check this:
- S. Uchida, Y. Egashira, K. Sato, "Exploring the World of Fonts for Discovering the Most Standard Fonts and the Missing Fonts", ICDAR, 2015.
```
# Generate 1000 randomly in ./result/gan/{trained}/recognition_test
python main.py --generate_test --gan_dir ./result/gan/{trained} --char_img_n 1000

# Pack generated fonts into ./result/gan/{trained}/recognition_test/generated_1000fonts.h5
python main.py --png2h5 --font_pngs ./result/gan/{trained}/recognition_test/generated_1000fonts.h5

# Evaluate generated fonts
python main.py --evaluate --gan_dir ./result/gan/{trained} --font_h5 ./src/realfonts.h5 --generated_h5 ./result/gan/{trained}/recognition_test/generated_1000fonts.h5
```
Results are saved in `./result/gan/{trained}/evaluated`.

### Options

There are many options. Check following table.

|For|Option|Description|Default|
|:-|:-|:-|:-|
|Prepare, Train|`--img_width`|width of images.|64|
|Prepare, Train|`--img_height`|height of images.|64|
|Prepare, Train|`--img_dim`|dimension of images.|3|
|Prepare, Train|`--chars_type`|you can choose characters type, "caps", "hiragana" or "caps,hiragana"|"caps"|
|Train, Generate|`--gpu_ids`|GPU IDs you use. this type is string (ex. "0, 1")|(set automatically)|
|Train, Generate|`--batch_size`|batch size for GAN|256|
|Train|`--arch`|architecture of GAN models. choose "DCGAN" or "ResNet"|"DCGAN"|
|Train|`--style_ids_n`|# of style IDs|256|
|Train|`--style_z_size`|size of style_z|100|
|Train|`--gan_epoch_n`|# of epoch iterations|10000|
|Train|`--critic_n`|# of critic iterations|5|
|Train|`--sample_imgs_interval`|interval of saving sample images|10|
|Train|`--sample_col_n`|# of sample image's columns|26|
|Train|`--keep_ckpt_interval`|interval of keeping TensorFlow's dumps|250|
|Train|`--run_tensorboard`|run tensorboard or not|True|
|Train|`--tensorboard_port`|port for tensorboard page|6006|
|Generate|`--change_align`|Change IDs alignment|False|
|Classifier|`--train_rate`|train:test = train_rate:(1 - train_rate)|0.9|
|Classifier|`--c_epoch_n`|# of epoch for training Classifier|10|
|Classifier|`--labelacc`|Save accuracies of each labels|False|
|Intermediate|`--plot_method`|Method of plotting for visualization, 'TSNE' or 'MDS' or 'PCA'|'TSNE'|
|Intermediate|`--tsne_p`|tSNE's perplexity|30|
