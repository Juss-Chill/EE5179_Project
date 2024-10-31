# Image Denoising
- [Real Image Denoising](#real-image-denoising)
  * [Training](#training-1)
  * [Evaluation](#evaluation-1)
      - [Testing on MVTECAD dataset](#testing-on-sidd-dataset)

# Real Image Denoising

## Training

- Download MVTECAD dataset and extract the Denoising_Dataset_train_val folder in the main directory.

- Change MVTECAD dataset structure to SIDD dataset structure, run

```
python mvtec2sidd_train.py
```
```
python mvtec2sidd_val.py
```

- Copy MVTEC/train to Denoising/Datasets/Downloads and rename the folder as MVTECAD

- Copy MVTEC/val to Denoising/Datasets/Downloads and rename the folder as MVTECAD_val

- Generate image patches from full-resolution training images, run
```
cd Denoising
python generate_patches_mvtecad.py
python generate_patches_mvtecad_val.py
 
```

- Train Restormer
```
cd EE5179_Project-main
./train.sh Denoising/Options/RealDenoising_Restormer_MVTECAD.yml
```

**Note:** This training script uses 1 GPU by default. To use any other number of GPUs, modify [Restormer/train.sh](../train.sh) and [Denoising/Options/RealDenoising_Restormer_MVTECAD.yml](Options/RealDenoising_Restormer_MVTECAD.yml)

## Evaluation on MVTECAD dataset

- Download the pre-trained [model](https://drive.google.com/drive/folders/1rRjmaTms2f3sO5PGNvmp8mLZahCiXK_D?usp=sharing) and place it in `./pretrained_models/`

- Evaluate Restormer (input_dir for the test images directory should be in same format as train and val images directory in Denoising_Dataset_train_val)
```
python eval_mvtecad.py --input_dir <your_input_images_dir> --result_dir results --task Real_Denoising --tile 256 --tile_overlap 64
```

- Results along with object wise PSNR and SSIM will be updated in results directory.


## Inference
- 'app.py' is for inference purpose.
- you can upload a denoise image it will give you clear image.
- It is written in streamlit frame work of python.
- To run
  ```
  streamlit run app.py
  ```
