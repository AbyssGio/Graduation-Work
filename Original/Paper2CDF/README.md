# Transparent Depth Completion

code for paper: Consistent Depth Prediction for Transparent Object Reconstruction from RGB-D Camera (ICCV 23')


## Usage

Environment requirement:

1. Install python environment for running depth prediction code

`conda env create -f environment.yml`

2. Install ElasticFusion for SLAM reconstruction

Please refer to [here](https://github.com/mp3guy/ElasticFusion) and follow the project build instruction

3. Install libtorch for running nerual network in C++

You can find more help in [here](https://github.com/AllentDan/LibtorchTutorials/tree/main/lesson1-Environment)

4. Install data converter: pngtoklg

Please refer to [here](https://github.com/HTLife/png_to_klg) for detailed information

## Running

- training and validating network using config file in `experiments/xx.yml`

```
python main.py
```

- example for testing network in new dataset

```
./experiments/test_glass.py --cfg=experiments/dfnet_pt.yml --test_scene=val_seq1 --output_filename=depth_pt
```

- predict depth and run reconstruction seperately

```
bash run_efusion.sh
```

### Running PersudoSLAM

**make sure you can sucessfully compile ElasticFusion**

- generate associate.txt

```
run_elasticfusion.py --base_dir=/to/your/data/path --depth_file=depth --rgb_file=rgb --mask_file=mask --drop_num=1 --depth_scale=4000
```

- export checkpoint to libtorch

```
python utils/ckpt2script.py
```

- generate klg with depth data for logger in ElasticFusion

```
python utils/png2klg.py
```

- compile code in `PersudoSLAM` as ElasticFusion and run it

## Dataset

synthetic dataset is available at [here](https://drive.google.com/file/d/1fgnoVxZFtUuDBs5EGbtgFdx0ScVSybLL/view?usp=sharing)


