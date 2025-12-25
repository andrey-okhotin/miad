# MiAD: Mirage Atom Diffusion for De Novo Crystal Generation
by Andrey Okhotin, Maksim Nakhodnov, Nikita Kazeev, Andrey E Ustyuzhanin, Dmitry Vetrov

<p align="center">
  <img src="https://github.com/andrey-okhotin/miad/blob/main/pictures_from_paper/miad_method_scheme.png" width="700" height="220">
</p>

This repo contains the official PyTorch implementation for the paper [MiAD: Mirage Atom Diffusion for De Novo Crystal Generation](https://arxiv.org/abs/2511.14426) -- approach to incorporate an ability into diffusion models for crystals to change number of atoms in crystal during trajectory generation. Mirage Infusion leads to a substantial boost in the rate of stable, unqiue and novel materials and achieves state-of-the-art results using well-known [DiffCSP](https://arxiv.org/abs/2309.04475) as a backbone. 

We provide pack of crystals found via MiAD and processed via DFT ([link)](https://drive.google.com/drive/folders/1fERECKwNbzgI2kx8M6q5w2PwfQiQk5xN?usp=sharing)). Also, we provide crystals, that are generated using MiAD that trained on MP-20 using this repo ([link](https://drive.google.com/file/d/1NtQy5YnxoiyFCSvXIcF9DHy7DZ1ZvszW/view?usp=sharing)), prerelaxed via CHGNet ([link](https://drive.google.com/file/d/1m4pjx0ALtrAUcZSXU5Gd_kwBiXnuJw4O/view?usp=sharing)) or prerelaxed via eq-V2 ([link](https://drive.google.com/file/d/1OI8f_Vx5jmBj_inxLkDBBOl4QJzR3bJs/view?usp=sharing)).

The code in this repo mostly reimplement original DiffCSP. Parts of code, where mirage infusion is used, are allocated into separate 'if' blocks inside the code and could be found by string 'miad:add_mirage_atoms_upto25'. All following scripts reproduce results from the paper and require running from the root of the repo (directory 'miad').

### Environment setup
```
conda env create -f env/environment.yml
conda activate miad
pip install -r env/requirements.txt --no-dependencies
pip install --index-url https://download.pytorch.org/whl/cu121 \
  --extra-index-url https://pypi.org/simple \
  "torch==2.4.1+cu121" "torchvision==0.19.1+cu121" "torchaudio==2.4.1+cu121"
pip install -f https://data.pyg.org/whl/torch-2.4.1+cu121.html torch-scatter -U --force
```

### Downloading datasets
```
pip install gdown py7zr
gdown --fuzzy https://drive.google.com/file/d/1BLI3VtvzfIIXlH6UHQ4o-gQaCIOZ1UR7/view?usp=sharing
py7zr x datasets.7z
rm datasets.7z
```

### Training
Run MiAD training on MP-20 and copy checkpoint. The progress can be tracked from file 'logs/progress_logs/train_miad_mp20.txt'. Checkpoints will be saved in the folder 'checkpoint/train_miad_mp20'. This pipeline uses 2 gpus by default, but also can be run on one, if argument -gpu is changed from '0_1' to '0' (also you can increase number of gpus for speed up '0_1_2', '0_1_2_3' and so on).
```
python lib/run.py -gpu 0_1 -ignore_warnings 1 -config train_miad_mp20.yaml
cp checkpoints/train_miad_mp20/CSPNet_epoch8000_model.pt saved_models/miad_mp20_epoch8000.pt
```

### Pretrained checkpoint
We also provide this checkpoint in the case if you don't want to train model by yourself.
```
pip install gdown
gdown --fuzzy https://drive.google.com/file/d/1KyD6KzvjYFPfU8lutFyO_0b8EbeHSGqf/view?usp=sharing
mv miad_mp20_epoch8000.pt saved_models/miad_mp20_epoch8000.pt
```

### Generating
Run MiAD generating using checkpoint saved_models/miad_mp20_epoch8000.pt. The progress can be tracked from file 'logs/progress_logs/generate_miad_mp20.txt'. Results will be saved in the folder 'saved_results/generate_miad_mp20'. This pipeline uses 2 gpus by default, but also can be run on one, if argument -gpu is changed from '0_1' to '0' (also you can increase number of gpus for speed up '0_1_2', '0_1_2_3' and so on).
```
python lib/run.py -gpu 0_1 -ignore_warnings 1 -config generate_miad_mp20.yaml 
```

### S.U.N.
Run SUN estimation via CHGNet for the results of the previous step. Pipeline will read crystals from folder 'saved_results/generate_miad_mp20'. The progress can be tracked from file 'logs/progress_logs/sun-chgnet_miad_mp20.txt'. The results (prerelaxed structures, energy estimations, checks for the uniqueness and novely) will be saved in the folder 'saved_results/sun-chgnet_miad_mp20'. Main results will be provided at the end of the pipeline in the file 'logs/progress_logs/sun-chgnet_miad_mp20.txt' (as well as in the file 'saved_results/sun-chgnet_miad_mp20/metric_values.txt').
```
python lib/run.py -gpu 0 -ignore_warnings 1 -config sun-chgnet_miad_mp20.yaml
```
Estimation of SUN vie eq-V2 require checkpoint for eq-V2, that can be provided only by fairchem from their huggingface. In our experiments we used checkpoint 'eqV2_153M_omat_mp_salex.pt'. If you will get one, save it to the path 'saved_models/eqV2_153M_omat_mp_salex.pt'. Only then, you can run the following pipeline. This pipeline uses 2 gpus by default, but also can be run on one, if argument -gpu is changed from '0_1' to '0' (also you can increase number of gpus for speed up '0_1_2', '0_1_2_3' and so on). The results (prerelaxed structures, energy estimations, checks for the uniqueness and novely) will be saved in the folder 'saved_results/sun-eqv2_miad_mp20'. Main results will be provided at the end of the pipeline in the file 'logs/progress_logs/sun-eqv2_miad_mp20.txt' (as well as in the file 'saved_results/sun-eqv2_miad_mp20/metric_values.txt').
```
python lib/run.py -gpu 0_1 -ignore_warnings 1 -config sun-eqv2_miad_mp20.yaml
```

### Expected results
After execution of all scripts in their original form, an expected results are quite near to the results from the paper: 

- __S.U.N. (CHGNet): 12.21%__

- __S.U.N. (eq-V2): 5.64%__


### Citation
```
@article{okhotin2025miad,
  title={MiAD: Mirage Atom Diffusion for De Novo Crystal Generation},
  author={Andrey Okhotin, Maksim Nakhodnov, Nikita Kazeev, Andrey E Ustyuzhanin, Dmitry Vetrov},
  journal={arXiv preprint arXiv:2511.14426},
  year={2025},
  url={https://arxiv.org/abs/2511.14426}
}
```
