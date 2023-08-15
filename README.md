# PoetryDiffusion: Towards Jointly Semantic and Metrical Manipulation in Poetry Generation

-----------------------------------------------------
## Conda Setup:
```python 
conda install mpi4py
conda install pytorch torchvision torchaudio cudatoolkit=11.3 -c pytorch
pip install -e improved-diffusion/ 
pip install -e transformers/
pip install spacy==3.2.4
pip install datasets==1.8.0 
pip install huggingface_hub==0.4.0 
pip install wandb
```

-----------------------------------------------------
## Train Diffusion Model:

```
cd bash
bash ci_diffusion.sh	# or: bash sonnet_diffusion.sh
```


-------------------
## Decode Diffusion Model:
```
cd bash
bash decode.sh
```


-------------------
## Controllable Text Generation 
First, train the classsifier used to guide the generation (e.g. tone in SongCi) 

```
cd bash
bash ci_classifier.sh
```

Then, we can use the trained classifier to guide generation. 

```
cd bash
bash ci_control.sh
```


