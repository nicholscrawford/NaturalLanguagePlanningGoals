# Natural Language Planning Goals

## Install

Run ```pip install -e .``` to install the module in development mode. 

TODO: List required modules, and any other install reqs.

## Model Summary
We're interested in creating goals, in the form of Multi-Modal Entity Maps (MEMs). These are, in practice, segmeneted point clouds. We'll structure them here as object entity maps, and environment entity maps. We'll first train a network to select the relevant entities from any given MEM, and we can then temporarily move all the non-selected entities to the background.

We'll then train a realistic-configuration-diffusion model. This diffuses in the configuration space of relevant entities, and is able to then generate realistic configurations. We'll train this on a large simulated dataset. Refrence StructDiffusion for specifics on the formulation of entity noise/diffusion model. Our diffusion model will be able to then transform our point clouds to their altered poses.

Then, it must be conditioned on natrual language. We'll need a renderer -- trained on high quality RGBD streams -- and we can use that renderer to render variosly noised diffusion states. For each of those state-renders we can give it a language alignment score using CLIP 
```f(Image, language) = alignment score```
and use that score to push our configuration generator to generate language aligned scores.

The renderer casts a ray from each output pixel, and finds the k-nearest neighbors. Then we run a transformer/convolutional network to render.

## Unit Testing

These are meant largely to ensure the code is functioning as intended before training or evaluation procedures. UnitTests folders can be run collectively by navigating to the folder, and then running ```pytest``` or individual sections can be run with ```pytest section_name.py```.

## Data Storage

By default datasets will be downloaded and stored to ```~/data/NaturalLanguagePlanning/<dataset name>```. 
TODO: Make this configurable in a root level yaml file. 