


<h1 align="center">
Automatic Gain Tuning for Humanoid Robots Walking Architectures Using Gradient-Free Optimization Techniques
</h1>


<div align="center">

C.Sartore*, M. Rando *, G. Romualdi, C. Molinari, L.Rosasco, D.Pucci
_"Automatic Gain Tuning for Humanoid Robots Walking Architectures Using Gradient-Free Optimization Techniques"_
in 2024 IEEE-RAS International Conference on Humanoid Robotics (Humanoids)
Authors* contributed equally to this work. 
</div>

<p align="center">


[![Video](https://github.com/ami-iit/paper_sartore_2022_humanoids_ergonomic_design/assets/56030908/a0d66262-5539-481e-ac42-60219561b607)](https://github.com/ami-iit/paper_sartore_2022_humanoids_ergonomic_design/assets/56030908/6f73779e-5153-4048-bb1d-706c59b80490)

</p>

<div align="center">
  IEEE-RAS International Conference on Humanoid Robotics
</div>

<div align="center">
  <a href="#installation"><b>Installation</b></a> |
  <a href="https://arxiv.org/abs/2409.18649"><b>PrePrint</b></a> |
  <a href=https://www.youtube.com/watch?v=BccsReE9MpY><b>Video</b></a>
</div>

### Installation


:warning: The repository depends on [HSL for IPOPT (Coin-HSL)](https://www.hsl.rl.ac.uk/ipopt/), to correctly link the library please substitute [this](./Dockerfile#L110) line of the docker image with the absolute path to the `coinhsl.zip`. In particular for the paper experiments Coin-HSL 2019.05.21 have been used, but also later version should work fine. 

⚠️ This repository depends on [docker](https://docs.docker.com/)

To install the repo on a Linux terminal follow the following steps 

```
git clone https://github.com/ami-iit/paper_sartore_rando_2024_humanoids_zero_order_gain_tuning
cd paper_sartore_rando_2024_humanoids_zero_order_gain_tuning
docker build --tag sartore_rando_humanoids_2024 . 
```

### Running 
In the src folder, you can find: 
- optimize_fitness_ga: optimize the fitness function using genetic algorithm.
- optimize_fitness_es: optimize the fitness function using evolutionary strategies.
- optimize_fitness_tde: optimize the fitness function using differential evolution.
- optimize_fitenss_cmaes: optimize the fitness function using cmaes.

:warning: Each of the file run a repetition optimization: 10 independent optimization perfromed for the fitness that considers the torques, and 10 independent run performed for the fitenss that does not consider the torques.

:warning: the optmization run with a multiprocess and will take 100 CPU cores. 

### Citing this work

```bibtex
@INPROCEEDINGS{SartoreRando2024gainTuning,
  author={Sartore, Carlotta and Rando, Marco and Romualdi, Giulio and Molinari, Cesare and Rosasco, Lorenzo and Pucci, Daniele},
  booktitle={2024 IEEE-RAS 21st International Conference on Humanoid Robots (Humanoids)}, 
  title={Automatic Gain Tuning for Humanoid Robots Walking Architectures Using Gradient-Free Optimization Techniques}, 
  year={2024},
  volume={},
  number={},
  pages={},
  doi={}}
```

### Maintainer

This repository is maintained by:


|     [Carlotta Sartore](https://github.com/CarlottaSartore)    | [Marco Rando](https://github.com/Akatsuki96)
|-------------------------------------------------------|-------------------------------------------------------|
|<img src="https://user-images.githubusercontent.com/56030908/135461492-6d9a1174-19bd-46b3-bee6-c4dbaea9e210.jpeg" width="180">| <img src="https://github.com/ami-iit/element_hardware-intelligence/assets/56030908/def8f63d-0bc3-47fb-a64b-9626665c0f98" width="180">|



