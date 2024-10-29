# Evidential Occupancy
Official implementation for "Accurate Training Data for Occupancy Map Prediction in Automated Driving Using Evidence Theory" (CVPR 2024, [arXiv](https://arxiv.org/pdf/2405.10575)).


# Installation
We use [Pixi](https://pixi.sh/latest/) to manage the dependencies. To install the dependencies, run the following command:
```
pixi install
```
This will create the Python environment with all necessary dependencies..
To run a shell with the activated environment run `pixi shell` or to start a single task, use `pixi run python ...`.


# Data preparation

To use the default configuration, sturcture the data as follows:
```
./data/
├── nuscenes/  # (original nuScenes dataset goes here (required))
│   ├── samples/
│   ├── scenes/
│   ├── sweeps/
│   └── ...
└── nuscenes_extra/  # (additional data)
    ├── nuscenes_occ3d/  # (Occ3D (optional))
    │   └── gts/
    │       ├── scene-*
    │       └── ...
    ├── nuScenes-Occupancy-v0.1/  # (OpenOccupancy (optional))
    │   ├── scene_*
    │   └── ...
    ├── occ_gt_release_v1_0/  # (Scene as Occupancy (optional))
    │   └── trainval/
    │       ├── scene-*
    │       └── ...
    └── surround_occ_occupancy/  # (SurroundOcc (optional))
        └── samples/
            └── ...
```


# Data Processing

We use the YAML file `./conf/default.yaml` to configure the processing steps.
For testing purposes, the nuScenes mini dataset is selected by default.
To use the entire nuScenes dataset, set `version` to `v1.0-train`, `v1.0-val` or `v1.0-trainval` (entire dataset) and set the appropriate split (`train`,`val` or `trainval`).

The data processing requires multiple steps.
The necessary task are defined in the `pixi.toml`.
You can run the entire pipeline with the following command:
```
pixi run data-processing
```
This will create the following file structure:
```
./data/
└── nuscenes_extra/
   ├── reflection_and_transmission_spherical/
   │   ├── scene-*
   │   └── ...
   ├── scene_flow/
   │   ├── scene-*
   │   └── ...
   ├── sample_annotation_cache/
   │   └── ...
   └── reflection_and_transmission_multi_frame/
       ├── scene-*
       └── ...
```


**OR**

run the following commands step by step:

## Relfections and Transmissions
Run the following command to calculate the reflections and transmissions:
```
pixi run transmissions-reflections
```

## Scene Flow
Run the following command to calculate the scene flow information:
```
pixi run scene-flow
```

## Temporal Accumulation
Run the following command to generate the accumulated reflections and transmissions:
```
pixi run temporal-accumulation
```

# Evaluation
The different occupancy methods can be evaluated separately using the following commands:
```
pixi run eval-bba
pixi run eval-bba04
pixi run eval-occ3d
pixi run eval-open-occupancy
pixi run eval-scene-as-occupancy
pixi run eval-surround-occ
```
Note: This requires the corresponding data to be processed / setup up properly.

To evaluate all methods at once run:
```
pixi run eval-all
```

The results are stored in the `./workdir` directory.