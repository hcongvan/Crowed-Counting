## Dataset for crowded counting
# How to label:
- Just dot all objects which you think runing on road, follow example below:

![origin|256x144](./imgs/260.jpg)
![label|256x144](./imgs/260_label.jpg)
# Generate density
- After label and save you dataset, you must have take next action to get density map. Following command below:
```
    ptyhon3 density.py -p [path/to/dataset]
```
- `-p`: path to dataset labeled (contain label file - txt)
# Store h5py
- After run command above, tool will generate `dataset.hdf5` contain all density map of dataset
- contruct of `dataset.hdf5`:

    `density`: groups dataset

        `name image 1`: value density 1

        `name image 2`: value density 2

        `name image 3`: value density 3

         .

         .

         .
         
# Architect folder dataset crowded counting
- folder dataset:
    image 1

    image 2
    
    .

    .

    .

    image n

    dataset.hdf5