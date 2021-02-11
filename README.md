YASS: Yet Another Spike Sorter
================================


[UPDATE March 2020] 
------------------
Yass ver. 2.0 has now been released and is available in the master branch. A manuscript is now available on Biorxiv (https://www.biorxiv.org/content/10.1101/2020.03.18.997924v1). 


YASS - Spike sorting retinal recordings
------------------
YASS is a spike sorting pipeline developed for high-firing rate, high-collision rate retinal recordings.  YASS employs a number of largely automated approaches to isolate single neuron templates and match them to the raw data. In monkey retinal recordings we found that YASS can identify dozens to hundreds of additional neurons not identified by human or other sorters and that such additional neurons have receptive fields and can be used to more accurately decode images.

<p float="center"> 
<img src="https://raw.githubusercontent.com/wiki/paninski-lab/yass/images/rfs.png" width="500" height="310">
<img src="https://raw.githubusercontent.com/wiki/paninski-lab/yass/images/decoding.png" width="330" height="310">
</p>


YASS - Spike sorting Neuropixel and other cortical/subcortical recordings
------------------
YASS is currently undergoing development to be extended to Neuropixel datasets as well as other electrode layouts (16 channel/4 shank probes, Utah arrays, and others) where electrode drift or unique electrode layouts are present.  YASS can currently be run on these types of datasets using the example neural networks provided (or retraining; see https://github.com/paninski-lab/yass/wiki/Neural-Networks---Loading-and-Retraining).  We will post updates regarding further development of dynamic drift in YASS in a separate branch: YASS-Neuropixels. In the mean time please feel free to reach out regarding novel layouts and datasets.


Installation and Running Instructions
---------
YASS can be run on AWS or installed on local workstations that have GPUs. Please review the YASS-Wiki (https://github.com/paninski-lab/yass/wiki) for more information. Brief installation instructions are also here:

### 1. Installing Anaconda and creating YASS environment:
```conda create -n yass python=3.7
source activate yass
```

### 2. Cloning and installing YASS python code:
```git clone https://github.com/paninski-lab/yass
 pip install numpy
 cd yass
 pip --no-cache-dir install -e .
 conda install pytorch==1.2
 ```

### 3. Compiling CUDA code:
```cd src/gpu_bspline_interp
python setup.py install --force
cd ..
cd gpu_rowshift
python setup.py install --force
```

```cd ../..
pip install .
```
 

Reference
---------

YASS 2.0 manuscript is available on biorxiv: https://www.biorxiv.org/content/10.1101/2020.03.18.997924v1

The older version can be found here: 

Lee, J. et al. (2017). YASS: Yet another spike sorter. Neural Information Processing Systems. Available in biorxiv: https://www.biorxiv.org/content/early/2017/06/19/151928

------------
