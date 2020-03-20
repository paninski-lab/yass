YASS: Yet Another Spike Sorter
================================


[UPDATE March 2020] 
------------------
Yass ver. 2.0 has now been released and is available in the master branch. A manuscript is now available on Biorxiv (https://www.biorxiv.org/content/10.1101/2020.03.18.997924v1). 


YASS - Spike sorting 
------------------
YASS is a spike sorting pipeline developed for high-firing rate, high-collision rate retinal recordings.  YASS employs a number of largely automated approaches to isolate single neuron templates and match them to the raw data. In monkey retinal recordings we found that YASS can identify dozens to hundreds of additional neurons not identified by human or other sorters and that such additional neurons have receptive fields and can be used to more accurately decode images.

<p float="center"> 
<img src="https://raw.githubusercontent.com/wiki/paninski-lab/yass/images/rfs.png" width="500" height="310">
<img src="https://raw.githubusercontent.com/wiki/paninski-lab/yass/images/decoding.png" width="330" height="310">
</p>


YASS - Spike sorting Neuropixel and other cortical/subcortical recordings
------------------
YASS is currently undergoing development to be extended to Neuropixel datasets as well as other electrode layouts (16 channel/4 shank probes, Utah arrays, and others).  YASS can be run on these datasets (provided that the neural-networks are retrained on the novel layouts) when substantial drift is not present in the recordings.  We will post updates regarding further development of dynamic drift in YASS in a separate branch: YASS-Neuropixels. In the mean time please feel free to reach out should you have questions.


Installation and Running Instructions
---------
YASS can be run on AWS or installed on local workstations that have GPUs. Please review the YASS-Wiki (https://github.com/paninski-lab/yass/wiki) for more information.


 

Reference
---------

YASS 2.0 manuscript is available on biorxiv: https://www.biorxiv.org/content/10.1101/2020.03.18.997924v1

The older version can be found here: 

Lee, J. et al. (2017). YASS: Yet another spike sorter. Neural Information Processing Systems. Available in biorxiv: https://www.biorxiv.org/content/early/2017/06/19/151928

------------
