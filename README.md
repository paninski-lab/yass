YASS: Yet Another Spike Sorter
================================

------------------------------
***[UPDATE Sep 2022]***

Please note that default pyyaml no longer works and the following command needs to be run after all installation steps:

 !pip install pyyaml==5.4.1   

------------------------------------
***[UPDATE Feb 2021]***

Yass is now available on colab including installation and operating instructions:

https://colab.research.google.com/drive/1Qp7RAnPFj8zrhfEGkV7nHbzFiQ8w-4FX?usp=sharing

Yass is also avialable on Spike Interface:

https://github.com/SpikeInterface

------------------------------------
***[UPDATE Mar 2020]***

Yass ver. 2.0 has now been released and is available in the master branch. A manuscript is now available on Biorxiv (https://www.biorxiv.org/content/10.1101/2020.03.18.997924v1). 

YASS is a spike sorting pipeline developed for high-firing rate, high-collision rate retinal recordings.  YASS employs a number of largely automated approaches to isolate single neuron templates and match them to the raw data. In monkey retinal recordings we found that YASS can identify dozens to hundreds of additional neurons not identified by human or other sorters.

<p float="center"> 
<img src="https://raw.githubusercontent.com/wiki/paninski-lab/yass/images/rfs.png" width="500" height="310">
<img src="https://raw.githubusercontent.com/wiki/paninski-lab/yass/images/decoding.png" width="330" height="310">
</p>

YASS is currently undergoing development to be extended to Neuropixel datasets as well as other electrode layouts (16 channel/4 shank probes, Utah arrays, and others) where electrode drift or unique electrode layouts are present.  

--------------------------
**Installation and Running Instructions**

We recommend using provided colab notebooks:

https://colab.research.google.com/drive/1Qp7RAnPFj8zrhfEGkV7nHbzFiQ8w-4FX?usp=sharing

Or the existing implementation on Spike Interface:

https://github.com/SpikeInterface

Alternatively, you can use the local install options or the AWS provided images:

https://github.com/paninski-lab/yass/wiki
 

------------------
**Reference**

YASS 2.0 manuscript is available on biorxiv: https://www.biorxiv.org/content/10.1101/2020.03.18.997924v1

The older version can be found here: 

Lee, J. et al. (2017). YASS: Yet another spike sorter. Neural Information Processing Systems. Available in biorxiv: https://www.biorxiv.org/content/early/2017/06/19/151928

------------
