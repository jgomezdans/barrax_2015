<img src=http://www.nceo.ac.uk/images/NCEO_logo_lrg.jpg />
# Material for the NERC STFC Field Spectroscopy Course
## Radiative Transfer theory and model inversion

### J Gomez-Dans (NCEO & UCL) `<j.gomez-dans@ucl.ac.uk>`

This repository contains a number of presentations and practicals used in the 
NERC funded Field Spectroscopy Course that took place in Albacete (Spain) in
July 2015. The presentations can be viewed as HTML files witin the browser, or
you can also peruse the original IPython notebooks that were used to create
them.

The practicals are in the form of IPython notebooks. Currently, they require 
you to have access to a computer that runs Linux, and to have several different
packages installed. If you don't have access to this, you can also use a 
VirtualBox image with all the required packages installed.

##### Presentations

You can find the following presentations:

* A motivation for RT modelling [[slides](http://jgomezdans.github.io/barrax_2015/RTMotivation.slides.html)][[IPython notebook](http://github.io/jgomezdans/barrax_2015/RTMotivation.ipynb)]
* A brief introduction to leaf RT models [[slides](http://jgomezdans.github.io/barrax_2015/LeavesRTpres.slides.html)][[IPython notebook](http://github.io/jgomezdans/barrax_2015/LeavesRTpres.ipynb)]
* An introduction to canopy RT models [[slides](http://jgomezdans.github.io/barrax_2015/CanopyRTpres.slides.html)][[IPython notebook](http://github.io/jgomezdans/barrax_2015/CanopyRTpres.ipynb)]
* An introduction to RT model inversion [[slides](http://jgomezdans.github.io/barrax_2015/InversionPres.slides.html)][[IPython notebook](http://github.io/jgomezdans/barrax_2015/InversionPres.ipynb)]

##### Tutorials

You can also find a number of tutorials. These are 

* A lab session on the PROSPECT leaf RT model [[IPython notebook]
(http://github.io/jgomezdans/barrax_2015/leaf_optics.ipynb)] [[HTML](http://jgomezdans.github.io/barrax_2015/leaf_optics.html)]
* A lab session on the SAIL (and PROSPECT) RT model [[IPython notebook](http://github.io/jgomezdans/barrax_2015/PROSAIL_experiments.ipynb)] [[HTML](http://jgomezdans.github.io/barrax_2015/PROSAIL_experiments.html)]

##### Installing the VirtualBox Image

The practicals are written in Python using the IPython notebook. 
While in principle this is a portable solution, compiling the PROSAIL model 
Python bindings has proven challenging on some systems, so we provide a 
Virtual Box Linux image that you can install on your computer and use to
run the contents of the course. 

The [Virtual Box file is here](http://www.geog.ucl.ac.uk/ucl-vm), but mind
you: **it is a 4.4Gb file**. You will need to download the [VirtualBox software](https://www.virtualbox.org/wiki/Downloads)
for your computer, and install the appliance. Note that you will need a
fairly modern computer to use the VBox comfortably: ideally more than 
4Gb of RAM.
