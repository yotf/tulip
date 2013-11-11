Code documentation can be viewed here (under construction ) [tulipko](http://itrustedyou.bitbucket.org/tulipko)

#Tulip
Tulip is an interactive visualization and graph plotting software application written in Python, used for studying the thermodynamical characteristics of classical Heisenberg model in singlepath and multipath approach.

Application details can be found at [wiki].

## Dependencies and installation

For Ubuntu 12.04 installationa, a binary
can be downloaded at the [downloads](https://bitbucket.org/iTrustedYOu/tulipko/downloads) page. Otherwise, follow the instructions bellow and have fun!


first, run this in terminal, it will install all the dependencies and packages needed
except wxpython

```
sudo add-apt-repository ppa:fkrull/deadsnakes && sudo apt-get update && sudo apt-get install python2.7 && sudo apt-get install python-pip && sudo pip install numpy && sudo pip install pandas && sudo apt-get build-dep python-matplotlib && sudo pip-install matplotlib && sudo pip-install scipy
```
INSTALLING WXPYTHON:

  -   Install wxGTK 2.8 with the command, 'sudo apt-get install python-wxgtk2.8'
  -   Run the command, 'apt-get source -d wxwidgets2.8'
  -   Now run, 'dpkg-source -x wxwidgets2.8_2.8.12.1-6ubuntu2.dsc'
  -   cd wxwidgets2.8-2.8.12.1
  -   cd wxPython
  -   Now run the command, 'sudo python setup.py install'
  -   wxPython and wxWidgets are now successfully installed!


Once the previous steps were completed successfully you can go 
ahead and clone this repository with the command

```
 git clone https://iTrustedYOu@bitbucket.org/iTrustedYOu/tulipko.git
```

And voila, tulipko is installed - that easy!!! 


## Running the application

Once installed, you can go ahead and run the application
from the folder you cloned tulipko into (tulipko by default)
with this command:

```
python src/gui.py
```

If you are using the binary, just run it by double clicking it.


## Quick intro

Once started tulipko will ask for the path to your TDF (Tulip Data Folder),
that contains one or multiple 
simulation folders (folders with different parameters other than the standards
L, T, SP, LS). The directory structure looks like this, so make sure to select
your TDF folder (won't work otherwise!)

+---3D-N3-MP   {TDF}
|   +---g0
|   |   +---LT dirs
|   |
|   +---g0.005
|   |   +---LT dirs
|   ...
|
+---3D-N3-SP   {TDF}
|   +---g0
|   |   +---LT dirs
...

