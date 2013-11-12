Code documentation can be viewed here (under construction ) [tulipko](http://itrustedyou.bitbucket.org/tulipko)

#Tulip
Tulip is an interactive visualization and graph plotting software application written in Python, used for studying the thermodynamical characteristics of classical Heisenberg model in singlepath and multipath approach.

Application details can be found at [wiki].

## Dependencies and installation

For Ubuntu 12.04 installationa, a binary
can be downloaded at the [downloads](https://bitbucket.org/iTrustedYOu/tulipko/downloads/tulipko) page. 

For Windows, you can download a self contained .exe file from the [downloads page](https://bitbucket.org/iTrustedYOu/tulipko/downloads/gui.exetulipko-win.exe) as well.

For other linux distrubitions and OS-es, you can try and  follow the instructions belllow. Should be fun. 

#### Installing Python2.7
run the following comands in terminal:
```
sudo add-apt-repository ppa:fkrull/deadsnakes
sudo apt-get update
sudo apt-get install python2.7 
```
That will install Python2.7 as ``python``

#### Installing matplotlib dependencies
`` sudo apt-get build-dep python-matplotlib ``

### Installing pip
The Python Package installer pip is needed...
``sudo apt-get pip install``

#### Installing WxPython:

  -   Install wxGTK 2.8 with the command, 'sudo apt-get install python-wxgtk2.8'
  -   Run the command, 'apt-get source -d wxwidgets2.8'
  -   Now run, 'dpkg-source -x wxwidgets2.8_2.8.12.1-6ubuntu2.dsc'
  -   cd wxwidgets2.8-2.8.12.1
  -   cd wxPython
  -   Now run the command, 'sudo python setup.py install'
  -   wxPython and wxWidgets are now successfully installed!

### Installing tulipko
after cloning the repo with command:
::
  
   git clone https://iTrustedYOu@bitbucket.org/iTrustedYOu/tulipko.git

go into the cloned directory

::

   cd tulipko

And install tulipko

::

  python setup.py install

Now you can run tulipko with command ``tulipko`` in the terminal


## Running the application

Once installed you can go ahead and run tulipko by typing
the command ``tulipko`` in terminal or

If you are using the binary, just run it by double clicking it.


## Quick intro

Once started tulipko will ask for the path to your TDF (Tulip Data Folder),
that contains one or multiple 
simulation folders (folders with different parameters other than the standards
L, T, SP, LS). The directory structure looks like this, so make sure to select
your TDF folder (won't work otherwise!)
```
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

```