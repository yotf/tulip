Code documentation can be viewed here (under construction ) [tulipko](http://itrustedyou.bitbucket.org/tulipko)

#Tulip
Tulip is an interactive visualization and graph plotting software application written in Python, used for studying the thermodynamical characteristics of classical Heisenberg model in singlepath and multipath approach.

Application details can be found at [wiki].

## Binaries for Linux and Windows

No installation needed, just download and run the binaries.

### Linux (Tested on Ubuntu 12.04)

You can download the executable file from
the [downloads](https://bitbucket.org/iTrustedYOu/tulipko/downloads/tulipko)page. 

Run it by double clicking on it.

### Windows (Tested on Windows 7)

For Windows, you can download an exe file from the [downloads page](https://bitbucket.org/iTrustedYOu/tulipko/downloads/gui.exetulipko-win.exe).

Run it by double clicking on it!


If you are unable to run tulipko from the binaries provided
above, please report it as an [issue](https://bitbucket.org/iTrustedYOu/tulipko/issues). In the meantime, try the method described below. Should be fun.

## Dependencies and installation
If you wish to download and run, and contribute to the development
version, you will have to install the 'whole package',
meaning the python development version, multiple packages
that matplotlib is dependent on (including font packages,
latex, packages for manipulatin png images..), wxpython with wxwidgets
and the third party packages used for calculations and
statistical analysis - pandas, numpy and scipy.

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

  
   ```git clone https://iTrustedYOu@bitbucket.org/iTrustedYOu/tulipko.git```

go into the cloned directory



```   cd tulipko ```

And install tulipko



```  python setup.py install ```

or alternatively if you don't want to install
it on your system but run it as a python script,
you can download all the dependancies via ```pip install package_name```
and then run the ```gui.py``` script with ```python src/gui.py```

If you installed tulipko, you should be able to run it
by typing ```tulipko``` in your terminal. 


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