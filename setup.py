from setuptools import setup, find_packages
from os.path import join,dirname
import src
import os
import sys
# if sys.argv[1].strip() == 'install':
#     os.system('sudo apt-get install python-wxgtk2.8')
#     os.system('sudo apt-get build-dep python-matplotlib')
#     os.system('apt-get source -d wxwidgets2.8')
#     os.system('dpkg-source -x wxwidgets2.8_2.8.12.1-6ubuntu2.dsc')
#     os.system('cd wxwidgets2.8-2.8.12.1')
#     os.system('cd wxPython')
#     os.system('sudo python setup.py install')
setup(
    
    name='tulipko',
    version=src.__version__,
    packages=find_packages(),
    long_description = open(join(dirname(__file__),'README.md')).read(),
    entry_points= {
        'console_scripts':
        ['tulip = src.gui:main']
        },
    install_requires=['numpy==1.7.1','scipy==0.9.0','matplotlib==1.2.1','pandas==0.11.0']
    )



