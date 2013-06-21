#!/usr/bin/python
# -*- coding: utf-8 -*-
"""
=================================================================
Nalazi sve simulacije koje imaju isti L T i THERM i spaja njihove
MC korake u jednu datoteku. 
=================================================================

Usage: util.py 


"""
import numpy as np
import pandas as pd
import sys
import os
from os.path import join
from unittest import TestCase
import re
from docopt import docopt



glregex = \
    re.compile(r"""
^           #Pocetak stringa poklapam, ovo bi trebalo da radi i \n
(L\d+       #Poklapam L sa bilo kojim int brojem posle
T\d+        #Poklapam T sa bilo kojim int brojem posle
THERM\d+)   #Poklapan THERM sa bilo kojim int brojem posle
MC(\d+)     #Poklapam MC sa bilo kojim int broje posle
.*\.dat$    #uzimamo .dat fajlove
"""
               , re.VERBOSE)

mcregex = r'%sMC\d+.*?\.dat'




def check_seeds(seeds):
    """Checks if first column of a whitespace seperated file
    has unique values of longlong's.
    """
    duplicates = seeds[seeds.duplicated()]
    if len(duplicates)>0:
        print 'Ima duplikata medju seedovima!! ABORT ABORT!!'
        print 'Duplikati su:\n' + str(duplicates)
        sys.exit()

def main(ltdir):
    print __doc__
    print "IN DIRECTORY: ",ltdir
    all_files = [f for f in os.listdir(ltdir) if glregex.match(f)]
    all_files.sort()
    group_file = dict()
    while all_files:
        grupa = glregex.search(all_files[0]).groups()[0]
        group_file[grupa] = re.findall(mcregex % grupa, ' '.join(all_files))
        all_files = all_files[len(group_file[grupa]):]
    # Mnogo jednostavnije, a dozvoljava postojanje proizvoljnog broja 
    # linija komentara na proizvoljnim mestima (ili nepostojanja komentara).
    for key, value in group_file.items():
        print 'Simulacija', key
        ofName = join(ltdir,"%s.all") % key   # Output file name
        with open(ofName, 'a') as of:
            for sim_file in value:
                with open(sim_file) as f:
                    for l in f:
                        if l[0] != '#':
                            of.write(l)

        # Proveri da li je fajl ispravan (bar donekle):
        # ako je neka od simulacija proizvela neispravan simulation output
        # (manje od 5 kolona) -> prijavi gresku
        # (dtype garantuje da ce prva kolona biti interpretirana kao long long).
        all_frame =  pd.read_table(ofName,names=['seed', 'E', 'Mx', 'My', 'Mz'],
                                   delim_whitespace=True, 
                                   dtype={'seed':np.int64})
        check_seeds(all_frame.seed)
        
        
        
        


if __name__=="__main__":
    docopt(__doc__)

    main(os.getcwd())
