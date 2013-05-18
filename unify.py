
#!/usr/bin/ipython
# -*- coding: utf-8 -*-
"""
=================================================================
Nalazi sve simulacije koje imaju isti L T i THERM i spaja njihove
MC korake u jednu datoteku. 
=================================================================

Usage: util.py 


"""
#  -a --append  Dodavanje na vec postojace fajlove [default:False].
# [-a |--append]
import numpy as np
import sys
import os
from unittest import TestCase
import re
import glob
from docopt import docopt



glregex = \
    re.compile(r"""
^           #Pocetak stringa poklapam, ovo bi trebalo da radi i \n
(L\d+       #Poklapam L sa bilo kojim int brojem posle
T\d+        #Poklapam T sa bilo kojim int brojem posle
THERM\d+    #Poklapan THERM sa bilo kojim int brojem posle
MC)(\d+)    #Poklapam MC sa bilo kojim int broje posle
.*\.dat$    #uzimamo .dat fajlove
"""
               , re.VERBOSE)

mcregex = r'%s\d+.*?\.dat'




def check_seeds(sim_file):
    """Checks if first column of a whitespace seperated file
    has unique values of longlong's.
    Returns: True if unique, False otherwise
    """

    sim_file.seek(0)
    seeds = np.loadtxt(sim_file, usecols=(0, ), dtype=np.str)
    sim_file.seek(0)
    (uniqueVal, i) = np.unique(seeds, return_inverse=True)
    duplicates = uniqueVal[np.bincount(i) > 1]
    if duplicates:
        print 'Ima duplikata medju seedovima!! ABORT ABORT!!'
        print 'Duplikati su' + str(duplicates)
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
    for (key, value) in group_file.items():
        print 'Simulacija', key
        # ako su vec napravljeni all fajlovi
        prev_alls = glob.glob("%s[0-9]*.all" % key)
        # nesto nije u redu ako ima vise .all fajlova
        assert(len(prev_alls)<=1)
        # ako vec postoji all fajl za dato lt i therm
        # onda se dodaje na njega i menja mu se ime
        ze_all = "temp" if not prev_alls else prev_alls[0]
        with open(ze_all, "a+") as svi:
            for sim_file in value:
                with open(sim_file) as dat_file:
                    svi.write(''.join([line for line in dat_file.readlines()
                                       if not line.strip().startswith("#") ]) )
                    os.remove(sim_file)
            check_seeds(svi)
            os.rename(ze_all, key + str(len(svi.readlines())) + '.all')

        # ##I one CESGA logove

if __name__=="__main__":
    arguments = docopt(__doc__)
#    mode = ('a+' if arguments['--append'] else 'w+')
    main(os.getcwd())