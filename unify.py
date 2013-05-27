
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
import pandas as pd
import sys
import os
from os.path import join
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




def check_seeds(seeds):
    """Checks if first column of a whitespace seperated file
    has unique values of longlong's.
    """
    duplicates = seeds[seeds.duplicated()]
    
    if len(duplicates)>0:
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
    for key, value in group_file.items():
        print 'Simulacija', key
        # ako su vec napravljeni all fajlovi
        prev_alls = glob.glob(join(ltdir,"%s[0-9]*.all") % key)
        # nesto nije u redu ako ima vise .all fajlova
        assert(len(prev_alls)<=1)
        # ako vec postoji all fajl za dato lt i therm
        #ako postoji vec onda ga ucitavamo, u suprotnom pravimo prazan
        #:((((((( iff
        if not prev_alls:
            all_frame=pd.DataFrame(columns=['seed', 'E', 'Mx', 'My', 'Mz'])
        else:
            all_frame =  pd.read_table(prev_alls[0],delim_whitespace=True,names=['seed', 'E', 'Mx', 'My', 'Mz'])
            os.remove(prev_alls[0])

        # znaci za svaki dat fajl. on ga cita
        # zadaje mu kolone
        #numpy je jako spor :(
        for sim_file in value:
            dat = pd.read_table(sim_file,names=['seed', 'E', 'Mx', 'My', 'Mz'],skiprows =
                                1,delim_whitespace=True)
#            dat =pd.DataFrame(np.loadtxt(sim_file,dtype=np.str),columns=['seed', 'E', 'Mx', 'My', 'Mz'])
            all_frame=pd.concat([all_frame,dat])
            os.remove(sim_file)      
        check_seeds(all_frame.seed)
        all_frame.to_csv(join(ltdir,"%s%s.all" %(key,len(all_frame.seed))) ,sep=" ",header=False,index=False)
        
        
        
        


if __name__=="__main__":
    arguments = docopt(__doc__)

    main(os.getcwd())