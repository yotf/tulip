#!/usr/bin/python
# -*- coding: utf-8 -*-
"""
=================================================================
Statistical calculations
=================================================================

Usage: 
  stat.py  [N]
  stat.py  -h | --help

Arguments:
    N    Take into account just first n results.
Options:
    -h --help
"""


from docopt import docopt
import sys
import pandas as pd
import numpy as np
import os
import re
from os.path import join
import logging


def create_mat(in_file,out_file_base,therm,mc_count,booli=False):
    """Dobija ime all_fajla,
    """
    out_file = "{}MC{}.mat".format(out_file_base,mc_count)
    log.debug("out file: %s" %out_file)
    data = pd.read_table(in_file, nrows=mc_count, delim_whitespace=True, names=['seed', 'E', 'Mx', 'My', 'Mz'])
    data.pop('seed')
    booli = booli if booli else [True]*len(data.index)
    data = data.loc[booli]
    log.debug("matdata booled :\n %s",data)
    N = len(data.index)
    log.debug("broj rezultata: %s" % N)
    Eavg = data.E.mean()
    stdE = data.E.std()
    stdMeanE = stdE / np.sqrt(N)
    E2 = data.E ** 2
    E2avg = E2.mean()
    stdE2 = E2.std()
    stdMeanE2 = stdE2 / np.sqrt(N)
    MAG = data[['Mx', 'My', 'Mz']]
    MAG2 = MAG ** 2
    M2 = MAG2.Mx + MAG2.My + MAG2.Mz
    M1 = np.sqrt(M2)
    M4 = M2 ** 2
    M2avg = M2.mean()
    M1avg = M1.mean()
    M4avg = M4.mean()
    stdM1 = M1.std()
    stdM2 = M2.std()
    stdM4 = M4.std()
    stdMeanM1 = stdM1 / np.sqrt(N)
    stdMeanM2 = stdM2 / np.sqrt(N)
    stdMeanM4 = stdM4 / np.sqrt(N)
    val_names = [
        'THERM',
        'Eavg',
        'stdE',
        'stdMeanE',
        'E2avg',
        'stdE2',
        'stdMeanE2',
        'M1avg',
        'stdM1',
        'stdMeanM1',
        'M2avg',
        'stdM2',
        'stdMeanM2',
        'M4avg',
        'stdM4',
        'stdMeanM4',
        ]
    values = pd.Series([
        therm,
        Eavg,
        stdE,
        stdMeanE,
        E2avg,
        stdE2,
        stdMeanE2,
        M1avg,
        stdM1,
        stdMeanM1,
        M2avg,
        stdM2,
        stdMeanM2,
        M4avg,
        stdM4,
        stdMeanM4,
        ], index=val_names)

    values.to_csv(out_file, sep=' ')


glregex = \
    re.compile(r"""
^           #Pocetak stringa poklapam
(L\d+       #L i bilo koji int posle
T\d+        #T sa bilo kojim int brojem posle
THERM(\d+))  #THERM sa bilo kojim int brojem posle
\.all$      #uzimamo .all fajlove
"""
               , re.VERBOSE)

def main(ltdir,n=None):
    from collections import defaultdict
    logging.basicConfig(level=logging.DEBUG)
    global log
    log = logging.getLogger(__name__)
    unified = [f for f in os.listdir(ltdir) if glregex.match(f)]
    therm_mc_dict = defaultdict(list)
    for u in unified:
        self.log.debug('Unified: %s '%u)
        base_name, therm_count = glregex.match(u).groups()

        mc_count=int(len(open(join(ltdir,u)).readlines()))
        #znaci ako je prosledjeno n koje je vece od mc
        # ono ne zadovoljava, i ne generise se ( u suprotnom bi
        # se samo prebrisao postojeci fajl )
        if n > mc_count:
            continue
        mc_count = n or mc_count
        # znaci ako generismo vec fajl
        # trebace da nam bude u ovom dictu
        therm_mc_dict[therm_count].append(mc_count)
        log.debug("mc_count=%s" % mc_count)
        out_file_base = join(ltdir,base_name)

        log.debug("Out .mat file base:%s" %out_file)
        in_file = join(ltdir,u)
        log.debug("In .all file: %s" % in_file )
        create_mat(in_file ,therm_count,out_file,mc_count)
    return therm_mc_dict


if __name__=="__main__":
    from voluptuous import All, Range, Schema, Invalid, Coerce
    schema = Schema({'N': All(Coerce(int), Range(min=1))}, extra=True)

    args = docopt(__doc__, version='0')

    try:
        args = schema(args)
    except Invalid, ex:
        print '\n'.join([e.msg for e in ex.errors])
        sys.exit()
    except TypeError, terr:
        n = 0
    else:
        n = args['N']
        
    main(os.getcwd(),n) 