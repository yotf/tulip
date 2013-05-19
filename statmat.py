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

from voluptuous import All, Range, Schema, Invalid, Coerce
from docopt import docopt
import sys
import pandas as pd
import numpy as np
import os
import re
from os.path import join

# nije sigurno potrebno ovoliko svega



def create_mat(data, tmc, base,ltdir):
    N = len(data.index)
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
        tmc,
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

    values.to_csv(join(ltdir,base) + '.mat', sep=' ')



def create_mat_raw(data, tmc, base,ltdir):

    N = len(data.index)

    avg = data.mean()
    std_ = data.std()
    stdMean = std_ / np.sqrt(N)


    d = {
        'THERM': tmc,
        'mean': avg,
        'std': std_,
        'stdMean': stdMean,
        }
    exp = pd.DataFrame(index=data.columns, data=d)
    exp.to_csv(join(ltdir,base) + '.raw', sep=' ')




def create_stat(data, tmc, base,ltdir):

    N = len(data.index)
    avg = data.mean()
    std_ = data.std()
    stdMean = std_ / np.sqrt(N)
    MAG = data[['Mx', 'My', 'Mz']]
    MAG2 = MAG ** 2
    M2 = MAG2.Mx + MAG2.My + MAG2.Mz
    M1 = np.sqrt(M2)
    M4 = M2 ** 2
    M2avg = M2.mean()
    M1avg = M1.mean()
    M4avg = M4.mean()

    stdMeanM2 = 2 * np.sqrt(avg['Mx'] ** 2 * std_['Mx'] ** 2 
                            + avg['My'] ** 2 * std_['My'] ** 2 
                            + avg['Mz'] ** 2 * std_['Mz'] ** 2) / np.sqrt(N)

    # ili ovako

    stdMeanM2 = 2 * np.sqrt(avg['Mx'] ** 2 * stdMean['Mx'] ** 2
                            + avg['My'] ** 2 * stdMean['My'] ** 2
                            + avg['Mz'] ** 2 * stdMean['Mz'] ** 2)
    stdMeanM4 = M1avg * stdMeanM2

    out = pd.Series([
        tmc,
        M1avg,
        M2avg,
        stdMeanM2,
        M4avg,
        stdMeanM4,
        ], 
        index=[
        'THERM',
        'M1avg',
        'M2avg',
        'stdMeanM2',
        'M4avg',
        'stdMeanM4',
        ])
    out.to_csv(join(ltdir,base + '.stat'), sep=' ')


def create_output(data, tmc, base,ltdir):
    create_mat_raw(data, tmc, base,ltdir)
    create_mat(data, tmc, base,ltdir)
    create_stat(data, tmc, base,ltdir)



glregex = \
    re.compile(r"""
^           #Pocetak stringa poklapam
(L\d+       #L i bilo koji int posle
T\d+        #T sa bilo kojim int brojem posle
THERM(\d+)  #THERM sa bilo kojim int brojem posle
MC)(\d+)    #MC sa bilo kojim int broje posle
\.all$      #uzimamo .all fajlove
"""
               , re.VERBOSE)

def main(ltdir,n=None):
    unified = [f for f in os.listdir(ltdir) if glregex.match(f)]

    for u in unified:
        print 'Unified:  ', u
        base_name, therm_count, mc_count = glregex.match(u).groups()
        mc_count=int(mc_count)
        # jedino ako nije prosledjeno n ili ako je prosledjeno n
        # koje je manje ili jednako broj mc koraka u fajlu ima
        # smisla da bilo sta radimo
        if n is None or n<=mc_count:
            mc_count = n or mc_count
            print mc_count
            agregate = base_name + str(mc_count)
            print "Aggregate: ", agregate
            # hm, verovatno postoji bolji nacin odbacivanja prve kolone
            #proveri da li moze beline da gleda za separator
            data = pd.read_table(join(ltdir,u), nrows=mc_count, delim_whitespace=True, names=['seed', 'E', 'Mx', 'My', 'Mz'])[['E', 'Mx', 'My', 'Mz']]
            create_output(data, therm_count, agregate,ltdir)


if __name__=="__main__":
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