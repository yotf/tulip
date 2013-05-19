#!/usr/bin/python
# -*- coding: utf-8 -*-

"""
=================================================================
 Od statisticki obradjenih podataka (.mat) treba napraviti agregate - 
 fajlove iz kojih se mogu crtati grafici 
 (sadrze podatke za isto L, T i MC u funkciji THERM).
 Oblik imena: LnTnMcn  (ekstenzija '_THERM.plot' ce biti kasnije dodata)
=================================================================

Usage: compose.py  [-h |--help]

"""

from docopt import docopt
import re
import os
from os.path import join
import pandas as pd
docopt(__doc__)


def create_plot(df, cv_name,ltdir):
    out = { 'abs(cv(E1))':abs(df.ix['stdMeanE'] / df.ix['Eavg']),
            'cv(E2)':df.ix['stdMeanE2']/df.ix['E2avg'],
            'cv(M1)':df.ix['stdMeanM1']/df.ix['M1avg'],
            'cv(M2)':df.ix['stdMeanM2']/df.ix['M2avg'],
            'cv(M4)':df.ix['stdMeanM4']/df.ix['M4avg']}
    out = pd.DataFrame(out)
    out = pd.concat([df,out.T])
    print out
    out.to_csv(join(ltdir,cv_name))
    #cisto da mogu da se igram sa njim is ipythona
    #mada organizacija bi mogla da mi bude drugcija
    #da vracam npr cv i  onda ih tamo konkateniram
    # stagod
    return out


    #ok hocu ovde da samo pravim compose onih novih
    # ako vec zovem tamo iz onoga
    #znaci ako nismo specificirali trazi .mat sa bilo
    # kojim mc. ako smo specificirali, onda trazi samo te
    # odredjene
def main(ltdir,mcsteps="\d+"):
    glregex = \
        re.compile(r"^(L\d+)(T\d+)THERM\d+(MC%s)\.mat$" % mcsteps)
    file_list = [f for f in os.listdir(ltdir) if glregex.match(f)]
    plots = list(set([glregex.match(f).groups() for f in file_list]))
    dataf= "nesto"
    for plot in plots:
        data = dict()
        for mat in re.findall("^%s%sTHERM.*%s\.mat$" % plot,
                              '\n'.join(file_list), re.MULTILINE):
            ser = pd.read_csv(join(ltdir,mat), sep=' ', index_col=0, names=['t'])['t']
            data[ser.ix['THERM']] = ser

        base = '%s%s%s_THERM' % plot
        plotf = base + '.plot'
        print "making",plotf,"..."
        dataf = pd.DataFrame(data)
        zdata = create_plot(dataf, plotf,ltdir)


if __name__=="__main__":
    main(os.getcwd())
