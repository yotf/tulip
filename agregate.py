#!/usr/bin/python
# -*- coding: utf-8 -*-

"""
Usage: 
    agregate.py PLOT_NAME DIRECTORY ...
    agregate.py -h | --help


Arguments:
    PLOT_NAME  L Number will be used for name of .aplot file
    DIRECTORY  Input directories to aggregate over. 
Options:
    -h --help
"""

import os
from os.path import join
import pdb
import re
import glob
import pandas as pd
import numpy as np
from docopt import docopt

def fname(mat_files):
    matf = dict()
    for mat in mat_files:
        therm_count = int(re.match(r"^.*?THERM(\d+).*?\.mat",
                          mat).groups()[0])
        matf[therm_count] = mat
    return matf[max(matf.keys())]

def agg2(agg_mat, aplot):

    T = agg_mat.ix['T'] / 100
    susc = (agg_mat.ix['M2avg'] - agg_mat.ix['M1avg'] ** 2) / T

    Tcap = (agg_mat.ix['E2avg'] - agg_mat.ix['Eavg'] ** 2) / T ** 2
    U = 1.0 - 1.0 / 3.0 * (agg_mat.ix['M4avg'] / agg_mat.ix['M2avg'] ** 2)
    print 'KUMULANT', U
    print 'AGG_MAT\n', agg_mat
    out_index = [
        'T',
        'M1avg',
        'M2avg',
        'M4avg',
        'susc',
        'Eavg',
        'E2avg',
        'Tcap',
        'U'        
        ]
   
    #ove Z kolone mi veerovatno ne trebaju??
    # out_data={
    #     'T': T,
    #     'M1avg':agg_mat.ix['M1avg'],
    #     'M2avg':agg_mat.ix['M2avg'],
    #     'M4avg': agg_mat.ix['M4avg'],
    #     'susc':susc,
    #     'Eavg':agg_mat.ix['Eavg'],
    #     'E2avg':agg_mat.ix['E2avg'],
    #     'Tcap':Tcap,
    #     'U':U}

    
#    out = pd.DataFrame(out_data, index = agg_mat.columns)

    out = pd.DataFrame([
        T,
        agg_mat.ix['M1avg'],
        agg_mat.ix['M2avg'],
        agg_mat.ix['M4avg'],
        susc,
        agg_mat.ix['Eavg'],
        agg_mat.ix['E2avg'],
        Tcap,
        U
        ], columns=agg_mat.columns, index=out_index)

    # out = out.T
    print "OUT",out
    #prvo se sortira, posto po kolonama ne radi sort. ocigledno.
    #mozda bi bilo bolje da se tako i salje da ne radim ovo T
    out.sort_index(axis=1,inplace=True)

    out.sort_index(axis=1,inplace=True)

    maps = {value:"T"+str(value) for value in out.columns }
    out.rename(columns = maps,inplace=True)

    print "MAPS",maps
    print out
    print aplot
    out.to_csv(aplot)

def agg(agg_mat,aplot):

    T = agg_mat.ix['T'] / 100
    susc = (agg_mat.ix['M2avg'] - agg_mat.ix['M1avg'] ** 2) / T

    Tcap = (agg_mat.ix['E2avg'] - agg_mat.ix['Eavg'] ** 2) / T ** 2
    U = 1.0 - 1.0 / 3.0 * (agg_mat.ix['M4avg'] / agg_mat.ix['M2avg'] ** 2)
    print 'KUMULANT', U
    print 'AGG_MAT\n', agg_mat
    out_index = [
        'T',
        'M1avg',
        'M2avg',
        'M4avg',
        'susc',
        'Eavg',
        'E2avg',
        'Tcap',
        'U'        
        ]
   
    #ove Z kolone mi veerovatno ne trebaju??
    # out_data={
    #     'T': T,
    #     'M1avg':agg_mat.ix['M1avg'],
    #     'M2avg':agg_mat.ix['M2avg'],
    #     'M4avg': agg_mat.ix['M4avg'],
    #     'susc':susc,
    #     'Eavg':agg_mat.ix['Eavg'],
    #     'E2avg':agg_mat.ix['E2avg'],
    #     'Tcap':Tcap,
    #     'U':U}

    
#    out = pd.DataFrame(out_data, index = agg_mat.columns)

    out = pd.DataFrame([
        T,
        agg_mat.ix['M1avg'],
        agg_mat.ix['M2avg'],
        agg_mat.ix['M4avg'],
        susc,
        agg_mat.ix['Eavg'],
        agg_mat.ix['E2avg'],
        Tcap,
        U
        ], columns=agg_mat.columns, index=out_index)

   
    print "OUT",out
    print out
    print aplot
    out.to_csv(aplot)
    

def main(mat_dict,simd):
    """Prima za parametar dict koji sadrzi apsolutne
    putanje do izabranih .mat fajlova"""
    #za svako L ce imati vise T-ova, znaci vise foldera
    # za odredjeno L, i za svaki od tih izabran mat
    # prebaci ga u regularni dict pre nego sto ga prosledis
    for l,tdict in mat_dict.items():
        agg_mat = list()
        for t,path in tdict.items():
            data_mat = pd.read_table(path,index_col=0,delim_whitespace=True,names=[t])
            data_mat.rename(index={'THERM': 'T'},inplace=True)
            data_mat.ix['T']=int(t[1:])
            agg_mat.append(data_mat)
        #uvek cemo imati samo jednu kolonu, naravno
        agg_mat = sorted(agg_mat, key=lambda x: int(x.columns[0][1:]))
        print agg_mat
        
        if(tdict.items()):
            agg_mat = pd.concat(agg_mat, axis=1)
            agg(agg_mat,join(simd,"%s.aplot" % l ))


        

def main2(sim_dir,PLOT_NAME,DIRECTORY):
    """Ide kroz foldere bira najbolje od mat i stata,
    i pravi agregat od najboljeg.Umesto broja termalizacionh ciklusa cuva temperature.
    Pre njega se pozvala stat.py skripta tako da su matovi i statovi
    najazurniji, je l'? Ako se poziva zasebno mora se pozivati is foldera gde su svi rezultati"""
    print sim_dir

    agg_mat = list()
    agg_stat = list()

    # pravimo .aplot fajl za prosledjeni L
    aplot =join(sim_dir,PLOT_NAME +".aplot")

    print DIRECTORY
    dirs = [d for d in glob.glob(join(sim_dir,DIRECTORY)) if os.path.isdir(d)]
    print dirs
    for dirc in dirs:
        os.chdir(join(sim_dir,dirc))
        print "CURRENTLY IN:",os.getcwd()
        new_mats = glob.glob('*.mat')
        best_mat = fname(new_mats)
        T = int(re.match(r'^L\d+T(\d{1,4})', best_mat).groups()[0])
        print best_mat
        best_stat = best_mat.replace('.mat', '.stat')
             
        print "BEST STAT\n", best_stat
        data_mat = pd.read_table(best_mat, index_col=0,
                                 delim_whitespace=True, names=[T])
        print "DATA_MAT\n", data_mat
        data_stat = pd.read_table(best_stat, index_col=0,
                                  delim_whitespace=True, names=[T])
        data_mat.rename(index={'THERM': 'T'}, inplace=True)
        data_stat.rename(index={'THERM': 'T'}, inplace=True)
        data_mat.ix['T'] = T
        data_stat.ix['T'] = T
        print data_mat[T]
        
        agg_mat.append(data_mat)
        agg_stat.append(data_mat)
             
    agg_mat = pd.concat(agg_mat, axis=1)
    print agg_mat
    agg_stat = pd.concat(agg_stat, axis=1)
    agg2(agg_mat, aplot)

if __name__=="__main__":
    arguments = docopt(__doc__)
    main2(sim_dir=os.getcwd(),PLOT_NAME=arguments['PLOT_NAME'],DIRECTORY=arguments['DIRECTORY'][0])

