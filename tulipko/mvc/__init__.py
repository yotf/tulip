"""

.. module:: mvc
   :platform: Unix, Windows
   :synopsis:  MVC of the appliaction

.. moduleauthor:: Tara Petric <tara.petric@gmail.com>

"""

import wx
import mvc_skelet
import re
import logging
import glob
import os
import sys
from os.path import join
import util
import unify
from collections import defaultdict
import pickle
import pandas as pd
import numpy as np
from profile import *

class Choices(mvc_skelet.Model):
    base_regex = r'^L\d+T\d+(?P<therm>THERM\d+)'
    single_base = r'%sMC\d+.*' % base_regex
    sp_regex = re.compile(r'%s\.sp$' % single_base)
    mp_regex = re.compile(r'%s(\.all|MC\d+.*\.mp|MC\d+.*\.dat)$' % base_regex)
    all_regex = re.compile(r'%s\.all$' %base_regex)
    lt_regex = re.compile(r'(L\d+)(T\d+)$')
    statmc_regex = re.compile(r'^MC\d+$')
    def __init__(self):
        """
        Initializes the logger
        and sets all the dictionaries used to None
        for clarity
        : returns : Nothing
        """
        mvc_skelet.Model.__init__(self)
        self.log = logging.getLogger("Choices")
        self.bestmats = None
        self.mags = None
        self.files = None

        
    def init_model(self):
        """Initializes the model
        Runs when GUI is up
        Maps the current filesystem

         Sets the quantaties of interest that can be plotted:
        
        * susc
        * Tcap
        * M1avg
        * M2avg
        * M4avg
        * Eavg
        * E2avg
        * U

        .. todo::
        
           Custom formulae?
        
        :returns: Nothing
        
        :raises: BaseException If there was an exception \
        when mapping the filesystem, it is propagated to the app \
        and it exits with code 0
        
        """
        try:
            self.files = self._map_filesystem()
        except BaseException as e:
            raise e
        self.load_state()
        
        self.mags = [
        'susc',
        'Tcap',
        'M1avg',
        'M2avg',
        'M4avg',
        'Eavg',
        'E2avg',
        'U'        
        ]


    def add_virtual_file(self,dir_,l,t,mc,therm,booli):
        """Calculates the mat file for the given parameters
        Updates the direcotry structure to show the new
        number of SPs and the appropriate number of LSs

        Notifies observers of change in model
        
        :param dir_: The simulation directory.
        :param l: The linear lattice size.
        :type l: str.
        :type dir_: str.
        :param t: The given temperature.
        :type t: str.
        :param mc: The number of Simulation Paths.
        :type mc: str.
        :param therm: The number of Lattice Sweeps.
        :type therm: str.
        :param booli: When 'cutting' bad results.
        :type booli: boolean index.
        
        """
        mat = self.create_mat(dir_,l,t,therm,booli=booli)
        therm_ch = dict()
        therm_ch[therm]=mat
        self.files[dir_][l][t][mc].update(therm_ch)
        self.notify_observers()
        
    def add_bestmat(self,dir_,l,t,mc,therm):
        """Adds a file as the best choice
        to be used for later plotting.
        **Notifies** *observers*.
        
        :param dir_: The simulation directory.
        :param l: The linear lattice size.
        :type l: str.
        :type dir_: str.
        :param t: The given temperature.
        :type t: str.
        :param mc: The number of Simulation Paths.
        :type mc: str.
        :param therm: The number of Lattice Sweeps.
        :type therm: str.
        
        """
        
        
        self.bestmats[dir_] = self.bestmats[dir_] if self.bestmats.has_key(dir_) else dict()
        self.bestmats[dir_][l]=self.bestmats[dir_][l] if self.bestmats[dir_].has_key(l) else dict()
        self.bestmats[dir_][l][t] = {'mc':mc,'therm':therm}
        self.save_mats()
        self.notify_observers()

    def random_bestmats(self,dir_,**kwargs):
        """Chooses the \'best' results, by choosing the ones
        with the most LSs and SPs for a chosen linear lattice
        size

        Raises :class: BaseException, maybe 

        :returns: Nothing.
        
        :Parameters:
           - `dir_`: The simulation directory
           - `**kwargs`: Should be linear lattice size

        .. todo:: Make activated for combobox L select, instead \
        of matchooser L
        
        
        """
        try:
            l = kwargs['l']
            for t,mcdict in self.files[dir_][l].items():
                mc = max(mcdict.keys(),key=lambda x:util.extract_int(x))
                therm = max(mcdict[mc].keys(),key=lambda x:util.extract_int(x))
                self.add_bestmat(dir_,l,t,mc,therm)
        except IndexError:
            util.show_error('No Lattice Size selected','Contact developer to disable Random button')
        except BaseException as e:
            util.show_error("Don't know",str(e))
            raise e

    def save_mats(self):
        """Saves a deepcopy of the math dictionary, \
        excluding the \'virtual' ones

        :keyword virtual: The ones created on the *fly*
                          not in the filsystem
        
        """
        from copy import deepcopy
        for_save = deepcopy(self.bestmats)
        for dir_,ldict in for_save.items():
            for l,tdict in ldict.items():
                for t,val in tdict.items():
                    if '[' in val['mc']:
                        del tdict[t]
        self.log.debug('for save:%s' %for_save)
        self.log.debug('bestmat:%s' %self.bestmats)
        with open(join(self.simdir,'bestmat.dict'),'wb') as bmatfile:
            pickle.dump(dict(for_save),bmatfile)
            
    @logg
    def remove_bestmat(self,**kwargs):
        """Removes the bestmat for given **L** and **T**

        Cleans the bestmat dictionary, saves it, and notifies \
        observers of change in model
        

        :Parameters:
           - `**kwargs`:
              - `dir_` (req.)
              - `l` (req.)
              - `t` (req.)

        .. todo:: Implement for whole **L**. Hm, should work, but doesn't..

        """
        dir_= kwargs['dir_']
        l = kwargs['l']
        try:
            t = kwargs['t']
            del self.bestmats[dir_][l][t]
            if not self.bestmats[dir_][l]:
                self.log.debug('Sad je prazno za taj l, brisem ga celog')
                del self.bestmats[dir_][l]
            self.log.debug('Brisem iz bestmata za l:{} i t:{}'.format(l,t))
        except KeyError:
            try:
                self.bestmats.pop(l)
                self.log.debug('Brisem iz bestmata za l:{}'.format(l))
                self.log.debug('Sada bestmat izgleda ovako: {}'.format(self.bestmats))
            except KeyError:
                self.log.debug('Kliknuo korisnik na ne bestmat stvar. Idiot!')

        self.__clean_bmat()
        self.save_mats()
        self.notify_observers()
        
    def add_mc(self,mc,**kwargs):
        """
        .. _add_mc:
        
        Adds new SP, for drawing different
        precision. Adds it to the map of filesystem, while
        at the same time, taking care that the new file
        has the same LS choices as the actual file on the
        filesystem. Meaning that we create a virtual \'file'
        in memory that corresponds to the actual file, just
        *cut* in smaller peace so that it takes into account
        fewer SPs, for reduced precision, and observation sake

        :Parameters:
           -`**kwargs`: **L**inear lattice size, and **T**emperature .
        
        """

        statmc = self.__get_static_mcs(**kwargs)[0]
        kwargs['mc']=statmc
        therms = self.choices(**kwargs)
        tdict = {key:None for key in therms}
        self.log.debug("adding mc %s" % tdict )
        kwargs['mc']="MC%s" % mc
        self.__add_to_map(tdict = tdict,**kwargs)


    def __add_to_map(self,dir_,l,t,mc,tdict):
        """Adds to filesystem map, and notifies observers
        Called exclusively by add_mc_
        
        """
        self.files[dir_][l][t][mc]=tdict
        self.notify_observers()
        
    def get_maxmc(self,dir_,l,t):
        """
        :returns: The max number of Simulation Paths \
                  for chosen directory.

        """
        self.log.debug("Vracam max mc")
        return util.extract_int(max(self.choices(dir_,l,t),key =lambda x: util.extract_int(x)))
        
    @logg
    def choices(self,dir_=None,l=None,t=None,mc=None):
        """Depending on the parameters passed,
        this method returns the subsequent choices
        e.g. if dir_ was passed, it returns choices of
        Linear lattice size, etc.

        :param dir_: Simulation directory (opt.).
        :param l_: Linear lattice size (opt.).
        :param t_: Temperature(opt.).
        :param mc_: Number of Simulation Paths(opt.).
        
        """
        ls = [dir_,l,t,mc]
        x = self.files
        for val in ls:
            if not val:
                break
            x = x[val]
        return x.keys()

    def bestmat_choices(self,dir_=None,l=None,t=None,which='mc'):
        """
        Depending on the parameters passed, returns the subsequent
        optimal choices (made by the user) for the given parameters
        e.g. if only dir_ was passed returns a list of L choices.
        This method is used mostly by the \'view', since the display
        of many components, annotations, legends etc depend on this.
        When it happens that all arguments are passed, the components
        to be returned are plain strings (only one choices for given
        parameters) then it is wrapped in a list for uniformity's sake.

        :param dir_: The simulation directory.
        :param l: The linear lattice size.
        :type l: str.
        :type dir_: str.
        :param t: The given temperature.
        :type t: str.
        :param which: Do we want the choice of optimal SPs or LSs ('therm'|'mc')



        .. todo:: Define keywords used throughout documentation.

        .. todo:: Figure out what else is used for determening bestmats
                  Since this is only used by the \'view' part.


        """
        
        ls = [dir_,l,t,which]
        x = self.bestmats
        for val in ls:
            if not val:
                break
            try:
                x = x[val]
            except:
                return []
        self.log.debug("Vracam bestmat izbore :{} za l:{} t:{} mctherm:{} ".format(x,l,t,which))
        self.log.debug('type od povratne vrednosti je {}'.format(type(x)))
        
        return [x] if isinstance(x,basestring) else x.keys()
        
        
    def __get_static_mcs(self,**kwargs):
        """
        Called exclusively by the add_mc_ method

        :returns: All the SPs that correspond to actual simulation
                  precision on the disk.
        
        """
        statmcs = [ mc for mc in self.choices(**kwargs) if self.statmc_regex.match(mc)]
        self.log.debug("Vracam staticke mc-ove:{} za dir:{dir_} l:{l} t:{t}".format(statmcs,**kwargs))
        return statmcs
        
    def load_state(self):
        """
        Loads the state (for now only bestmat choices)
        that the user last left the application in..

        :returns: Nothing
        
        """
        self.load_bestmat()

    def __clean_bmat(self):
        """
        \'Cleans' the dictionary contatining
        user choices for bestmats. As in, removes
        the empty keys from the dictionary
        for presentation sake. Is called solely from
        methods that remove bestmat choices and the ones
        that loads them, since those are the only ones
        who can leave the dictionary *dirty*
        """
        
        for dir_,ldict in self.bestmats.items():
            if not ldict:
                del self.bestmats[dir_]
            for l,tdict in ldict.items():
                if not tdict:
                    del ldict[l]
                    continue
                for t,tmc_dict in tdict.items():
                    if not tmc_dict:
                        del tdict[t]
                        continue
                        
        

    def load_choices(self,fname):
        self.log.debug("Loading {}".format(fname))
        with open(join(self.simdir,fname),mode="ab+") as hashf:
            try:
                fcontent =  dict(pickle.load(hashf))
            except EOFError:
                fcontent = dict()
        return fcontent

    def load_sp_data(self,dir_,l,n=None):
        """Ucitava podatke za isrcatavanje na
        scatter panelu, i vraca u formatu pogodnom
        za plotovanje


        .. versionadded: 2.0
        
        """
        all_data = dict()
        print self.bestmats[dir_][l].items()
        for t,tmc in self.bestmats[dir_][l].items():
            print self.simdir,dir_,l,t,tmc['therm']
            filename = self.get_filename(dir_,l,t,tmc['therm'])
            self.log.debug("Loading data from file %s" % filename)
            # ne znam da li mi treba ovde neki try catch hmhmhmhmhmhmhmmhhh
            data = self.read_tdfile(filename,n)
            self.log.debug("rows read: %s" % data.E.count())
            data.set_index(np.arange(data.E.count()),inplace=True)
            all_data[t] = data
        
        data = pd.concat(all_data,axis=0)
        return data
            
        
    def unify(self,lt_dir):
        try:
            unify.main(lt_dir)
        except NotImplementedError :
            util.show_error("Duplicate Seeds","Duplicated seeds found while unifying! What to do, what to do????!?!?!")

    def spify(self,lt_dir):
        """Proverava da nema dva fajla sa istim thermovima
        i razlicitim mc-ovima. Blahhhhhh"""
        print 'spifyinggg'
        ls = [re.match(self.base_regex,f).groupdict()['therm'] for f in os.listdir(lt_dir) if re.match(self.base_regex,f)]
        print ls
        if len(ls)> len(set(ls)):
            util.show_error('Single path error','Two files with same therm different mcs in %s. Please delete one. Exiting now' %lt_dir)
            try:
                os.system("nautilus %s" % lt_dir)
            except:
                pass
            raise BaseException
        
        
    def therm_count(self,dir_,l,t,mc):
        """Vraca koliko ima tacaka za dato mc, tj.trebace kod
        ovog therm panela da vidi neko da ne plotuje nebulozne
        stvari. znaci ovo ce biti u nekim viticastim zagradama
        iza broja mc-ova"""
        therms = self.choices(dir_,l,t,mc)
        self.log.debug("Vracam broj thermova:{} za mc:{}".format(len(therms),mc))
        return len(therms)

    def name_gen(self):
        cntr=0
        constants = ['seed','E']
        while(True):
            if cntr>len(constants)-1:
                yield 'M%s' % (cntr-len(constants))
            else:
                yield constants[cntr]
            cntr=cntr+1

    def read_tdfile(self,filename,first_mcs):
        data = pd.read_table(filename,nrows=first_mcs,delim_whitespace=True,header=None)
        for col in data:
            if len(data[col].dropna())==0:
                data.pop(col)
        self.log.debug('len of columns %s' %len(data.columns))
        print data
        ngen = self.name_gen()
        names = [ngen.next() for col in data]
        data.columns = names
        data.pop('seed')
        return data

    def mag_components(self,data):
        """izvlaci kolone koje sadrze
        komponente magnetizacije i vraca
        ih u novoj tabeli"""
        ix = self.mag_index(data)
        MAG = data[ix]
        return MAG

    def mag_index(self,data):
        ix = []
        for col in data:
            if col.startswith('M'):
                ix.append(col)
        return ix

    def calculate_magt(self,data):
        """za datafrejm koji se sastoji
        od nekoliko kolona gde su neke od njih
        magnetne komponente, i pocinju sa M, on racuna
        intenzitet"""
        mag = self.mag_components(data)
        mag2 = mag ** 2
        # Mx^2 + My^2 + Mz^2 ...
        magt = mag2.sum(1)
        # koren iz toga, intenzitet
        return np.sqrt(magt)

    def get_filename(self,dir_,l,t,therm):
        regex = re.compile(r'(^%s%s%s(MC.*\.sp|\.all))' %(l,t,therm))
        filenames = [f for f in os.listdir(join(self.simdir,dir_,'%s%s' %(l,t))) if regex.match(f)]
        if len(filenames)!=1:
            util.show_error('Emergency','Report issue asap with code 1100010001110**__??')
            raise BaseException
        filename = filenames[0]
        return join(self.simdir,dir_,'%s%s' %(l,t),filename)


    def create_mat(self,dir_,l,t,therm,mc=None,booli=False):
        """Cita all fajlove i obradjuje ih statisticki. U slucaju
        da je prosledjen mc to znaci da smo prethodno dodali u onaj
        choices dictionary odgovarajuc mc, i da sad radimo za njega
        ako ne prosledjujemo, sve regularno citamo sve, a ne samo
        mc prvih redova. Ako je prosledjen booli to znaci da su izbaceni
        ODREDJENI rezultati, pa koristimo taj argument kao indeks koji
        ce izdvojiti samo zeljenje rezultate
        """
        mc = util.extract_int(mc) if mc else None
        filename = None
        try:
            
            filename = self.get_filename(dir_,l,t,therm)
        except BaseException as e:
            util.show_error('Yeah yeah',str(e))
            return
        self.log.debug("Creating mat from file {} for first {} rows".format(filename,mc))
        data = self.read_tdfile(filename,mc)
        booli = booli if booli else [True]*len(data.index)
        data = data.loc[booli]
        self.log.debug("matdata booled :\n %s",data)
        N = len(data.index)
        self.log.debug("broj rezultata: %s" % N)
        Eavg = data.E.mean()
        stdE = data.E.std()
        stdMeanE = stdE / np.sqrt(N)
        E2 = data.E ** 2
        E2avg = E2.mean()
        stdE2 = E2.std()
        stdMeanE2 = stdE2 / np.sqrt(N)
        self.calculate_magt(data)
        #Ovo mozda da bude unutrasnja funkcija
        # Nem pojma, treba videti sta ti treba
        # gde pa onda da znas
        MAG = self.mag_components(data)
        MAG2 = MAG ** 2
        M2 = MAG2.sum(1)
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
            util.extract_int(therm),
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
        return values

    @logg
    def compose(self,dir_,l,t,mc):
        """Modifikuje odgovarajuce elemente u self.files [sa izracunatim matovima],
        ako je potrebno, i vraca strukturu pogodnu za plotovanje"""
        data = dict()
        self.log.debug("Composing for dir_:{} l:{} t:{} mc:{}".format(dir_,l,t,mc))
        self.log.debug("Idem kroz {}".format(self.files[dir_][l][t][mc].items()))
        for therm,mat in self.files[dir_][l][t][mc].items():
            # ne znam da li ce biti problem sto menjam ovaj dictionary
            # dok iteriram kroz njega???
            self.log.debug("Trenutni mat je %s" %mat)
            therm_int = util.extract_int(therm)
            try:
                self.files[dir_][l][t][mc][therm] = mat if mat.any() else "weird"
            except:
                self.files[dir_][l][t][mc][therm] = self.create_mat(dir_,l,t,therm,mc)
            else:
                self.log.debug("Vec je izracunat mat")
            data[therm_int] = self.files[dir_][l][t][mc][therm]
        df = pd.DataFrame(data)
        out = { 'abs(cv(E1))':abs(df.ix['stdMeanE'] / df.ix['Eavg']),
            'cv(E2)':df.ix['stdMeanE2']/df.ix['E2avg'],
            'cv(M1)':df.ix['stdMeanM1']/df.ix['M1avg'],
            'cv(M2)':df.ix['stdMeanM2']/df.ix['M2avg'],
            'cv(M4)':df.ix['stdMeanM4']/df.ix['M4avg']}
        out = pd.DataFrame(out)
        out = pd.concat([df,out.T])
        self.log.debug("Vracam za plotovanje : \n %s" %out)
        return out

    def get_plot_dict(self,dir_):
        """Vraca dictionary koji
        sadrzi podatke o tome sta
        ce se agregirate, tj. sta ce
        se plotovati. Moze samo matove,
        moze samo altove, a moze i trenutno
        izabrane"""
        return self.bestmats[dir_]
        
    @logg
    def agregate(self,dir_,alt=False,mat=False):
        plot_data = dict()
        for_plotting = self.get_plot_dict(dir_)
        for l,tdict in for_plotting.items():
            agg_mat = list()
            for t,val in tdict.items():
                mc,therm =val['mc'],val['therm']
                try:
                    booli = val['booli']
                except:
                    booli = False


                self.log.debug('checking whether calculated : %s' % self.files[dir_][l][t][mc][therm])
                if self.files[dir_][l][t][mc][therm] is None:
                    self.files[dir_][l][t][mc][therm]=self.create_mat(dir_,l,t,therm)
                data_mat = self.files[dir_][l][t][mc][therm]
                self.log.debug(type(data_mat))
                data_mat.rename({'THERM': 'T'},inplace=True)
                data_mat.ix['T']=int(t[1:])
                agg_mat.append((data_mat,t))
            #uvek cemo imati samo jednu kolonu, naravno
            print agg_mat
            agg_mat = sorted(agg_mat, key=lambda x: util.extract_int(x[1]))
            try:
                dat,keys = zip(*agg_mat)
            except:
                util.show_error('No points','No points selected. Ask developer to disable draw button')
            else:
                print 'dat,keys',dat,keys
                agg_mat = pd.concat(dat, axis=1,keys=keys)
                plot_data[l] = self.agg(agg_mat)

        self.log.debug('plot data \n:%s' %plot_data)
        return plot_data
        
    @logg
    def agg(self,agg_mat):
        T = agg_mat.ix['T'] / 10000.0
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
        print out
        return out

    def remap_fsystem(self):
        """BICE PROBLEM SA NOVO GENERISANIM MC-OVIMA!!!"""
        self.init_model()
        self.notify_observers()

    def load_bestmat(self):
        """Proverava da li postoji u strukturi fajlova
        odgovarajuci podaci iz bestmat-a, ako nema - brise ih
        vraca azuran bestmat dict"""
        bmat =  self.load_choices('bestmat.dict')
        for dir_,ldict in bmat.items():
            if dir_ not in self.files.keys():
                del bmat[dir_]
                continue
            for l,tdict in ldict.items():
                if l not in self.files[dir_].keys():
                    del ldict[l]
                    continue
                for t,tmc_dict in tdict.items():
                    mc = tmc_dict['mc']
                    if t not in self.files[dir_][l].keys() or mc not in self.files[dir_][l][t].keys() or tmc_dict['therm'] not in self.files[dir_][l][t][mc].keys():
                        del tdict[t]
                        continue

        self.bestmats = bmat
        self.__clean_bmat()

    def _map_filesystem(self):
        """Ide kroz sve direktorijume ( za sada jedan sim dir)
        u simdir-u i izvlaci l,t,i therm.
        """
        funcs = {'sp':self.spify,'mp':self.unify}
        def check_kind(filename):
            """Ustvari bih mogla da napravim regex
            za sva tri i onda po grupi da vracam
            nesto, al ajde, ovo je jednostavno"""
            if self.sp_regex.match(filename):
                return 'sp'
            elif self.mp_regex.match(filename):
                return 'mp'
                
        filess = dict()
        dirs_ = [dir_ for dir_ in os.listdir(self.simdir) if os.path.isdir(join(self.simdir,dir_))]
        self.log.debug('directories in sim folder: %s' % dirs_)
        for dir_ in dirs_:
            path = join(self.simdir,dir_)
            choices = defaultdict(dict)
            mc_to_all = defaultdict(list)
            self.log.debug('Currently mapping directory: %s' %dir_)
            # za svaki folder, kind ce biti drugciji, potencijalno
            kind = None
            for lt_dir,dirs,files in os.walk(path):
                
                ltmatch = self.lt_regex.search(lt_dir)
                if not ltmatch:
                    #u slucaju da je folder drugog formata
                    continue
                l,t = ltmatch.groups()
                #samo gledamo one fajlove koji su nam odgovarajuceg formata
                mp_and_sp_files = [f for f in files if self.sp_regex.match(f) or self.mp_regex.match(f)]
                try:
                    first_file = mp_and_sp_files[0]
                except IndexError:
                    #ne postoji ni jedan sp,mp ili dat ili all
                    util.show_error('Empty LT folder','The folder %s is empty' %lt_dir)
                else:
                    #obradjujemo fajlove
                    kind = check_kind(first_file) if kind is None else kind
                
                    different_kind = [f for f in mp_and_sp_files if check_kind(f)!=kind]
                    if different_kind:
                        #znaci ako je bilo koji fajl drugog tipa
                        # izaci ce iz programa
                        util.show_error('Mixed pathsies',"""Please seperate multipaths from singlepaths in
                        \n foldeR '%s', LT folder '%s'.\n Exiting now, bye!""" %( dir_,lt_dir))
                        try:
                            os.system("nautilus %s" %lt_dir)
                        except:
                            pass
                        raise BaseException
                    funcs[kind](lt_dir)

                #ovo radimo kad smo obradili sve
                new_files = os.listdir(join(self.simdir,dir_,lt_dir))
                matched = [f for f in new_files if self.sp_regex.match(f) or self.all_regex.match(f)]
                mct_choices = defaultdict(dict)
                for all_ in matched:
                    therm = re.search(self.base_regex,all_).groupdict()['therm']
                    #znaci za svaki therm ce dodavati u prikacenu listu
                    # mc-ove tako sto ce otvarati all fajl i brojati linije
                    mc = "MC{sps}".format(sps=len(open(join(lt_dir,all_)).readlines()))
                    mct_choices[mc][therm] = None
                choices[l][t] = mct_choices
            if choices:
                filess[dir_]=choices
                
        self.log.debug('mapped fsystem: %s' % filess)
        return filess



class FileManager(mvc_skelet.Controller):
    """Ova klasa ce brinuti o postojacim l,t,mc, i therm
    i takodje o izborima razlicitim, znaci originalnim statickim
    i dinamickim, i onima sa izbacenim rezultatima.Inicijalizuje se sa
    postojacim l-ovima, postojacim t-ovima, postojacim mc i thermovima
    Prosledjivace mu se lt_dict, ali lt_dict nece biti globalna varijabla
    sada, vec ce se ovde cuvati, samo u drugom obliku

    """
   
    # extr_regex = re.compile(r'(?P<base>L\d+T\d+THERM(?P<therm>\d+))MC(?P<mc>\d+)\.mat$')
    
    def __init__(self):
        """morace da bude iniciijalizovan sa napravljenim
        posto ce se praviti kako se budu pozivali matovi, tako ce ovaj
        vracati, a mi cemo stavljati u nas lt dict, cemo appendovati na nase
        tuple"""
        #  logging.basicConfig(level=logging.DEBUG)
        self.log = logging.getLogger("FileManager")
        self.gui = None
        self.init_components()
        
        self.match_stuff ={'l':{'result':[],'choice':None,'func':self.l_gen},
                           't':{'result':[],'choice':None,'func':self.t_gen},
                           'mc':{'result':[],'choice':None,'func':self.mc_gen},
                           'therm':{'result':[],'choice':None,'func':self.therm_gen}
        }

    def model_updated(self,subject=None):
        self.update_gui()

    def init_gui(self,thermPanel,aggPanel,scatterPanel):
        """Kaci gui, kad napravi sve komponente gui-ja
        iz TabContainera"""
        self.tp = thermPanel
        self.ap = aggPanel
        self.sp = scatterPanel
        
    def init_model(self):
        self.model = Choices()

    def init_view(self):
        """Ne znam da li je ok da imam
        vise view-a, ili da li je potrebno
        """
        self.view = ChoicesConverter(model = self.model, controller = self)

    def set_simdir(self,simdir):
        """Prosledjuje modelu simdir
        ovo je pre nego sto je aplikacija
        pocela"""
        self.simdir = simdir
        
    def application_started(self):
        """Posto je sve vec napravljeno, mozemo
        da inicijalizujemo model
        i da postavljamo vrednosti gui elemenata i to
        """
        self.model.simdir = self.simdir
        try:
            self.model.init_model()
        except BaseException as e:
            self.tp.stop()
        self.update_gui()
       
       # self.ap.reader.populateListControls(**{'l':lch})


    def update_gui(self):
        """Ako je promenjen model
        updejtuju se svi elementi gui-ja
        """
        dir_choices = self.view.choices()
        self.tp.set_choices(**{'dir_':dir_choices})
        #ovaj ce gledati iz bestmatdicta kako sta
        self.ap.set_dir_choices(dir_choices)
        
        # samo ce morati da vidi ako ima, ako nema jbg
        # bice prazno
        self.ap.set_mag_choices(self.view.mags)
        self.refresh_matchooser()

    def remap_fsystem(self):
        self.model.remap_fsystem()
    @logg
    def refresh_matchooser(self):
        ch = self.current_choices()
        self.l_gen(**ch)
        self.t_gen(**ch)
        self.mc_gen(**ch)
        lch = self.view.matchooseritems(dir_=self.ap_dir)
        self.match_stuff['l']['result']=lch
        self.populate_matchooser()
        
    ####  GUI INTERFACE ####
    ### Ove metode poziva gui ###
    
    
    ######### THERM PANEL #############
                       
    def add_mc(self,mc,**kwargs):
        """Dodaje mc u choices, thermove ce morati da
        preuzme od prvog suseda koji nije 'specijalan'
        Hm, verovatno je pogresno sto iz modela prikazujem
        errore. verovatno bi trebalo samo da raisujem errore
        i to a ovde da ih hvatam. hmm
         """
        try:
            self.model.add_mc(mc,**kwargs)
        except IndexError:
            util.show_error("Simdir error","An empty folder in your simulation directory??")
            
    def on_l_select_tp(self,l):
        """Kada korisnik selektuje l u thermpanelu,
        namestaju se odredjeni t"""
        tch = self.view.choices(l)
        self.tp.set_t_choices(tch)

    @logg
    def tp_on_select(self,**kwargs):
        """prosledjuju se sve sto je selektovano.
        gleda koja je prva kontrola koja nije prosledjena
        i u nju stavlja rezultat. oslanja se na sekv. cmbord"""
        kwargs['therm_count']=True
        choices = self.view.choices(**kwargs)
        self.tp.reset_chkboxes()
        for cmb in self.tp.cmbord:
            self.log.debug('Nalazim prvi izbor koji nije prosledjen, trenutni: %s' %cmb)
            try:
                dir_ = kwargs['dir_']
                l = kwargs['l']
                t = kwargs['t']
            except:
                pass
            else:
                uplimit = self.model.get_maxmc(dir_,l,t)
                self.tp.set_mc_range(uplimit)
            if cmb not in kwargs.keys():
                self.log.debug('Ovaj nije: %s' %cmb)
                self.tp.set_choices(**{cmb:choices})
                return

        util.show_error('What the select?','You selected %s, what\'s that?' %kwargs)
        

      
      


    
    def get_plot_data(self,dir_,l,t,mc):
        """Vraca datafrejm za plotovanje"""
        return self.model.compose(dir_,l,t,mc)
    
        

    ########################################
    ########### AGGREGATE PANEL ############

    def get_agg_plot_data(self,dir_):
        return self.model.agregate(dir_)

    def annotate_agg(self,dir_, l,t):
        """
        Vraca anotaciju za odredjeno l i t
        """
        return self.view.annotate_agg(dir_,l,t)

    @logg
    def ap_dir_selected(self,val,changed):
        self.ap.set_l_choices(self.view.bmat_choices(dir_=val))
        self.ap_dir = val
        if changed:
            self.l_gen()
            self.refresh_matchooser()
    

        

    ########################################
    ######## READER INTERFACE ##############

    def set_matchooser_lists(self,listctrl_names):
        """CUva reference na listcontrols"""
        raise NotImplementedError

           
    def listctrl_selected(self,choice):
        typ = util.extract_name(choice)
        self.log.debug('listctrl selected name:{}'.format(typ))
        self.match_stuff[typ]['func'](**{typ:choice})
        if typ!='therm':
            self.populate_matchooser(choice = choice)

    @logg
    def populate_matchooser(self,choice = None):
        self.log.debug('populating matchooser, choice made: %s' %choice)
        lala = ['l','t','mc','therm']
        cmpr = lala.index(util.extract_name(choice)) if choice else -1
        pop = {key:self.match_stuff[key]['result'] for key in self.match_stuff if cmpr < lala.index(key)}
        self.log.debug('population : {}'.format(pop))
        self.ap.reader.populateListControls(**pop)

    def add_bestmat(self):
        bestm = self.current_choices()
        self.model.add_bestmat(self.ap_dir,**bestm)

    def random_bestmats(self,dir_):
        bestm = self.current_choices()
        self.model.random_bestmats(dir_,**bestm)
                
            

    def remove_bestmat(self,**kwargs):
        typ = kwargs.keys()[0]
        self.match_stuff[typ]['choice'] = kwargs[typ]
        ch = self.current_choices()
        ch['dir_'] = self.ap_dir
        self.model.remove_bestmat(**ch)
        

    def current_choices(self):
        """Vraca trenutno selektovan izbor
        u bestmat chooseru """
        return {key:self.match_stuff[key]['choice'] for key in self.match_stuff}

    @logg
    def l_gen(self,**kwargs):
        """L izbor prima za parametar
        stavlja ga medju atribut choices,
        generise sve izbore za taj i stavlja
        i zove sve ostale funkcije za atribute
        zato sto jbg, ne znam kako drugcije
        """
        dir_ = self.ap_dir
        lch = kwargs['l'] if 'l' in kwargs.keys() else None
        self.match_stuff['l']['choice'] = lch
        self.match_stuff['t']['result'] = self.view.matchooseritems(dir_=dir_,l=lch) if lch else []
        self.t_gen(**kwargs)

    def t_gen(self,**kwargs):
        tch =kwargs['t'] if 't' in kwargs.keys() else None
        lch = self.match_stuff['l']['choice']
        self.log.debug('generating mc results for choice %s' %tch)
        self.match_stuff['t']['choice'] = tch
        self.match_stuff['mc']['result'] = self.view.matchooseritems(dir_=self.ap_dir,l=lch,t=tch) if tch else []
        self.mc_gen(**kwargs)

    def mc_gen(self,**kwargs):
        mch = kwargs['mc'] if 'mc' in kwargs.keys() else None
        lch = self.match_stuff['l']['choice']
        tch = self.match_stuff['t']['choice']
        self.log.debug('generating therm results for choice %s' %mch)
        self.match_stuff['mc']['choice'] = mch
        self.match_stuff['therm']['result'] = self.view.matchooseritems(dir_=self.ap_dir,l = lch,t=tch,mc = mch) if mch else []
                                    
    def therm_gen(self,**kwargs):
        self.match_stuff['therm']['choice'] = kwargs['therm'] 
        self.add_bestmat()

    
    #############SCATTER PANEL###################

    def init_scatter(self):
        bmat_ch = self.view.bmat_choices()
        self.sp.set_dir_choices(bmat_ch)

    def sp_on_dir_select(self,val):
        lch = self.view.bmat_choices(dir_=val)
        self.sp.set_l_choices(lch)

    def load_sp_data(self,dir_,l,n=None):
        return self.model.load_sp_data(dir_,l,n)

    def get_maxmc(self,dir_,l,t):
        return self.model.get_maxmc(dir_,l,t)

    def get_scat_title(self,dir_,l,t):
        self.log.debug('Getting scat title for dir_:{} l:{} t:{}'.format(dir_,l,t))
        return self.view.get_scat_title(dir_,l,t)

    def calculate_magt(self,data):
        return self.model.calculate_magt(data)

    def mag_components(self,data):
        return self.model.mag_components(data)

    def remove_faulty(self,dir_,l,t,booli):
        """Za odredjeno l i t izracunava novi
        mat, i stavlja ovaj izbor u sve moguce izbore
        i stavlja taj izbor u izbor za bestmat, sto je
        problem je l' posto ne postoji na disku teh teh
        a kako bi se cuvalo, cuvalo bi se nekako brate
        znaci samo taj booli treba da sacuvamo ali jbg
        kad nije uniformno sa bestmatom mozda u taj dict
        da ubacimo samo jos jedno polje hmmm hmmm i da ga
        tako cuvamo. samo kad bi se to onda racunalo mmm
        i kako bi se ucitalo pri pokretanju. moralo bi. ajd
        za sada nek se lepo racuna i dodaje tamo gde treba
        a posle cemo videti"""
        mc=sum(booli)
        prev_mc = len(booli)
        mc = 'MC%s[%s]' % (mc,prev_mc)
        therm = self.model.bestmat_choices(dir_,l,t,which='therm')[0]
        self.model.add_virtual_file(dir_=dir_,l=l,t=t,mc=mc,therm=therm,booli=booli)

    

class ChoicesConverter(mvc_skelet.View):
    """Ova klasa sprema za prikaz
    za gui izbore inace i u matchooseru"""

    def __init__(self,model,controller):
        mvc_skelet.View.__init__(self,model,controller)
        self.log = logging.getLogger("ChoicesConverter")

    def choices(self,dir_=None,l=None,t=None,mc=None,therm_count=False):
        reverse = True if not t else False
        if t and not mc and therm_count:
            return self.mc_choices(dir_,l,t)
        else:
            return sorted(self.model.choices(dir_=dir_,l=l,t=t,mc=mc), key= lambda x:util.extract_int(x) ,reverse=reverse)

    def get_scat_title(self,dir_,l,t):
       
        therm = self.model.bestmat_choices(dir_=dir_,l=l,t=t,which='therm')[0]
        print therm
        therm = util.extract_int(therm)
        mc = util.extract_int(self.model.bestmat_choices(dir_=dir_,l=l,t=t,which='mc')[0])
        return "T={:.4f}\nLS={}\n SP={}".format(float(util.extract_int(t))/10000.0,therm,mc)

    @counter
    @benchmark
    @logg
    def matchooseritems(self,dir_,l=None,t=None,mc=None):
        items = list()
        # ako je prosledjen mc onda zelimo therm
        which = 'therm' if mc else 'mc'
        bestmat_ch = self.model.bestmat_choices(dir_,l,t,which)
        self.log.debug('bestmatch %s' %bestmat_ch)
        for x in self.choices(dir_,l,t,mc):
            fdict = {'pointSize':10,'family':wx.FONTFAMILY_DEFAULT,
                     'style':wx.FONTSTYLE_NORMAL,'weight':wx.FONTWEIGHT_NORMAL}
            self.log.debug('uredjujem izgleda za :%s' %x)
            item = wx.ListItem()
            item.SetText(x)
            if x in bestmat_ch and (mc is None or mc in self.model.bestmat_choices(dir_,l,t,'mc')):
                self.log.debug('boldovao')
                fdict['weight']=wx.FONTWEIGHT_BOLD
            
            font = wx.Font(**fdict)
            # if x in self.model.altmat_choices():
            #     font.SetWeight(wx.FONTWEIGHT_BOLD)
            item.SetFont(font)
            items.append(item)
        return items

    def bmat_choices(self,dir_=None,l=None,t=None,which = 'mc'):
        return sorted(self.model.bestmat_choices(dir_,l,t,which),key = lambda x:util.extract_int(x))
        
    def _l_choices_agg(self):
        return sorted(self.model.bestmat_ls(), key = lambda x:util.extract_int(x))

    @property
    def mags(self):
        return self.model.mags

    def mc_choices(self,dir_,l,t):
        """Vraca sve raspolozive mc-ove, formatirane za prikaz. Oni ujedno i odredjuju moguce
        plotove za therm panel. Ovo podize pitanje da li je bolje da
        se gleda za koje mc-ove postoje koji thermovi, i obrnuto"""
        mc_choices = self.model.choices(dir_=dir_,l=l,t=t)
        
        sorted_mcs =  sorted(mc_choices,key=lambda x: util.extract_int(x))
        return ["{mc} ({tmc})".format(mc=mc,tmc=self.model.therm_count(dir_,l,t,mc)) for mc in sorted_mcs ]


    def annotate_agg(self,dir_,l,t):
        ls =self.model.bestmat_choices(dir_=dir_,l=l,t=t,which='therm')[0]
        sp =self.model.bestmat_choices(dir_=dir_,l=l,t=t,which='mc')[0]
        return 'LS={}\nSP={}'.format(util.extract_int(ls),util.extract_int(sp))

   




