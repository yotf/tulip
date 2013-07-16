#!/usr/bin/python
# -*- coding: utf-8 -*-
"""
=================================================================
Opis aplikacije....
=================================================================
   
Usage:
    tulip.py [SIMDIR]
    tulip.py -h | --help
Arguments:
    SIMDIR     Path to sim directory
Options:
    -h --help
"""
   
#import pdb
import wxversion
wxversion.select('2.8')
import wx
import pandas as pd
import os
from os.path import join
import glob
import re
import pickle
import logging
from collections import defaultdict
import itertools
import unify
import agregate
import compose
import statmat
import matplotlib
from matplotlib import widgets   
#PRETPOSTAVKA: SVI SKRIPTOVI SE NALAZE U ISTOM DIREKTORIJUMU
# PRETPOSTAVKA : OVAJ (GLANI) SKRIPT SE UVEK POKRECE IZ DIREKTORIJUMA
# U KOM SE NALAZI



   
matplotlib.use('WXAgg')
from matplotlib.figure import Figure
from matplotlib import cm
   
from matplotlib.backends.backend_wxagg import FigureCanvasWxAgg as FigCanvas, \
    NavigationToolbar2WxAgg as NavigationToolbar
import numpy as np
import pylab
from mpl_toolkits.mplot3d import Axes3D
from simp_zoom import zoom_factory
from itertools import cycle
from scipy import stats   
from matplotlib.widgets import RectangleSelector


regexf = re.compile(r'^(L\d{1,3}).aplot$')
   
SIM_DIR = ""

LATTICE_MC = os.getcwd()
fmt_strings = ['g+-','r*-','bo-','y+-']
fmt_cycle = itertools.cycle(fmt_strings)


########################################
##########POMOCNE KLASE################
class twoway_cycle():
    """Uzima iterable, i na zvanje prev i next
    vraca odgovarajuce clanove. ovo je verovatno
    moglo lepse preko nekih dekoratora, generatora
    nesto """
    def __init__(self,it):
        if not np.iterable(it):
            raise ValueError("Must be an iterable")
        
        self.log=logging.getLogger("twoway_cycle")
        self.it=it
        self.i=0
    def next(self):
        self.i = self.i+1
        self.i = 0 if self.i==len(self.it) else self.i
        return self.it[self.i]

    def prev(self):
        self.i = len(self.it) if self.i==0 else self.i
        self.i = self.i-1
        return self.it[self.i]

    def curr(self):
        self.log.debug("returning curr element:{}".format(self.it[self.i]))
        return self.it[self.i]
            


class FileManager():
    """Ova klasa ce brinuti o postojacim l,t,mc, i therm
    i takodje o izborima razlicitim, znaci originalnim statickim
    i dinamickim, i onima sa izbacenim rezultatima.Inicijalizuje se sa
    postojacim l-ovima, postojacim t-ovima, postojacim mc i thermovima
    Prosledjivace mu se lt_dict, ali lt_dict nece biti globalna varijabla
    sada, vec ce se ovde cuvati, samo u drugom obliku

    """
    #ovaj regex nam izvlaci therm i mc iz imena mat fajla
    # fazon je sto imam redudantnost u smislu da imam LT i kao
    # kljuceve dicta, a i kao ime fajla. Pa dobro, sta da mu radim
    # necu sve kao regex, ovo je jednostavnije. Da sam sve kao regex
    # onda bih morala da mmm, stavljam u stringove regexa, ove vrednosti
    # ma, siugrno postoji zasto je dobro dict. dict je fleksibilniji, ali doduse
    # da nam nije dict bih mogla da radim presek skupa ili tako nesto. hm, sigurno
    # postoji nacin da na pametan nacin izvucem kao neki presek, ali to nije presek
    # to jeste ako je u preseku ta vrednost uzimamo iz drugog, a ako nije uzimamo iz
    # prvog
    extr_regex = re.compile(r'(?P<base>L\d+T\d+THERM(?P<therm>\d+))MC(?P<mc>\d+)\.mat$')
    all_regex = re.compile(r'^L\d+T\d+(?P<therm>THERM\d+)\.all$')
    lt_regex = re.compile(r'(L\d+)(T\d+)$')
    
    def __init__(self):
        """morace da bude iniciijalizovan sa napravljenim
        posto ce se praviti kako se budu pozivali matovi, tako ce ovaj
        vracati, a mi cemo stavljati u nas lt dict, cemo appendovati na nase
        tuple"""
        #  logging.basicConfig(level=logging.DEBUG)
        self.log = logging.getLogger("FileManager")
        self.choices = self._get_choices()
        self.log.debug("Glavni izbori, choices : %s" %self.choices)
        print "FileManager:self.choices",self.choices
        self.bestmatd = self._load_bmatdict()

    #def _load_mc_to_all(self):

    def add_mc(self,l,t,mc):
        """Dodaje mc u choices, thermove ce morati da
        kupi od suseda, ili ti komsije"""
        therms = self.choices[l][t].values()[0].keys()
        tdict = {key:None for key in therms}
        self.log.debug("adding mc %s" % tdict )
        self.choices[l][t]["MC%s" %mc]=tdict


    def create_mat(self,l,t,therm,mc=None,booli=False):
        """Cita all fajlove i obradjuje ih statisticki. U slucaju
        da je prosledjen mc to znaci da smo prethodno dodali u onaj
        choices dictionary odgovarajuc mc, i da sad radimo za njega
        ako ne prosledjujemo, sve regularno citamo sve, a ne samo
        mc prvih redova. Ako je prosledjen booli to znaci da su izbaceni
        ODREDJENI rezultati, pa koristimo taj argument kao indeks koji
        ce izdvojiti samo zeljenje rezultate
        """
        mc = int(re.search(r'\d+',mc).group())
        filename = join(SIM_DIR,"{l}{t}".format(l=l,t=t),"{l}{t}THERM{therm}.all".format(l=l,t=t,therm=therm))
        
        self.log.debug("Creating mat from file {} for first {} rows".format(filename,mc))
        data = pd.read_table(filename,nrows=mc,delim_whitespace=True, names= ['seed', 'E','Mx','My','Mz'])
        data.pop('seed')
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
        return values


    def compose(self,l,t,mc):
        """Modifikuje odgovarajuce elemente u self.choices,
        ako je potrebno, i vraca strukturu pogodnu za plotovanje"""
        data = dict()
        self.log.debug("Composing for l:{} t:{} mc:{}".format(l,t,mc))
        self.log.debug("Idem kroz {}".format(self.choices[l][t][mc].items()))
        for therm,mat in self.choices[l][t][mc].items():
            # ne znam da li ce biti problem sto menjam ovaj dictionary
            # dok iteriram kroz njega???
            self.log.debug("Trenutni mat je %s" %mat)
            therm_int = int(re.search(r'\d+',therm).group())
            try:
                self.choices[l][t][mc][therm] = mat if mat.any() else "weird"
            except:
                self.choices[l][t][mc][therm] = self.create_mat(l,t,therm_int,mc)
            else:
                self.log.debug("Vec je izracunat mat")
            data[therm_int] = self.choices[l][t][mc][therm]
        df = pd.DataFrame(data)
        self.log.debug("Choices izgleda ovako:\n %s" % self.choices)
        out = { 'abs(cv(E1))':abs(df.ix['stdMeanE'] / df.ix['Eavg']),
            'cv(E2)':df.ix['stdMeanE2']/df.ix['E2avg'],
            'cv(M1)':df.ix['stdMeanM1']/df.ix['M1avg'],
            'cv(M2)':df.ix['stdMeanM2']/df.ix['M2avg'],
            'cv(M4)':df.ix['stdMeanM4']/df.ix['M4avg']}
        out = pd.DataFrame(out)
        out = pd.concat([df,out.T])
        self.log.debug("Vracam za plotovanje : \n %s" %out)
        return out
        
    def t_choices(self,l):
        """Vraca sve moguce t-ove za prosledjeno L"""
        self.log.debug("returning t_choices {} for l:{}".format(self.choices[l].keys(),l))
        return sorted(self.choices[l].keys(), key=lambda x:int(x[1:]))
                       
    def l_choices(self):
        """Vraca sve moguce L-ove u sim_dir"""
        return sorted(self.choices.keys(),key=lambda x: int(x[1:]))

    def mc_choices(self,l,t):
        """Vraca sve raspolozive mc-ove. Oni ujedno i odredjuju moguce
        plotove za therm panel. Ovo podize pitanje da li je bolje da
        se gleda za koje mc-ove postoje koji thermovi, i obrnuto"""
        mc_choices = self.choices[l][t].keys()
        self.log.debug("returing mc_chioices {}".format(mc_choices))
        return mc_choices

    def therm_count(self,l,t,mc):
        """Vraca koliko ima tacaka za dato mc, tj.trebace kod
        ovog therm panela da vidi neko da ne plotuje nebulozne
        stvari. znaci ovo ce biti u nekim viticastim zagradama
        iza broja mc-ova"""
        therms = self.therm_choices(l,t,mc)
        self.log.debug("Vracam broj thermova:{} za mc:{}".format(len(therms),mc))
        return len(therms)

    def therm_choices(self,l,t,mc):
        """Vraca sve thermove za odredjeno l,t i mc"""
        therms = self.choices[l][t][mc]
        self.log.debug("Vracam thermove: {} za mc:{}".format(therms,mc))
        return therms
        
        
        
    # def get_choices(self):
    #     """Ide kroz imena svih direktorijum i za svaki L prilepljuje
    #     odgovarajuce t-ove."""
    #     regex = re.compile(r"^(L\d+)(T\d+)$")
    #     dirlist = [d for d in os.walk(SIM_DIR).next()[1] if regex.match(d)]
    #     dirlist.sort()
    #     dct = defaultdict(list)
    #     for d in dirlist:
    #         l,t = regex.match(d).groups()
    #         dct[l].append(t)
    #     return dct
                       
    def get_alt(self):
        """Pravi dictionary koji uzima
        sve vrednosti koje postoje u alt dictu
        i prelepljuje ih preko vrednosti iz bmatdicta
        i to vraca"""
        import copy
        altm = copy.deepcopy(self.bmatdict)
        print "REPAIRED DICT:\n", self.repdict
        for l,val in self.repdict.items():
            for t,path in val.items():
                altm[l][t] = path
        return altm
        
    def get_all_file(self,l,t):
        """Vraca all fajl za zeljeni bmat, za prosledjeno l i t"""
        base = self.get_base(l,t)
        self.log.debug("get_all_file:vracam %s" % ("%s.all" % base))
        return "%s.all" % base

    def get_base(self,l,t):
        """Vraca osnovu imena u obliku L[0-9]*T[0-9]*THERM[0-9]*
        iz bmatdicta za prosledjeno l i t
        """
        extr_regex.search(self.bestmatd[l][t]).groupdict()["base"]

    def get_files(self,l,t,ext="*.plot"):
        """Vraca sve plot fajlove u zadatom folderu (L i T))"""
        folder_name = join(SIM_DIR,"%s%s" %(l,t),ext)
        self.log.debug("get_files:%s" %folder_name)
        files = glob.glob(folder_name)
        self.log.debug(files)
        return files
   
    # def get_therms(self,l,t):
    #     """Mislim da ovaj iz best_mat_dicta vraca, pa da
    #     uglavnom su nam relevantne informacije da je iz matova
    #     i da su ili mc ili therms, mozemo napraviti dve funkcije
    #     nece vise biti komplikovano, posto cemo ovu informaciju odmah
    #     na pocetku zapisati"""

    #     return self.choices[l][t].keys()


    # def get_therms(self,l,t):
    #     """Vraca sve moguce thermove za dato l i t"""
    #     therms = self.choices[l][t].keys()
    #     self.log.debug("getting therms:{}".format(therms))
    #     return therms
        
    # def get_mcs(self,l,t,therm):
    #     """
    #     Znaci vraca listu mc-ova za dato l t i therm
    #     Razlikuje se od mc_choices posto se onome ne
    #     prosledjuje therms. Ovaj je specificniji.
    #     """
    #     mcs =sorted(self.choices[l][t][therm],key=lambda x: int(x[2:]))
    #     self.log.debug("vracam mcove: {} za therm: {}".format(mcs,therm))
    #     return mcs

                       
    def get_maxmc(self,l,t):
        """
        Vraca max mc u okviru 
        """
        #ok, uzimamo therm, posto je to nesto konstantno u
        # best mat dictu, tj. ne konstantno nego je odredjeno
        # sa l i t jedinstveno
        return max(self.mc_choices(l,t))
        
    def best_therm(self,l,t):
       """Vraca therm koji se nalazi u best_mat_dict,
       samo treba videti kako ce se ovo ponasati kod ovih drugih
       da li cu imati neki currdict ili nesto pa iz njega vracati
       dobro, svakako bi trebalo da postoji neko stanje ja mislim
       u zavisnosti od toga sta su kako su, ajd videcu"""
       self.log.debug("vracam best_therm za l:{} i t:{} mat:{}".format(l,t,self.bmatdict[l][t]))
       print self.extr_regex
       return self.extr_regex.search(self.bestmatd[l][t]).groupdict()["therm"]
        
    def best_mc(l,t):
        """Vraca iz bmatdicta izabrani mc
        za odredjene l i t"""

        return extr_regex.search(self.bestmatd[l][t]).groupdict()["mc"]
    def _get_choices(self):
        """Ide kroz sve direktorijume ( za sada jedan sim dir)
        u simdir-u i izvlaci l,t,i therm.Pretpostavljamo da nema
        mat fajlova. mc izvlacimo iz broja linija. Ko mi? Vraca
        te taj dict koji predstavlja direktorijume, neki dirtree
        i takodje vraca dict koji mapira broj mc-ova sa odgovarajucim
        all fajlovima pa cemo kad plotujemo. Hm, ali to bi sve moglo da bude
        u ovom choices dictionary, znaci da vezemo za svaki therm mc kombinaciju
        i fajl. I onda cemo kad plotujemo hmmm, ali pazi nama treba kad plotujemo
        cista slika fajlova sa odredjenim mc, hm, kako to da napravim. Znaci u ovom
        slucaju mi treba cisto mapiranje mc-ova sa fajlovima a tamo mi treba cisto
        mapiranje thermova sa mc-ovima. A zasto prvo therm ide? Zasto ne izaberu prvo mc
        pa onda therm?"""
        choices = defaultdict(dict)
        mc_to_all = defaultdict(list)
        for root,dirs,files in os.walk(SIM_DIR):
            ltmatch = self.lt_regex.search(root)
            if not ltmatch:
                #u slucaju da je folder drugog formata
                continue
            l,t = ltmatch.groups()
            matched = [f for f in files if self.all_regex.match(f)]
            mct_choices = defaultdict(dict)
            for all_ in matched:
                therm = self.all_regex.match(all_).groupdict()['therm']
                #znaci za svaki therm ce dodavati u prikacenu listu
                # mc-ove tako sto ce otvarati all fajl i brojati linije
                mc = "MC{sps}".format(sps=len(open(join(root,all_)).readlines()))
                mct_choices[mc][therm] = None
            choices[l][t] = mct_choices
        return choices
  
    def get_changed_mat_name(self,l,t,new_mc):
        """Ovo je zarad izbacivanja 'nepozeljnih' rezultata
        prosledjuju mu se l i t, i novi mc. Vraca novo ime"""
        return "{base}[MC{mc}].mat".format(base=self.get_base(l,t), mc=new_mc)
                                         
    def _load_bmatdict(self):
        """Ucitava bmatdict iz memorije"""
        with open(join(SIM_DIR,"mat.dict"),mode="ab+") as hashf:
           try:
               fcontent =  defaultdict(dict,pickle.load(hashf))
           except EOFError:
               fcontent = defaultdict(dict)
        print "bmatdict",fcontent
        self.bmatdict = fcontent
        self.repdict = defaultdict(dict)

    def clean_mat_dict(self):
        """Gleda koji unosi u dict nemaju vrednost
        tj. samo su dodati zbog defaultdict svojstva
        i skida ih. Ovo ce se samo desiti u mat chooseru
        tako da ovo tada samo treba da zovem.Nisam sigurna
        da ce mi biti potreban"""
        # ovo ce proci posto mi ne iteriramo kroz sam dict
        # a skidamo sa samog dicta. ova lista items nece biti
        # vise up to date, ali to nam nije bitno, posto nije
        # ciklicna
        self.log.debug("cleaning bmatdict : %s" % self.bmatdict)
        for key,value in self.bmatdict.items():
            if not value:
                self.bmatdict.pop(key)
                self.log.debug("removing {}.aplot".format(key))
                try:
                    os.remove(join(SIM_DIR,'{}.aplot'.format(key)))
                except Exception:
                    self.log.warning("Exception!!!")
        self.serialize_mat()

    def add_repaired(self,l,t,val):
        """Mislim ovo mi je glupost
        sigurno je lakse samo tamo
        ali ovako mogu menjati. Mada
        mozda bi mi propertyji dobro
        dosli
        """
        self.repdict[l][t]=val

    # def get_alt_base(self,l,t):
        
    def add_to_mat_dict(self,l,t,therm,mc):
        """Dodaje u dictionary 'reprezentativnih' matova, ispisuje poruku
        u status baru, i cuva novo stanje best_mat_dict-a na disk"""
        best_mat = self.get_files(l=l,t=t,ext="*%s%s*.mat" %(therm,mc))
        # ne bi smelo da ima fajlova u okviru jednog foldera sa istim MC i THERM
        assert len(best_mat)==1
        self.bmatdict[l][t] = best_mat[0]
        self.log.debug("added %s to bmatdict : %s " %(best_mat[0], self.bmatdict))
    #    self.parent.flash_status_message("Best .mat for %s%s selected" % (l,t))
        #mhm, nisam sigurna nisam sigurna nista. al ajd. nekako cu popraviti sve
        self.clean_mat_dict()
        #onda mi ne treba ovde serialize
#        self.serialize_mat()

    def remove_from_bmatdict(self,l,t):
        """Brise zapis za prosledjeno l i t iz bestmatdicta"""
        self.log.debug("Brisem zapis za l:{} t:{} i bmatdicta".format(l,t))
        del self.bmatdict[l][t]
        self.serialize_mat()
    def serialize_mat(self):
        with open(join(SIM_DIR,"mat.dict") ,"wb") as matdictfile:
            pickle.dump(dict(self.bmatdict),matdictfile)
    #matDictEmpty
    def bmatdict_empty(self):
        """ne znam da li mi je ova f-ja relevantna
        sad kad imam ovaj clean mm"""
        return not self.bmatdict.keys()
        # for key in self.best_mat_dict.keys():
        #     if self.best_mat_dict[key]:
        #         return False
        # return True
        
class ScatterPanel(wx.Panel):
    xlabel = 'Mx'    
    ylabel = 'My'
    zlabel = 'Mz'
    all_data = dict()
    ylim = None
    xlim = None
    chk_mc_txt = "Show first %d SPs"
    firstn = 1000
    def __init__(self,parent):
        wx.Panel.__init__(self,parent=parent,id=wx.ID_ANY)
        self.parent = parent
        logging.basicConfig(level=logging.DEBUG)
        self.log = logging.getLogger("ScatterPanel")
        self.tooltip = wx.ToolTip("'n':next\n'p':previous\nscroll:zoom")
        self.tooltip.Enable(False)
        self.dpi = 100
        fig_width = self.parent.GetParent().width / self.dpi
        fig_height = (self.parent.GetParent().height / self.dpi) * 3.2 / 4.0
        self.log.debug("fig width:{} fig height:{}".format(fig_width,fig_height))
        self.fig = Figure(figsize=(fig_width,fig_height),dpi=self.dpi,facecolor='white')
        self.canvas = FigCanvas(self,-1,self.fig)
        # self.ax = Axes3D(self.fig)
        #self.ax_3d = self.fig.add_subplot(121,projection="3d")
        self.ax_3d = self.fig.add_axes([0,0,0.5,1],projection="3d")
        print type(self.ax_3d)
        self.ax_hist = self.fig.add_axes([0.53,0.55,0.40,0.43])
        self.ax_qq = self.fig.add_axes([0.53,0.05,0.40,0.43])
              

        self.canvas.SetToolTip(self.tooltip)
        self.canvas.mpl_connect('key_press_event',self.on_key_press)
        self.canvas.mpl_connect('figure_enter_event',self.on_figure_enter)
        self.zoomf = zoom_factory(self.ax_3d,self.canvas)

        self.selector = RectangleSelector(self.ax_hist, self.selector_callback,
                             drawtype='box', useblit=True,
                             button=[1,3], # don't use middle button
                             minspanx=5, minspany=5,
                             spancoords='pixels')
   
        self.ax_3d.mouse_init()
        self.init_gui()
      
        if ():
            self.load_data(l=self.cmb_l.GetValue())

    def selector_callback(self,eclick,erelease):
        x_begin = eclick.xdata
        x_end = erelease.xdata
        if x_begin > x_end:
            x_begin,x_end = x_end,x_begin
        # znaci bice true ako se ne nalazi u ovoj regiji
        self.log.debug("x_begin {} x_end {}".format(x_begin,x_end))
        print self.magt
                       
        booli = [ not (mag>=x_begin and mag<=x_end) for mag in self.magt ]
        self.log.debug("booli:\n",booli)
        #znaci prosledjujemo mu za sta da generise mat
        curr_t = self.temprs.curr()
        curr_l = self.cmb_l.GetValue()
        print "AGGPANEL.aggd\n", AggPanel.aggd[curr_l][curr_t]

        mc = sum(booli)
        print "Neodstranjenih rezultata ima:\n", mc
        #!!!stavi ovde da bude statmat!!!
        #!!! posto necemo da dupliciramo kod
        #!!! tj, promenices statmat
        therm = file_manager.best_therm(curr_l,curr_t)
        print "Racuna za therm = ",therm
        fname = file_manager.get_all_file(curr_l,curr_t)
        mat = statmat.create_mat(fname,file_manager.get_alt_base(),therm,mc,booli)
        newmatfname =join(SIM_DIR,file_manager.get_changed_mat_name(curr_l,curr_t,mc))
        
        print "newmatfname:",newmatfname
        mat.to_csv(newmatfname, sep=' ')
        global repaired_dict
        print "repaired dict",repaired_dict
        file_manager.add_repaired(curr_l,curr_t,newmatfname)
        print "repaired dict",repaired_dict
        print "NEW MAT",mat
        print "BEST MAT DICT",file_manager.bmatdict
        self.step("dummy",curr=True,booli=booli)
        
    
    def change_dist(self,event):
        print type(event)
        dist= event.GetEventObject().GetLabel()
        self.radio_selected=dist;
        self.plot_qq(dist)

    def plot_qq(self,dist):
        self.ax_qq.cla()
        (x,y),(slope,inter,cor) = stats.probplot(self.magt,dist=dist)
        self.log.debug("type of x is %s" %type(x))
        pylab.setp(self.ax_qq.get_xticklabels(),fontsize=10)
        pylab.setp(self.ax_qq.get_yticklabels(),fontsize=10)
        osmf = x.take([0, -1])  # endpoints
        osrf = slope * osmf + inter
        lines = self.ax_qq.plot(osmf,osrf,'-', linewidth=0.2)
        lines = self.ax_qq.plot(x,y,',')
        self.canvas.draw()
        
        
    def save_figure(self, *args):
        filetypes, exts, filter_index = self.canvas._get_imagesave_wildcards()
        default_file = self.canvas.get_default_filename()
        dlg = wx.FileDialog(self, "Save to file", "", default_file,
                            filetypes,
                            wx.SAVE|wx.OVERWRITE_PROMPT)
        dlg.SetFilterIndex(filter_index)
        if dlg.ShowModal() == wx.ID_OK:
            dirname  = dlg.GetDirectory()
            filename = dlg.GetFilename()
            format = exts[dlg.GetFilterIndex()]
            basename, ext = os.path.splitext(filename)
            if ext.startswith('.'):
                ext = ext[1:]
            if ext in ('svg', 'pdf', 'ps', 'eps', 'png') and format!=ext:
                #looks like they forgot to set the image type drop
                #down, going with the extension.
                #warnings.warn('extension %s did not match the selected image type %s; going with %s'%(ext, format, ext), stacklevel=0)
                format = ext
            try:
                self.canvas.print_figure(
                    os.path.join(dirname, filename), format=format)
            except Exception as e:
                pass

    def on_figure_enter(self,event):
        self.tooltip.Enable(True)
        print "entered figure"
    
    def on_key_press(self,event):
        if event.key =='n':
            self.step(event)
        elif event.key =='p':
            self.step(event,backwards=True)
   
    def init_gui(self):
        self.vbox = wx.BoxSizer(wx.VERTICAL)
        
        self.vbox.Add(self.canvas,1,wx.EXPAND)
        value = "" if file_manager.bmatdict_empty() else file_manager.bmatdict.keys()[0]
        self.cmb_l = wx.ComboBox(self, size=(70, -1),
                                 choices=sorted(file_manager.bmatdict.keys(),key=lambda x: int(x[1:])),
                                 style=wx.CB_READONLY,
                                 value=value)

        self.reload_button = wx.Button(self, -1, 'Reload')
        self.draw_button = wx.Button(self, -1, '>', size=(40,-1))
        self.prev_button = wx.Button(self, -1, '<',size=(40,-1))
        self.save_button = wx.Button(self, -1, 'Save')
        self.chk_ylim = wx.CheckBox(self,-1,"Global ylim",size=(-1,30))
        self.chk_xlim = wx.CheckBox(self,-1,"Global xlim",size=(-1,30))
        self.chk_mcs = wx.CheckBox(self,-1,self.chk_mc_txt %1000 ,size=(-1,30))
        self.chk_mcs.SetValue(True)

        self.chk_hist = wx.CheckBox(self,-1,"Histogram",size=(-1,30))
        self.chk_qq = wx.CheckBox(self,-1,"QQPlot",size=(-1,30))
        self.chk_hist.SetValue(True)
        self.chk_qq.SetValue(True)

        self.rb_norm = wx.RadioButton(self, -1, 'norm', (10, 10), style=wx.RB_GROUP)
        self.rb_uniform = wx.RadioButton(self, -1, 'uniform', (10, 30))

        self.rb_norm.Bind(wx.EVT_RADIOBUTTON,self.change_dist)
        self.rb_uniform.Bind(wx.EVT_RADIOBUTTON,self.change_dist)
        self.radio_selected = "norm"

        self.mc_txt = wx.SpinCtrl(self,size=(80,-1))
        self.load_button = wx.Button(self,-1,'Load')
        
        self.Bind(wx.EVT_CHECKBOX,self.on_chk_mcs, self.chk_mcs)
        #self.Bind(wx.EVT_CHECKBOX,self.on_chk_qqhist, self.chk_qq)
        #self.Bind(wx.EVT_CHECKBOX,self.on_chk_qqhist, self.chk_hist)
        self.Bind(wx.EVT_CHECKBOX,self.on_chk_lim, self.chk_ylim)
        self.Bind(wx.EVT_CHECKBOX,self.on_chk_lim, self.chk_xlim)
        self.Bind(wx.EVT_BUTTON, self.on_load_button, self.load_button)
        self.Bind(wx.EVT_BUTTON, self.on_reload_button, self.reload_button)
        self.Bind(wx.EVT_BUTTON, self.step, self.draw_button)
        self.Bind(wx.EVT_BUTTON, self.on_prev_press, self.prev_button)
        self.Bind(wx.EVT_BUTTON, self.save_figure, self.save_button)
        self.Bind(wx.EVT_COMBOBOX,self.on_selectl,self.cmb_l)
        self.hbox1 = wx.BoxSizer(wx.HORIZONTAL)

        self.hbox1.Add(self.cmb_l, border=5, flag=wx.ALL
                       | wx.ALIGN_CENTER_VERTICAL)
        self.hbox1.Add(self.reload_button, border=5, flag=wx.ALL
                       | wx.ALIGN_CENTER_VERTICAL)
        self.hbox1.Add(self.prev_button, border=5, flag=wx.BOTTOM | wx.TOP | wx.LEFT
                       | wx.ALIGN_CENTER_VERTICAL)
        self.hbox1.Add(self.draw_button, border=5, flag=wx.BOTTOM | wx.TOP | wx.RIGHT
                       | wx.ALIGN_CENTER_VERTICAL)

        self.hbox1.Add(self.save_button, border=5, flag=wx.ALL
               | wx.ALIGN_CENTER_VERTICAL)
       
        self.hbox1.Add(self.mc_txt, border=5, flag=wx.ALL
               | wx.ALIGN_CENTER_VERTICAL)
        self.hbox1.Add(self.load_button, border=5, flag=wx.ALL
               | wx.ALIGN_CENTER_VERTICAL)
        self.hbox1.Add(self.chk_mcs, border=5, flag=wx.ALL
               | wx.ALIGN_CENTER_VERTICAL)
        self.hbox1.Add(self.chk_ylim, border=5, flag=wx.ALL
               | wx.ALIGN_CENTER_VERTICAL)
        self.hbox1.Add(self.chk_xlim, border=5, flag=wx.ALL
               | wx.ALIGN_CENTER_VERTICAL)

        self.hbox1.Add(self.rb_norm, border=5, flag=wx.ALL
               | wx.ALIGN_CENTER_VERTICAL)
        self.hbox1.Add(self.rb_uniform, border=5, flag=wx.ALL
               | wx.ALIGN_CENTER_VERTICAL)
        
        self.hbox1.AddSpacer(20)
        self.vbox.Add(self.hbox1, 0, flag=wx.ALIGN_LEFT | wx.TOP)
        self.SetSizer(self.vbox)
        self.vbox.Fit(self)

  

    def on_prev_press(self,event):
        self.step("dummy",backwards=True)

    def on_load_button(self,event):
        """Ucita n zadatih i stavi checkbox
        na odredjen tekst i otkaci ga posto pp.
        da ce neko ako vec ide na load, hteti odmah
        da vidi rezultate, a posle ga moze uncheck"""
        self.chk_mcs.SetValue(True)
        self.chk_mcs.SetLabel(self.chk_mc_txt % self.mc_txt.GetValue())
        self.firstn = self.mc_txt.GetValue()
        self.on_chk_mcs("dummy")
        

    def on_chk_mcs(self,event):
        firstn = self.firstn if self.chk_mcs.IsChecked() else None
        self.log.debug("Loading first {} sp for {}".format(firstn,None))
        self.load_data(l=self.cmb_l.GetValue(),n = firstn,keep=True)

    def on_chk_lim(self,event):
        self.ylim = self.global_ylim if self.chk_ylim.IsChecked() else self.local_ylim
        self.xlim = self.global_xlim if self.chk_xlim.IsChecked() else self.local_xlim
        self.log.debug("Setting ylim:{} and xlim:{}".format(self.ylim,self.xlim))
        self.ax_hist.set_ylim(self.ylim)
        self.ax_hist.set_xlim(self.xlim)
        self.canvas.draw()

    def on_reload_button(self,event):
        choices = sorted(file_manager.bmatdict.keys(),key=lambda x: int(x[1:]))
        self.cmb_l.SetItems(choices)
        self.cmb_l.SetValue(choices[0])
        self.load_data(l=choices[0])
        
    def on_selectl(self,event):
        print "Loading data for {}...".format(self.cmb_l.GetValue())
        self.load_data(l=self.cmb_l.GetValue())
        self.setup_plot()
   
    def load_data(self,l="L10",n=1000, keep=False):
        """Ucitava podatke za animaciju"""
        flist=glob.glob(join(SIM_DIR,"{}T*".format(l)))
        self.all_data= dict()
        
        flist = [f for f in flist if os.path.isdir(f)]
        self.log.debug("file list: %s " %flist)

        for f in file_manager.bmatdict[l].values():
            fname,t,self.curr_tmc =re.match(r"(.*%s(T\d+)THERM(\d+))" % l,f).groups()
            print f
            
            self.log.debug("Loading data for tempearture {}".format(t))
            self.curr_all = '%s.all' % fname
            self.log.debug("Loading data from file %s" % self.curr_all)
            # ne znam da li mi treba ovde neki try catch hmhmhmhmhmhmhmmhhh
            data = pd.read_table(self.curr_all,delim_whitespace=True,nrows=n, names=['seed', 'e', 'x', 'y', 'z'])
            data.pop('seed')
            self.log.debug("rows read: %s" % data.e.count())
            data.set_index(np.arange(data.e.count()),inplace=True)
            self.all_data[t] = data
        
        self.data = pd.concat(self.all_data,axis=0)
        self.ts = self.all_data.keys()
        ylims = list()
        xlims = list()
        for t in self.ts:
            self.ax_hist.cla()
            x,y,z = self.data.ix[t,'x'],self.data.ix[t,'y'],self.data.ix[t,'z']
            magt =np.sqrt( x ** 2 + y ** 2 + z ** 2)
            self.ax_hist.hist(magt,bins=100)
            ylims.append(self.ax_hist.get_ylim())
            xlims.append(self.ax_hist.get_xlim())
            self.log.debug("ylim for {} is {}".format(t,self.ax_hist.get_ylim()))
            self.log.debug("xlim for {} is {}".format(t,self.ax_hist.get_xlim()))
        print zip(*ylims)
        self.log.debug("xlims: {}".format(xlims))
        self.global_ylim = (min(zip(*ylims)[0]), max(zip(*ylims)[1]))
        self.global_xlim = (min(zip(*xlims)[0]), max(zip(*xlims)[1]))
        self.log.debug("Global maximum for {} is {}".format(l,self.global_ylim))
        self.log.debug("Global maximum for {} is {}".format(l,self.global_xlim))
            
        self.mc_txt.SetRange(0,file_manager.get_maxmc(l))

        key = lambda x: int(x[1:])
        self.ts = sorted(self.ts,key = lambda x: int(x[1:]))
        self.temprs = self.temprs if keep else twoway_cycle(self.ts)
        self.setup_plot(curr=True)
        self.canvas.mpl_connect('draw_event',self.forceUpdate)
        
        print DEBUG,"self.ts reversed",self.ts

    def setup_plot(self,curr=False):
        "Initial drawing of scatter plot"
        from matplotlib import cm
        self.step("dummy",curr=curr)

    
    def step(self,event, backwards=False,curr=False,booli=False):
        """Crta za sledece, proslo ili trenutno t"""
        t= (curr and self.temprs.curr()) or (self.temprs.next() if not backwards else self.temprs.prev())
        self.log.debug("t u step-u je ispalo :{}".format(t))
        x,y,z = self.data.ix[t,'x'],self.data.ix[t,'y'],self.data.ix[t,'z']
        self.magt =np.sqrt( x ** 2 + y ** 2 + z ** 2)
        colors = np.where(self.magt>np.mean(self.magt),'r','b')
        
        if booli:
            x=x[booli]
            y=y[booli]
            z=z[booli]
            self.magt = self.magt[booli]
                
        self.ax_3d.cla()
        self.ax_hist.cla()
        pylab.setp(self.ax_3d.get_xticklabels(),fontsize=8, color='#666666')
        pylab.setp(self.ax_3d.get_yticklabels(),fontsize=8, color='#666666')
        pylab.setp(self.ax_3d.get_zticklabels(),fontsize=8, color='#666666')


        pylab.setp(self.ax_hist.get_xticklabels(),fontsize=10)
        pylab.setp(self.ax_hist.get_yticklabels(),fontsize=10)
        pylab.setp(self.ax_qq.get_xticklabels(),fontsize=10)
        pylab.setp(self.ax_qq.get_yticklabels(),fontsize=10)
        
        self.ax_3d.set_xlabel(self.xlabel, fontsize=8)
        self.ax_3d.set_ylabel(self.ylabel,fontsize=8)
        self.ax_3d.set_zlabel(self.zlabel,fontsize=8)
        self.log.debug("Magt has {} elements".format(self.magt.count()))
        
        self.scat =  self.ax_3d.scatter(x,y,z,s=10,c = self.magt,cmap=cm.RdYlBu)
        therm = file_manager.best_therm(self.cmb_l.GetValue(),t)
        sp = file_manager.best_mc(self.cmb_l.GetValue(),t)
        title ="T={:.2f}\nLS={}\n SP={}".format((float(t[1:])/100),therm,sp)
        self.ax_3d.set_title(title, fontsize=10, position=(0.1,0.95))
        
        self.log.debug("Maksimum magt je {}".format(self.magt.max()))
#        self.ax_hist.set_ylim(0,magt.max()*1000)
        self.log.debug("MAGT:\n %s"% self.magt)
        z = self.ax_hist.hist(self.magt,bins=100,facecolor='green',alpha=0.75)
        print "ret. histograma\n",z
        print "size of z is{}".format(len(z[0]))
        self.local_ylim = self.ax_hist.get_ylim()
        self.local_xlim = self.ax_hist.get_xlim()
        self.log.debug("local xlim: {} local ylim: {}".format(self.local_ylim, self.local_xlim))
        self.on_chk_lim("dummy")
        self.ax_hist.set_ylim(self.ylim)
        self.ax_hist.set_xlim(self.xlim)
        self.plot_qq(self.radio_selected)
        self.canvas.draw()
               
    def forceUpdate(self,event):
        self.scat.changed()
    



            
class ThermPanel(wx.Panel):
    def __init__(self, parent):
        wx.Panel.__init__(self, parent=parent, id=wx.ID_ANY)
        self.log = logging.getLogger("ThermPanel")
        self.tooltip=wx.ToolTip("r:color to red\ng:color to green\n")
        self.parent = parent
        self.init_plot()
        self.canvas.SetToolTip(self.tooltip)
                       

        self.mc_txt = wx.SpinCtrl(self, size = (80,-1))
        self.add_button = wx.Button(self,-1,'Generate')
        self.clear_button = wx.Button(self,-1,"Clear")
        self.draw_button = wx.Button(self,-1,"Draw")
        self.btn_savesep = wx.Button(self,-1,"Save Axes")
#        self.draw_button.Enable(False)
  
        self.Bind(wx.EVT_BUTTON, self.on_add_button,self.add_button)
        self.Bind(wx.EVT_BUTTON, self.on_draw_button,self.draw_button)
        self.Bind(wx.EVT_BUTTON, self.on_save_button,self.btn_savesep)
        self.Bind(wx.EVT_BUTTON, self.on_clear_button, self.clear_button)
        self.chk_l = wx.CheckBox(self,-1,"L", size=(-1, 30))
        self.chk_t = wx.CheckBox(self,-1,"T",size=(-1, 30))
        self.chk_mc = wx.CheckBox(self,-1,"SP",size=(-1, 30))
        self.chk_l.Enable(False)
        self.chk_t.Enable(False)
        self.chk_mc.Enable(False)

        #mm, moram obezbediti da ovi imaju istu x i y. valjda samo zajedno
        # da ih uvek zovem, ili da namestim ono sharex i sharey. mada mozda to nece
        # neko hteti da ima
        self.tick_txt = wx.StaticText(self,label="ticks")
        self.label_txt = wx.StaticText(self,label="labels")
        self.lbl_slider = wx.Slider(self,value = self.ax_mag.xaxis.get_ticklabels()[0].get_fontsize(),
                                    minValue=5,maxValue=20,size=(100,-1),style=wx.SL_HORIZONTAL)

        self.xylbl_slider = wx.Slider(self,value = self.ax_mag.xaxis.get_ticklabels()[0].get_fontsize(),
                                    minValue=5,maxValue=20,size=(100,-1),style=wx.SL_HORIZONTAL)
        self.lbl_slider.Bind(wx.EVT_SCROLL,self.on_slider_scroll)
        self.xylbl_slider.Bind(wx.EVT_SCROLL,self.on_xyslider_scroll)
        self.Bind(wx.EVT_CHECKBOX,self.draw_legend,self.chk_l)
        self.Bind(wx.EVT_CHECKBOX,self.draw_legend,self.chk_t)
        self.Bind(wx.EVT_CHECKBOX,self.draw_legend,self.chk_mc)
        
        plot_choices = ['M1', 'M2', 'M4']
        self.cmb_plots = wx.ComboBox(self, size=(100, -1),
                choices=plot_choices, style=wx.CB_READONLY,
                value=plot_choices[0])
        l_choices = file_manager.l_choices()
        self.cmb_L = wx.ComboBox(self, size=(70, -1),
                                 choices=l_choices,
                                 style=wx.CB_READONLY,
                                 value=l_choices[0])
        t_choices = file_manager.t_choices(self.cmb_L.GetValue())
        self.cmb_T = wx.ComboBox(self, size=(100, -1),
                                 choices=t_choices,
                                 value=t_choices[0])


        self.cmb_pfiles = wx.ComboBox(self, size=(300, -1),
                choices=[], value='--')
        self.set_cmb_pfiles()
        self.update_combos()
        #self.Bind(wx.EVT_COMBOBOX, self.on_select, self.cmb_plots)
        self.Bind(wx.EVT_COMBOBOX, self.update_combos, self.cmb_L)
        self.Bind(wx.EVT_COMBOBOX, self.on_select_T, self.cmb_T)
        self.Bind(wx.EVT_COMBOBOX, self.on_select_pfiles,
                  self.cmb_pfiles)
        self.hbox1 = wx.BoxSizer(wx.HORIZONTAL)
        
        self.hbox1.AddSpacer(20)
   
        self.hbox1.Add(self.cmb_L, border=5, flag=wx.ALL
                       | wx.ALIGN_CENTER_VERTICAL)
        self.hbox1.Add(self.cmb_T, border=5, flag=wx.ALL
                       | wx.ALIGN_CENTER_VERTICAL)
        self.hbox1.Add(self.cmb_pfiles, border=5, flag=wx.ALL
                       | wx.ALIGN_CENTER_VERTICAL)
        self.hbox1.Add(self.cmb_plots, border=5, flag=wx.ALL
                       | wx.ALIGN_CENTER_VERTICAL)


        self.hbox1.Add(self.draw_button, border=5, flag=wx.ALL
               | wx.ALIGN_CENTER_VERTICAL)

        self.hbox1.Add(self.clear_button, border=5, flag=wx.ALL
                       | wx.ALIGN_CENTER_VERTICAL)
        self.hbox1.Add(self.mc_txt, border=5, flag=wx.ALL
               | wx.ALIGN_CENTER_VERTICAL)
        self.hbox1.Add(self.add_button, border=5, flag=wx.ALL
                       | wx.ALIGN_CENTER_VERTICAL)
   
        
        
   
        self.toolhbox =wx.BoxSizer(wx.HORIZONTAL)
   
        self.toolhbox.Add(self.toolbar)
        self.toolhbox.AddSpacer(20)
      
#        self.toolhbox.Add(self.btn_savesep, border=5, flag=wx.ALL
 #                      | wx.ALIGN_CENTER_VERTICAL)
        self.toolhbox.AddSpacer(20)
        self.toolhbox.Add(self.chk_l, border=5, flag=wx.LEFT | wx.BOTTOM | wx.TOP
                       | wx.ALIGN_CENTER_VERTICAL)
        self.toolhbox.Add(self.chk_t, border=5, flag=wx.TOP | wx.BOTTOM| wx.ALIGN_CENTER_VERTICAL)
        self.toolhbox.Add(self.chk_mc, border=5, flag=wx.RIGHT | wx.BOTTOM | wx.TOP
                       | wx.ALIGN_CENTER_VERTICAL)
        self.toolhbox.Add(self.tick_txt, border=5, flag=wx.ALL
               | wx.ALIGN_CENTER_VERTICAL)
        self.toolhbox.Add(self.lbl_slider, border=5, flag=wx.ALL
                       | wx.ALIGN_CENTER_VERTICAL)
        self.toolhbox.Add(self.label_txt, border=5, flag=wx.ALL
                       | wx.ALIGN_CENTER_VERTICAL)        
        self.toolhbox.Add(self.xylbl_slider, border=5, flag=wx.ALL
                       | wx.ALIGN_CENTER_VERTICAL)
   
        self.vbox = wx.BoxSizer(wx.VERTICAL)
        self.vbox.Add(self.canvas, 1, flag=wx.LEFT | wx.TOP)
        self.vbox.Add(self.toolhbox)
        self.vbox.Add(self.hbox1, 0, flag=wx.ALIGN_LEFT | wx.TOP)
        self.SetSizer(self.vbox)
        self.vbox.Fit(self)


    def on_save_button(self,event):
        filetypes, exts, filter_index = self.canvas._get_imagesave_wildcards()
        default_file = self.canvas.get_default_filename()
        dlg = wx.FileDialog(self, "Save to file", "", default_file,
                            filetypes,
                            wx.SAVE|wx.OVERWRITE_PROMPT)
        dlg.SetFilterIndex(filter_index)
        if dlg.ShowModal() == wx.ID_OK:
            dirname  = dlg.GetDirectory()
            filename = dlg.GetFilename()
            format = exts[dlg.GetFilterIndex()]
            basename, ext = os.path.splitext(filename)
            if ext.startswith('.'):
                ext = ext[1:]
            if ext in ('svg', 'pdf', 'ps', 'eps', 'png') and format!=ext:
                #looks like they forgot to set the image type drop
                #down, going with the extension.
                #warnings.warn('extension %s did not match the selected image type %s; going with %s'%(ext, format, ext), stacklevel=0)
                format = ext
               
            try:
                extent_mag = self.ax_mag.get_window_extent().transformed(self.fig.dpi_scale_trans.inverted())
                extent_cv = self.ax_cv.get_window_extent().transformed(self.fig.dpi_scale_trans.inverted()) 
                self.fig.savefig(os.path.join(dirname,filename+"_mag"),format=format,bbox_inches=extent_mag.expanded(1.2,1.5))
                self.fig.savefig(os.path.join(dirname,filename+"_cv"),format=format,bbox_inches=extent_cv.expanded(1.2,1.5))
            except Exception as e:
                raise e

    def set_ticklabelfontsize(self,size):
        pylab.setp(self.ax_mag.get_xticklabels(), fontsize=size)
        pylab.setp(self.ax_cv.get_xticklabels(), fontsize=size)
        pylab.setp(self.ax_mag.get_yticklabels(), fontsize=size)
        pylab.setp(self.ax_cv.get_yticklabels(), fontsize=size)
        self.canvas.draw()

    def set_labelfontsize(self,size):
        pylab.setp(self.ax_mag.xaxis.get_label(), fontsize=size)
        pylab.setp(self.ax_cv.xaxis.get_label(), fontsize=size)
        pylab.setp(self.ax_mag.yaxis.get_label(), fontsize=size)
        pylab.setp(self.ax_cv.yaxis.get_label(), fontsize=size)
        self.canvas.draw()
        

    def on_xyslider_scroll(self,e):
        fontsize = e.GetEventObject().GetValue()
        print "FONTSIZE", fontsize
        self.set_labelfontsize(fontsize)
        
    def on_slider_scroll(self,e):
        fontsize = e.GetEventObject().GetValue()
        print "FONTSIZE", fontsize
        self.set_ticklabelfontsize(fontsize)
        
   
   
    def on_clear_button(self,event):
        self.reset_chkboxes()
        self.chk_l.Enable(False)
        self.chk_t.Enable(False)
        self.chk_mc.Enable(False)
        self.ax_cv.cla()
        self.ax_mag.cla()
        self.set_labelfontsize(self.xylbl_slider.GetValue())
        self.set_ticklabelfontsize(self.lbl_slider.GetValue())
      #  self.ax_mag.set_title('Magnetisation',fontsize=10,family='monospace')
       # self.ax_cv.set_title('Coefficient of Variation',fontsize=10,family='monospace')
        self.canvas.draw()
   
  
        
        
    def on_add_button(self,event):
        """Pravi novi .plot fajl u zeljenom direktorijumu
        tj. u zavisnosti od vrednosti L i T combo boxova"""
        mcs =int( self.mc_txt.GetValue())
        # gledamo sta je korisnik selektovao sto se tice L i T
        # i onda u tom folderu pravimo nove .mat fajlove
        # i radimo compose nad novim .mat fajlovima
        l = self.cmb_L.GetValue()
        t = self.cmb_T.GetValue()
        file_manager.add_mc(l,t,mcs)
        self.set_cmb_pfiles()

    def set_cmb_pfiles(self):
        """Posto je malo komplikovanije,napraviti prikaz
        ovde ce se dovlaciti moguci mc-ovi i za njih broj
        thermova. U da, zamalo zaboravih, treba da se updejtuje
        choices kad se dodaju novi!!!"""
        l = self.cmb_L.GetValue()
        t = self.cmb_T.GetValue()               
        mcs = file_manager.mc_choices(l,t)
        mc_choices = ["{mc} [{tmc}]".format(mc=mc,tmc=file_manager.therm_count(l,t,mc)) for mc in mcs]
        self.log.debug("setting mc_choices:{}".format(mc_choices))
        self.cmb_pfiles.SetItems(mc_choices)
        self.cmb_pfiles.SetValue(mc_choices[0])
        
        
    def on_select_T(self,event="dummy"):
        self.set_cmb_pfiles()

#        self.draw_button.Enable(False)
        self.reset_chkboxes()
        # treba i checkboxovi da su disabled dok god nije nacrtan plot
        # znaci kad se pozove draw onda se enable-uju a kad se cler pozove onda
        # se disejbluju. valjda su to svi slucajevi 
        # trazimo najveci mc od svih .all fajlova.  inace
        # nece se desiti nista pri odabiru generisanja za taj mc
        l = self.cmb_L.GetValue()
        t = self.cmb_T.GetValue()
        int_regex = re.compile(r'(\d+)')
        uplimit = int(int_regex.search(file_manager.get_maxmc(l,t)).group(0))
        
        print "uplimit",uplimit
        self.mc_txt.SetRange(0,uplimit)
    
   
    def update_combos(self,event="dummy"):
        """Ova metoda azurira kombinacijske kutije koje zavise od stanja
        drugih elemenata/kontrola """
        t_items = file_manager.t_choices(self.cmb_L.GetValue())
        self.cmb_T.SetItems(t_items)
        self.cmb_T.SetValue(t_items[0])
        #selektovali smo T, pa pozivamo odgovarajucu metodu
        self.on_select_T()

      
         
    def get_files(self):
        """Kada zovemo iz klase ovu metodu, uglavnom nista ne tweakujemo i to, vec
        stavljamo defaultnu vrednost "*.plot" """
        return file_manager.get_files(ext="*.plot",l=self.cmb_L.GetValue(),t=self.cmb_T.GetValue())
        

    def get_mc(self):
        """Vraca selektovan mc"""
        return re.search(r'MC(\d+)', self.cmb_pfiles.GetValue()).group(0)

    def on_select_pfiles(self, event):
        """Kada korisnik izabere .plot fajl, iscrtava se """
        
        mc = self.get_mc()
        l = self.cmb_L.GetValue()
        t  = self.cmb_T.GetValue()
        self.log.debug("On select pfiles, ali neradi!!!")
        self.data = file_manager.compose(l,t,mc)
        self.log.debug("Loaded data for plot:\n %s" % self.data)
        self.draw_plot()
        
    def init_plot(self):
        self.dpi = 100
        fig_width = (self.parent.GetParent().width / self.dpi) 
        fig_height = self.parent.GetParent().height / self.dpi * 3 / 4

        self.fig = Figure((fig_width, fig_height), dpi=self.dpi,facecolor='w')



        self.ax_mag = self.fig.add_subplot(1, 2, 1)
        self.ax_cv = self.fig.add_subplot(1, 2, 2)

        # self.ax_mag.set_title('Magnetisation',fontsize=10,family='monospace')
        # self.ax_cv.set_title('Coefficient of Variation',fontsize=10,family='monospace')

        # pylab.setp(self.ax_mag.get_xticklabels(), fontsize=5)
        # pylab.setp(self.ax_cv.get_xticklabels(), fontsize=5)
        # pylab.setp(self.ax_mag.get_yticklabels(), fontsize=5)
        # pylab.setp(self.ax_cv.get_yticklabels(), fontsize=5)
        self.canvas = FigCanvas(self, -1, self.fig)
        self.toolbar = NavigationToolbar(self.canvas)
        tw,th = self.toolbar.GetSizeTuple()
        fw,fh = self.canvas.GetSizeTuple()
        self.toolbar.SetSize(wx.Size(fw,20))
        self.toolbar.Realize()

        self.canvas.mpl_connect('key_press_event',self.on_key_press)
        self.canvas.mpl_connect('pick_event', self.on_pick)

    def on_pick(self,event):
        artist = event.artist
        artist.set_color(np.random.random(3))
        print "pick"
        self.canvas.draw()

    def on_key_press(self,event):
        if event.key in 'rgbcmyk':
            self.error_line.get_children()[0].set_color(event.key)
            self.semilog_line.set_color(event.key)
        elif event.key in "-.," :
            self.error_line.get_children()[0].set_linestyle(event.key)
            self.semilog_line.set_linestyle(event.key)
        elif event.key in "ov^<>1234s*hHDd|":
            self.error_line.get_children()[0].set_marker(event.key)
            self.semilog_line.set_marker(event.key)
        elif event.key =="+":
            self.semilog_line.set_linewidth(self.semilog_line.get_linewidth()+0.1)
        elif event.key =="-":
            self.semilog_line.set_linewidth(self.semilog_line.get_linewidth()-0.1)
        elif event.key=="p":
            print "pp"
            self.multi=widgets.MultiCursor(self.canvas,[self.ax_mag,self.ax_cv],useblit=True,color='gray',lw=1)
        elif event.key=="x":
            del self.multi
                              
                              
        self.canvas.draw()

        

    # def on_select(self, event):
    #     # item = self.cmb_plots.GetValue()
    #     # self.draw_plot(item=item)
    #     self.draw_button.Enable(True)

    def on_draw_button(self,event):
        mc = self.get_mc()
        l = self.cmb_L.GetValue()
        t  = self.cmb_T.GetValue()
        self.log.debug("on_draw_button")
        self.data = file_manager.compose(l,t,mc)
        self.log.debug("Loaded data for plot:\n %s" % self.data)
        self.draw_plot()
        # self.draw_plot(self.cmb_plots.GetValue())
        
    def draw_legend(self,event):
        lbl_mc = self.get_mc()
        lbl_mc ="%s=%s" %("SP",lbl_mc[2:])
        lbl_t ="%s=%.2f" %(self.cmb_T.GetValue()[0],float(self.cmb_T.GetValue()[1:])/100)
        lbl_l ="%s=%s"% (self.cmb_L.GetValue()[0],self.cmb_L.GetValue()[1:])
        lchk  = self.chk_l.IsChecked()
        mcchk  = self.chk_mc.IsChecked()
        tchk  = self.chk_t.IsChecked()
        lbl = "%s %s %s" %(lbl_l if lchk else "", lbl_t if tchk else "",lbl_mc if mcchk else "")
        print lbl


        error_line = self.ax_mag.get_lines()[-1]
        semilog_line = self.ax_cv.get_lines()[-1]
        error_line.set_label(lbl)
        semilog_line.set_label(lbl)
        mag_leg=self.ax_mag.legend(loc="best",fontsize=10,frameon=False,shadow=True)
        mag_leg.draggable(True)
        cv_leg=self.ax_cv.legend(loc="best",fontsize=10,frameon=False,shadow=True)
        cv_leg.draggable(True)
        self.canvas.draw()

    def reset_chkboxes(self):
        self.chk_l.SetValue(False)
        self.chk_mc.SetValue(False)
        self.chk_t.SetValue(False)
    def draw_plot(self, item='M1'):
        """Redraws the plot
        """
        # self.ax_mag.cla()
        # self.ax_cv.cla()
        self.reset_chkboxes()
        self.log.debug("Crtam grafik za{}".format(self.get_mc()))
        
        
        
        fmt =fmt_cycle.next()
        self.error_line = self.ax_mag.errorbar(x=self.data.ix['THERM'],
                             y=self.data.ix[item + 'avg'],
                             yerr=self.data.ix['stdMean' + item],fmt=fmt,fillstyle='none',
                                           picker=5)
        
        self.semilog_line = self.ax_cv.semilogx(self.data.ix['THERM'], self.data.ix['cv(%s)'
                             % item],fmt,fillstyle='none',picker=5)[0]

        self.ax_mag.set_xscale('log')
        self.ax_cv.grid(True, color='red', linestyle=':')
        self.ax_mag.grid(True, color='red', linestyle=':')
        self.ax_cv.set_xlabel('Number of lattice sweeps')
        self.ax_mag.set_xlabel('Number of lattice sweeps')
        # ne znam da li ovo moze bolje. najbolje bi bilo da izmenim kako se zapisuju
        # ustvari. sta ce nam M1. Samo sto ce onda sve menjati. Onda  bi bio i lepsi
        # combobox. Combobox bi svakako trebao da bude lepsi za ove vrednosti. hm.
        # kako se uopste ugradjuje support za tex . mogla bih to da uradim za cmb box
        mlanglerangle = "%s^%s" %(item[0],item[1]) if int(item[1])!=1 else "%s" % item[0]
        self.ax_cv.set_ylabel(r'Coefficient of variation for $\langle{%s}\rangle$'
                               % mlanglerangle)
        self.ax_mag.set_ylabel(r'$\langle{%s}\rangle$' % (mlanglerangle))
        self.set_labelfontsize(self.xylbl_slider.GetValue())
        self.set_ticklabelfontsize(self.lbl_slider.GetValue())
        
        self.canvas.draw()
        self.chk_l.Enable(True)
        self.chk_t.Enable(True)
        self.chk_mc.Enable(True)



    
class AggPanel(wx.Panel):

    name_dict = {'susc':r"$\mathcal{X}$",'Tcap':'$C_V$','T':'$T$',
                 'M1avg':r"$\langle{M}\rangle$",'M2avg':r"$\langle{M^2}\rangle$",'M4avg':r"$\langle{M^4}\rangle$",'Eavg':r"$\langle{H}\rangle$",'E2avg':r"$\langle{H^2}\rangle$",'U':r'$U_{L}$'}
    def __init__(self, parent):

        wx.Panel.__init__(self, parent=parent, id=wx.ID_ANY)
        self.parent = parent
        self.init_plot()
        self.init_gui()        
        
    def init_gui(self):
        self.bestmat_button=wx.Button(self,-1,"Choose best .mats ...")
        self.Bind(wx.EVT_BUTTON, self.on_chooser,self.bestmat_button)
        self.agregate_btn = wx.Button(self,-1,"Aggregate!")
        self.agregate_btn.Enable( not file_manager.bmatdict_empty())
        self.Bind(wx.EVT_BUTTON,self.on_agg_button,self.agregate_btn)

        self.alt_btn = wx.Button(self,-1,"Alt agg!")
#        self.alt_btn.Enable( not matDictEmpty())
        self.Bind(wx.EVT_BUTTON,self.on_alt_button,self.alt_btn)
        self.cmb_L = wx.ComboBox(
            self,
            -1,
            value="<Aggregate first!>",
         #   value=L_choices[0],
            size=(150, -1),
#            choices=L_choices,
            style=wx.CB_READONLY,
            )
        self.cmb_mag = wx.ComboBox(
            self,
            -1,
 #           value=mag_choices[0],
            value="<Aggregate First!>",
            size=(150, -1),
#            choices=mag_choices,
            style=wx.CB_READONLY,
            )
        self.draw_button = wx.Button(self, -1, 'Draw')
        self.draw_button.Enable(False)
        self.Bind(wx.EVT_BUTTON, self.on_draw_button, self.draw_button)
        self.clear_button = wx.Button(self, -1, 'Clear')
        self.clear_button.Enable(True)
        self.Bind(wx.EVT_BUTTON, self.on_clear_button,
                  self.clear_button)


        self.tick_txt = wx.StaticText(self,label="ticks")
        self.label_txt = wx.StaticText(self,label="labels")
        self.lbl_slider = wx.Slider(self,value = self.ax_agg.xaxis.get_ticklabels()[0].get_fontsize(),
                                    minValue=5,maxValue=20,size=(100,-1),style=wx.SL_HORIZONTAL)

        self.xylbl_slider = wx.Slider(self,value = self.ax_agg.xaxis.get_ticklabels()[0].get_fontsize(),
                                    minValue=5,maxValue=20,size=(100,-1),style=wx.SL_HORIZONTAL)
        self.lbl_slider.Bind(wx.EVT_SCROLL,self.on_slider_scroll)
        self.xylbl_slider.Bind(wx.EVT_SCROLL,self.on_xyslider_scroll)
        self.chk_ann = wx.CheckBox(self,-1,"Annotate", size=(-1,30))
        self.chk_ann.Enable(False)
        self.Bind(wx.EVT_CHECKBOX, self.on_chk_ann, self.chk_ann)
        
        self.hbox1 = wx.BoxSizer(wx.HORIZONTAL)
        self.hbox1.Add(self.bestmat_button, border=5, flag=wx.ALL
                       | wx.ALIGN_CENTER_VERTICAL) 
        self.hbox1.Add(self.agregate_btn, border=5, flag=wx.ALL
                       | wx.ALIGN_CENTER_VERTICAL)
        self.hbox1.Add(self.alt_btn, border=5, flag=wx.ALL
                       | wx.ALIGN_CENTER_VERTICAL) 
        self.hbox1.Add(self.cmb_L, border=5, flag=wx.ALL
                       | wx.ALIGN_CENTER_VERTICAL)
        self.hbox1.AddSpacer(20)
        self.hbox1.Add(self.cmb_mag, border=5, flag=wx.ALL
                       | wx.ALIGN_CENTER_VERTICAL)
        self.hbox1.AddSpacer(20)
        self.hbox1.Add(self.draw_button, border=5, flag=wx.ALL
                       | wx.ALIGN_CENTER_VERTICAL)
        self.hbox1.AddSpacer(20)
        self.hbox1.Add(self.clear_button, border=5, flag=wx.ALL
                       | wx.ALIGN_CENTER_VERTICAL)
        self.toolhbox =wx.BoxSizer(wx.HORIZONTAL)
        self.toolhbox.Add(self.toolbar)
        self.toolhbox.Add(self.tick_txt, border=5, flag=wx.ALL
               | wx.ALIGN_CENTER_VERTICAL)
        self.toolhbox.Add(self.lbl_slider, border=5, flag=wx.ALL
                       | wx.ALIGN_CENTER_VERTICAL)
        self.toolhbox.Add(self.label_txt, border=5, flag=wx.ALL
                       | wx.ALIGN_CENTER_VERTICAL)        
        self.toolhbox.Add(self.xylbl_slider, border=5, flag=wx.ALL
                       | wx.ALIGN_CENTER_VERTICAL)
        self.toolhbox.Add(self.chk_ann, border=5, flag=wx.ALL
                       | wx.ALIGN_CENTER_VERTICAL)
        self.vbox = wx.BoxSizer(wx.VERTICAL)
        self.vbox.Add(self.canvas, 1,  wx.EXPAND)
        self.vbox.Add(self.toolhbox)
        self.vbox.Add(self.hbox1, 0, flag=wx.ALIGN_LEFT | wx.TOP)
        self.SetSizer(self.vbox)
        self.vbox.Fit(self)


    def on_chk_ann(self,event):
        for ann in self.annotations:
            ann.set_visible(self.chk_ann.IsChecked())
        self.canvas.draw()
    def set_ticklabelfontsize(self,size):
        pylab.setp(self.ax_agg.get_xticklabels(), fontsize=size)

        pylab.setp(self.ax_agg.get_yticklabels(), fontsize=size)

        self.canvas.draw()

    def set_labelfontsize(self,size):
        pylab.setp(self.ax_agg.xaxis.get_label(), fontsize=size)
        pylab.setp(self.ax_agg.yaxis.get_label(), fontsize=size)
        self.canvas.draw()
        
    def on_chooser(self,event):
        self.chooser = Reader(self,-1,"Chooser")
        self.chooser.ShowModal()
        file_manager.clean_mat_dict()
        print "BEST MAT DICT", file_manager.bmatdict
        self.chooser.Destroy()
        self.agregate_btn.Enable(not file_manager.bmatdict_empty())
        
    def on_agg_button(self,event):
        self.agregate(file_manager.bmatdict)

    def on_alt_button(self,event):
        altm = file_manager.get_alt()
        print "altm:",altm
        self.agregate(altm)
        
    def agregate(self,bmatdict):
        #!!! hm,dobro ove choices resi
        #kad budes resila ono sa lazy evaluation
        #hmmmhhhhh
        agregate.main(dict(bmatdict),SIM_DIR)

        aggd = load_agg()
        # stavljamo ovde kao staticku variablu
        # valjda je ovo ok
        self.L_choices = zip(*aggd.columns)[0]
        print self.L_choices
        
        self.L_choices = list(set(self.L_choices))
        print self.L_choices
        
        self.mag_choices = list(aggd.index)
        self.cmb_L.SetItems(self.L_choices)
        self.cmb_L.SetValue(self.L_choices[0])
        self.cmb_mag.SetItems(self.mag_choices)
        self.cmb_mag.SetValue(self.mag_choices[0])
        self.draw_button.Enable(True)
        #ovome ce moci da se pristupi i preko self i to
        # samo sto ako ga prebrisemo, nece biti dobro
        # samo nam je potrebno da ponovo izracunamo za jedno
        # L i T mat i to je to
        AggPanel.aggd = aggd;

    def on_xyslider_scroll(self,e):
        fontsize = e.GetEventObject().GetValue()
        print "FONTSIZE", fontsize
        self.set_labelfontsize(fontsize)
        
    def on_slider_scroll(self,e):
        fontsize = e.GetEventObject().GetValue()
        print "FONTSIZE", fontsize
        self.set_ticklabelfontsize(fontsize)


    
    def on_draw_button(self, event):
        int_regex = re.compile(r'(\d+)')
        L_select = self.cmb_L.GetValue()
        mag_select = self.cmb_mag.GetValue()
        print L_select, mag_select
        lbl = "$%s_{%s}$" %(L_select[0],L_select[1:])
        print  self.aggd[L_select].ix['T'].index
        self.ax_agg.plot(self.aggd[L_select].ix['T'],
                         self.aggd[L_select].ix[mag_select],fmt_cycle.next(), label=lbl,fillstyle='none',picker=5)
        
        self.annotations = list()
        for t,m in zip(self.aggd[L_select].ix['T'].index,self.aggd[L_select].ix[mag_select]):
            print t
            self.annotations.append(self.ax_agg.annotate('LS={}\nSP={}'.format(file_manager.best_therm(L_select,t),file_manager.best_mc(L_select,t),xy=(float(t[1:])/100,m),xytext=(float(t[1:])/100,m), visible=False,fontsize=8)))

        self.chk_ann.SetValue(False)
        self.chk_ann.Enable(True)
        
        self.ax_agg.set_xlim(right=1.55)
        leg = self.ax_agg.legend(loc="best",frameon=False,shadow=True)
        leg.draggable(True)
        self.ax_agg.set_xlabel("$T$")
        self.ax_agg.set_ylabel(self.name_dict[mag_select])
        self.set_labelfontsize(self.xylbl_slider.GetValue())
        self.set_ticklabelfontsize(self.lbl_slider.GetValue())
        self.ax_agg.grid(True, color='red', linestyle=':')
        self.canvas.draw()
        
    def on_clear_button(self,event):
        self.ax_agg.cla()
        self.set_labelfontsize(self.xylbl_slider.GetValue())
        self.set_ticklabelfontsize(self.lbl_slider.GetValue())
        self.chk_ann.SetValue(False)
        self.chk_ann.Enable(False)
        
        self.canvas.draw()
        
    def init_plot(self):
        self.dpi = 100
        fig_width = self.parent.GetParent().width / self.dpi
        fig_height = self.parent.GetParent().height / self.dpi * 3 / 4
        self.fig = Figure((fig_width, fig_height), dpi=self.dpi,facecolor='w')
        self.ax_agg = self.fig.add_subplot(111)
        
#        self.ax_agg.set_title('Aggregate')
      
        self.canvas = FigCanvas(self, -1, self.fig)
        self.toolbar = NavigationToolbar(self.canvas)
        tw,th = self.toolbar.GetSizeTuple()
        fw,fh = self.canvas.GetSizeTuple()
        self.toolbar.SetSize(wx.Size(fw,20))
        self.toolbar.Realize()


class TabContainer(wx.Notebook):

    def __init__(self, parent):

        wx.Notebook.__init__(self, parent, id=wx.ID_ANY,
                             style=wx.BK_DEFAULT)
        self.AddPage(ThermPanel(self), 'Therm')
        self.AddPage(AggPanel(self), 'Aggregate')
        self.AddPage(ScatterPanel(self),"Scatter")

    def flash_status_message(self,message):
        self.GetParent().flash_status_message(message,3000)


class GraphFrame(wx.Frame):

    """The main frame of the application
    """

    title = 'tulip'

    def __init__(self):
        wx.Frame.__init__(self, None, -1, title=self.title)

##        self.size = tuple(map(operator.mul, wx.DisplaySize(),(3.0/4.0)))

        self.width = float(wx.DisplaySize()[0]) * 4.0 / 4.0
        self.height = float(wx.DisplaySize()[1]) * 3.5 / 4.0
        self.SetSize((self.width, self.height))
        self.Center()

        self.create_menu()
        self.create_status_bar()
        self.create_main_panel()

    def create_menu(self):
        self.menubar = wx.MenuBar()
        menu_file = wx.Menu()

        m_exit = menu_file.Append(-1, 'E&xit\tCtrl-X', 'Exit')
        self.Bind(wx.EVT_MENU, self.on_exit, m_exit)

        self.menubar.Append(menu_file, '&File')
        self.SetMenuBar(self.menubar)

    def create_main_panel(self):
        self.notebook = TabContainer(self)
        self.notebook.SetPadding(wx.Size(self.width / 6.0
                                 - self.notebook.GetFont().GetPixelSize()[0]
                                 * 5, -1))
        sizer = wx.BoxSizer(wx.VERTICAL)

        sizer.Add(self.notebook, 1, wx.ALL | wx.EXPAND, 5)
        self.SetSizer(sizer)
        self.Layout()

       
    def create_status_bar(self):
        self.statusbar = self.CreateStatusBar()

    def on_exit(self, event):
        self.Destroy()

    def flash_status_message(self, msg, flash_len_ms=1500):
        self.statusbar.SetStatusText(msg)
        self.timeroff = wx.Timer(self)
        self.Bind(wx.EVT_TIMER, self.on_flash_status_off, self.timeroff)
        self.timeroff.Start(flash_len_ms, oneShot=True)

    def on_flash_status_off(self, event):
        self.statusbar.SetStatusText('')



    


# def compose(ltdir,mcsteps="\d+",save=False):
#     """Po thermovima redja vrednosti iz matova,
#     ako treba da gledamo samo za odredjen broj mc-ova, onda
#     prosledjujemo parametar mcsteps (inace se svi citaju), ako
#     hocemo da cuvamo rezultat onda stavljamo save na True. Samo prvi
#     put ce save biti true, ovo ostalo nece biti volataile, vec samo da
#     se gleda"""
#     glregex = \
#         re.compile(r"^(L\d+)(T\d+)THERM\d+(MC%s)\.mat$" % mcsteps)
#     file_list = [f for f in os.listdir(ltdir) if glregex.match(f)]
#     plots = list(set([glregex.match(f).groups() for f in file_list]))
#     df= "nesto"
#     for plot in plots:
#         data = dict()
#         for mat in re.findall("^%s%sTHERM.*%s\.mat$" % plot,
#                               '\n'.join(file_list), re.MULTILINE):
#             ser = pd.read_csv(join(ltdir,mat), sep=' ', index_col=0, names=['t'])['t']
#             data[ser.ix['THERM']] = ser

#         base = '%s%s%s_THERM' % plot
#         plotf = base + '.plot'
#         print "making",plotf,"..."
#         df = pd.DataFrame(data)

#         out = { 'abs(cv(E1))':abs(df.ix['stdMeanE'] / df.ix['Eavg']),
#                 'cv(E2)':df.ix['stdMeanE2']/df.ix['E2avg'],
#                 'cv(M1)':df.ix['stdMeanM1']/df.ix['M1avg'],
#                 'cv(M2)':df.ix['stdMeanM2']/df.ix['M2avg'],
#                 'cv(M4)':df.ix['stdMeanM4']/df.ix['M4avg']}
#         out = pd.DataFrame(out)
#         out = pd.concat([df,out.T])

#         if save:
#             out.to_csv(join(ltdir,plotf))
#         return out

#ili je mozda umesto ovih save-ova bolje da imam neki temp
# folder, nisam sigurna, posto mm, brze ce ici ako se ne cuva
# fazon jeste sto ce to samo trebati da se radi za odredjeno
# t, za pocetak, shvatas? znaci mi cemo praviti samo za jedno t
# i onda njega ubacivati sa ostalim agregatima hmmmmmmmmmmmmmmmmm
# ti aplot fajlovi mi onako nisu nesto vazni a ovi compose ustvari
# nam ni ne trebaju. posto aggregate radi za mat fajlovima da
# ok, sta je fazon, fazon je sto nam trebaju novi matovi za odredjeno
# t i onda da zamenimo agregat sa tim. znaci kad kliknemo aggregate da

def load_agg():
    aplot_files = [f for f in os.listdir(SIM_DIR) if regexf.match(f)]
    aplot_files.sort()
    agg = dict()
    for aplot in aplot_files:
        L = regexf.match(aplot).groups()[0]
        data = pd.read_csv(join(SIM_DIR,aplot), index_col=0)
        agg[L] = data

    agg_data = pd.concat(agg, axis=1)
    agg_data.columns.names = ['L', 'T']
    print agg_data
    return agg_data

        



def IsBold(l,itemText):
    """Proverava da li je item u listi bold
    ili ne"""
#    font = item.GetFont()
    print "checking if item '{}' is bold or not...".format(itemText)
    print "or is in {}".format(file_manager.bmatdict[l].keys())
    return itemText in file_manager.bmatdict[l].keys()
    
def MakeBold(self,item):
        font = item.GetFont()
        font.SetWeight(wx.FONTWEIGHT_BOLD)
        item.SetFont(font)
        self.SetItem(item)
        
def UnBold(self,item):
        font = item.GetFont()
        font.SetWeight(wx.FONTWEIGHT_NORMAL)
        item.SetFont(font)
        self.SetItem(item)
        
class ListCtrlLattice(wx.ListCtrl):
    def __init__(self, parent, id):
        wx.ListCtrl.__init__(self, parent, id, style=wx.LC_REPORT | wx.LC_HRULES | 
		wx.LC_NO_HEADER | wx.LC_SINGLE_SEL)
       
        self.parent = parent

        self.Bind(wx.EVT_SIZE, self.OnSize)
        self.Bind(wx.EVT_LIST_ITEM_SELECTED, self.OnSelect)
        self.InsertColumn(0, '')
        cntr = 0
        for l in file_manager.l_choices():
            self.InsertStringItem(cntr,l)
            
            if l in file_manager.bmatdict.keys() and file_manager.bmatdict[l]:
                item = self.GetItem(cntr)
                MakeBold(self,item)

                
            cntr = cntr+1    

    
        
    def OnSize(self, event):
        size = self.parent.GetSize()
        self.SetColumnWidth(0, size.x-5)
        event.Skip()

    
    def OnSelect(self, event):
        print "parent of window is ",self.parent
        print "grand parent of window is",self.parent.GetGrandParent()
        selected = event.GetIndex()
        
        self.window = getSiblingByName(self,"ListControlTemperature")
        self.parent.GetGrandParent().GetParent().FindWindowByName('rightSplitter').FindWindowByName('ListControlTherm').DeleteAllItems()
        self.parent.GetGrandParent().GetParent().FindWindowByName('rightSplitter').FindWindowByName('ListControlMC').DeleteAllItems()
        self.parent.GetGrandParent().GetGrandParent().disable_choose()
        self.window.LoadData(self.GetItemText(selected))
        self.parent.GetGrandParent().GetGrandParent().unchoose_enabled(False);

    def OnDeSelect(self, event):
        index = event.GetIndex()
        self.SetItemBackgroundColour(index, 'WHITE')

    def OnFocus(self, event):
        self.SetItemBackgroundColour(0, 'red')

def getSiblingByName(listctrl,name):
    return listctrl.parent.GetGrandParent().FindWindowByName(name)
    

class ListCtrlTempr(wx.ListCtrl):
    def __init__(self, parent, id):
        wx.ListCtrl.__init__(self, parent, id, style=wx.LC_REPORT | wx.LC_HRULES | 
		wx.LC_NO_HEADER | wx.LC_SINGLE_SEL)

        self.parent = parent

        self.Bind(wx.EVT_SIZE, self.OnSize)
        self.Bind(wx.EVT_LIST_ITEM_SELECTED,self.OnSelect)

        self.InsertColumn(0, '')


    def OnSize(self, event):
        size = self.parent.GetSize()
        self.SetColumnWidth(0, size.x-5)
        event.Skip()

    def OnSelect(self, event):
        print "parent of window is ",self.parent
        print "grand parent of window is",self.parent.GetGrandParent().GetParent()
        print "grand parent of window is",self.parent.GetGrandParent().GetParent()
        window = self.parent.GetGrandParent().GetParent().FindWindowByName('rightSplitter').FindWindowByName('ListControlTherm')
        print "window",window
#        selected = event.GetIndex()
        selected = self.GetFirstSelected()
        print "first selected",selected
        self.parent.GetGrandParent().GetParent().FindWindowByName('rightSplitter').FindWindowByName('ListControlMC').DeleteAllItems()
        self.parent.GetGrandParent().GetGrandParent().disable_choose()
        self.parent.GetGrandParent().GetGrandParent().unchoose_enabled(IsBold(self.l,self.GetItemText(selected)));
        
        window.LoadData(self.GetItemText(selected),self.l)
        print "selected item",self.GetItemText(selected)

    def LoadData(self, item):
        self.DeleteAllItems()
        self.l = item
        cntr = 0
        for t in file_manager.t_choices(item):
            self.InsertStringItem(cntr,t)
            if t in file_manager.bmatdict[item].keys():
                MakeBold(self,self.GetItem(cntr))

                
            cntr = cntr+1


class ListCtrlTherm(wx.ListCtrl):
    def __init__(self, parent, id):
        wx.ListCtrl.__init__(self, parent, id, style=wx.LC_REPORT | wx.LC_HRULES | 
		wx.LC_NO_HEADER | wx.LC_SINGLE_SEL)

        self.parent = parent

        self.Bind(wx.EVT_SIZE, self.OnSize)
        self.Bind(wx.EVT_LIST_ITEM_SELECTED,self.OnSelect)

        self.InsertColumn(0, '')
    
        self.SetName('ListControlTherm')
    def OnSelect(self, event):
        print "parent of window is ",self.parent
        print "grand parent of window is",self.parent.GetGrandParent().GetParent()
        print "grand parent of window is",self.parent.GetGrandParent().GetParent()
        window = self.parent.GetGrandParent().FindWindowByName('ListControlMC')
        print "window",window
        selected = event.GetIndex()
        print "first selected",selected
        self.parent.GetGrandParent().GetGrandParent().disable_choose()
        #self.parent.GetGrandParent().GetGrandParent().unchoose_enabled(False);
        window.LoadData(self.GetItemText(selected),self.l,self.t)
    
    def OnSize(self, event):
        size = self.parent.GetSize()
        self.SetColumnWidth(0, size.x-5)
        event.Skip()

    def LoadData(self, t,l):
        self.DeleteAllItems()
        self.t = t
        self.l = l
        print "gparent",self.parent.GetGrandParent().GetGrandParent().GetParent()
        cntr = 0
        for therm in sorted(file_manager.get_therms(l,t),key=lambda x: int(x[5:])):
            self.InsertStringItem(cntr,therm)
            print therm
            try:
                if file_manager.best_therm(self.l,self.t)==therm:
                    MakeBold(self,self.GetItem(cntr))
                    getSiblingByName(self,"ListControlMC").LoadData(therm=therm,l=self.l,t=self.t)
            except Exception:
                pass
            cntr = cntr + 1


class ListCtrlMC(wx.ListCtrl):
    def __init__(self, parent, id):
        wx.ListCtrl.__init__(self, parent, id, style=wx.LC_REPORT | wx.LC_HRULES | 
		wx.LC_NO_HEADER | wx.LC_SINGLE_SEL)

        self.parent = parent

        self.Bind(wx.EVT_SIZE, self.OnSize)
        self.Bind(wx.EVT_LIST_ITEM_SELECTED,self.OnSelect)

        self.InsertColumn(0, '')


    def OnSize(self, event):
        size = self.parent.GetSize()
        self.SetColumnWidth(0, size.x-5)
        event.Skip()

    def LoadData(self,therm,l,t):
        self.DeleteAllItems()
        cntr=0
        for mc in sorted(file_manager.get_mcs(l,t,therm),key=lambda x: int(x[2:])):
            self.InsertStringItem(cntr,mc)
            try:
                if file_manager.best_mc(l,t) == mc:
                    MakeBold(self,self.GetItem(cntr))
            except Exception:
                pass
            cntr = cntr + 1

    def OnSelect(self,event):
        self.parent.GetGrandParent().GetGrandParent().enable_choose()
        #self.parent.GetGrandParent().GetGrandParent().unchoose_enabled(False);



class Reader(wx.Dialog):
    def __init__(self, parent, id, title):
        wx.Dialog.__init__(self, parent, id, title)
        self.parent = parent
        self.SetSize((500,500))
        vbox = wx.BoxSizer(wx.VERTICAL)
        splitter = wx.SplitterWindow(self, -1, style=wx.SP_LIVE_UPDATE|wx.SP_NOBORDER)
        leftSplitter = wx.SplitterWindow(splitter,-1,style=wx.SP_LIVE_UPDATE | wx.SP_NOBORDER,name='leftSplitter')
        rightSplitter = wx.SplitterWindow(splitter,-1,style=wx.SP_LIVE_UPDATE | wx.SP_NOBORDER,name='rightSplitter')

        vboxL = wx.BoxSizer(wx.VERTICAL)
        panelL = wx.Panel(leftSplitter, -1)
        panelLTxt = wx.Panel(panelL, -1, size=(-1, 40))
        panelLTxt.SetBackgroundColour('#53728c')
        stL = wx.StaticText(panelLTxt, -1, 'Lattice size', (5, 5))
        stL.SetForegroundColour('WHITE')

        panelLList = wx.Panel(panelL, -1, style=wx.BORDER_SUNKEN)
        vboxLList = wx.BoxSizer(wx.VERTICAL)
        self.listL = ListCtrlLattice(panelLList, -1)
        self.listL.SetName('ListControlLattice')

        vboxLList.Add(self.listL, 1, wx.EXPAND)
        panelLList.SetSizer(vboxLList)
        panelLList.SetBackgroundColour('WHITE')


        vboxL.Add(panelLTxt, 0, wx.EXPAND)
        vboxL.Add(panelLList, 1, wx.EXPAND)

        panelL.SetSizer(vboxL)

        vboxT = wx.BoxSizer(wx.VERTICAL)
        panelT = wx.Panel(leftSplitter, -1)
        panelTTxt = wx.Panel(panelT, -1, size=(-1, 40), style=wx.NO_BORDER)
        stT = wx.StaticText(panelTTxt, -1, 'Temperature', (5, 5))
        stT.SetForegroundColour('WHITE')

        panelTTxt.SetBackgroundColour('#53728c')

        panelTList = wx.Panel(panelT, -1, style=wx.BORDER_RAISED)
        vboxTList = wx.BoxSizer(wx.VERTICAL)
        self.listT = ListCtrlTempr(panelTList, -1)
        self.listT.SetName('ListControlTemperature')
        vboxTList.Add(self.listT, 1, wx.EXPAND)
        panelTList.SetSizer(vboxTList)


        panelTList.SetBackgroundColour('WHITE')
        vboxT.Add(panelTTxt, 0, wx.EXPAND)
        vboxT.Add(panelTList, 1, wx.EXPAND)

        panelT.SetSizer(vboxT)
        vboxTherm = wx.BoxSizer(wx.VERTICAL)
        panelTherm = wx.Panel(rightSplitter,-1)
        panelThermTxt = wx.Panel(panelTherm,-1,size=(-1,40),style=wx.NO_BORDER)
        panelThermList = wx.Panel(panelTherm, -1, style=wx.BORDER_RAISED)
        vboxThermList = wx.BoxSizer(wx.VERTICAL)
        self.listTherm = ListCtrlTherm(panelThermList, -1)
        vboxThermList.Add(self.listTherm, 1, wx.EXPAND)
        panelThermList.SetSizer(vboxThermList)
        panelThermList.SetBackgroundColour('WHITE')
        
        stTherm = wx.StaticText(panelThermTxt, -1, 'Therms', (5, 5))
        stTherm.SetForegroundColour('WHITE')
        panelThermTxt.SetBackgroundColour('#53728c')
        vboxTherm.Add(panelThermTxt,0,wx.EXPAND)
        vboxTherm.Add(panelThermList,1,wx.EXPAND)
        panelTherm.SetSizer(vboxTherm)
        
        
        
        
        panelMC = wx.Panel(rightSplitter,-1)
        panelMCTxt = wx.Panel(panelMC,-1,size=(-1,40),style=wx.NO_BORDER)
        vboxMC = wx.BoxSizer(wx.VERTICAL)
        stMC = wx.StaticText(panelMCTxt, -1, 'MCs', (5, 5))
        stMC.SetForegroundColour('WHITE')
       
        panelMCTxt.SetBackgroundColour('#53728c')
        panelMCList = wx.Panel(panelMC, -1, style=wx.BORDER_RAISED)
        vboxMCList = wx.BoxSizer(wx.VERTICAL)
        self.listMC = ListCtrlMC(panelMCList, -1)
        self.listMC.SetName('ListControlMC')
        vboxMCList.Add(self.listMC, 1, wx.EXPAND)
        panelMCList.SetSizer(vboxMCList)
        panelMCList.SetBackgroundColour('WHITE')
        vboxMC.Add(panelMCTxt,0,wx.EXPAND)
        vboxMC.Add(panelMCList,1,wx.EXPAND)
        panelMC.SetSizer(vboxMC)
      
        self.Bind(wx.EVT_TOOL, self.ExitApp, id=1)
        rightSplitter.SplitVertically(panelTherm,panelMC)
    
        
        splitter.SplitHorizontally(leftSplitter,rightSplitter)
        self.button_choose = wx.Button(self,-1,"Choose",size=(70,30))
        self.button_unchoose = wx.Button(self,-1,"Remove",size=(100,30))
        self.button_choose.Enable(False)
        self.button_unchoose.Enable(False)
        self.button_done = wx.Button(self,-1,"Done",size=(70,30))
        self.Bind(wx.EVT_BUTTON, self.on_choose_button,self.button_choose)
        self.Bind(wx.EVT_BUTTON, self.on_unchoose_button,self.button_unchoose)
        self.Bind(wx.EVT_BUTTON, self.on_done_button,self.button_done)
        vbox.Add(splitter, 1,  wx.EXPAND | wx.TOP | wx.BOTTOM, 5 )
        hbox = wx.BoxSizer(wx.HORIZONTAL)
        hbox.AddStretchSpacer()
        hbox.Add(self.button_choose,1, wx.BOTTOM,5)
        hbox.Add(self.button_unchoose,1, wx.BOTTOM,5)
        hbox.Add(self.button_done,1,wx.BOTTOM,5)
        vbox.Add(hbox,flag=wx.ALIGN_RIGHT|wx.RIGHT,border=10)
        self.SetSizer(vbox)
        leftSplitter.SplitVertically(panelL,panelT)
      
        self.Centre()
        self.Show(True)

    
        

    def disable_choose(self):
        self.button_choose.Enable(False)
    def enable_choose(self):
        self.button_choose.Enable(True)
    def unchoose_enabled(self,is_enabled):
        self.button_unchoose.Enable(is_enabled)
    def on_unchoose_button(self,event):
        l = self.listL.GetItemText(self.listL.GetFirstSelected())
        t = self.listT.GetItemText(self.listT.GetFirstSelected())
        file_manager.remove_from_bmatdict(l,t)

        UnBold(self.listT, self.listT.GetItem(self.listT.GetFirstSelected()))
        self.listT.OnSelect("dummY")
        
    def on_done_button(self,event):
        self.SetReturnCode(wx.ID_OK)
        self.Close()
        
    def on_choose_button(self,event):
        l =self.listL.GetItemText(self.listL.GetFirstSelected())
        t =self.listT.GetItemText(self.listT.GetFirstSelected())
        therm =self.listTherm.GetItemText(self.listTherm.GetFirstSelected())
        mc =self.listMC.GetItemText(self.listMC.GetFirstSelected())

        file_manager.add_to_mat_dict(l=l,t = t,therm=therm,mc=mc)
        self.GetGrandParent().flash_status_message("Best .mat for %s%s selected" % (l,t))
    def ExitApp(self, event):
        self.Close()

        

if __name__ == '__main__':
    import sys
    try:
        from docopt import docopt
    except ImportError:
        SIM_DIR = None
    else:
        args = docopt(__doc__)
        SIM_DIR = args['SIMDIR']
    print SIM_DIR
    logging.basicConfig(level=logging.DEBUG) 
    app = wx.PySimpleApp()
    
    if not SIM_DIR or not os.path.isdir(SIM_DIR):
        dlg=wx.DirDialog(None,style=wx.DD_DEFAULT_STYLE,message="Where your simulation files are...")
        if dlg.ShowModal() == wx.ID_OK:
            SIM_DIR=dlg.GetPath()
        else:
            sys.exit(0)

        dlg.Destroy()
#    best_mat_dict = load_best_mat_dict()
 #   repaired_dict= defaultdict(dict)

    ########################################
    ############ INIT #####################
    #!!!!OVDE treba uraditi unify po svim direktorijumima
    file_manager = FileManager()
    app.frame = GraphFrame()
    app.frame.Show()
    app.MainLoop()
    

