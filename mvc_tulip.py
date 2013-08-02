import wx
import mvc
import re
import logging
import glob
import os
from os.path import join
import util
import unify
from collections import defaultdict
import pickle
import pandas as pd
import numpy as np
from profile import *

class ChoicesConverter(mvc.View):
    """Ova klasa sprema za prikaz
    za gui izbore inace i u matchooseru"""

    def __init__(self,model,controller):
        mvc.View.__init__(self,model,controller)
        self.log = logging.getLogger("ChoicesConverter")

    def model_updated(self,subject=None):
        """Nisam sigurna da mi ovo treba
        sto samo kontroler, zasto view prica
        direktno sa guijem a ne preko modela
        hmmm???? Sada mi dobijemo """
        pass
        # self.update_gui()

    def choices(self,l=None,t=None,mc=None):
        reverse = True if not t else False
        return sorted(self.model.choices(l=l,t=t,mc=mc), key= lambda x:util.extract_int(x) ,reverse=reverse)

    @counter
    @benchmark
    @logg
    def matchooseritems(self,l=None,t=None,mc=None):
        items = list()
        # ako je prosledjen mc onda zelimo therm
        which = 'therm' if mc else 'mc'
        bestmat_ch = self.model.bestmat_choices(l,t,which)
        self.log.debug('bestmatch %s' %bestmat_ch)
        for x in self.choices(l,t,mc):
            fdict = {'pointSize':10,'family':wx.FONTFAMILY_DEFAULT,
                     'style':wx.FONTSTYLE_NORMAL,'weight':wx.FONTWEIGHT_NORMAL}
            self.log.debug('uredjujem izgleda za :%s' %x)
            item = wx.ListItem()
            item.SetText(x)

            if x in bestmat_ch:
                self.log.debug('boldovao')
                fdict['weight']=wx.FONTWEIGHT_BOLD
            
            font = wx.Font(**fdict)
            # if x in self.model.altmat_choices():
            #     font.SetWeight(wx.FONTWEIGHT_BOLD)
            item.SetFont(font)
            items.append(item)
        return items

    def l_choices_agg(self):
        return sorted(self.model.bestmat_ls(), key = lambda x:util.extract_int(x))

    def mag_choices_agg(self):
        """Ovde cu morati da gledam u hm, pa to jeste fazon, sto
        da, cu morati da imam ovo, sto znaci da je pre toga uradjen
        bar jedan agregate. treba da je uradjen bar jedan agregate.
        posto nama se desava sledece:. a mozda da uradimo da u settings
        panelu mogu da se menjaju formule i tacno sve formule, samo onda bi trebalo
        da zapamtim neke formule, pa mogla bih sada da izvucim iz agregate-a formuel
        i da ih drzim zapamcene, tj, tu ce nam biti i onda kad radim agregate idem
        iz tih formula. Hoce formule biti u modelu ili kako??? Pa da, nema tu sta,
        ok, ajd lepo radi ovaj view. Pa radim sve lepo, samo eto preskacem stvari
        koje mislim da ne treba sada. tj, da, prioritizujem, sta hoces"""
        return self.model.mag_choices()
    def l_choices(self):
        """Vraca sve moguce L-ove u sim_dir
        valjda ovaj zna za model pa moze tako
        da mu gettuje stvari. tj, zna kako izgleda
        e pa nisam bas sigurna JBGt
        Jaoj moram pogledati od ovoga sve pre nego
        sto radim ovo. Nema smisla jednostavno. 
        """
        ls = sorted(self.model.l_choices(),key=lambda x: util.extract_int(x))
        
        return ls

    def t_choices(self,l):
        """Vraca sortirane sve moguce t-ove za prosledjeno L"""
        ts =  sorted(self.model.t_choices(l), key=lambda x:util.extract_int(x))
        return ts

    def mc_choices(self,l,t):
        """Vraca sve raspolozive mc-ove, formatirane za prikaz. Oni ujedno i odredjuju moguce
        plotove za therm panel. Ovo podize pitanje da li je bolje da
        se gleda za koje mc-ove postoje koji thermovi, i obrnuto"""
        mc_choices = self.model.mc_choices(l,t)
        
        sorted_mcs =  sorted(mc_choices,key=lambda x: util.extract_int(x))
        return ["{mc} [{tmc}]".format(mc=mc,tmc=self.model.therm_count(l,t,mc)) for mc in sorted_mcs ]

    def therm_choices(self,l,t,mc):
        """Vraca sve thermove za odredjeno l,t i mc,
        sortirane u obrnutom redosledu
        """
        therms = self.model.therm_choices(l,t,mc)
        self.log.debug("Vracam  pogled thermova")
        return sorted(therms,key=lambda x: util.extract_int(x),reverse=True)

    def annotate_agg(self,l,t):
        ls =self.model.bestmat_choices(l=l,t=t,which='therm')[0]
        sp =self.model.bestmat_choices(l=l,t=t,which='mc')[0]
        return 'LS={}\nSP={}'.format(extract_int(ls),extract_int(sp))

class Choices(mvc.Model):
    
    all_regex = re.compile(r'^L\d+T\d+(?P<therm>THERM\d+)\.all$')
    lt_regex = re.compile(r'(L\d+)(T\d+)$')
    statmc_regex = re.compile(r'^MC\d+$')
    def __init__(self):
        """Ne znam da li nam treba
        nesto posebno ovde. Mislim
        da se sve radi u runtime"""
        mvc.Model.__init__(self)
        self.log = logging.getLogger("Choices")
        self.bestmats = None
        self.altmats = None
        self.mags = None
        
    def init_model(self):
        """Ovo se poziva tek kada se pokrene aplikacija
        posto necemo imati progress bar niti ista"""
        
        self.unify()
        try:
            self.files = self._map_filesystem()
        except:
            util.show_error("IO error","Error occured while reading simfolder")
        self.load_state()
        # za sada cu se zadovoljiti ovime. ali !! !!! ! ! !!!!
        # ok, tu ce biti hardkodovane formule. ali ideja mi je da
        # mogu da se ukucaju formule. to bi bilo tako sto ce se zadati
        # pisati velicine i aritmeticki operator, najnormalnije ce sse pisat
        self.mags = [
        'M1avg',
        'M2avg',
        'M4avg',
        'susc',
        'Eavg',
        'E2avg',
        'Tcap',
        'U'        
        ]

    def mag_choices(self):
        self.log.debug("Vracam mag choices : %s" %self.mags)
        return self.mags


    def choices(self):
        ch = dict()
        ch['l'] = self.l_choices()
        ch['t'] = self.t_choices()
        ch['mc'] = self.mc_choices()
        ch['therm'] = self.therm_choices()        


    def set_simdir(self,simdir):
        """Ovo zove kontroler pre nego
        sto pozove init_model metodu. Mislim
        ovo nije bas najsrecnije. Treba neki
        communication diagram da napravim, ali
        nije strasno posto se samo na pocetku
        poziva
        """
        self.simdir = simdir

    def files_updated(self):
        """
        Ako se pritisne neko reload dugme
        ce se ovo ja mislim zvati, tj kontroler ce ga zvati
        """
        self.files = self._map_filesystem()
        self.notify_observers()
        
    def add_bestmat(self,l,t,mc,therm):
        self.log.debug('Adding bestmat!')
        self.bestmats[l][t] = {'mc':mc,'therm':therm}
        self.notify_observers()

    def add_alt(self,l,t,mc,therm,booli):
        self.alts[l][t] = {'mc':mc,'therm':therm,'booli':booli}

    def add_curr(self,l,t,mc,therm,booli=False):
        """Moram videti kako ce ovo funkcionisati,
        posto moram uzimati i iz bestmata i sve"""
        raise NotImplementedError
        
    def add_mc(self,l,t,mc):
        """Prima za koje l t da generise novo
        mc, i stavlja ga u sturkturu podataka
        """
        therms = self.therm_choices(l,t, self.get_static_mcs(l,t)[0])
        tdict = {key:None for key in therms}
        self.log.debug("adding mc %s" % tdict )
        self.files[l][t]["MC%s" %mc]=tdict
        self.notify_observers()

    def get_maxmc(self,l,t):
        """Vraca makisamalan BROJ
        mc-ova"""
        self.log.debug("Vracam max mc")
        return util.extract_int(max(self.mc_choices(l,t),key =lambda x: util.extract_int(x)))

    def get_mags(self):
        """Taj mags ce sadrzati i formule
        a i mozda ove stringove koji korespondiraju.
        Videcemo. Samo Nek ti proradi ovo i onda malo
        doteruj"""
        return self.mags

    def choices(self,l=None,t=None,mc=None):
        """Nemoj da prosledjujes van redosleda argumenta
        tj, nemoj da si prosledila therm ako nemas
        mc!"""
        ls = [l,t,mc]
        x = self.files
        for val in ls:
            if not val:
                break
            x = x[val]
        self.log.debug("Vracam izbore :{} za l:{} t:{} mc:{} ".format(x,l,t,mc))
        return x.keys()

    def bestmat_choices(self,l=None,t=None,which='mc'):
        """
        Ocekuje dobar format. ne znam kako to da omogucim
        uvek. sta bi ovaj rekao. oni treba da komuniciraju
        i dogovore format. hmh. mi kazemo sta hocemo a oni
        izkomuniciraju
        """
        ls = [l,t,which]
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
        
        return [x] if isinstance(x,basestring) else x
        
        
    def currmat_choices(self):
        #!!!TODO
        return []
        
    def bestmat_ls(self):
        """Vraca sve L-ove iz bestmata"""
        self.log.debug("returning l's from bestmat:{}".format(self.bestmats.keys()))
        return self.bestmats.keys()
        
    def bestmat_ts(self,l):
        """Vraca sve t-ove iz bestmata za odredjeno l"""
        self.log.debug("Vracam tove iz bestmata za l %s " %l)
        return self.bestmats[l].keys()
        
    def bestmat_mcs(self,l,t):
        return self.bestmats[l][t].keys()

    def bestmat_therms(self,l,t,mc):
        return self.bestmats[l][t][mc]

    def altmat_ls(self):
        """Vraca sve L-ove iz altmata"""
        self.log.debug("returning l's from altmat")
        return self.altmats.keys()
        
    def altmat_ts(self,l):
        """Vraca sve t-ove iz altmata za odredjeno l"""
        self.log.debug("Vracam tove iz altmata za l %s " %l)
        return self.altmats[l].keys()
        
    def altmat_mcs(self,l,t):
        return self.altmats[l][t].keys()

    def altmat_therms(self,l,t,mc):
        return self.altmats[l][t][mc].keys()
        
    def get_static_mcs(self,l,t):
        """
        Vraca samo one mc-ove koji se nalaze
        na disku. ili imaju isti format kao ti na disku
        jbg
        """
        statmcs = [ mc for mc in self.mc_choices(l,t) if self.statmc_regex.match(mc)]
        self.log.debug("Vracam staticke mc-ove:{} za l:{} t:{}".format(statmcs,l,t))
        return statmcs
        
    def load_state(self):
        """Postavlja tulip u stanje u kom
        ga je korisnik ostavio. Sto se tice
        izbora samo, doduse"""
        self.bestmats = self.load_choices("bestmat.dict")
        self.altmats = self.load_choices("altmat.dict")
        

    def load_choices(self,fname):
        self.log.debug("Loading {}".format(fname))
        with open(join(self.simdir,fname),mode="ab+") as hashf:
            try:
                fcontent =  defaultdict(dict,pickle.load(hashf))
            except EOFError:
                fcontent = defaultdict(dict)
        return fcontent
        
    def unify(self):
        dirlist = [d for d in glob.glob(join(self.simdir,"L[0-9]*[0-9]*")) if os.path.isdir(d)]
        dirlist.sort()
        for dir_ in dirlist:
            try:
                unify.main(dir_)
            except NotImplementedError :
                util.show_error("Duplicate Seeds","Duplicated seeds found while unifying! What to do, what to do????!?!?!")

    def therm_count(self,l,t,mc):
        """Vraca koliko ima tacaka za dato mc, tj.trebace kod
        ovog therm panela da vidi neko da ne plotuje nebulozne
        stvari. znaci ovo ce biti u nekim viticastim zagradama
        iza broja mc-ova"""
        therms = self.therm_choices(l,t,mc)
        self.log.debug("Vracam broj thermova:{} za mc:{}".format(len(therms),mc))
        return len(therms)

    def create_mat(self,l,t,therm,mc=None,booli=False):
        """Cita all fajlove i obradjuje ih statisticki. U slucaju
        da je prosledjen mc to znaci da smo prethodno dodali u onaj
        choices dictionary odgovarajuc mc, i da sad radimo za njega
        ako ne prosledjujemo, sve regularno citamo sve, a ne samo
        mc prvih redova. Ako je prosledjen booli to znaci da su izbaceni
        ODREDJENI rezultati, pa koristimo taj argument kao indeks koji
        ce izdvojiti samo zeljenje rezultate
        """

        mc = util.extract_int(mc) if mc else None
        filename = join(self.simdir,"{l}{t}".format(l=l,t=t),"{l}{t}{therm}.all".format(l=l,t=t,therm=therm))
        
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
        """Modifikuje odgovarajuce elemente u self.files [sa izracunatim matovima],
        ako je potrebno, i vraca strukturu pogodnu za plotovanje"""
        data = dict()
        self.log.debug("Composing for l:{} t:{} mc:{}".format(l,t,mc))
        self.log.debug("Idem kroz {}".format(self.files[l][t][mc].items()))
        for therm,mat in self.files[l][t][mc].items():
            # ne znam da li ce biti problem sto menjam ovaj dictionary
            # dok iteriram kroz njega???
            self.log.debug("Trenutni mat je %s" %mat)
            therm_int = util.extract_int(therm)
            try:
                self.files[l][t][mc][therm] = mat if mat.any() else "weird"
            except:
                self.files[l][t][mc][therm] = self.create_mat(l,t,therm_int,mc)
            else:
                self.log.debug("Vec je izracunat mat")
            data[therm_int] = self.files[l][t][mc][therm]
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

    def get_plot_dict(self,mat=False,alt=False):
        """Vraca dictionary koji
        sadrzi podatke o tome sta
        ce se agregirate, tj. sta ce
        se plotovati. Moze samo matove,
        moze samo altove, a moze i trenutno
        izabrane"""
        from itertools import chain
        from copy import deepcopy
        
        if mat:
            return self.bestmats
        if alt:
            raise NotImplementedError
        return self.bestmats
        
    @logg
    def agregate(self,alt=False,mat=False):
        plot_data = dict()
        for_plotting = self.get_plot_dict()
        for l,tdict in for_plotting.items():
            agg_mat = list()
            for t,val in tdict.items():
                mc,therm =val['mc'],val['therm']
                try:
                    booli = val['booli']
                except:
                    booli = False

                self.log.debug('checking whether calculated : %s' %self.files[l][t][mc][therm])
                if self.files[l][t][mc][therm] is None:
                    self.files[l][t][mc][therm]=self.create_mat(l,t,therm)
                data_mat = self.files[l][t][mc][therm]
                self.log.debug(type(data_mat))
                data_mat.rename({'THERM': 'T'},inplace=True)
                self.log.debug('TTTTT :%s',t)
                data_mat.ix['T']=int(t[1:])
                agg_mat.append((data_mat,t))
            #uvek cemo imati samo jednu kolonu, naravno
            print agg_mat
            agg_mat = sorted(agg_mat, key=lambda x: util.extract_int(x[1]))
            dat,keys = zip(*agg_mat)
            print 'dat,keys',dat,keys
            agg_mat = pd.concat(dat, axis=1,keys=keys)
            plot_data[l] = self.agg(agg_mat)

        self.log.debug('plot data \n:%s' %plot_data)
        return plot_data
        
    @logg
    def agg(self,agg_mat):
        T = agg_mat.ix['T'] / 10000.0
        print T
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


    def _map_filesystem(self):
        """Ide kroz sve direktorijume ( za sada jedan sim dir)
        u simdir-u i izvlaci l,t,i therm.
        """
        choices = defaultdict(dict)
        mc_to_all = defaultdict(list)
        for root,dirs,files in os.walk(self.simdir):
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

    def l_choices(self):
        """
        Vraca sve moguce L-ove
        u fajl sistemu. tj, ako postoji
        taj direktorijum - bice u vracenoj
        listi
        """
        return self.files.keys()
        
    def t_choices(self,l):
        """
        Vraca sve moguce t-ove za prosledjeno
        l, tj ako postoji ime foldera odgovarajuce
        to t ce biti u vracenoj listi
        """
        t_choices = self.files[l].keys()
        self.log.debug("Vracam t izbore:{} za l:{} ".format(t_choices,l))
        return t_choices

    def mc_choices(self,l,t):
        """
        Vraca sve mc-ove za odredjeno l i t,
        ako postoji .all fajl sa tim brojem linija
        taj mc ce biti u vracenoj listi
        """
        mc_choices = self.files[l][t].keys()
        self.log.debug("Vracam mc izbore : {} za l:{} i t:{}".format(mc_choices,l,t))
        return mc_choices

    def therm_choices(self,l,t,mc):
        """
        Vraca sve moguce thermove
        za prosledjeno l t i mc,
        ako postoji u fajl sistemu
         bice vracen u listi
        """
        therms = self.files[l][t][mc].keys()
        self.log.debug("Vracam thermove : {} za l:{} t:{} mc:{}".format(therms,l,t,mc))
        return therms


class FileManager(mvc.Controller):
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
        self.model.set_simdir(self.simdir)
        self.model.init_model()
        self.update_gui()
       
       # self.ap.reader.populateListControls(**{'l':lch})


    def update_gui(self):
        """Ako je promenjen model
        updejtuju se svi elementi gui-ja
        """
        lchoices = self.view.l_choices()
        self.tp.set_l_choices(lchoices)
        #ovaj ce gledati iz bestmatdicta kako sta
        self.ap.set_l_choices(self.view.l_choices_agg())
        # samo ce morati da vidi ako ima, ako nema jbg
        # bice prazno
        self.ap.set_mag_choices(self.view.mag_choices_agg())
        self.refresh_matchooser()
        #^^ u gui-ju ce se pozvati
        #odredjeni event handleri koji
        # ce ostatak updejtovati
        # self.ap.set_l_choices(self.view.best_ls())
        # self.sp.set_l_choices(self.view.best_ls())
        # Malo ovo demetrin zakon rusi!!!

    def refresh_matchooser(self):
        ch = self.current_choices()
        self.l_gen(**ch)
        self.t_gen(**ch)
        self.mc_gen(**ch)
       # self.therm_gen(self.match_stuff['therm']['choice'])
        lch = self.view.matchooseritems()
        self.match_stuff['l']['result']=lch
        self.populate_matchooser()
        
    ####  GUI INTERFACE ####
    ### Ove metode poziva gui ###
    
    
    ######### THERM PANEL #############
                       
    def add_mc(self,l,t,mc):
        """Dodaje mc u choices, thermove ce morati da
        preuzme od prvog suseda koji nije 'specijalan'
        Hm, verovatno je pogresno sto iz modela prikazujem
        errore. verovatno bi trebalo samo da raisujem errore
        i to a ovde da ih hvatam. hmm
         """
        try:
            self.model.add_mc(l,t,mc)
        except IndexError:
            util.show_error("Simdir error","An empty folder in your simulation directory??")
            
    def on_l_select_tp(self,l):
        """Kada korisnik selektuje l u thermpanelu,
        namestaju se odredjeni t"""
        tch = self.view.t_choices(l)
        self.tp.set_t_choices(tch)
        
    def on_t_select(self,l,t):
        """Kada korisnik u thermpanelu izabere
        t ova metoda se pozove i popune se odredjeni
        mc izbori
        """
        mch = self.view.mc_choices(l,t)
        self.tp.set_mc_choices(mch)
        self.tp.reset_chkboxes()
        uplimit = self.model.get_maxmc(l,t)
        self.tp.set_mc_range(uplimit)

        
    def get_plot_data(self,l,t,mc):
        """Vraca datafrejm za plotovanje"""
        return self.model.compose(l,t,mc)
    
        

    ########################################
    ########### AGGREGATE PANEL ############

    def get_agg_plot_data(self):
        return self.model.agregate()

    def annotate_agg(self, l,t):
        """
        Vraca anotaciju za odredjeno l i t
        """
        return self.view.annotate_agg(l,t)


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

    def populate_matchooser(self,choice = None):
        self.log.debug('populating matchooser, choice made: %s' %choice)
        lala = ['l','t','mc','therm']
        cmpr = lala.index(util.extract_name(choice)) if choice else -1
        pop = {key:self.match_stuff[key]['result'] for key in self.match_stuff if cmpr < lala.index(key)}
        self.log.debug('population : {}'.format(pop))
        self.ap.reader.populateListControls(**pop)

    def add_bestmat(self):
        bestm = self.current_choices()
        self.model.add_bestmat(**bestm)

    def current_choices(self):
        """Vraca trenutno selektovan izbor
        u bestmat chooseru """
        return {key:self.match_stuff[key]['choice'] for key in self.match_stuff}
        
    def l_gen(self,**kwargs):
        """L izbor prima za parametar
        stavlja ga medju atribut choices,
        generise sve izbore za taj i stavlja
        i zove sve ostale funkcije za atribute
        zato sto jbg, ne znam kako drugcije
        """
        lch = kwargs['l']
        self.match_stuff['l']['choice'] = lch
        self.match_stuff['t']['result'] = self.view.matchooseritems(l=lch) if lch else []
        self.t_gen(**kwargs)

    def t_gen(self,**kwargs):
        tch =kwargs['t'] if 't' in kwargs.keys() else None
        lch = self.match_stuff['l']['choice']
        self.log.debug('generating mc results for choice %s' %tch)
        self.match_stuff['t']['choice'] = tch
        self.match_stuff['mc']['result'] = self.view.matchooseritems(l=lch,t=tch) if tch else []
        self.mc_gen(**kwargs)

    def mc_gen(self,**kwargs):
        mch = kwargs['mc'] if 'mc' in kwargs.keys() else None
        lch = self.match_stuff['l']['choice']
        tch = self.match_stuff['t']['choice']
        self.log.debug('generating therm results for choice %s' %mch)
        self.match_stuff['mc']['choice'] = mch
        self.match_stuff['therm']['result'] = self.view.matchooseritems(l = lch,t=tch,mc = mch) if mch else []
                                    
    def therm_gen(self,**kwargs):
        self.match_stuff['therm']['choice'] = kwargs['therm'] 
        self.add_bestmat()

    

   

    # def on_select_listctr(self,selected):
        
        
        
                       
    # def get_alt(self):
    #     """Pravi dictionary koji uzima
    #     sve vrednosti koje postoje u alt dictu
    #     i prelepljuje ih preko vrednosti iz bmatdicta
    #     i to vraca"""
    #     import copy
    #     altm = copy.deepcopy(self.bmatdict)
    #     print "REPAIRED DICT:\n", self.repdict
    #     for l,val in self.repdict.items():
    #         for t,path in val.items():
    #             altm[l][t] = path
    #     return altm
        
    # def get_all_file(self,l,t):
    #     """Vraca all fajl za zeljeni bmat, za prosledjeno l i t"""
    #     base = self.get_base(l,t)
    #     self.log.debug("get_all_file:vracam %s" % ("%s.all" % base))
    #     return "%s.all" % base

    # def get_base(self,l,t):
    #     """Vraca osnovu imena u obliku L[0-9]*T[0-9]*THERM[0-9]*
    #     iz bmatdicta za prosledjeno l i t
    #     """
    #     extr_regex.search(self.bestmatd[l][t]).groupdict()["base"]

                       
    # def get_maxmc(self,l,t):
    #     """
    #     Vraca max mc za odredjeno l i t
    #     ne znam da li nam to treb
    #     """
    #     return max(self.mc_choices(l,t))

    # def add_bestmat(self,l,t,therm,mc):
    #     self.bmatdict.bestmat(l,t,therm,mc)
        
            


    
  
    # def get_changed_mat_name(self,l,t,new_mc):
    #     """Ovo je zarad izbacivanja 'nepozeljnih' rezultata
    #     prosledjuju mu se l i t, i novi mc. Vraca novo ime"""
    #     return "{base}[MC{mc}].mat".format(base=self.get_base(l,t), mc=new_mc)
                                         
    # def _load_bmatdict(self):
    #     """Ucitava bmatdict iz memorije"""
    #     with open(join(self.simdir,"mat.dict"),mode="ab+") as hashf:
    #        try:
    #            fcontent =  defaultdict(dict,pickle.load(hashf))
    #        except EOFError:
    #            fcontent = defaultdict(dict)
    #     print "bmatdict",fcontent

    # def clean_mat_dict(self):
    #     """Gleda koji unosi u dict nemaju vrednost
    #     tj. samo su dodati zbog defaultdict svojstva
    #     i skida ih. Ovo ce se samo desiti u mat chooseru
    #     tako da ovo tada samo treba da zovem.Nisam sigurna
    #     da ce mi biti potreban"""
    #     # ovo ce proci posto mi ne iteriramo kroz sam dict
    #     # a skidamo sa samog dicta. ova lista items nece biti
    #     # vise up to date, ali to nam nije bitno, posto nije
    #     # ciklicna
    #     self.log.debug("cleaning bmatdict : %s" % self.bmatdict)
    #     for key,value in self.bmatdict.items():
    #         if not value:
    #             self.bmatdict.pop(key)
    #             self.log.debug("removing {}.aplot".format(key))
    #             try:
    #                 os.remove(join(self.simdir,'{}.aplot'.format(key)))
    #             except Exception:
    #                 self.log.warning("Exception!!!")
    #     self.serialize_mat()


    

        
#     def add_to_mat_dict(self,l,t,therm,mc):
#         """Dodaje u dictionary 'reprezentativnih' matova, ispisuje poruku
#         u status baru, i cuva novo stanje best_mat_dict-a na disk"""
#         best_mat = self.get_files(l=l,t=t,ext="*%s%s*.mat" %(therm,mc))
#         # ne bi smelo da ima fajlova u okviru jednog foldera sa istim MC i THERM
#         assert len(best_mat)==1
#         self.bmatdict[l][t] = best_mat[0]
#         self.log.debug("added %s to bmatdict : %s " %(best_mat[0], self.bmatdict))
#     #    self.parent.flash_status_message("Best .mat for %s%s selected" % (l,t))
#         #mhm, nisam sigurna nisam sigurna nista. al ajd. nekako cu popraviti sve
#         self.clean_mat_dict()
#         #onda mi ne treba ovde serialize
# #        self.serialize_mat()

#     def remove_from_bmatdict(self,l,t):
#         """Brise zapis za prosledjeno l i t iz bestmatdicta"""
#         self.log.debug("Brisem zapis za l:{} t:{} i bmatdicta".format(l,t))
#         del self.bmatdict[l][t]
#         self.serialize_mat()
#     def serialize_mat(self):
#         with open(join(self.simdir,"mat.dict") ,"wb") as matdictfile:
#             pickle.dump(dict(self.bmatdict),matdictfile)
#     #matDictEmpty
#     def bmatdict_empty(self):
#         """ne znam da li mi je ova f-ja relevantna
#         sad kad imam ovaj clean mm"""
#         return not self.bmatdict.keys()
#         # for key in self.best_mat_dict.keys():
#         #     if self.best_mat_dict[key]:
#         #         return False