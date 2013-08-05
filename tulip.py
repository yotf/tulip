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
import logging
from collections import defaultdict
import itertools
import matplotlib
from matplotlib import widgets   
   
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
import mvc_tulip
import util


#!!!!Izbaci utility metode u jedan modul   

#mislim da je ovo robusnije od onog prethodnog
#tako treba valjda i sa importima, nem pojma
# to jos vidi
import tulip
PACKAGE_ABS_PATH = os.path.abspath(os.path.dirname(tulip.__file__))
#!!!Ove napravi kao dictionaryje. i nesto da mogu da se kombinuju
#nem pojma. da moze da se menja, znaci neka struktura koja ce se u zavisnosti
#od parametra ce advancovati jedan cycle++, od bilo koji od ovih
#znaci bice neki parameter dict i bice po jedan fmt cycle
#za svaki od njih
fmt_strings = ['g+-','r*-','bo-','y+-']
fmt_cycle = itertools.cycle(fmt_strings)


    
        
class UserChoices():
    """
    Ova klasa ce voditi racuna
    o svemu sto korisnik izabere,
    imace tri dicta koji ce predstavljati
    alt, curr i mat, i jos jedan dict
    koji ce se prosledjivati agregate-u.
    On ce sadrzati izracunate zeljene l-t kombinacije.
    Znaci agregateu trebaju u memoriji ti rezultati racunanja
    Mogu prvobitno da updejtujem taj dict sa matom, a onda
    po potrebi ga updejtujem sa ovima. Svaki od ova tri ce imati
    potrebne informacije, ovo ce biti samo plitka kopija, referenca
    a mozda i nece, ali moze ako hoce
    """
    def __init__(self):
        self.log = self.logging.getLogger("UserChoices")
        
    def currmat(self,l,t,mc,therm):
        """Dodaje u izbore za curr,
        ovi izbori ce se koristiti pri
        pokretanju slajdera
        """
        self.log.debug("Neko pomera slajder, l:{} t:{} mc:{} therm:{}".format(l,t,mc,therm))
        self.currmat[l][t][mc]=therm

    def bestmat(self,l,t,mc,therm):
        """
        Dodaje izbor bestmata. Oni se koriste
        pri pritisku na dugme draw
        """
        self.log.debug("Dodajem u bestmat, l:{} t:{} mc:{} therm:{}".format(l,t,mc,therm))
        self.bestmat[l][t][mc]=therm

    def altmat(self,l,t,mc,therm):
        """
        Dodaje u izbore za altmat,
        pre ovoga se dodao mc sa zvezdicom
        u glavne choices. mc sa zvezdicom
        se i prosledjuje ovde
        Hm,samo ovo nam sada ne treba
        posto ce korisnik stavljati u bestmat
        te altove. Hmh, ne znam koliko mi je to
        pametno. Ne znam sada sta ce se desiti
        ako izabere ove alternativne i koje izabere
        pa dobro pri izabiru treba proveriti goddddamamamfkdafljslk
        taman sam mislila da je sve ok.
        Mozda da imam informaciju o tome od kog mc-a je napravljen
        kao u viticastim. i onda ce onaj search valjda doci do prvog
        rezultata. I kad hoce da vrati, nek samo onda izabere to sto je
        u zagradi jbg, sta da mu radim. Samo ipak, sta onda cuvati u pm
        pa da cuvamo to jbg, da cuvamo to, pa ne mozemo da cuvamo to posto
        bmatdict nije komplikovan. Znaci sta samo treba Pa jedino da proveravamo
        kad se izabere da li je sa zvezdicom ili nije i onda da to cuvamo u odredjen
        ali kako onda da jbm li ga. Mozda da imam taj curr_dict. Mozda da se tu izabere
        i onda da ostaje u currdictu dok neko ne klikne na save as best mat. Znaci
        bira se u bestmatchooseru kako se sta stigne i bolduje se sve to i stagod
        i tu moze biti dugme, save as bestmat, i onda ako izabere covek drugcije
        znaci neke alternativne i ove sa mc* to ce se crtati, ali nece biti
        u bestmat, i onda ako se gleda. Znaci u trenutku kada korisnik klikne
        choose as bestmat, sve iz curr_dicta
        """
        self.log.debug("Neko pomera slajder, l:{} t:{} mc:{} therm:{}".format(l,t,mc,therm))
        self.altmat[l][t][mc]=therm 

    def best_therm(self,l,t):
        """Ovaj ce nam trebati tokom anotacija
        mozda treba da hmm, mozda treba iz onog
        trenutnog da izvlacim za anotacije.
        Hmh, mozda da nemam ovaj curr dict
        kao poseban nego da curr bude ono
        sto ce se trenutno crtati. sto znaci
        da ce se menjati ako vucemo slajdere
        a ako smo izabrali bestmatove ce biti
        oni, a ako smo izabrali alt matove onda
        ce biti oni kompletno. tj bice bestmat + oni
        hurr hurr. samo sto onda se ne mozemo  vracati na
        ove kako nam je volja. """
        return self.bestmat[l][t].therm
        

    def curr_therm(self,l,t):
        """Vraca onaj therm koji je trenutno
        iscrtan"""
        return self.currmat[l][t].therm

    def best_mc(self,l,t):
        """Vraca mc iz bestmat-a za
        prosledjeno l i t"""
        return self.bestmat[l][t].mc
    def curr_mc(self,l,t):
        """Vraca trenutno iscrtani mc
        za odredjeno l i t"""
        return self.currmat[l][t].mc
        
        
class ScatterPanel(wx.Panel):
    xlabel = 'Mx'    
    ylabel = 'My'
    zlabel = 'Mz'
    all_data = dict()
    ylim = None
    xlim = None
    chk_mc_txt = "Show first %d SPs"
    firstn = 1000
    def __init__(self,parent,controller):
        self.controller = controller
        
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
        newmatfname =join(self.simdir,file_manager.get_changed_mat_name(curr_l,curr_t,mc))
        
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
        flist=glob.glob(join(self.simdir,"{}T*".format(l)))
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
    
    def __init__(self, parent,controller):
        wx.Panel.__init__(self, parent=parent, id=wx.ID_ANY)
        self.controller = controller
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
        self.cmb_mags = wx.ComboBox(self, size=(100, -1),style=wx.CB_READONLY,
                                    choices = plot_choices, value= plot_choices[0])
        self.cmb_L = wx.ComboBox(self, size=(70, -1),
                                 style=wx.CB_READONLY,
                                 value = '--'
                                 )
        self.cmb_T = wx.ComboBox(self, size=(100, -1),style=wx.CB_READONLY, value='--')

        self.cmb_mcs = wx.ComboBox(self, size=(150, -1),
                choices=[], value='--')
        self.Bind(wx.EVT_COMBOBOX, self.on_l_select, self.cmb_L)
        self.Bind(wx.EVT_COMBOBOX, self.on_t_select, self.cmb_T)
        self.Bind(wx.EVT_COMBOBOX, self.on_select_mcs,
                  self.cmb_mcs)
        self.hbox1 = wx.BoxSizer(wx.HORIZONTAL)
        
        self.hbox1.AddSpacer(20)
   
        self.hbox1.Add(self.cmb_L, border=5, flag=wx.ALL
                       | wx.ALIGN_CENTER_VERTICAL)
        self.hbox1.Add(self.cmb_T, border=5, flag=wx.ALL
                       | wx.ALIGN_CENTER_VERTICAL)
        self.hbox1.Add(self.cmb_mcs, border=5, flag=wx.ALL
                       | wx.ALIGN_CENTER_VERTICAL)
        self.hbox1.Add(self.cmb_mags, border=5, flag=wx.ALL
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
        self.canvas.draw()
   
        
    def on_add_button(self,event):
        """Pravi rezultat manje preciznosti
        tj, za prvih n mc-ova"""
        firstnmcs =int( self.mc_txt.GetValue())
        l = self.cmb_L.GetValue()
        t = self.cmb_T.GetValue()
        self.controller.add_mc(l,t,firstnmcs)


    ### CONTROLER INTERFACE ####

    def set_l_choices(self,lch):
        try:
            value = lch[0] if self.cmb_L.GetValue()=='--' else self.cmb_L.GetValue()
        except IndexError:
            util.show_error("Simdir error", "No l choices! Wrong simulation directory?")
        else:
            self.cmb_L.SetItems(lch)
            self.cmb_L.SetValue(value)
            self.on_l_select()

    def on_l_select(self, *args):
        l = self.cmb_L.GetValue()
        self.controller.on_l_select_tp(l)
    
    def set_t_choices(self,ts):
        self.cmb_T.SetItems(ts)
        try:
            value = ts[0]
        except IndexError:
            util.show_error("Simdir error","No t choices! Something wrong with simdir?")
        else:
            self.cmb_T.SetValue(value)
            self.on_t_select()

    def on_t_select(self):
        l = self.cmb_L.GetValue()
        t = self.cmb_T.GetValue()
        self.controller.on_t_select(l,t)

    def set_mc_range(self,range_):
        self.mc_txt.SetRange(0,range_)
        
    def set_mc_choices(self,mch):
        self.cmb_mcs.SetItems(mch)
        try:
            value = mch[0]
        except IndexError:
            util.show_error("Simdir error","No mc choices! Something wrong with simdir?")
        else:
            self.cmb_mcs.SetValue(value)

    def get_mc(self):
        """Vraca selektovan mc"""
        return re.search(r'MC(\d+)', self.cmb_mcs.GetValue()).group(0)

    def on_select_mcs(self, event):
        """Nista se ne desava """
        self.log.debug("On select mcs")
        
    def init_plot(self):
        self.dpi = 100
        fig_width = (self.parent.GetParent().width / self.dpi)
        fig_height = self.parent.GetParent().height / self.dpi * 3 / 4

        self.fig = Figure((fig_width, fig_height), dpi=self.dpi,facecolor='w')


        self.ax_mag = self.fig.add_subplot(1, 2, 1)
        self.ax_cv = self.fig.add_subplot(1, 2, 2)

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
        try:
            artist.set_color(np.random.random(3))
        except:
            pass
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
        
    def on_draw_button(self,event):
        mc = self.get_mc()
        l = self.cmb_L.GetValue()
        t  = self.cmb_T.GetValue()
        self.log.debug("on_draw_button")
        self.data = self.controller.get_plot_data(l,t,mc)
        self.log.debug("Loaded data for plot:\n %s" % self.data)
        self.draw_plot()
        
    def draw_legend(self,event):
        lbl_mc = self.get_mc()
        lbl_mc ="%s=%s" %("SP",lbl_mc[2:])
        lbl_t ="%s=%.4f" %(self.cmb_T.GetValue()[0],float(self.cmb_T.GetValue()[1:])/100)
        lbl_l ="%s=%s"% (self.cmb_L.GetValue()[0],self.cmb_L.GetValue()[1:])
        lchk  = self.chk_l.IsChecked()
        mcchk  = self.chk_mc.IsChecked()
        tchk  = self.chk_t.IsChecked()
        lbl = "%s %s %s" %(lbl_l if lchk else "", lbl_t if tchk else "",lbl_mc if mcchk else "")

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
        
    def reset_and_disable_chkboxes(self):
        self.reset_chkboxes()
        self.chk_l.Enable(False)
        self.chk_mc.Enable(False)
        self.chk_t.Enable(False)
        
    def draw_plot(self, item='M1'):
        """Redraws the plot
        """
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
        mlanglerangle = "%s^%s" %(item[0],item[1]) if int(item[1])!=1 else "%s" % item[0]
        self.ax_cv.set_ylabel(r'Coefficient of variation for $\langle{%s}\rangle$'
                               % mlanglerangle)
        self.ax_mag.set_ylabel(r'$\langle{%s}\rangle$' % (mlanglerangle))
        self.set_labelfontsize(self.xylbl_slider.GetValue())
        self.set_ticklabelfontsize(self.lbl_slider.GetValue())
        
        self.canvas.draw()

        for c in self.checkboxes:
            c.Enable(True)

    def checkboxes(self):
        return [ self.chk_l,self.chk_t,self.chk_mc]
    
class AggPanel(wx.Panel):

    name_dict = {'susc':r"$\mathcal{X}$",'Tcap':'$C_V$','T':'$T$',
                 'M1avg':r"$\langle{M}\rangle$",'M2avg':r"$\langle{M^2}\rangle$",'M4avg':r"$\langle{M^4}\rangle$",'Eavg':r"$\langle{H}\rangle$",'E2avg':r"$\langle{H^2}\rangle$",'U':r'$U_{L}$'}
    def __init__(self, parent,controller):
        self.controller = controller
        wx.Panel.__init__(self, parent=parent, id=wx.ID_ANY)
        self.parent = parent
        self.log= logging.getLogger('AggPanel')
        self.init_plot()
        self.init_gui()        
        
    def init_gui(self):
        
        
        self.cmb_L = wx.ComboBox(
            self,
            -1,
            size=(150, -1),
            style=wx.CB_READONLY,
            )
         
        self.cmb_mag = wx.ComboBox(
            self,
            -1,
            size=(150, -1),
            style=wx.CB_READONLY,
            )
        self.draw_button = wx.Button(self, -1, 'Draw')
      
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
        # self.hbox1.Add(self.agregate_btn, border=5, flag=wx.ALL
        #                | wx.ALIGN_CENTER_VERTICAL)
        # self.hbox1.Add(self.alt_btn, border=5, flag=wx.ALL
                    #   | wx.ALIGN_CENTER_VERTICAL) 
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

        hboxmain = wx.BoxSizer(wx.HORIZONTAL)
        self.reader = Reader(parent=self,title='Chooser')
        # panelCanvas = wx.Panel(self,-1)
        # vboxCanvas = wx.BoxSizer(wx.HORIZONTAL)
        # vboxCanvas.Add(self.canvas)
        # panelCanvas.SetSizer(vboxCanvas)
        hboxmain.Add(self.canvas, flag = wx.EXPAND | wx.ALIGN_LEFT)
        hboxmain.Add(self.reader, flag= wx.EXPAND| wx.ALIGN_RIGHT)
        self.vbox.Add(hboxmain, 1, wx.EXPAND)
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
        self.chooser.Destroy()
        
    def on_agg_button(self,event):
        self.controller.agregate()

    def on_alt_button(self,event):
        raise NotImplementedError
        
    # def agregate(self,bmatdict):
    #     #!!! hm,dobro ove choices resi
    #     #kad budes resila ono sa lazy evaluation
    #     #hmmmhhhhh
    #     agregate.main(dict(bmatdict),self.simdir)

    #     aggd = load_agg()
    #     # stavljamo ovde kao staticku variablu
    #     # valjda je ovo ok
    #     self.L_choices = zip(*aggd.columns)[0]
    #     print self.L_choices
        
    #     self.L_choices = list(set(self.L_choices))
    #     print self.L_choices
        
    #     self.mag_choices = list(aggd.index)
    #     self.cmb_L.SetItems(self.L_choices)
    #     self.cmb_L.SetValue(self.L_choices[0])
    #     self.cmb_mag.SetItems(self.mag_choices)
    #     self.cmb_mag.SetValue(self.mag_choices[0])
    #     self.draw_button.Enable(True)
    #     #ovome ce moci da se pristupi i preko self i to
    #     # samo sto ako ga prebrisemo, nece biti dobro
    #     # samo nam je potrebno da ponovo izracunamo za jedno
    #     # L i T mat i to je to
    #     AggPanel.aggd = aggd;

    def on_xyslider_scroll(self,e):
        fontsize = e.GetEventObject().GetValue()
        print "FONTSIZE", fontsize
        self.set_labelfontsize(fontsize)
        
    def on_slider_scroll(self,e):
        fontsize = e.GetEventObject().GetValue()
        print "FONTSIZE", fontsize
        self.set_ticklabelfontsize(fontsize)

    
    def on_draw_button(self, event):
        L_select = self.cmb_L.GetValue()
        mag_select = self.cmb_mag.GetValue()
        lbl = "$%s_{%s}$" %(L_select[0],L_select[1:])
        agg_data = self.controller.get_agg_plot_data()
        
        self.ax_agg.plot(agg_data[L_select].ix['T'],
                         agg_data[L_select].ix[mag_select],fmt_cycle.next(), label=lbl,fillstyle='none',picker=5)
        self.annotations = list()
        self.log.debug('agg_data: \n %s' %agg_data)
        index_ts = agg_data[L_select].ix['T'].index
        real_ts = agg_data[L_select].ix['T']
        mag_values = agg_data[L_select].ix[mag_select]
        for ti,m,tr in zip(index_ts,mag_values,real_ts):
            print 'ti:{} m:{} tr:{}'.format(ti,m,tr)
            text = self.controller.annotate_agg(L_select,ti)
            self.annotations.append(self.ax_agg.annotate(text,xy=(tr,m),xytext=(tr,m), visible=False,fontsize=8,picker = 5))

        self.chk_ann.SetValue(False)
        self.chk_ann.Enable(True)
        
        #???!!!self.ax_agg.set_xlim(right=1.55)
        leg = self.ax_agg.legend(loc="best",frameon=False,shadow=True)

        self.xy_ann = self.ax_agg.annotate('dummy text',xy = (0.0,0.0),fontsize=9,color='gray')
        self.xy_ann.set_visible(False)
        
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
        fig_width = self.parent.GetParent().width / self.dpi / 1.3
        fig_height = self.parent.GetParent().height / self.dpi * 3 / 4
        self.fig = Figure((fig_width, fig_height), dpi=self.dpi,facecolor='w')
        self.ax_agg = self.fig.add_subplot(111)

        self.xy_ann = self.ax_agg.annotate('dummy text',xy = (0.0,0.0),fontsize=9,color='gray')
        self.xy_ann.set_visible(False)
        
        self.canvas = FigCanvas(self, -1, self.fig)
        self.toolbar = NavigationToolbar(self.canvas)
        tw,th = self.toolbar.GetSizeTuple()
        fw,fh = self.canvas.GetSizeTuple()
        self.toolbar.SetSize(wx.Size(fw,20))
        self.toolbar.Realize()

        self.canvas.mpl_connect('pick_event', self.on_pick)
        self.canvas.mpl_connect('key_press_event',self.on_key_press)
        self.mte = self.canvas.mpl_connect('motion_notify_event', self.mouse_moved)


        
        ########################################
        ########### MATPLOTLIB EVENTS ##########

    def on_key_press(self,event):

        if event.key == 'x':
            try:
                self.canvas.mpl_disconnect(self.mte)
                del self.cursor
            except:
                pass
        elif event.key == 'c':
            self.cursor = widgets.Cursor(self.ax_agg, useblit=True, color='gray', linewidth=0.5)
            
            
    def on_pick(self,event):
        print type(event)
        print event.artist.set_visible(False)
        self.canvas.draw()
        print type(event.artist)
        print event

        
    def mouse_moved(self,event):
        x,y = event.xdata,event.ydata
        if x is not None:
            print "XX",x
            x,y = np.asscalar(x),np.asscalar(y)
            text = 'x:{:.4f}\ny:{:.4f}'.format(event.xdata,event.ydata)
            self.xy_ann.xy = x,y
            xlim = self.ax_agg.get_xlim()[1]
            ylim = self.ax_agg.get_ylim()[1]
            print xlim
            #self.xy_ann.xytext= xlim+0.01, 0
            self.xy_ann.xytext= xlim+0.004,ylim
        
            self.xy_ann.set_text(text)
            self.xy_ann.set_visible(True)
        else:
            self.xy_ann.set_visible(False)
        self.canvas.draw()
    def exited_ax(self,event):
        pass
        ########################################
        ######### CONTROLER INTERFACE ##########


    def set_l_choices(self,lch):
        self.cmb_L.SetItems(lch)
        try:
            self.cmb_L.SetValue(lch[0])
        except:
            self.cmb_L.SetValue('--')

    def set_mag_choices(self,mch):
        self.cmb_mag.SetItems(mch)
        self.cmb_mag.SetValue(mch[0])



class TabContainer(wx.Notebook):

    def __init__(self, parent, controller):

        wx.Notebook.__init__(self, parent, id=wx.ID_ANY,
                             style=wx.BK_DEFAULT)
        self.controller  = controller
        tp = ThermPanel(self,controller)
        ag = AggPanel(self,controller)
        # scat = ScatterPanel(self,controller)
        scat = "dummy"
        self.controller.init_gui(tp,ag,scat)
        self.AddPage(tp, 'Therm')
        self.AddPage(ag, 'Aggregate')
        # self.AddPage(scat,"Scatter")

    def flash_status_message(self,message):
        self.GetParent().flash_status_message(message,3000)


class GraphFrame(wx.Frame):

    """The main frame of the application
    """

    title = 'tulip'

    def __init__(self,controller):
        wx.Frame.__init__(self, None, -1, title=self.title)

##        self.size = tuple(map(operator.mul, wx.DisplaySize(),(3.0/4.0)))

        self.controller = controller
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
        self.notebook = TabContainer(self,self.controller)
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
    aplot_files = [f for f in os.listdir(self.simdir) if regexf.match(f)]
    aplot_files.sort()
    agg = dict()
    for aplot in aplot_files:
        L = regexf.match(aplot).groups()[0]
        data = pd.read_csv(join(self.simdir,aplot), index_col=0)
        agg[L] = data

    agg_data = pd.concat(agg, axis=1)
    agg_data.columns.names = ['L', 'T']
    print agg_data
    return agg_data

        



# def IsBold(l,itemText):
#     """Proverava da li je item u listi bold
#     ili ne"""
# #    font = item.GetFont()
#     print "checking if item '{}' is bold or not...".format(itemText)
#     print "or is in {}".format(file_manager.bmatdict[l].keys())
#     return itemText in file_manager.bmatdict[l].keys()
    
# def ChangeFont(listctrl_,item,l,what,*args):
#     """Menja font odredjenog elementa
#     u matchooseru u odnosu na gde je ovaj,
#     ha,samo sta je fazon, fazon jeste
#     sto je ovo samo za l hahahahahahahha
#     to je najveci fazon od svih!! Ajd probamo ovo
#     znaci mogu joj se proslediti sta je znaci
#     l t mc ili therm, i proizvoljan broj argumenata"""
#     map_ = {'l':file_manager.best_ls,'t': lambda l: file_manager.best_ts(l),
#             'mc':lambda l,t:file_manager.best_mc(l,t),
#             'therm':lambda l,t,mc:file_manager.best_therm(l,t,mc)
#             }
#     fontweight = wx.FONTWEIGHT_NORMAL
#     fontstyle = wx.FONTSTYLE_NORMAL
#     if l in map[what](*args) and not in file_manager.alt_ls():
#         fontweight = wx.FONTWEIGHT_BOLD
#         fontstyle = wx.FONTSTYLE_ITALIC
#     elif l in file_manager.best_ls() and in file_manager.alt_ls():
#         fontweight = wx.FONTWEIGHT_BOLD
#     elif l not in file_manager.best_ls and in file_manager.alt_ls():
#         fontstyle = wx.FONTSTYLE_ITALIC
#         font = item.GetFont()
#         font.SetWeight(wx_fontweight)
#         font.SetStyle(wx_fontstyle)
#         item.SetFont(font)
#         listctrl_.SetItem(item)
        
# def UnBold(self,item):
#         font = item.GetFont()
#         font.SetWeight(wx.FONTWEIGHT_NORMAL)
#         item.SetFont(font)
#         self.SetItem(item)
        


class MyListCtrl(wx.ListCtrl):
    def __init__(self, parent, id,controller,mat_deletable=False):
        wx.ListCtrl.__init__(self, parent, id, style=wx.LC_REPORT | wx.LC_HRULES | 
		wx.LC_NO_HEADER | wx.LC_SINGLE_SEL)

        self.parent = parent
        self.controller = controller

        self.Bind(wx.EVT_SIZE, self.OnSize)
        self.Bind(wx.EVT_LIST_ITEM_SELECTED,self.OnSelect)
        self.Bind(wx.EVT_LIST_ITEM_RIGHT_CLICK,self.OnRightClick)

        self.InsertColumn(0, '')
        self.SetName(MyListCtrl.__name__)

    def OnRightClick(self,event):
        """U slucaju da smo na t,therm ili mc-u
        brise se bestmat iz bestmatova.Oslanjamo
        se na atribut controllera koji on drzi
        konzistentim, nadamo se"""
        selected = event.GetIndex()
        selected = self.GetItemText(selected)
        name = util.extract_name(selected)
        self.controller.remove_bestmat(**{name:selected})
        
    def LoadData(self,list_items):
        self.DeleteAllItems()
        for li in list_items:
            self.InsertItem(li)
            
    def OnSize(self, event):
        size = self.parent.GetSize()
        self.SetColumnWidth(0, size.x-5)
        event.Skip()

    def OnFocus(self, event):
        self.SetItemBackgroundColour(0, 'red')
        
    def OnSelect(self, event):
        selected = event.GetIndex()
        selected = self.GetItemText(selected)
        self.controller.listctrl_selected(selected)
        
    



class Reader(wx.Panel):
    
    def __init__(self, parent, title):
        wx.Panel.__init__(self, parent=parent,name=title)
        self.SetSize(wx.Size(500,-1))
        
        self.controller = parent.controller
        self.parent = parent
        self.list_controls = dict()
#        self.SetSize((500,500))
        vbox = wx.BoxSizer(wx.VERTICAL)
        splitter = wx.SplitterWindow(self, -1, style=wx.SP_LIVE_UPDATE|wx.SP_NOBORDER)
        leftSplitter = wx.SplitterWindow(splitter,-1,style=wx.SP_LIVE_UPDATE | wx.SP_NOBORDER,name='leftSplitter')
        rightSplitter = wx.SplitterWindow(splitter,-1,style=wx.SP_LIVE_UPDATE | wx.SP_NOBORDER,name='rightSplitter')

        panel_arg_dicts = ({'parent':leftSplitter,'name':'l','text':'Lattice Size'},
                           {'parent':leftSplitter,'name':'t','text':'Temperature'},
                           {'parent':rightSplitter,'name':'mc','text':'Simulation paths'},
                           {'parent':rightSplitter,'name':'therm','text':'Lattice Sweeps'})
        panels = dict()
        for panel_args in panel_arg_dicts:
            panels[panel_args['name']]= self.make_listctrl_panel(**panel_args)

        leftSplitter.SplitVertically(panels['l'],panels['t'])
        rightSplitter.SplitVertically(panels['mc'],panels['therm'])
        splitter.SplitHorizontally(leftSplitter,rightSplitter)

      
        self.Bind(wx.EVT_TOOL, self.ExitApp, id=1)
        
 #       self.button_choose = wx.Button(self,-1,"Choose",size=(70,30))

   #     self.button_done = wx.Button(self,-1,"Done",size=(70,30))
###        self.Bind(wx.EVT_BUTTON, self.on_choose_button,self.button_choose)

    #    self.Bind(wx.EVT_BUTTON, self.on_done_button,self.button_done)
        vbox.Add(splitter, 1,  wx.EXPAND | wx.TOP | wx.BOTTOM, 5 )
        hbox = wx.BoxSizer(wx.HORIZONTAL)
        hbox.AddStretchSpacer()
        #hbox.Add(self.button_choose,1, wx.BOTTOM,5)
#        hbox.Add(self.button_done,1,wx.BOTTOM,5)
        vbox.Add(hbox,flag=wx.ALIGN_RIGHT|wx.RIGHT,border=10)
        self.SetSizer(vbox)
        


    def make_listctrl_panel(self,parent,text,name):
        """Vraca panel, i dodaje konkretni listcontrol u dictionary
        parent: roditelj panela koji cemo vracati, u njega ce se ugraditi
        panel"""
        vboxL = wx.BoxSizer(wx.VERTICAL)
        panelL = wx.Panel(parent, -1)
        panelLTxt = wx.Panel(panelL, -1, size=(-1, 40))
        panelLTxt.SetBackgroundColour('#53728c')
        stL = wx.StaticText(panelLTxt, -1, text, (5, 5))
        stL.SetForegroundColour('WHITE')

        panelLList = wx.Panel(panelL, -1, style=wx.BORDER_SUNKEN)
        vboxLList = wx.BoxSizer(wx.VERTICAL)
        listL = MyListCtrl(panelLList, -1,controller = self.controller)
        listL.SetName(name)
        self.list_controls[name]=listL

        vboxLList.Add(listL, 1, wx.EXPAND)
        panelLList.SetSizer(vboxLList)
        panelLList.SetBackgroundColour('WHITE')
        vboxL.Add(panelLTxt, 0, wx.EXPAND)
        vboxL.Add(panelLList, 1, wx.EXPAND)

        panelL.SetSizer(vboxL)
        return panelL

    def disable_choose(self):
        self.button_choose.Enable(False)
    def enable_choose(self):
        self.button_choose.Enable(True)
    def unchoose_enabled(self,is_enabled):
        self.button_unchoose.Enable(is_enabled)
        
    def on_unchoose_button(self,event):
        raise NotImplementedError
        



    def ExitApp(self, event):
        self.Close()

    def populateListControls(self,**kwargs):
        """Prosledjuju se dictionary koji mapira
        l/t/therm/mc na novi sadrzaj. Uglavnom ce
        biti jedna prava lista, i ostale prazne.
        
        Keyword arguments su:"l,t,therm i mc"""
        
        for key,itemlist in kwargs.items():
            if itemlist is None:
                continue
            try:
                self.list_controls[key].LoadData(itemlist)
            except AttributeError,e:
                util.show_error("Pogresan kljuc za listcontrolu","Ovo nije smelo da se desi, report issue!\nDetails: %s" %e)


class App(wx.App):

    def __init__(self,controller,*args, **kwargs):
        self.controller = controller
        wx.App.__init__(self,*args,**kwargs)
    def OnInit(self):
        """
        Poziva wx.App u okviru inita. Ovde mozemo
        konfiguracione parametre da prosledimo,i
        prosledjujemo kontroler gui-ju
        """
        import sys
        try:
            from docopt import docopt
        except ImportError:
            simdir = None
        else:
            args = docopt(__doc__)
            simdir = args['SIMDIR']
        print simdir
        logging.basicConfig(level=logging.DEBUG)

        if not simdir or not os.path.isdir(simdir):
            dlg=wx.DirDialog(None,style=wx.DD_DEFAULT_STYLE,message="Where your simulation files are...")
            if dlg.ShowModal() == wx.ID_OK:
                simdir=dlg.GetPath()
            else:
                sys.exit(0)

            dlg.Destroy()
        self.controller.set_simdir(simdir)
        self.main = GraphFrame(self.controller)
        self.main.Show()
        self.SetTopWindow(self.main)
        wx.CallAfter(self.application_started)
        return True
    def application_started(self):
        self.controller.application_started()

def main():
    controller = mvc_tulip.FileManager()
    app = App(controller)
    #controller.init_gui(app.main)
    app.MainLoop()
if __name__ == '__main__':
    main()

  

