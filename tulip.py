

#!/usr/bin/python
# -*- coding: utf-8 -*-
"""
=================================================================
Opis aplikacije....
=================================================================
   
Usage:
    gui_tabbed.py [SIMDIR]
    gui_tabbed.py -h | --help
Arguments:
    SIMDIR     Path to sim directory
Options:
    -h --help
"""
   
#import pdb
import wx
import pandas as pd
import os
from docopt import docopt
from os.path import join
import glob
import re
import pickle
   
from collections import defaultdict
import itertools
import unify
import agregate
import compose
import statmat
import matplotlib
   
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
   
   
regexf = re.compile(r'^(L\d{1,3}).aplot$')
   
SIM_DIR = ""
DEBUG = "+++DEBUG INFO+++  "
DEBUGG = False
LATTICE_MC = os.getcwd()
fmt_strings = ['g+-','r*-','bo-','y+-']
fmt_cycle = itertools.cycle(fmt_strings)
def debug():
    if(DEBUGG):
        pdb.set_trace()
        
def get_files(l,t,ext="*.plot"):
    """Vraca sve plot fajlove u zadatom folderu (L i T))"""

    folder_name = join(SIM_DIR,"%s%s" %(l,t),ext)
    print "folder name",folder_name
    files = glob.glob(folder_name)
    return files
   
def get_mtherms(L,T,therms="THERM\d+",igroup=0,ext="*.mat"):
    """za sve moguce .mat fajlove za dato L i T
    vraca parove mc ili therm u zavisnosti od igroup parametra)"""
    return [re.match(r'.*(%s)(MC\d+)' % therms, f.split(os.path.sep)[-1]).groups()[igroup] for f in get_files(l=L,t=T,ext=ext)]
def get_mmcs(L,T,therms):
    
    return get_mtherms(L=L,T=T,therms=therms,igroup=1,ext="*%sMC*.mat" %therms)
        
class ScatterPanel(wx.Panel):
    xlabel = 'Mx'    
    ylabel = 'My'
    zlabel = 'Mz'
    all_data = dict()
    
    def __init__(self,parent):
        
        wx.Panel.__init__(self,parent=parent,id=wx.ID_ANY)
        self.parent = parent
        self.tooltip = wx.ToolTip("Press 'd' for next T, scroll to zoom")
        self.tooltip.Enable(False)
        self.load_data()
        self.dpi = 100
        fig_width = self.parent.GetParent().width / self.dpi
        fig_height = self.parent.GetParent().height / self.dpi * (3 / 4)
        #self.fig = Figure((fig_width, fig_height), dpi=self.dpi)
   
        self.fig = Figure(figsize=(20,7),facecolor='#595454')
        # self.ax = Axes3D(self.fig)
        self.ax_3d = self.fig.add_subplot(121,projection="3d")
        print type(self.ax_3d)
        self.ax_hist = self.fig.add_subplot(122)
        self.canvas = FigCanvas(self,-1,self.fig)
        self.canvas.SetToolTip(self.tooltip)
        self.canvas.mpl_connect('key_press_event',self.on_key_press)
        # self.canvas.mpl_connect('button_press_event',self.on_button_press)
        self.canvas.mpl_connect('figure_enter_event',self.on_figure_enter)
        self.zoomf = zoom_factory(self.ax_3d,self.canvas)
   
        self.ax_3d.mouse_init()
        self.init_gui()
        
        self.ts = self.all_data.keys()
        key = lambda x: int(x[1:])
        self.ts = sorted(self.ts,key = lambda x: int(x[1:]))
        self.temprs = cycle(self.ts)
        self.setup_plot()
        self.canvas.mpl_connect('draw_event',self.forceUpdate)
        
        print DEBUG,"self.ts reversed",self.ts
   
    def on_figure_enter(self,event):
        self.tooltip.Enable(True)
        print "entered figure"
        
    def on_button_press(self,event):
        self.step(event)
    def on_key_press(self,event):
        if event.key =='d':
            self.step(event)
   
    def init_gui(self):
        self.vbox = wx.BoxSizer(wx.VERTICAL)
        # self.vbox.Add(self.canvas, 1, flag=wx.LEFT | wx.TOP)
        self.vbox.Add(self.canvas)
       
        self.draw_button = wx.Button(self, -1, 'next')
        self.Bind(wx.EVT_BUTTON, self.step, self.draw_button)
        self.hbox1 = wx.BoxSizer(wx.HORIZONTAL)
        
        self.hbox1.Add(self.draw_button, border=5, flag=wx.ALL
                       | wx.ALIGN_CENTER_VERTICAL)
        self.hbox1.AddSpacer(20)
        self.vbox.Add(self.hbox1, 0, flag=wx.ALIGN_LEFT | wx.TOP)
        self.SetSizer(self.vbox)
   
    
   
    def load_data(self):
        """Ucitava podatke za animaciju"""
        import os
        flist=glob.glob(join(SIM_DIR,"L10T*"))
        flist = [f for f in flist if os.path.isdir(f)]
        print DEBUG,"flist",flist
   
        for f in flist:
            print glob.glob(join(f,"*THERM1000MC*.all"))
            try:
                allf= glob.glob(join(f,"*THERM1000MC*.all"))[0]
            except Exception, e:
                continue
            data = pd.read_table(allf,delim_whitespace=True,nrows=1000, names=['seed', 'e', 'x', 'y', 'z'])
            data.pop('seed')
            data.set_index(np.arange(1000),inplace=True)
            temp =re.match(r".*L10(T\d{2,4})$",f).groups()[0]
            self.all_data[temp] = data
   
        self.data = pd.concat(self.all_data,axis=0)
    def setup_plot(self):
        
        "Initial drawing of scatter plot"
        from matplotlib import cm
        self.cbar=False
        self.step("dummy")
   
    def step(self,event):
        import time
        t=self.temprs.next()
        x,y,z = self.data.ix[t,'x'],self.data.ix[t,'y'],self.data.ix[t,'z']
        magt =np.sqrt( x ** 2 + y ** 2 + z ** 2)
        colors = np.where(magt>np.mean(magt),'r','b')
   
        self.ax_3d.cla()
        self.ax_hist.cla()
                
        self.scat =  self.ax_3d.scatter(x,y,z,s=10,c = magt,cmap=cm.RdYlBu)
        # self.scat=self.ax_3d.scatter3D(x,y,z,s=10,c=colors)
        if self.cbar :
            self.cbar.draw_all()
        else:
            self.cbar = self.fig.colorbar(self.scat)  
        
        self.ax_3d.set_title(t)
        self.ax_hist.set_ylim(0,40)
        self.ax_hist.hist(magt,bins=100,normed=1,facecolor='green',alpha=0.75)
        self.ax_3d.set_xlabel(self.xlabel)
        self.ax_3d.set_ylabel(self.ylabel)
        self.ax_3d.set_zlabel(self.zlabel)
        self.canvas.draw()
            
    def forceUpdate(self,event):
        self.scat.changed()
   
   
def load_best_mat_dict():
    with open(join(SIM_DIR,"mat.dict"),mode="ab+") as hashf:
       try:
           fcontent =  defaultdict(dict,pickle.load(hashf))
       except EOFError:
           fcontent = defaultdict(dict)
    print "best mat dict",fcontent
    return fcontent
   
class ThermPanel(wx.Panel):
   
    def __init__(self, parent):
        wx.Panel.__init__(self, parent=parent, id=wx.ID_ANY)
        self.parent = parent
        self.init_plot()
        self.mc_txt = wx.SpinCtrl(self, size = (80,-1))
        self.add_button = wx.Button(self,-1,'Gen stat for this MC')
        self.bestmat_button=wx.Button(self,-1,"Choose best .mats ...")
        self.clear_button = wx.Button(self,-1,"Clear plot")
        self.Bind(wx.EVT_BUTTON, self.on_chooser,self.bestmat_button)
        self.Bind(wx.EVT_BUTTON, self.on_add_button,self.add_button)
        self.Bind(wx.EVT_BUTTON, self.on_clear_button, self.clear_button)
        self.chk_l = wx.CheckBox(self,-1,"Lattice size", size=(-1, 30))
        self.chk_t = wx.CheckBox(self,-1,"Temperature",size=(-1, 30))
        self.chk_mc = wx.CheckBox(self,-1,"MC steps",size=(-1, 30))
        plot_choices = ['M1', 'M2', 'M4']
        self.cmb_plots = wx.ComboBox(self, size=(100, -1),
                choices=plot_choices, style=wx.CB_READONLY,
                value=plot_choices[0])
        self.cmb_L = wx.ComboBox(self, size=(70, -1),
                                 choices=sorted(lt_dict.keys(),key=lambda x: int(x[1:])),
                                 style=wx.CB_READONLY,
                                 value=lt_dict.keys()[0])
        t_choices = sorted(lt_dict[self.cmb_L.GetValue()],key= lambda x: int(x[1:]))
        self.cmb_T = wx.ComboBox(self, size=(100, -1),
                                 choices=t_choices,
                                 value=t_choices[0])
        self.cmb_pfiles = wx.ComboBox(self, size=(300, -1),
                choices=self.get_files(), value='<Choose plot file>')
        self.update_combos()
                                    
   
        self.Bind(wx.EVT_COMBOBOX, self.on_select, self.cmb_plots)
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
   
        self.hbox1.Add(self.clear_button, border=5, flag=wx.ALL
                       | wx.ALIGN_CENTER_VERTICAL)
        self.hbox1.Add(self.mc_txt, border=5, flag=wx.ALL
               | wx.ALIGN_CENTER_VERTICAL)
        self.hbox1.Add(self.add_button, border=5, flag=wx.ALL
                       | wx.ALIGN_CENTER_VERTICAL)
   
        self.hbox1.Add(self.bestmat_button, border=5, flag=wx.ALL
                       | wx.ALIGN_CENTER_VERTICAL)
        
   
        self.toolhbox =wx.BoxSizer(wx.HORIZONTAL)
   
        self.toolhbox.Add(self.toolbar)
        self.toolhbox.AddSpacer(20)
      
        self.toolhbox.AddSpacer(20)
        self.toolhbox.Add(self.chk_l,5, wx.TOP,3)
        self.toolhbox.Add(self.chk_t,5,wx.TOP,3)
        self.toolhbox.Add(self.chk_mc,5,wx.TOP,3)
   
        self.vbox = wx.BoxSizer(wx.VERTICAL)
        self.vbox.Add(self.canvas, 1, flag=wx.LEFT | wx.TOP)
        self.vbox.Add(self.toolhbox)
        self.vbox.Add(self.hbox1, 0, flag=wx.ALIGN_LEFT | wx.TOP)
        self.SetSizer(self.vbox)
        self.vbox.Fit(self)
    
   
   
   
    def on_clear_button(self,event):
        self.ax_cv.cla()
        self.ax_mag.cla()
      #  self.ax_mag.set_title('Magnetisation',fontsize=10,family='monospace')
       # self.ax_cv.set_title('Coefficient of Variation',fontsize=10,family='monospace')
        self.canvas.draw()
   
    def on_chooser(self,event):
        self.chooser = Reader(self,-1,"Chooser")
        self.chooser.ShowModal()
        self.chooser.Destroy()
        
    def add_to_mat_dict(self,l,t,therm,mc):
        """Dodaje u dictionary 'reprezentativnih' matova, ispisuje poruku
        u status baru, i cuva novo stanje best_mat_dict-a na disk"""
        best_mat = get_files(l=l,t=t,ext="*%s%s*.mat" %(therm,mc))
        # ne bi smelo da ima fajlova u okviru jednog foldera sa istim MC i THERM
        assert len(best_mat)==1
        best_mat_dict[l][t]=best_mat[0]
        print "BEST MAT DICT:",best_mat_dict
        self.parent.flash_status_message("Best .mat for %s%s selected" % (l,t))
        with open(join(SIM_DIR,"mat.dict") ,"wb") as matdictfile:
            pickle.dump(dict(best_mat_dict),matdictfile)
        
    def on_add_button(self,event):
        """Pravi novi .plot fajl u zeljenom direktorijumu
        tj. u zavisnosti od vrednosti L i T combo boxova"""
        mcs =int( self.mc_txt.GetValue())
        # gledamo sta je korisnik selektovao sto se tice L i T
        # i onda u tom folderu pravimo nove .mat fajlove
        # i radimo compose nad novim .mat fajlovima
        mcs_dir =join(SIM_DIR,self.cmb_L.GetValue()+self.cmb_T.GetValue())
        print "Making new plot files in dir %s for %d MCs" % (mcs_dir,mcs)
        statmat.main(mcs_dir,mcs)
        compose.main(mcs_dir,mcs)
        self.cmb_pfiles.SetItems(self.get_files())
      
    def on_select_T(self,event="dummy"):
        self.cmb_pfiles.SetItems(self.get_files())
        self.cmb_pfiles.SetValue('<Choose plot file>')
        # trazimo najveci mc od svih .all fajlova.  inace
        # nece se desiti nista pri odabiru generisanja za taj mc
        uplimit = get_mtherms(L=self.cmb_L.GetValue(),T=self.cmb_T.GetValue(),igroup=1,ext="*.all")
        uplimit = int(sorted(uplimit,key=lambda x: int(x[2:]))[-1][2:])
        print "uplimit",uplimit
        self.mc_txt.SetRange(0,uplimit)
   
    def update_combos(self,event="dummy"):
        """Ova metoda azurira kombinacijske kutije koje zavise od stanja
        drugih elemenata/kontrola """
        t_items=lt_dict[self.cmb_L.GetValue()]
        self.cmb_T.SetItems(t_items)
        self.cmb_T.SetValue(t_items[0])
        #selektovali smo T, pa pozivamo odgovarajucu metodu
        self.on_select_T()

      
         
    def get_files(self):
        """Kada zovemo iz klase ovu metodu, uglavnom nista ne tweakujemo i to, vec
        stavljamo defaultnu vrednost "*.plot" """
        return get_files(ext="*.plot",l=self.cmb_L.GetValue(),t=self.cmb_T.GetValue())
        


    def on_select_pfiles(self, event):
        """Kada korisnik izabere .plot fajl, iscrtava se """
        path = self.cmb_pfiles.GetValue()
        self.data = pd.read_csv(path, index_col=0)
        print "self data"
        print self.data
        #self.draw_plot()
        # gledamo da li je selektovano M M^2 ili M^4
        # tako ili mozemo vratiti vrednost uvek na M
        self.on_select("dummy")

    def init_plot(self):
        self.dpi = 100
        fig_width = (self.parent.GetParent().width / self.dpi) 
        fig_height = self.parent.GetParent().height / self.dpi * (3 / 4)

        self.fig = Figure((fig_width, fig_height), dpi=self.dpi,facecolor='w')



        self.ax_mag = self.fig.add_subplot(1, 2, 1)
        self.ax_cv = self.fig.add_subplot(1, 2, 2)

        # self.ax_mag.set_title('Magnetisation',fontsize=10,family='monospace')
        # self.ax_cv.set_title('Coefficient of Variation',fontsize=10,family='monospace')

        pylab.setp(self.ax_mag.get_xticklabels(), fontsize=5)
        pylab.setp(self.ax_cv.get_xticklabels(), fontsize=5)
        pylab.setp(self.ax_mag.get_yticklabels(), fontsize=5)
        pylab.setp(self.ax_cv.get_yticklabels(), fontsize=5)
        self.canvas = FigCanvas(self, -1, self.fig)
        self.toolbar = NavigationToolbar(self.canvas)
        tw,th = self.toolbar.GetSizeTuple()
        fw,fh = self.canvas.GetSizeTuple()
        self.toolbar.SetSize(wx.Size(fw,20))
        self.toolbar.Realize()

    def on_select(self, event):
        item = self.cmb_plots.GetValue()
        self.draw_plot(item=item)

    def draw_plot(self, item='M1'):
        """Redraws the plot
        """
        # self.ax_mag.cla()
        # self.ax_cv.cla()
        print self.cmb_pfiles.GetValue().split(os.path.sep)[-1]
        lbl_mc=re.match(r"^(L\d+T\d+)(MC\d+).*\.plot$",self.cmb_pfiles.GetValue().split(os.path.sep)[-1]).groups()[1]
        lbl_mc ="%s=%s" %(lbl_mc[:2],lbl_mc[2:])
        lbl_t ="%s=%s" %(self.cmb_T.GetValue()[0],self.cmb_T.GetValue()[1:])
        lbl_l ="%s=%s"% (self.cmb_L.GetValue()[0],self.cmb_L.GetValue()[1:])
        lchk  = self.chk_l.IsChecked()
        mcchk  = self.chk_mc.IsChecked()
        tchk  = self.chk_t.IsChecked()
        lbl = "%s %s %s" %(lbl_l if lchk else "", lbl_t if tchk else "",lbl_mc if mcchk else "")
        print lbl
        
        
        fmt =fmt_cycle.next()
        line = self.ax_mag.errorbar(x=self.data.ix['THERM'],
                             y=self.data.ix[item + 'avg'],
                             yerr=self.data.ix['stdMean' + item],fmt=fmt,fillstyle='none',label=lbl)

        mag_leg=self.ax_mag.legend(loc="best",fontsize=10,frameon=False,shadow=True)
        mag_leg.draggable(True)
        self.ax_cv.semilogx(self.data.ix['THERM'], self.data.ix['cv(%s)'
                             % item],fmt,label =lbl,fillstyle='none')
        cv_leg=self.ax_cv.legend(loc="best",fontsize=10,frameon=False,shadow=True)
        cv_leg.draggable(True)

        self.ax_mag.set_xscale('log')
        self.ax_cv.grid(True, color='red', linestyle=':')
        self.ax_mag.grid(True, color='red', linestyle=':')
        self.ax_cv.set_xlabel('Number of lattice sweeps', size=10)
        self.ax_mag.set_xlabel('Number of lattice sweeps',size=10)
        self.ax_cv.set_ylabel(r'Coefficient of variation for $\langle{%s^%s}\rangle$'
                               % (item[0],item[1]))
        self.ax_mag.set_ylabel(r'$\langle{%s^%s}\rangle$' % (item[0],item[1]))
        pylab.setp(self.ax_mag.get_xticklabels(), fontsize=5)
        pylab.setp(self.ax_mag.get_yticklabels(), fontsize=5)
        pylab.setp(self.ax_cv.get_xticklabels(), fontsize=5)
        pylab.setp(self.ax_cv.get_yticklabels(), fontsize=5)
        
        self.canvas.draw()


class AggPanel(wx.Panel):

    name_dict = {'susc':r"Magnetic Susceptibility",'Tcap':'Tcap???????','T':'Temperature',
                 'M1avg':r"Average value of $\langle{M}\rangle$",'M2avg':r"Average value of $\langle{M^2}\rangle$",'M4avg':r"Average value of $\langle{M^4}\rangle$",'Eavg':r"Average value of $\langle{E}\rangle$",'E2avg':r"Average value of $\langle{E^2}\rangle$",'U':r'Binder Cumulant U$_{L}$'}
    def __init__(self, parent):

        wx.Panel.__init__(self, parent=parent, id=wx.ID_ANY)
        self.parent = parent
        self.init_plot()
        self.init_gui()        
        
    def init_gui(self):
        self.agregate_btn = wx.Button(self,-1,"Aggregate!")
        self.Bind(wx.EVT_BUTTON,self.on_agg_button,self.agregate_btn)
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
        self.draw_button = wx.Button(self, -1, 'Draw plot')
        self.draw_button.Enable(False)
        self.Bind(wx.EVT_BUTTON, self.on_draw_button, self.draw_button)
        self.clear_button = wx.Button(self, -1, 'Clear plot')
        self.clear_button.Enable(True)
        self.Bind(wx.EVT_BUTTON, self.on_clear_button,
                  self.clear_button)
        
        self.hbox1 = wx.BoxSizer(wx.HORIZONTAL)
        self.hbox1.Add(self.agregate_btn, border=5, flag=wx.ALL
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
        
        
        self.vbox = wx.BoxSizer(wx.VERTICAL)
        self.vbox.Add(self.canvas, 1,  wx.EXPAND)
        self.vbox.Add(self.toolhbox)
        self.vbox.Add(self.hbox1, 0, flag=wx.ALIGN_LEFT | wx.TOP)
        self.SetSizer(self.vbox)
        self.vbox.Fit(self)

    def on_agg_button(self,event):
        agregate.main(dict(best_mat_dict),SIM_DIR)
        self.aggd = load_agg()
        self.L_choices = zip(*self.aggd.columns)[0]
        print self.L_choices
        
        self.L_choices = list(set(self.L_choices))
        print self.L_choices
        
        self.mag_choices = list(self.aggd.index)
        self.cmb_L.SetItems(self.L_choices)
        self.cmb_L.SetValue(self.L_choices[0])
        self.cmb_mag.SetItems(self.mag_choices)
        self.cmb_mag.SetValue(self.mag_choices[0])
        self.draw_button.Enable(True)
        
    def on_draw_button(self, event):
        L_select = self.cmb_L.GetValue()
        mag_select = self.cmb_mag.GetValue()
        print L_select, mag_select
        lbl = "$%s_{%s}$" %(L_select[0],L_select[1:])
        print self.aggd[L_select].ix['T']
        self.ax_agg.plot(self.aggd[L_select].ix['T'],
                         self.aggd[L_select].ix[mag_select],fmt_cycle.next(), label=lbl,fillstyle='none')
        self.ax_agg.set_xlim(right=1.55)
        leg = self.ax_agg.legend(loc="best",frameon=False,shadow=True)
        leg.draggable(True)
        self.ax_agg.set_xlabel("Temperature",size=10)
        self.ax_agg.set_ylabel(self.name_dict[mag_select],size=10)
        self.ax_agg.grid(True, color='red', linestyle=':')
        self.canvas.draw()
        
    def on_clear_button(self,event):
        self.ax_agg.cla()
        pylab.setp(self.ax_agg.get_xticklabels(), fontsize=5)
        pylab.setp(self.ax_agg.get_yticklabels(), fontsize=5)
        self.canvas.draw()
        
    def init_plot(self):
        self.dpi = 100
        fig_width = self.parent.GetParent().width / self.dpi
        fig_height = self.parent.GetParent().height / self.dpi * (3 / 4)
        self.fig = Figure((fig_width, fig_height), dpi=self.dpi,facecolor='w')
        self.ax_agg = self.fig.add_subplot(111)
        
#        self.ax_agg.set_title('Aggregate')
        pylab.setp(self.ax_agg.get_xticklabels(), fontsize=5)
        pylab.setp(self.ax_agg.get_yticklabels(), fontsize=5)
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


def writemd5hash(dir_md5):
    """Serijalizuje dir_md5 defaultdict,
    pre toga ga pretvara u regularni dict"""
   
    with open(hfpath,mode="wb") as file:
        pickle.dump(dict(dir_md5),file)


def remove_old_calcs(d):
    """Posto je utvrdio da je direktorijum sa simulacijama
    izmenjem, izbrise sva prethodna izracunavanja. Ovo treba
    da napravim na nivou pojedinacnih LT direktorijuma"""
    
    os.chdir(d)
    # ovo kad smislis sta ces sa agregatima
    #os.system("rm *.aplot")
    os.system("rm -f *.mat *.stat *.raw *.cv *.plot ")
    
def handle_simfiles(dir_md5):
    """Za svaki folder iz sim foldera on proverava da li stari
    hashmd5 odgovara novo izracunatom, za svakog racuna novi hash
    i stavlja ga u hash.txt (trebalo bi ove neizmenjene da ne dira, tj
    resi to nekako da zaobidje neki if i da se samo ponovo ispise, da ne moras
    da rewritujes u fajlovima). Mislim da je bolje da proverava za sve fajlove
    a ne samo za .dat fajlove, posto mozda hoce neko da izbrise .mat fajlove, i
    ocekuje da ce ovaj to prepoznati i ponovo ih napraviti. da. """
   
    # dirlist = os.walk(SIM_DIR).next()[1]
   
    dirlist = [d for d in glob.glob(join(SIM_DIR,"L[0-9]*[0-9]*")) if os.path.isdir(d)]
    dirlist.sort()
    
    
    maxi = len(dirlist)

    prbar = wx.ProgressDialog("Please wait, doing statistics 'n stuff...",message="starting",maximum=maxi,parent=None,style=0| wx.PD_APP_MODAL| wx.PD_CAN_ABORT)
    prbar.SetMinSize(wx.Size(350,100))
    count =itertools.count()
    for d in dirlist:
        hashmd5 = getmd5(d)
        
        #ako je doslo do promena u direktorijumu
        #ili po prvi put pisemo
        # posto radimo sa defaultdictom, postoje tri slucaja
        # 1. ne postoji uopste zapis za ovaj direktorijum, u tom slucaju
        # defaultdict vraca prazan '', i jednakost nije zadovoljena
        # 2. postoji zapis koji ima razlicit md5 za d - jednoakost nezadovoljena
        # 3. nista nije promenjeno - True

        print dir_md5[d]
        if dir_md5[d]!=hashmd5:
            remove_old_calcs(d)
            prbar.Update(count.next(),d.split(os.path.sep)[-1])
            unify.main(d)
            statmat.main(d)
            compose.main(d)
            
            #ponovo se racuna hash, mada mozda je ovo prebrzo
            # posto ce korisnici mozda generisati nove .mat fajlove
            # ali dobro, otom potom
            # pa da, sa ovim dictom ce sada to biti lakse, posto stalno
            # moze biti u memoriji, pa kad se dodaju novi fajlovi, ponovo se racuna
            # dictovi su kul
   
            # stavljamo novu vrednost
            # mislim da sacuvam ipak ovaj dict na pocetku
            # ako se desi do nekog prekida programa, da ipak sacuvamo
            dir_md5[d] = getmd5(d)
    #izracunato sve
    writemd5hash(dir_md5)
    prbar.Destroy()
        
def getmd5(d):
    """Izracunava md5 za direktorijum cija
    je apsolutna putanja prosledjena"""
    import hashlib
    m = hashlib.md5()
    for f in glob.glob(join(d,"*")):
        for line in  open(f,"rb").readlines():
            m.update(line)
    return m.hexdigest()
        

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

        
def read_hashf(hfpath):
    """Ako postoji fajl na hashf putanji
    ucitava ga u dict i vraca, ako ne postoji- pravi ga
    i vraca prazan dict"""

    # stavljam ab+ cisto da bi ga napravio ako ga nema
    # ne znam koliko je to pametno
    with open(hfpath,mode="ab+") as hashf:
       try:
           fcontent =  defaultdict(str,pickle.load(hashf))
       except EOFError:
           fcontent = defaultdict(str)
    return fcontent

def get_choices():
    regex = re.compile(r"^(L\d+)(T\d+)$")
    dirlist = [d for d in os.walk(SIM_DIR).next()[1] if regex.match(d)]
    dirlist.sort()
    dct = defaultdict(list)
    # ide kroz imena svih direktorijuma i za svaki L prilepljuje odgovarajuc
    # T-ove. prvi put kad naidje na odredjen L, to ce biti prazan niz, pa ce
    # nalepljivati dalje nove clanove
    for d in dirlist:
        l,t = regex.match(d).groups()
        dct[l].append(t)

    return dct
        
class ListCtrlLattice(wx.ListCtrl):
    def __init__(self, parent, id):
        wx.ListCtrl.__init__(self, parent, id, style=wx.LC_REPORT | wx.LC_HRULES | 
		wx.LC_NO_HEADER | wx.LC_SINGLE_SEL)
       
        self.parent = parent

        self.Bind(wx.EVT_SIZE, self.OnSize)
        self.Bind(wx.EVT_LIST_ITEM_SELECTED, self.OnSelect)
        self.InsertColumn(0, '')
        for l in sorted(lt_dict.keys(), key = lambda x: int(x[1:]),reverse=True):
            self.InsertStringItem(0,l)
      
    def OnSize(self, event):
        size = self.parent.GetSize()
        self.SetColumnWidth(0, size.x-5)
        event.Skip()

    def OnSelect(self, event):
        print "parent of window is ",self.parent
        print "grand parent of window is",self.parent.GetGrandParent()
        window = self.parent.GetGrandParent().FindWindowByName('ListControlTemperature')
        selected = event.GetIndex()
        print "first selected",selected
        self.parent.GetGrandParent().GetParent().FindWindowByName('rightSplitter').FindWindowByName('ListControlTherm').DeleteAllItems()
        self.parent.GetGrandParent().GetParent().FindWindowByName('rightSplitter').FindWindowByName('ListControlMC').DeleteAllItems()
        self.parent.GetGrandParent().GetGrandParent().disable_choose()
        window.LoadData(self.GetItemText(selected))

    def OnDeSelect(self, event):
        index = event.GetIndex()
        self.SetItemBackgroundColour(index, 'WHITE')

    def OnFocus(self, event):
        self.SetItemBackgroundColour(0, 'red')

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
        selected = event.GetIndex()
        print "first selected",selected
        self.parent.GetGrandParent().GetParent().FindWindowByName('rightSplitter').FindWindowByName('ListControlMC').DeleteAllItems()
        self.parent.GetGrandParent().GetGrandParent().disable_choose()
        window.LoadData(self.GetItemText(selected),self.l)

    def LoadData(self, item):
        self.DeleteAllItems()
        self.l = item
        for t in sorted(lt_dict[item],key=lambda x: int(x[1:]),reverse=True):
            self.InsertStringItem(0,t)


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
        for t in sorted(get_mtherms(L=l,T=t),key=lambda x: int(x[5:]),reverse=True):
            self.InsertStringItem(0,t)


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
        for t in sorted(get_mmcs(L=l,T=t,therms=therm),key=lambda x: int(x[2:]),reverse=True):
            self.InsertStringItem(0,t)

    def OnSelect(self,event):
        self.parent.GetGrandParent().GetGrandParent().enable_choose()



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
        self.button_choose.Enable(False)
        self.button_done = wx.Button(self,-1,"Done",size=(70,30))
        self.Bind(wx.EVT_BUTTON, self.on_choose_button,self.button_choose)
        self.Bind(wx.EVT_BUTTON, self.on_done_button,self.button_done)
        vbox.Add(splitter, 1,  wx.EXPAND | wx.TOP | wx.BOTTOM, 5 )
        hbox = wx.BoxSizer(wx.HORIZONTAL)
        hbox.AddStretchSpacer()
        hbox.Add(self.button_choose,1, wx.BOTTOM,5)
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
    def on_done_button(self,event):
        self.SetReturnCode(wx.ID_OK)
        self.Close()
    def on_choose_button(self,event):
        l =self.listL.GetItemText(self.listL.GetFirstSelected())
        t =self.listT.GetItemText(self.listT.GetFirstSelected())
        therm =self.listTherm.GetItemText(self.listTherm.GetFirstSelected())
        mc =self.listMC.GetItemText(self.listMC.GetFirstSelected())

        self.GetParent().add_to_mat_dict(l=l,t = t,therm=therm,mc=mc)

    def ExitApp(self, event):
        self.Close()
       
    
    
if __name__ == '__main__':
    import sys
    args = docopt(__doc__)
    SIM_DIR = args['SIMDIR']
    print SIM_DIR 
    app = wx.PySimpleApp()

    # ok, mozda bi bolje bilo da koristim dirpickerctrl???
    # mm, ifovi. Ako nije prosledjen argument, ili nije validan
    # pokazuje se dijalog
    if not SIM_DIR or not os.path.isdir(SIM_DIR):
        dlg=wx.DirDialog(None,style=wx.DD_DEFAULT_STYLE,message="Where your simulation files are...")
        if dlg.ShowModal() == wx.ID_OK:
            SIM_DIR=dlg.GetPath()
        else:
            sys.exit(0)

        dlg.Destroy()
    best_mat_dict = load_best_mat_dict()   

    ########################################
    ############ INIT #####################
    hfpath = join(LATTICE_MC,"md5_hash.dict")
    dir_md5  = read_hashf(hfpath)
    assert type(dir_md5) is defaultdict
    handle_simfiles(dir_md5)
    # ova globalna varijabla ce sadrzati kombinacije L-ova i T-ova koje
    # su nam dostupne. sto se tice foldera
    lt_dict = get_choices()
    app.frame = GraphFrame()
    app.frame.Show()
    app.MainLoop()
    
