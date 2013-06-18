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
   
   
regexf = re.compile(r'^(L\d{1,3}).aplot$')
   
SIM_DIR = ""
DEBUG = "+++DEBUG INFO+++  "
DEBUGG = False
LATTICE_MC = os.getcwd()
fmt_strings = ['g+-','r*-','bo-','y+-']
fmt_cycle = itertools.cycle(fmt_strings)

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
        
        self.fig = Figure(figsize=(fig_width,fig_height),dpi=self.dpi,facecolor='#595454')
        self.canvas = FigCanvas(self,-1,self.fig)
        # self.ax = Axes3D(self.fig)
        #self.ax_3d = self.fig.add_subplot(121,projection="3d")
        self.ax_3d = self.fig.add_axes([0,0,0.5,1],projection="3d")
        print type(self.ax_3d)
        self.ax_hist = self.fig.add_subplot(122)

        self.canvas.SetToolTip(self.tooltip)
        self.canvas.mpl_connect('key_press_event',self.on_key_press)
        self.canvas.mpl_connect('figure_enter_event',self.on_figure_enter)
        self.zoomf = zoom_factory(self.ax_3d,self.canvas)
   
        self.ax_3d.mouse_init()
        self.init_gui()        # mozda da napravim da svakako mora load?
        # ma kakvi. ovo je ok za sada
        if best_mat_dict.keys():
            self.load_data(l=self.cmb_l.GetValue())
        
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
                warnings.warn('extension %s did not match the selected image type %s; going with %s'%(ext, format, ext), stacklevel=0)
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
        
        self.vbox.Add(self.canvas)
        value = "" if not best_mat_dict.keys() else best_mat_dict.keys()[0]
        self.cmb_l = wx.ComboBox(self, size=(70, -1),
                                 choices=sorted(best_mat_dict.keys(),key=lambda x: int(x[1:])),
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
        self.mc_txt = wx.SpinCtrl(self,size=(80,-1))
        self.load_button = wx.Button(self,-1,'Load')
        
        self.Bind(wx.EVT_CHECKBOX,self.on_chk_mcs, self.chk_mcs)
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
        self.hbox1.AddSpacer(20)
        self.vbox.Add(self.hbox1, 0, flag=wx.ALIGN_LEFT | wx.TOP)
        self.SetSizer(self.vbox)
        self.vbox.Fit(self)

    def on_prev_press(self,event):
        self.step("dumm",backwards=True)

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
        choices = sorted(best_mat_dict.keys(),key=lambda x: int(x[1:]))
        self.cmb_l.SetItems(choices)
        self.cmb_l.SetValue(choices[0])
        self.load_data(l=choices[0])
        
        

    def on_selectl(self,event):
        print "Loading data for {}...".format(self.cmb_l.GetValue())
        self.load_data(l=self.cmb_l.GetValue())
        self.setup_plot()
   
    def load_data(self,l="L10",n=1000, keep=False):
        """Ucitava podatke za animaciju"""
        import os
        flist=glob.glob(join(SIM_DIR,"{}T*".format(l)))
        self.all_data= dict()
        
        flist = [f for f in flist if os.path.isdir(f)]
        self.log.debug("file list: %s " %flist)

        for f in best_mat_dict[l].values():
            temp =re.match(r".*%s(T\d{2,4})" % l,f.split(os.path.sep)[-1]).groups()[0]
            self.log.debug("Loading data for tempearture {}".format(temp))
            self.log.debug("Loading data from file %s.all" % f[:-4])
            # ne znam da li mi treba ovde neki try catch hmhmhmhmhmhmhmmhhh
            data = pd.read_table("%s.all" % f[:-4],delim_whitespace=True,nrows=n, names=['seed', 'e', 'x', 'y', 'z'])
            data.pop('seed')
            print data.count()
            data.set_index(np.arange(data.e.count()),inplace=True)
            self.all_data[temp] = data
        
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
            
        self.mc_txt.SetRange(0,getmaxmc(l))

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



    
    def step(self,event, backwards=False,curr=False):
        """Crta za sledece, proslo ili trenutno t"""
        t= (curr and self.temprs.curr()) or (self.temprs.next() if not backwards else self.temprs.prev())
        print "magic t",t
        x,y,z = self.data.ix[t,'x'],self.data.ix[t,'y'],self.data.ix[t,'z']
        magt =np.sqrt( x ** 2 + y ** 2 + z ** 2)
        colors = np.where(magt>np.mean(magt),'r','b')
        
        self.ax_3d.cla()
        self.ax_hist.cla()
        pylab.setp(self.ax_3d.get_xticklabels(),fontsize=8, color='#666666')
        pylab.setp(self.ax_3d.get_yticklabels(),fontsize=8, color='#666666')
        pylab.setp(self.ax_3d.get_zticklabels(),fontsize=8, color='#666666')
        self.ax_3d.set_xlabel(self.xlabel, fontsize=8)
        self.ax_3d.set_ylabel(self.ylabel,fontsize=8)
        self.ax_3d.set_zlabel(self.zlabel,fontsize=8)
        self.log.debug("Magt has {} elements".format(magt.count()))
        
        self.scat =  self.ax_3d.scatter(x,y,z,s=10,c = magt,cmap=cm.RdYlBu)
        therm,sp = getthermmc(self.cmb_l.GetValue(),t)

        title ="T={:.2f}\nTHERM={}\n SP={}".format((float(t[1:])/100),therm,sp)
        self.ax_3d.set_title(title, fontsize=10, position=(0.1,0.95))
        
        self.log.debug("Maksimum magt je {}".format(magt.max()))
#        self.ax_hist.set_ylim(0,magt.max()*1000)
        self.log.debug(magt)
        z = self.ax_hist.hist(magt,bins=100,facecolor='green',alpha=0.75)
        print "zz\n",z
        print "size of z is{}".format(len(z[0]))
        self.local_ylim = self.ax_hist.get_ylim()
        self.local_xlim = self.ax_hist.get_xlim()
        self.log.debug("local xlim: {} local ylim: {}".format(self.local_ylim, self.local_xlim))
        self.on_chk_lim("dummy")
        self.ax_hist.set_ylim(self.ylim)
        self.ax_hist.set_xlim(self.xlim)
        
        
        self.canvas.draw()

               
    def forceUpdate(self,event):
        self.scat.changed()


def getmaxmc(l):
    mcs = list()
    for t in best_mat_dict[l].keys():
        mcs.append(int(getthermmc(l,t)[1]))
    return max(mcs)
    
def getthermmc(l,t):
    return re.match(r'.*THERM(\d+)MC(\d+)',best_mat_dict[l][t].split(os.path.sep)[-1]).groups()
   
def load_best_mat_dict():
    with open(join(SIM_DIR,"mat.dict"),mode="ab+") as hashf:
       try:
           fcontent =  defaultdict(dict,pickle.load(hashf))
       except EOFError:
           fcontent = defaultdict(dict)
    print "best mat dict",fcontent
    return fcontent

def clean_mat_dict():
    """Gleda koji unosi u dict nemaju vrednost
    tj. samo su dodati zbog defaultdict svojstva
    i skida ih. Ovo ce se samo desiti u mat chooseru
    tako da ovo tada samo treba da zovem"""
    # ovo ce proci posto mi ne iteriramo kroz sam dict
    # a skidamo sa samog dicta. ova lista items nece biti
    # vise up to date, ali to nam nije bitno, posto nije
    # ciklicna
    for key,value in best_mat_dict.items():
        if not value:
            best_mat_dict.pop(key)
            print "removing {}.aplot".format(key)
            try:
                os.remove(join(SIM_DIR,'{}.aplot'.format(key)))
            except Exception:
                print "...NOT"
                pass
    serialize_mat()
        
def add_to_mat_dict(l,t,therm,mc):
    """Dodaje u dictionary 'reprezentativnih' matova, ispisuje poruku
    u status baru, i cuva novo stanje best_mat_dict-a na disk"""
    best_mat = get_files(l=l,t=t,ext="*%s%s*.mat" %(therm,mc))
    # ne bi smelo da ima fajlova u okviru jednog foldera sa istim MC i THERM
    assert len(best_mat)==1
    best_mat_dict[l][t]=best_mat[0]
    print "BEST MAT DICT:",best_mat_dict
#    self.parent.flash_status_message("Best .mat for %s%s selected" % (l,t))
    serialize_mat()

def serialize_mat():
    with open(join(SIM_DIR,"mat.dict") ,"wb") as matdictfile:
        pickle.dump(dict(best_mat_dict),matdictfile)
            
class ThermPanel(wx.Panel):
    def __init__(self, parent):
        self.tooltip=wx.ToolTip("r:color to red\ng:color to green\n")
        wx.Panel.__init__(self, parent=parent, id=wx.ID_ANY)
        self.parent = parent
        self.init_plot()
        self.canvas.SetToolTip(self.tooltip)
       

        self.mc_txt = wx.SpinCtrl(self, size = (80,-1))
        self.add_button = wx.Button(self,-1,'Generate .plot')
        self.clear_button = wx.Button(self,-1,"Clear")
        self.draw_button = wx.Button(self,-1,"Draw")
        self.draw_button.Enable(False)
  
        self.Bind(wx.EVT_BUTTON, self.on_add_button,self.add_button)
        self.Bind(wx.EVT_BUTTON, self.on_draw_button,self.draw_button)
        self.Bind(wx.EVT_BUTTON, self.on_clear_button, self.clear_button)
        self.chk_l = wx.CheckBox(self,-1,"Lattice size", size=(-1, 30))
        self.chk_t = wx.CheckBox(self,-1,"Temperature",size=(-1, 30))
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
        self.toolhbox.Add(self.chk_l, border=5, flag=wx.ALL
                       | wx.ALIGN_CENTER_VERTICAL)
        self.toolhbox.Add(self.chk_t, border=5, flag=wx.ALL
                       | wx.ALIGN_CENTER_VERTICAL)
        self.toolhbox.Add(self.chk_mc, border=5, flag=wx.ALL
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
        mcs_dir =join(SIM_DIR,self.cmb_L.GetValue()+self.cmb_T.GetValue())
        print "Making new plot files in dir %s for %d MCs" % (mcs_dir,mcs)
        statmat.main(mcs_dir,mcs)
        compose.main(mcs_dir,mcs)
        self.cmb_pfiles.SetItems(self.get_files())
      
    def on_select_T(self,event="dummy"):
        self.cmb_pfiles.SetItems(self.get_files())
        self.cmb_pfiles.SetValue('<Choose plot file>')
        self.draw_button.Enable(False)
        self.reset_chkboxes()
        # treba i checkboxovi da su disabled dok god nije nacrtan plot
        # znaci kad se pozove draw onda se enable-uju a kad se cler pozove onda
        # se disejbluju. valjda su to svi slucajevi 
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

        

    def on_select(self, event):
        # item = self.cmb_plots.GetValue()
        # self.draw_plot(item=item)
        self.draw_button.Enable(True)

    def on_draw_button(self,event):
        self.draw_plot(self.cmb_plots.GetValue())
        
    def draw_legend(self,event):
        lbl_mc=re.match(r"^(L\d+T\d+)(MC\d+).*\.plot$",self.cmb_pfiles.GetValue().split(os.path.sep)[-1]).groups()[1]
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
        print self.cmb_pfiles.GetValue().split(os.path.sep)[-1]
        
        
        
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


def matDictEmpty():
    for key in best_mat_dict.keys():
        if best_mat_dict[key]:
            return False
    return True
    
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
        self.agregate_btn.Enable( not matDictEmpty())
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
        clean_mat_dict()
        print "BEST MAT DICT", best_mat_dict
        self.chooser.Destroy()
        self.agregate_btn.Enable(not matDictEmpty())
        
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
        print L_select, mag_select
        lbl = "$%s_{%s}$" %(L_select[0],L_select[1:])
        print  self.aggd[L_select].ix['T'].index
        self.ax_agg.plot(self.aggd[L_select].ix['T'],
                         self.aggd[L_select].ix[mag_select],fmt_cycle.next(), label=lbl,fillstyle='none')
        self.annotations = list()
        for t,m in zip(self.aggd[L_select].ix['T'].index,self.aggd[L_select].ix[mag_select]):
            self.annotations.append(self.ax_agg.annotate('$THERM={groups[0]}$\n $SP={groups[1]}$'.format(groups=
                re.match(r'.*THERM(\d+)MC(\d+)',best_mat_dict[L_select][t].split(os.path.sep)[-1]).groups()),xy=(float(t[1:])/100,m),xytext=(float(t[1:])/100,m), visible=False,fontsize=10))

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


def writemd5hash(dir_md5):
    """Serijalizuje dir_md5 defaultdict,
    pre toga ga pretvara u regularni dict"""
   
    with open(hfpath,mode="wb") as file:
        pickle.dump(dict(dir_md5),file)


def remove_old_calcs(d):
    """Posto je utvrdio da je direktorijum sa simulacijama
    izmenjem, izbrise sva prethodna izracunavanja. Ovo treba
    da napravim na nivou pojedinacnih LT direktorijuma"""
    import fnmatch
    os.chdir(d)
    # ovo kad smislis sta ces sa agregatima
    #os.system("rm *.aplot")
    # sranje sto ne rade brace ekspanzije . tek od pajtona 3.3, mozda?
    oldies = [f for f in os.listdir(os.getcwd()) if fnmatch.fnmatch(f,"*.mat") or fnmatch.fnmatch(f,"*.stat") or fnmatch.fnmatch(f,"*.raw") or fnmatch.fnmatch(f,"*.cv") or fnmatch.fnmatch(f,"*.plot") ]
    print "removing files in {} directory...".format(os.getcwd())
    
    for f in oldies:
        print "removing {}...".format(f)
        os.remove(f)
#    os.system("rm -f *.mat *.stat *.raw *.cv *.plot ")
    
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
    je apsolutna putanja prosledjena. Za svaki
    fajl iz tog direktorijuma gleda kad je
    izmenjem, i te vrednosti stavlja u md5"""
    import hashlib
    m = hashlib.md5()
    for f in glob.glob(join(d,"*")):
        m.update(str(os.path.getmtime(f)))
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

def IsBold(l,itemText):
    """Proverava da li je item u listi bold
    ili ne"""
#    font = item.GetFont()
    print "checking if item '{}' is bold or not...".format(itemText)
    print "or is in {}".format(best_mat_dict[l].keys())
    return itemText in best_mat_dict[l].keys()
    
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
        for l in sorted(lt_dict.keys(), key = lambda x: int(x[1:])):
            self.InsertStringItem(cntr,l)
            
            if l in best_mat_dict.keys() and best_mat_dict[l]:
                #ako je vec izabran
                #6f92a4
                #2b434f
#                self.SetItemBackgroundColour(cntr,'#d0dae0')
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
        for t in sorted(lt_dict[item],key=lambda x: int(x[1:])):
            self.InsertStringItem(cntr,t)
            if t in best_mat_dict[item].keys():
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
        for t in sorted(get_mtherms(L=l,T=t),key=lambda x: int(x[5:])):
            self.InsertStringItem(cntr,t)
            print t
            try:
                if re.match(r"^%s%s%sMC\d+\.mat$" % (self.l,self.t,t), best_mat_dict[self.l][self.t].split(os.path.sep)[-1]):
                    MakeBold(self,self.GetItem(cntr))
                    getSiblingByName(self,"ListControlMC").LoadData(therm=t,l=self.l,t=self.t)
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
        for mc in sorted(get_mmcs(L=l,T=t,therms=therm),key=lambda x: int(x[2:])):
            self.InsertStringItem(cntr,mc)
            try:
                if '%s%s%s%s.mat' % (l,t,therm,mc) == best_mat_dict[l][t].split(os.path.sep)[-1].strip():
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
        print "deleting '{}' from dictionary...".format(best_mat_dict[l][t])
        del best_mat_dict[l][t]
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

        add_to_mat_dict(l=l,t = t,therm=therm,mc=mc)
        self.GetGrandParent().flash_status_message("Best .mat for %s%s selected" % (l,t))
    def ExitApp(self, event):
        self.Close()
       
    
    
if __name__ == '__main__':
    import sys
    from docopt import docopt
    args = docopt(__doc__)
    SIM_DIR = args['SIMDIR']
    print SIM_DIR
    logging.basicConfig(level=logging.DEBUG)
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
    hfpath = join(SIM_DIR,"md5_hash.dict")
    dir_md5  = read_hashf(hfpath)
    assert type(dir_md5) is defaultdict
    handle_simfiles(dir_md5)
    # ova globalna varijabla ce sadrzati kombinacije L-ova i T-ova koje
    # su nam dostupne. sto se tice foldera
    lt_dict = get_choices()
    app.frame = GraphFrame()
    app.frame.Show()
    app.MainLoop()
    

