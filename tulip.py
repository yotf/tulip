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
#PACKAGE_ABS_PATH = os.path.abspath(os.path.dirname(tulip.__file__))
#!!!Ove napravi kao dictionaryje. i nesto da mogu da se kombinuju
#nem pojma. da moze da se menja, znaci neka struktura koja ce se u zavisnosti
#od parametra ce advancovati jedan cycle++, od bilo koji od ovih
#znaci bice neki parameter dict i bice po jedan fmt cycle
#za svaki od njih
fmt_strings = ['g+-','r*-','bo-','y+-']
fmt_cycle = itertools.cycle(fmt_strings)

        
        
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

    def selector_callback(self,eclick,erelease):
        x_begin = eclick.xdata
        x_end = erelease.xdata
        if x_begin > x_end:
            x_begin,x_end = x_end,x_begin
        # znaci bice true ako se ne nalazi u ovoj regiji
        self.log.debug("x_begin {} x_end {}".format(x_begin,x_end))
                       
        booli = [ not (mag>=x_begin and mag<=x_end) for mag in self.magt ]
        self.log.debug("booli:\n %s",booli)
        #znaci prosledjujemo mu za sta da generise mat
        curr_t = self.temprs.curr()
        curr_l = self.cmb_l.GetValue()
        dir_ = self.cmb_dirs.GetValue()
        
        self.controller.remove_faulty(dir_,curr_l,curr_t,booli)
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
        self.cmb_l = wx.ComboBox(self, size=(70, -1),
                                 style=wx.CB_READONLY)

        self.cmb_dirs = wx.ComboBox(self, size=(150, -1),
                                         style=wx.CB_READONLY)

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
        self.Bind(wx.EVT_BUTTON, self.step, self.draw_button)
        self.Bind(wx.EVT_BUTTON, self.on_prev_press, self.prev_button)
        self.Bind(wx.EVT_BUTTON, self.save_figure, self.save_button)
        self.Bind(wx.EVT_COMBOBOX,self.on_selectl,self.cmb_l)
        self.Bind(wx.EVT_COMBOBOX,self.on_select_dir,self.cmb_dirs)
        self.hbox1 = wx.BoxSizer(wx.HORIZONTAL)


        self.hbox1.Add(self.cmb_dirs, border=5, flag=wx.ALL
                       | wx.ALIGN_CENTER_VERTICAL)
        self.hbox1.Add(self.cmb_l, border=5, flag=wx.ALL
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

    def able_buttons(self,enable=False):
        """u zavisnosti od enable
        disabluje ili enabluje dugmice u sizeru"""
        buttons = self.hbox1.GetChildren()
        for b in buttons:
            try:
                b.GetWindow().Enable(enable)
            except:
                pass

    def on_chk_mcs(self,event):
        firstn = self.firstn if self.chk_mcs.IsChecked() else None
        dir_ = self.cmb_dirs.GetValue()
        self.log.debug("Loading first {} sps".format(firstn))
        self.data = self.controller.load_sp_data(dir_=dir_,l=self.cmb_l.GetValue(),n = firstn)
        self.step('dummy',curr=True)

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

    def on_select_dir(self,event):
        vl = self.cmb_dirs.GetValue()
        self.controller.sp_on_dir_select(vl)
        
        
    def on_selectl(self,event):
        print "Loading data for {}...".format(self.cmb_l.GetValue())
        self.load_data(l=self.cmb_l.GetValue())
        self.setup_plot()
   
    def set_lims(self,l):

        ylims = list()
        xlims = list()
        ts = set(zip(*self.data.index)[0])
        for t in ts:
            self.ax_hist.cla()
            magt = self.controller.calculate_magt(self.data.ix[t])
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
            
        # self.mc_txt.SetRange(0,self.controller.get_maxmc(l))
        self.ts = sorted(ts,key = lambda x: int(x[1:]))
        # self.temprs = self.temprs if keep else twoway_cycle(self.ts)
        self.temprs = util.twoway_cycle(self.ts)
        self.setup_plot(curr=True)
        self.canvas.mpl_connect('draw_event',self.forceUpdate)
        

    def setup_plot(self,curr=False):
        "Initial drawing of scatter plot"
        self.mc_txt
        self.step("dummy",curr=curr)

    
    def step(self,event, backwards=False,curr=False,booli=False):
        """Crta za sledece, proslo ili trenutno t"""
        dir_ = self.cmb_dirs.GetValue()
        t= (curr and self.temprs.curr()) or (self.temprs.next() if not backwards else self.temprs.prev())
        l = self.cmb_l.GetValue()
        self.magt =self.controller.calculate_magt(self.data.ix[t])
        self.magt = self.magt[booli] if booli else self.magt
        print 'MAGT', self.magt

        self.mc_txt.SetRange(0,self.controller.get_maxmc(dir_,l,t))
        self.log.debug("t u step-u je ispalo :{}".format(t))
        comps = self.controller.mag_components(self.data.ix[t])

        self.ax_hist.cla()
        
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
        if len(comps.columns)!=3:
            util.show_error('Non 3d data','Scatter unavailable for non 3D data')
            self.canvas.draw()
            return
        colors = np.where(self.magt>np.mean(self.magt),'r','b')
        x,y,z = comps.icol(0),comps.icol(1),comps.icol(2)
        if booli:
            x=x[booli]
            y=y[booli]
            z=z[booli]
            
                
        self.ax_3d.cla()
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
        title = self.controller.get_scat_title(dir_,l,t)
        self.ax_3d.set_title(title, fontsize=10, position=(0.1,0.95))
        
        self.log.debug("Maksimum magt je {}".format(self.magt.max()))
        self.log.debug("MAGT:\n %s"% self.magt)
        self.canvas.draw()
               
    def forceUpdate(self,event):
        try:
            self.scat.changed()
        except AttributeError:
            pass

    def on_selectl(self,event):
        val = self.cmb_l.GetValue()
        self.set_l(val)

    def set_l(self,l):
        self.cmb_l.SetValue(l)
        dir_ = self.cmb_dirs.GetValue()
        self.data = self.controller.load_sp_data(dir_,l,n=self.firstn)
        self.set_lims(l)

    ############## CONTROLLER INTERFACE ################

    def set_dir_choices(self,dirch):
        self.cmb_dirs.SetItems(dirch)
        try:
            dir_ = dirch[0]
        except:
            self.cmb_dirs.SetValue('--')
            self.able_buttons(False)
        else:
            self.cmb_dirs.SetValue(dir_)
            self.able_buttons(True)
            self.controller.sp_on_dir_select(dir_)
            

    
    def set_l_choices(self,lch):
        """stavlja izbore za l, i stavlja
        maksimalno mc za izabrano l """
        self.cmb_l.SetItems(lch)
        try:
            l = lch[0]
        except:
            self.cmb_l.SetValue('--')
        else:
            self.set_l(l)


class ExpPanel(wx.Panel):
    def __init__(self,parent,controller):
        wx.Panel.__init__(self, parent=parent, id=wx.ID_ANY)
        self.controller = controller
        self.log = logging.getLogger("ExpPanel")
        self.l = wx.SpinCtrl(self, size = (80,-1))
        self.t = wx.SpinCtrl(self, size = (80,-1))
        self.therm = wx.SpinCtrl(self, size = (80,-1))
        self.so = wx.SpinCtrl(self, size = (80,-1))

class ThermPanel(wx.Panel):

    cmbord = ['dir_','l','t','mc']
    cmbsize=[150,70,70,130]
    plts = ['M1', 'M2', 'M4']
    
    def __init__(self, parent,controller):
        wx.Panel.__init__(self, parent=parent, id=wx.ID_ANY)
        self.controller = controller
        self.log = logging.getLogger("ThermPanel")
        self.tooltip = wx.ToolTip("rainbow click")
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
        
        

        self.hbox1 = wx.BoxSizer(wx.HORIZONTAL)
        
        self.hbox1.AddSpacer(20)
                
        self.make_combos()
        
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


    def make_combos(self):
        self.cmb_dict = dict()
        for name,size in zip(self.cmbord,self.cmbsize):
            combo = wx.ComboBox(self, size=(size, -1),
                                style=wx.CB_READONLY,
                                value = '--',
                                name=name
            )
            self.cmb_dict[name]=combo

            self.Bind(wx.EVT_COMBOBOX, self.on_select, combo)
                
            self.hbox1.Add(combo, border=5, flag=wx.ALL
                               | wx.ALIGN_CENTER_VERTICAL)


        self.cmb_mags = wx.ComboBox(self, size=(100, -1),style=wx.CB_READONLY,
                                    choices = self.plts, value= self.plts[0],name='mags')
        
        self.hbox1.Add(self.cmb_mags, border=5, flag=wx.ALL
                               | wx.ALIGN_CENTER_VERTICAL)



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
        vals = self.get_cmbvals('t')
        self.controller.add_mc(firstnmcs,**vals)


    ### CONTROLER INTERFACE ####

        
    def set_choices(self,**kwargs):
        """Prosledjuje se jedan choice
        koji god i setuje se onda sta treba
        """
        assert len (kwargs.keys())==1
        name,items =kwargs.items()[0]
        cmb = self.cmb_dict[name]
        self.log.debug('Setting items for %s' %name)
        cmb.SetItems(items)

        curr_val = cmb.GetValue()
        try:
            value = items[0] if curr_val=='--' or curr_val not in items else curr_val
        except IndexError:
            util.show_error("Simdir error", "No %s choices! Wrong simulation directory?" %name)
        else:
            cmb.SetValue(value)
            self.on_select(**kwargs)
        
            

    def on_select(self, event=None,**kwargs):
        """
        Znaci uzimamo id od objekta koji je
        izazvao event i onda uzimamo name-ove
        Kwargsovi koje prima su sledeci:
           dir-->nad direktorijumi koji sadrzi L-ove
           l--> koji l je selektovan
           t--> temperature
           mc--> simulation paths
           samo jedno od te 4 sme da se prosledi, jbg
        ako je mc onda nista ne uradi, za sada
        """
        
        try:
            self.log.debug("event: %s" % event)
            name = event.GetEventObject().GetName()
        except Exception as e:
            print e
            assert len(kwargs.keys())==1
            
            name = kwargs.keys()[0]
        if name=='mc':
            return
        self.log.debug('getting vals for combo by name: %s' %name)
        vals = self.get_cmbvals(name)
        self.log.debug('Vrednosti za generisanje sledeceg komba: %s' % vals)
        self.controller.tp_on_select(**vals)


    def get_cmbvals(self,name):
        """Vraca dictionary vrednosti
        svih comboboxova koji su potrebni
        za odredjivanje sledeceg. tj, ako je
        l selektovano onda se prosledjuje njegova
        vrednost i vrednost g-a"""
        try:
            val_dict = {k:combo.GetValue() for k,combo in self.cmb_dict.items() if self.cmbord.index(name)>=self.cmbord.index(k)}
        except ValueError as ve:
            util.show_error('ValueError',str(ve))
        else:
            return val_dict
        
      
    def set_mc_range(self,range_):
        self.mc_txt.SetRange(0,range_)
        
    def get_mc(self):
        """Vraca selektovan mc"""
        return re.search(r'MC(\d+)', self.cmb_dict['mc'].GetValue()).group(0)
        
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
        self.log.debug("on_draw_button")
        plt = {name:cmb.GetValue() for name,cmb in self.cmb_dict.items()}
        plt['mc'] = self.get_mc()
        self.data = self.controller.get_plot_data(**plt)
        self.log.debug("Loaded data for plot:\n %s" % self.data)
        self.draw_plot()
        
    def draw_legend(self,event):
        lbl_mc = self.get_mc()
        lbl_mc ="%s=%s" %("SP",util.extract_name(lbl_mc).upper())
        lbl_t ="%s=%.4f" %(util.extract_name(self.cmb_dict['t'].GetValue()).upper(),float(util.extract_int(self.cmb_dict['t'].GetValue())/10000.0))
        l = self.cmb_dict['l'].GetValue()
        namel = util.extract_name(l).upper()
        lbl_l ="%s=%s"% (namel,util.extract_int(l))
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
        print self.data.ix[item+'avg']*100
        print self.data.ix['stdMean'+item]
        self.error_line = self.ax_mag.errorbar(x=self.data.ix['THERM'],
                             y=self.data.ix[item + 'avg']*100,
                             yerr=self.data.ix['stdMean' + item]*100,fmt=fmt,fillstyle='none',
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

    
    @property
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

        self.cmb_dirs =  wx.ComboBox(
            self,
            -1,
            size=(150, -1),
            style=wx.CB_READONLY,
            value='--',
            )
        
        self.cmb_L = wx.ComboBox(
            self,
            -1,
            size=(150, -1),
            style=wx.CB_READONLY,
            value='--'
            )
         
        self.cmb_mag = wx.ComboBox(
            self,
            -1,
            size=(150, -1),
            style=wx.CB_READONLY,
            )
        self.draw_button = wx.Button(self, -1, 'Draw')
        self.Bind(wx.EVT_COMBOBOX, self.on_select_dir, self.cmb_dirs)
        self.Bind(wx.EVT_BUTTON, self.on_draw_button, self.draw_button)
        self.clear_button = wx.Button(self, -1, 'Clear')
        self.random_button = wx.Button(self, -1, 'Random')
        self.clear_button.Enable(True)
        self.Bind(wx.EVT_BUTTON, self.on_clear_button,
                  self.clear_button)

        self.Bind(wx.EVT_BUTTON, self.on_random_button,
                  self.random_button)

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
        self.hbox1.Add(self.cmb_dirs, border=5, flag=wx.ALL
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
        self.hbox1.Add(self.random_button, border=5, flag=wx.ALL
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
        dir_ = self.cmb_dirs.GetValue()
        mag_select = self.cmb_mag.GetValue()
        lbl = "$%s_{%s}$" %(L_select[0],L_select[1:])
        agg_data = self.controller.get_agg_plot_data(dir_)
        
        self.ax_agg.plot(agg_data[L_select].ix['T'],
                         agg_data[L_select].ix[mag_select],fmt_cycle.next(), label=lbl,fillstyle='none',picker=5)
        self.annotations = list()
        self.log.debug('agg_data: \n %s' %agg_data)
        index_ts = agg_data[L_select].ix['T'].index
        real_ts = agg_data[L_select].ix['T']
        mag_values = agg_data[L_select].ix[mag_select]
        for ti,m,tr in zip(index_ts,mag_values,real_ts):
            print 'ti:{} m:{} tr:{}'.format(ti,m,tr)
            text = self.controller.annotate_agg(dir_,L_select,ti)
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

    def on_random_button(self,event):
        dir_ = self.cmb_dirs.GetValue()
        self.controller.random_bestmats(dir_)
        
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
        print help(event)
        try:
            x,y = event.artist.xy

        except Exception as e:
            print e
        else:
            print x,y
            t =  'T%s' % int(x*10000)
            l = self.cmb_L.GetValue()
            print t,l
            #therm = 
        if event.mouseevent.button == 1:
            print event.artist.set_visible(False)
        elif event.mouseevent.button ==3:
            pass
        self.canvas.draw()

        
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

        ########################################
        ######### CONTROLER INTERFACE ##########


    def set_dir_choices(self,dirch):
        self.cmb_dirs.SetItems(dirch)
        try:
            curr_val = self.cmb_dirs.GetValue()
            val = dirch[0] if  curr_val=='--' or curr_val not in dirch else curr_val
        except IndexError as e:
            self.cmb_dirs.SetValue('--')
        except Exception as e:
            util.show_error('Some error!',str(e))

        else:
            self.cmb_dirs.SetValue(val)

            self.on_select_dir(val=val,changed=curr_val!=val)

    def on_select_dir(self,event=None,val=None,changed=True):
        if event!=None:
            val = event.GetEventObject().GetValue()
        self.controller.ap_dir_selected(val,changed)
            
    def set_l_choices(self,lch):
        self.cmb_L.SetItems(lch)
        try:
            val = self.cmb_L.GetValue()
            self.cmb_L.SetValue(lch[0] if val=='--' or not val or val not in lch else val)
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
        scat = ScatterPanel(self,controller)
        exp = ExpPanel(self,controller)
        self.controller.init_gui(tp,ag,scat)
        self.AddPage(tp, 'Therm')
        self.AddPage(ag, 'Aggregate')
        self.AddPage(scat,"Scatter")
        self.AddPage(exp,"Exp")
        self.Bind(wx.EVT_NOTEBOOK_PAGE_CHANGING, self.OnPageChanging)

    def flash_status_message(self,message):
        self.GetParent().flash_status_message(message,3000)

    def OnPageChanging(self,event):
        new = event.GetSelection()
        if new==2:
            self.controller.init_scatter()


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
        m_reload = menu_file.Append(-1,'&Reload\tCtrl-R','Reload')
        self.Bind(wx.EVT_MENU, self.on_exit, m_exit)
        self.Bind(wx.EVT_MENU,self.on_reload,m_reload)

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

    def on_reload(self,event):
        self.controller.remap_fsystem()

    def flash_status_message(self, msg, flash_len_ms=1500):
        self.statusbar.SetStatusText(msg)
        self.timeroff = wx.Timer(self)
        self.Bind(wx.EVT_TIMER, self.on_flash_status_off, self.timeroff)
        self.timeroff.Start(flash_len_ms, oneShot=True)

    def on_flash_status_off(self, event):
        self.statusbar.SetStatusText('')



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
        
        vbox.Add(splitter, 1,  wx.EXPAND | wx.TOP | wx.BOTTOM, 5 )
        hbox = wx.BoxSizer(wx.HORIZONTAL)
        hbox.AddStretchSpacer()

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

  

