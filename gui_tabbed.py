#!/usr/bin/python
# -*- coding: utf-8 -*-

import pdb
import wx
import pandas as pd
import os
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


regexf = re.compile(r'(L\d{1,2}).aplot$')
SIM_DIR = ""
DEBUG = "+++DEBUG INFO+++  "
DEBUGG = False
LATTICE_MC = os.getcwd()

def debug():
    if(DEBUGG):
        pdb.set_trace()

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

        self.fig = Figure(figsize=(20,7))
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
       
        self.draw_button = wx.Button(self, -1, 'Next T')
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

        self.step("dummy")
        # t=self.temprs.next()
        # x,y,z =  self.data.ix[t,'x'],self.data.ix[t,'y'],self.data.ix[t,'z']
        # # magt =np.sqrt( x ** 2 + y ** 2 + z ** 2)
  
        # self.scat = self.ax.scatter(x,y,z,s=6 )       
  
    def step(self,event):
        import time
        from matplotlib import cm
        
        t=self.temprs.next()
        x,y,z = self.data.ix[t,'x'],self.data.ix[t,'y'],self.data.ix[t,'z']
        magt =np.sqrt( x ** 2 + y ** 2 + z ** 2)
        print magt
        print np.mean(magt)
        colors = np.where(magt>np.mean(magt),'r','b')
        # self.scat._offsets3d = (x,y,z)
        self.ax_3d.cla()
        self.ax_hist.cla()

             
        self.scat =  self.ax_3d.scatter(x,y,z,s=10,c = magt,cmap=cm.RdYlBu)
        # self.scat=self.ax_3d.scatter3D(x,y,z,s=10,c=colors)
        self.ax_3d.set_title(t)
        self.ax_hist.set_ylim(0,100)
        self.ax_hist.hist(magt,bins=100,normed=1,facecolor='green',alpha=0.75)
        self.ax_hist.plot(kind="kde",style="k--")
       
       
        
        self.canvas.draw()
            
    def forceUpdate(self,event):
       
        self.scat.changed()




class ThermPanel(wx.Panel):

    xlabel = 'Thermalisation Cycles'

    best_mat_dict = dict()
    def __init__(self, parent):
        wx.Panel.__init__(self, parent=parent, id=wx.ID_ANY)
        self.parent = parent
        self.init_plot()
        # self.save_button = wx.Button(self, -1, 'Save plot')

        self.mc_txt = wx.TextCtrl(self, size = (80,-1))
        self.add_button = wx.Button(self,-1,'Add')
        self.bestmat_button=wx.Button(self,-1,"'s the best .mat ma..")
        self.Bind(wx.EVT_BUTTON, self.on_bestmat_button,self.bestmat_button)
        self.Bind(wx.EVT_BUTTON, self.on_add_button,self.add_button)
#        self.Bind(wx.EVT_BUTTON, self.on_save_button, self.save_button)
        plot_choices = ['M1', 'M2', 'M4']
        self.cmb_plots = wx.ComboBox(self, size=(150, -1),
                choices=plot_choices, style=wx.CB_READONLY,
                value=plot_choices[0])
        self.cmb_L = wx.ComboBox(self, size=(70, -1),
                                 choices=L_choices,
                                 style=wx.CB_READONLY,
                                 value=L_choices[0])
        cmb_T_choices = LT_combo[self.cmb_L.GetValue()]
        self.cmb_T = wx.ComboBox(self, size=(100, -1),
                                 choices=cmb_T_choices,
                                 value=cmb_T_choices[0])
        self.cmb_pfiles = wx.ComboBox(self, size=(300, -1),
                choices=self.get_files(), value='<Choose plot file>')

        self.cmb_mats = wx.ComboBox(self,size=(300,-1),choices = self.get_files(ext="*.mat"),
                                    value = '<Choose best mat>')
        
                                    

        self.Bind(wx.EVT_COMBOBOX, self.on_select, self.cmb_plots)
        self.Bind(wx.EVT_COMBOBOX, self.update_combos, self.cmb_L)
        self.Bind(wx.EVT_COMBOBOX, self.update_combos, self.cmb_T)
        self.Bind(wx.EVT_COMBOBOX, self.on_select_pfiles,
                  self.cmb_pfiles)
        self.hbox1 = wx.BoxSizer(wx.HORIZONTAL)
#        self.hbox1.Add(self.save_button, border=5, flag=wx.ALL
                     #  | wx.ALIGN_CENTER_VERTICAL)
        self.hbox1.AddSpacer(20)

        self.hbox1.Add(self.cmb_L, border=5, flag=wx.ALL
                       | wx.ALIGN_CENTER_VERTICAL)
        self.hbox1.Add(self.cmb_T, border=5, flag=wx.ALL
                       | wx.ALIGN_CENTER_VERTICAL)
        self.hbox1.Add(self.cmb_pfiles, border=5, flag=wx.ALL
                       | wx.ALIGN_CENTER_VERTICAL)
        self.hbox1.Add(self.cmb_plots, border=5, flag=wx.ALL
                       | wx.ALIGN_CENTER_VERTICAL)
        self.hbox1.Add(self.mc_txt, border=5, flag=wx.ALL
               | wx.ALIGN_CENTER_VERTICAL)
        self.hbox1.Add(self.add_button, border=5, flag=wx.ALL
                       | wx.ALIGN_CENTER_VERTICAL)
        self.hbox1.Add(self.cmb_mats, border=5, flag=wx.ALL
                       | wx.ALIGN_CENTER_VERTICAL)
        self.hbox1.Add(self.bestmat_button, border=5, flag=wx.ALL
                       | wx.ALIGN_CENTER_VERTICAL)
        
        
        
        self.toolhbox =wx.BoxSizer(wx.HORIZONTAL)
        self.toolhbox.Add(self.toolbar)
        

        self.vbox = wx.BoxSizer(wx.VERTICAL)
        self.vbox.Add(self.canvas, 1, flag=wx.LEFT | wx.TOP)
        self.vbox.Add(self.toolhbox)
        self.vbox.Add(self.hbox1, 0, flag=wx.ALIGN_LEFT | wx.TOP)
        self.SetSizer(self.vbox)
        self.vbox.Fit(self)

    def on_bestmat_button(self,event):
        LT = self.cmb_L.GetValue()+self.cmb_T.GetValue()
        self.best_mat_dict[LT]=self.cmb_mats.GetValue()
        print "BEST MAT DICT:",self.best_mat_dict
        self.parent.flash_status_message("Best .mat for %s selected" % LT)
    def on_add_button(self,event):
        
        mcs =int( self.mc_txt.GetValue())
        # gledamo sta je korisnik selektovao sto se tice L i T
        # i onda u tom folderu pravimo nove .mat fajlove
        # i radimo compose nad novim .mat fajlovima
        # ovde bi bilo dobro da napravim da se gleda .all fajl
        # sa najvise MC-ova i da se ogranici upis na to
        mcs_dir =join(SIM_DIR,self.cmb_L.GetValue()+self.cmb_T.GetValue())
        print "Making new plot files in dir %s for %d MCs" % (mcs_dir,mcs)
        statmat.main(mcs_dir,mcs)
        compose.main(mcs_dir,mcs)
        self.cmb_pfiles.SetItems(self.get_files())


    def update_combos(self,event):
        """Ova metoda azurira kombinacijske kutije koje zavise od stanja
        drugih elemenata/kontrola """
        self.cmb_T.SetItems(LT_combo[self.cmb_L.GetValue()])
        self.cmb_pfiles.SetItems(self.get_files())
        self.cmb_pfiles.SetValue('<Choose plot file>')
        self.cmb_mats.SetItems(self.get_files("*.mat"));
        self.cmb_mats.SetValue('<Choose best .mat  file>')


    def get_files(self,ext="*.plot"):
      
        """Vraca sve plot fajlove u zadatom folderu (L i T))"""
        folder_name = join(SIM_DIR,self.cmb_L.GetValue() +  self.cmb_T.GetValue())+os.path.sep
        print "folder name",folder_name
        print DEBUG,"glob for get_files",folder_name+ext
        files = glob.glob(folder_name + ext)
        print DEBUG,"files",files
        return files

    def on_select_pfiles(self, event):
        """Kada korisnik izabere .plot fajl, iscrtava se """
        path = self.cmb_pfiles.GetValue()
        self.data = pd.read_csv(path, index_col=0)
        #self.draw_plot()
        # gledamo da li je selektovano M M^2 ili M^4
        # tako ili mozemo vratiti vrednost uvek na M
        self.on_select("dummy")

    def init_plot(self):
        self.dpi = 100
        fig_width = self.parent.GetParent().width / self.dpi
        fig_height = self.parent.GetParent().height / self.dpi * (3 / 4)
        self.fig = Figure((fig_width, fig_height), dpi=self.dpi)



        self.ax_mag = self.fig.add_subplot(1, 2, 1)
        self.ax_cv = self.fig.add_subplot(1, 2, 2)

        self.ax_mag.set_title('Magnetisation')
        self.ax_cv.set_title('Coefficient of Variation')

        self.ax_mag.set_xlabel(self.xlabel)
        self.ax_cv.set_xlabel(self.xlabel)

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

        self.ax_mag.cla()
        self.ax_cv.cla()

        line = self.ax_mag.errorbar(x=self.data.ix['THERM'],
                             y=self.data.ix[item + 'avg'], linewidth=1,
                             color='#564453',
                             yerr=self.data.ix['stdMean' + item])[0]
###        line.set_markerfacecolor('w')
        self.ax_cv.semilogx(self.data.ix['THERM'], self.data.ix['cv(%s)'
                             % item], linewidth=1, color='#56ae53')
        self.ax_mag.set_xscale('log')
        self.ax_cv.grid(True, color='red', linestyle=':')
        self.ax_mag.grid(True, color='red', linestyle=':')
        self.ax_cv.set_xlabel('Thermalisation cycles')
        self.ax_mag.set_xlabel('Thermalisation cycles')
        self.ax_cv.set_ylabel(r'Coefficient of variation for $\langle{%s}\rangle$'
                               % item)
        self.ax_mag.set_ylabel(r'$\langle{%s}\rangle$' % item)
        self.canvas.draw()

    # def on_save_button(self, event):
    #     file_choices = ' EPS (*.eps)|*.eps| PNG (*.png)|*.png'

    #     dlg = wx.FileDialog(
    #         self,
    #         message='Save plot as...',
    #         defaultDir=os.getcwd(),
    #         defaultFile='plot.eps',
    #         wildcard=file_choices,
    #         style=wx.SAVE,
    #         )

    #     if dlg.ShowModal() == wx.ID_OK:
    #         path = dlg.GetPath()
    #         self.canvas.print_figure(path, dpi=self.dpi)
    #         self.flash_status_message('Saved to %s' % path)


class AggPanel(wx.Panel):

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
        self.clear_button.Enable(False)
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
        pass
        
    # def on_save_button(self, event):
    #     file_choices = ' EPS (*.eps)|*.eps| PNG (*.png)|*.png'
        
    #     dlg = wx.FileDialog(
    #         self,
    #         message='Save plot as...',
    #         defaultDir=os.getcwd(),
    #         defaultFile='plot.eps',
    #         wildcard=file_choices,
    #         style=wx.SAVE,
    #         )
        
    #     if dlg.ShowModal() == wx.ID_OK:
    #         path = dlg.GetPath()
    #         self.canvas.print_figure(path, dpi=self.dpi)
    #         self.flash_status_message('Saved to %s' % path)
        
    def on_draw_button(self, event):
        L_select = self.cmb_L.GetValue()
        mag_select = self.cmb_mag.GetValue()
        print L_select, mag_select
        print agg_data[L_select].ix['T']
        self.ax_agg.plot(agg_data[L_select].ix['T'],
                         agg_data[L_select].ix[mag_select],
                         linewidth=1, color='#563f33', label=L_select)
        debug()
        self.ax_agg.set_xlim(right=1.55)
        self.ax_agg.legend(loc='upper left')
        self.ax_agg.set_xlabel("Temperature")
        self.ax_agg.grid(True, color='red', linestyle=':')
        self.canvas.draw()
        
    def on_clear_button(self,event):
        self.ax_agg.cla()
        self.canvas.draw()
        
    def init_plot(self):
        self.dpi = 100
        fig_width = self.parent.GetParent().width / self.dpi
        fig_height = self.parent.GetParent().height / self.dpi * (3 / 4)
        self.fig = Figure((fig_width, fig_height), dpi=self.dpi)
        self.ax_agg = self.fig.add_subplot(111)
        
        self.ax_agg.set_title('Aggregate thingie')
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

    title = 'Naslov aplikacije...'

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
        self.notebook.SetPadding(wx.Size(self.width / 4.0
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
    import pickle
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
        print type(dir_md5)
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
        
   
#AGREGATE, mozda za kasnije treba
    # for L in Ls:
        
    #     print DEBUG,"Ls", Ls
    #     print L,L+"*"
    #     agregate.main(SIM_DIR,L,L+"T*")
    #     count= count+1
    # prbar.Destroy()
    # #posto smo generisali sve fajlove ovo se pise
    # #ako korisnik bude dodavao nove, opet ce se zvati
    # writemd5hash()

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
    debug()
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
    
if __name__ == '__main__':
    import sys
    app = wx.PySimpleApp()
    # dijalog za odabir simulacijske datoteke
    # ako se opovrgne- izlazimo
    # ovo sve treba u main metodu
    # samo cu morati da koristim global SIM_DIR direktivu
    dlg=wx.DirDialog(None,style=wx.DD_DEFAULT_STYLE,message="Where your simulation files are...")
    if dlg.ShowModal() == wx.ID_OK:
        SIM_DIR=dlg.GetPath()
    else:
        sys.exit(0)

    dlg.Destroy()
    #ovde cemo drzati stanje hash.txt fajla, kakvo je trenutno
    hfpath = join(LATTICE_MC,"md5_hash.dict")
    
    dir_md5  = read_hashf(hfpath)
    # ako nije dict nesto nije u redu! 
    assert type(dir_md5) is defaultdict
    # mozda bi bilo bolje da ovaj sim_dir samo prosledjujem
    # nem pojma
    handle_simfiles(dir_md5)

    # I OVO CE SE TEK KAD NAPRAVE AGREGATE IZVRSITI
    # TJ, TREBA DA POGLEDAMO DA LI POSTOJI DICTIONARY U FAJLU NEKOM
    agg_data = load_agg()

    L_choices = zip(*agg_data.columns)[0]
    L_choices = list(set(L_choices))

    LT_combo = dict()

    for L in L_choices:
        LT_combo[L] = [t for (l, t) in agg_data.columns if l == L]

    print DEBUG,"LT_combo", LT_combo

    mag_choices = list(agg_data.index)
    app.frame = GraphFrame()

    app.frame.Show()
    app.MainLoop()
    

