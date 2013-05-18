#!/usr/bin/python
# -*- coding: utf-8 -*-

import pdb
import wx
import pandas as pd
import os
from os.path import join
import glob
import re

import matplotlib
matplotlib.use('WXAgg')
from matplotlib.figure import Figure
from matplotlib.backends.backend_wxagg import FigureCanvasWxAgg as FigCanvas, \
    NavigationToolbar2WxAgg as NavigationToolbar
import numpy as np
import pylab
import mpl_toolkits.mplot3d.axes3d as p3

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
        self.load_data()
        self.fig = Figure()
        self.ax = p3.Axes3D(self.fig)
        self.canvas = FigCanvas(self,-1,self.fig)
        self.ax.mouse_init()

       
        self.init_gui()
        self.ts = self.all_data.keys()
        key = lambda x: int(x[1:])
        sorted(self.ts,key = lambda x: int(x[1:]))
        self.ts.reverse()
        print DEBUG,"self.ts reversed",self.ts


    def init_gui(self):
        self.vbox = wx.BoxSizer(wx.VERTICAL)
        self.vbox.Add(self.canvas, 1, flag=wx.LEFT | wx.TOP)
       
        self.draw_button = wx.Button(self, -1, 'Draw plot')
        self.Bind(wx.EVT_BUTTON, self.animate, self.draw_button)
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
        t=self.ts[0]
        self.scat = self.ax.scatter(self.data.ix[t,'x'],self.data.ix[t,'y'],self.data.ix[t,'z'],s=1,cmap =cm.jet )       

    def animate(self,event):
        self.setup_plot()
        import time
            
        for t in self.ts:
            x,y,z = self.data.ix[t,'x'],self.data.ix[t,'y'],self.data.ix[t,'z']
            self.scat._offsets3d = (x,y,z)
            self.ax.set_title(t)
            self.canvas.draw()
            time.sleep(1)



class ThermPanel(wx.Panel):

    xlabel = 'Thermalisation Cycles'

    def __init__(self, parent):
        wx.Panel.__init__(self, parent=parent, id=wx.ID_ANY)
        self.parent = parent
        self.init_plot()
        self.save_button = wx.Button(self, -1, 'Save plot')

        self.Bind(wx.EVT_BUTTON, self.on_save_button, self.save_button)
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

        self.Bind(wx.EVT_COMBOBOX, self.on_select, self.cmb_plots)
        self.Bind(wx.EVT_COMBOBOX, self.on_select_L, self.cmb_L)
        self.Bind(wx.EVT_COMBOBOX, self.on_select_T, self.cmb_T)
        self.Bind(wx.EVT_COMBOBOX, self.on_select_pfiles,
                  self.cmb_pfiles)
        self.hbox1 = wx.BoxSizer(wx.HORIZONTAL)
        self.hbox1.Add(self.save_button, border=5, flag=wx.ALL
                       | wx.ALIGN_CENTER_VERTICAL)
        self.hbox1.AddSpacer(20)

        self.hbox1.Add(self.cmb_L, border=5, flag=wx.ALL
                       | wx.ALIGN_CENTER_VERTICAL)
        self.hbox1.Add(self.cmb_T, border=5, flag=wx.ALL
                       | wx.ALIGN_CENTER_VERTICAL)
        self.hbox1.Add(self.cmb_pfiles, border=5, flag=wx.ALL
                       | wx.ALIGN_CENTER_VERTICAL)
        self.hbox1.Add(self.cmb_plots, border=5, flag=wx.ALL
                       | wx.ALIGN_CENTER_VERTICAL)
        self.toolhbox =wx.BoxSizer(wx.HORIZONTAL)
        self.toolhbox.Add(self.toolbar)
        

        self.vbox = wx.BoxSizer(wx.VERTICAL)
        self.vbox.Add(self.canvas, 1, flag=wx.LEFT | wx.TOP)
        self.vbox.Add(self.toolhbox)
        self.vbox.Add(self.hbox1, 0, flag=wx.ALIGN_LEFT | wx.TOP)
        self.SetSizer(self.vbox)
        self.vbox.Fit(self)

    def on_select_L(self, event):
        self.cmb_T.SetItems(LT_combo[self.cmb_L.GetValue()])
        self.cmb_pfiles.SetItems(self.get_files())
        self.cmb_pfiles.SetValue('<Choose plot file>')

    def on_select_T(self, event):
        self.cmb_pfiles.SetItems(self.get_files())
        self.cmb_pfiles.SetValue('<Choose plot file>')

    def get_files(self):
        from os.path import join
        """Vraca sve plot fajlove u zadatom folderu (L i T))"""
        folder_name = join(SIM_DIR,self.cmb_L.GetValue() +  self.cmb_T.GetValue())+os.path.sep
        print "folder name",folder_name
        print DEBUG,"glob for get_files",folder_name+'*.plot'
        files = glob.glob(folder_name + '*.plot')
        print DEBUG,"files",files
        return files

    def on_select_pfiles(self, event):
        """Kada korisnik izabere .plot fajl, iscrtava se """
        path = self.cmb_pfiles.GetValue()
        self.data = pd.read_csv(path, index_col=0)
        self.draw_plot()

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

    def on_save_button(self, event):
        file_choices = ' EPS (*.eps)|*.eps| PNG (*.png)|*.png'

        dlg = wx.FileDialog(
            self,
            message='Save plot as...',
            defaultDir=os.getcwd(),
            defaultFile='plot.eps',
            wildcard=file_choices,
            style=wx.SAVE,
            )

        if dlg.ShowModal() == wx.ID_OK:
            path = dlg.GetPath()
            self.canvas.print_figure(path, dpi=self.dpi)
            self.flash_status_message('Saved to %s' % path)


class AggPanel(wx.Panel):

    def __init__(self, parent):

        wx.Panel.__init__(self, parent=parent, id=wx.ID_ANY)
        self.parent = parent
        self.init_plot()
        self.init_gui()        
        
    def init_gui(self):
        self.cmb_L = wx.ComboBox(
            self,
            -1,
            value=L_choices[0],
            size=(150, -1),
            choices=L_choices,
            style=wx.CB_READONLY,
            )
        self.cmb_mag = wx.ComboBox(
            self,
            -1,
            value=mag_choices[0],
            size=(150, -1),
            choices=mag_choices,
            style=wx.CB_READONLY,
            )
        self.draw_button = wx.Button(self, -1, 'Draw plot')
        self.Bind(wx.EVT_BUTTON, self.on_draw_button, self.draw_button)
        self.clear_button = wx.Button(self, -1, 'Clear plot')
        self.Bind(wx.EVT_BUTTON, self.on_clear_button,
                  self.clear_button)
        
        self.hbox1 = wx.BoxSizer(wx.HORIZONTAL)
        
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
        
    def on_save_button(self, event):
        file_choices = ' EPS (*.eps)|*.eps| PNG (*.png)|*.png'
        
        dlg = wx.FileDialog(
            self,
            message='Save plot as...',
            defaultDir=os.getcwd(),
            defaultFile='plot.eps',
            wildcard=file_choices,
            style=wx.SAVE,
            )
        
        if dlg.ShowModal() == wx.ID_OK:
            path = dlg.GetPath()
            self.canvas.print_figure(path, dpi=self.dpi)
            self.flash_status_message('Saved to %s' % path)
        
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

    
def getmd5hash():
    """Vraca trenutni md5hash sim direktorijuma"""
    import hashlib
    m = hashlib.md5()
    for root, dirs, files in os.walk(SIM_DIR):
        for file_read in files:
            full_path = join(root, file_read)
            for line in open(full_path).readlines():
                m.update(line)
    return  m.hexdigest()

def writemd5hash():
    """Kada je zavrsio sa statistickom obradom datoteka
    dodaje u fajl  podatak u obliku --> PUTANJA_DO_OBRADJENOG_SIM_DIR:NJEGOV_MD5"""
    hashmd5 = getmd5hash()
    with open(join(LATTICE_MC,"hash.txt"),mode="a") as file:
        file.write("%s:%s\n" %(SIM_DIR,hashmd5))


def del_old_md5hash():
    # mm, nisam sigurna koji je nabolji nacin da izbrisem liniju u fajlu
    # ako se ne podudaraju podaci onda brisemo (ako postoji) tu liniju iz fajla
    # posto cemo dodati svoje podatke. Samo da li je ovo dobro, pa jeste, posto svejedno
    # cemo morati ponovo obradjivati fajlove, i ovaj zapis je ocigledno zastareo tako
    # da nam ne treba tu, a kad zavrsi ce regularno ispisati tamo onda
    import fileinput
    for line in fileinput.input(join(LATTICE_MC,"hash.txt"),inplace=True):
        if line.split(":")[0]!=SIM_DIR:
            print line,


def remove_old_calcs():
    """Posto je utvrdio da je direktorijum sa simulacijama
    izmenjem, izbrise sva prethodna izracunavanja. Ovo treba
    da napravim na nivou pojedinacnih LT direktorijuma"""
    from subprocess import call
    os.chdir(SIM_DIR)
    os.system("rm *.aplot")
    os.system( "for dir in L[0-9]*T[0-9]* ;do  [ -d $dir ] || continue; cd $dir; rm -f *.mat *.stat *.raw *.cv *.plot; cd ..; done | bash -x")
#    os.system("ls -1 | grep -v '.all$' | xargs -I {} rm {}")
    
def check_modified():
    """Cita iz datoteke koja ima linije u sledecem obliku:
    PUTANJA_DO_SIM_DATOTEKE:NJEN_MD5 i gleda da li ijedan zapis
    odgovara paru na kom trenutno radimo. Ako postoji putanja
    a md5 je razlicit - brise se zapis"""
    hashmd5 = getmd5hash()
    with open(join(LATTICE_MC,"hash.txt"),mode = "r") as file:
        for line in file.readlines():
            if line.split(":")==[SIM_DIR,hashmd5]:
                return False
    del_old_md5hash()
    remove_old_calcs()
    return True

def handle_sfiles ():
    import unify
    import agregate
    import compose
    import statmat
    rxdir = re.compile(r'^(L\d*)T(\d*)$')

    """Obradjuju se sve datoteke simulacije, ako je potrebno"""
    # lista svih direktorijuma ako se poklapaju sa patternom
    # drugi nacin:
    #dlist = filter(rxdir.match,os.listdir(SIM_DIR))
    #print dlist
    dlist = [d  for d in os.listdir(SIM_DIR) if rxdir.match(d) and os.path.isdir(join(SIM_DIR,d))]
    dlist.sort()
    print dlist
    Ls = [rxdir.match(d).groups()[0] for d in dlist]
    Ls = list(set(Ls))
    maxi = len(dlist)+len(Ls)
    print DEBUG,"prog bar len: ", maxi
    prbar = wx.ProgressDialog("Please wait, doing statistics 'n stuff...",message="starting",maximum=maxi,parent=None,style=0| wx.PD_APP_MODAL| wx.PD_CAN_ABORT)
    prbar.SetMinSize(wx.Size(350,100))
    dlg.Destroy()
    # preko brojaca updejtujemo progressbar
    count = 0
    for d in dlist:
        #trenutni direktorijum koji obradjujemo
        ltdir=join(SIM_DIR,d)
        print ltdir
        prbar.Update(count,d)
        count = count+1
        unify.main(ltdir)
        statmat.main(ltdir)
        compose.main(ltdir=ltdir)
    for L in Ls:
        prbar.Update(count,"Aggregating: %s" % L)
        print DEBUG,"Ls", Ls
        print L,L+"*"
        agregate.main(SIM_DIR,L,L+"T*")
        count= count+1
    prbar.Destroy()
    #posto smo generisali sve fajlove ovo se pise
    #ako korisnik bude dodavao nove, opet ce se zvati
    writemd5hash()

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
    # ako su izmenjeni fajlovi..
    if(check_modified()):
        handle_sfiles()
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
    

