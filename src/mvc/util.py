import wx
import re
import numpy as np
import logging
import os
from os.path import join
def extract_int(string):
    """Vraca prvi int iz stringa, ako ga
    ima - u suprotnom vraca nulu!
    """
    try:
        return int(re.search(r'\d+',string).group())
    except:
        return 0 
    
def extract_name(string):
    """Vraca ime izbora
    L1010->l i tako"""
    try:
        name = re.search(r'(L|THERM|MC|T)',string).group().lower()
        assert name in ['l','t','therm','mc']
    except:
        show_error('Wrong choice format','New format? Please contact developer!')
    else:
        return name

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

class MyMessageBox(wx.MessageDialog):
    
    def __init__(self,message,title,style):
        wx.MessageBox.__init__(self,None,message,title,style)
        self.issue_button = wx.Button(self,-1,'Report Issue')
        self.Bind(wx.EVT_BUTTON,self.on_report_issue,self.issue_button)

    def on_report_issue(self,event):
        import webbrowser
        webbrowser.open('https://bitbucket.org/iTrustedYOu/tulip/issues/new')

def show_error(title,message):
    result = wx.MessageBox(message,title,style=wx.OK | wx.ICON_ERROR)


def absolute_listdir_dir(pardir,regex=None):
    """
    Given an absolute path to a directory, returns a list of
    the subdirectories contained within.
    If a regex was passed, returned directory names have to match it
    """
    
    dirs = [join(pardir,dir_) for dir_ in os.listdir(pardir)
                 if
            os.path.isdir(join(pardir, dir_))
            and
            (not regex or regex.search(dir_))]
    return dirs

def absolute_listfiles(pardir,regex=None):
    """
    Given an absolute path to direcotry, returns a list
    of files contained within that are not directories
    themselves. Ifa  regex is passed , it has to be
    matched.
    """
    files = [join(pardir,f) for f in os.listdir(pardir) if (not regex or regex.match(f))]
    return files
    