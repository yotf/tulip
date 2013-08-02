import wx
import re
def extract_int(string):
    """Vraca prvi int iz stringa
    nadam se
    """
    return int(re.search(r'\d+',string).group())
    
def extract_name(string):
    """Vraca ime izbora
    L1010->l i tako"""
    name = re.search(r'(L|THERM|MC|T)',string).group().lower()
    assert name in ['l','t','therm','mc']
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

def show_error(title,message):
    result = wx.MessageBox(message,title,style=wx.OK | wx.ICON_ERROR)