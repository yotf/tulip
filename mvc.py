class Subject(object):
    
    """Observer pattern"""

    def __init__(self):
        """Prosiri u podklasama"""
        self.observers = []
        """Funkcije koje ce se pozvati kad se podaci azuriraju"""

    def attach_observer(self,observer):
        """Prima callback funkciju koju
        dodaje na listu observera
        """
        self.observers.append(observer)
        
    def detach_observer(self,observer):
        self.observers.remove(observer)
        
    def notify_observers(self):
        for observer in self.observers:
            observer(subject=self)

class Model(Subject):
    """Stanje aplikacije i poslovna logika"""
    
class View(object):
    """Logika za prikaz"""
    def __init__(self,model,controller):
        self.model = model
        self.controller = controller
        self.model.attach_observer(self.model_updated)

    def model_updated(self,subject=None):
        """Ovu metodu poziva model kad se promeni,
        more se implementirati u podklasama"""
        raise NotImplementedError

class Controller(object):
    """Ova klasa se brine o korisnickom unosu"""
    model = None
    """Instanca 'Model'-a, koju dodeli 'init_model()'."""
    view = None
    """Instanca 'View'-a, koju pripremi 'init_view()'."""
    def init_components(self):
        """
        Priprema komponente modela i prikaza, i kaci kao observera
        modelu svoju metodu. Ovo treba da se pozove iz __init__ metode
        podklase
        """
        self.init_model()
        self.init_view()
        self.model.attach_observer(self.model_updated)

    def init_model(self):
        """
        Dodeljuje instancu 'Model'-a self.modelu, implementirati
        u podklasi
        """
        raise NotImplementedError
    def init_view(self):
        """
        Dodeljuje instancu 'View'-a self.view-u,implementirati
        u podklasi. Ali koliko cemo kontrolera imati. i zar ce svaki
        imati samo jedan view???
        """
        raise NotImplementedError
    def model_updated(self):
        """
        Ovo Model zove kad se updejtuje. Implementiraj u
        podklasama
        """
        raise NotImplementedError


