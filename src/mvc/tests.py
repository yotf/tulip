import unittest
import os
import sys
print __package__
print __name__
print __file__

sys.path.append(os.path.abspath(os.path.dirname(__file__)))
# import mvc_test

import types
import mvc_test

class ChoicesTest(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        """
        Initializes the model
        and sets the TDF path.
        Should add a list of simdirs
        for testing
        """
        tdfs = ["/media/123AEBFC3AEBDAAD/sim_output/"]
        tdf  = "/media/123AEBFC3AEBDAAD/sim_output/"
        cls.choices = mvc_test.Choices()
        cls.choices.simdir=tdf

    def __test_dict(self,dct,meq=None,leq=None,eq=None):
        """Ovo moze nekako da se automatizuje???
        TJ, da prosledimo operator kao string???
        """
        import collections
        self.assertTrue(isinstance(dct,collections.Mapping))
        for key in dct:
            self.assertTrue(isinstance(key,basestring))
        if meq:
            self.assertTrue(len(dct.keys())>=meq)
        elif leq:
            self.assertTrue(len(dct.keys())<=leq)
        else:
            self.assertTrue(len(dct.keys())==eq)


    def test_load_choices(self):
        """Checks if structure of read
        files is a good 'choices' dictionary
        structure
        ```
        +---DIR_  
        |   +---L1
        |   |   +---T1
                |  +--['mc':str.,'therm':str]
                   |
        
        |
        ...
        ```
        """
        dct = self.choices.load_choices('bestmat.dict')
        self.assertEqual(type(dct),types.DictType)
        if dct:
            self.__test_dict(dct,meq=1)
            for dir_ in dct.keys():
                self.__test_dict(dct[dir_],meq=1)
                for l_ in dct[dir_].keys():
                    self.__test_dict(dct[dir_][l_],meq=1)
                    for t_ in dct[dir_][l_].keys():
                        self.__test_dict(dct[dir_][l_][t_],eq=2)
                        self.assertEqual(['therm','mc'],dct[dir_][l_][t_].keys())

        print "Bestmat tests passed - you are awesome!!!"
        return True

    def test_init_model(self):
        """
        Ovde bi trebalo da je loadovano stanje
        tj. da bestmatdict ima nesto u njemu
        proverili smo da li ima odgovarajucu strukturu
        i proslom testu. i filesystem bi trebalo da je
        mapiran. Njega treba da testiram
        
        """
        self.choices.init_model()
        self.assertTrue(self.choices.files and self.choices.bestmats)
        print "init_model tests passed - you are awesome!"

    def test_map_filesystem(self):
        """ Treba da bude odredjena struktura
        pa treba da bude, bez prljavih gluposti
        + simdir1
        |
        +-simdir2
        """
        import collections
        files = self.choices.map_filesystem()
        for simd_name,l_map in files.iteritems():
            self.assertTrue(isinstance(simd_name,basestring))
            self.__test_dict(l_map,meq=1)
            for l,t_map in l_map.iteritems():
                self.assertTrue(isinstance(l,basestring))
                self.__test_dict(t_map,meq=1)
            for t,mc_map in t_map.iteritems():
                self.assertTrue(isinstance(t,basestring))
                self.__test_dict(mc_map,meq=1)
                for mc,therm_map in mc_map.iteritems():
                    self.assertTrue(isinstance(mc,basestring))
                    self.__test_dict(l_map,meq=1)


        print "Map filesystem tests passed - you are Awesome!")
                
                
            
        # znaci treba d prodjem kroz dictionary
        # i da vidim da li ima dobru strukturu
        
        


unittest.main()

    


        