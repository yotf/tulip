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

    def setUp(self):
        """
        Initializes the model
        and sets the TDF path.
        Should add a list of simdirs
        for testing
        """
        simdirs = ["/media/123AEBFC3AEBDAAD/sim_output/"]
        simdir = "/media/123AEBFC3AEBDAAD/sim_output/"
        self.choices = mvc_test.Choices()
        self.choices.simdir=simdir

    def __test_dict(self,dct,meq=None,leq=None,eq=None):
        """Ovo moze nekako da se automatizuje???
        TJ, da prosledimo operator kao string???
        """
        self.assertEqual(type(dct),types.DictType)
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


unittest.main()

    


        