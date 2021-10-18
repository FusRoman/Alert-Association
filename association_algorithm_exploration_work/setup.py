# -*- coding: utf-8 -*-
"""
Created on Wed Aug  4 17:20:14 2021

@author: Roman
"""

from distutils.core import setup

setup(name='SSOTrajectoriesPrediction',
      version='0.1',
      description='A python package that use deep learning to associate solar system candidates alert produce by Fink and create trajectory',
      author='Roman le Montagner',
      author_email='roman.lemontagner@gmail.com',
      url='',
      packages=['src', 'src.graph', 'src.model', 'src.script',
      "src.model.LSTM.Siamese_LSTM", "src.model.LSTM.Quality_LSTM"]
      )