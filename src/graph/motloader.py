# -*- coding: utf-8 -*-
"""
Created on Wed Aug  4 18:11:34 2021

@author: Roman
"""

from spektral.data.loaders import BatchLoader
import numpy as np

class MOTLoader(BatchLoader):
    
    def __init__(self, dataset, mask=False, batch_size=1, epochs=None, shuffle=True):
        self.mask = mask
        super().__init__(dataset, batch_size=batch_size, epochs=epochs, shuffle=shuffle)
    
    def collate(self, batch):
        # fonctionne uniquement si batch_size vaut 1
        
        packed = super().pack(batch, return_dict=False)
        past_index = np.array([g.past_index for g in batch])
        futur_index = np.array([g.futur_index for g in batch])
        y = np.array([g.y for g in batch]) if "y" in self.dataset.signature else None
        
        output = list(packed[:-1]) + [past_index, futur_index]
        
        if y is None:
            return output
        else:
            return output, y