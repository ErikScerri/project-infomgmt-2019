# -*- coding: utf-8 -*-
"""
Created on Mon Mar 18 09:34:46 2019

@author: Erik
"""
import numpy as np;

class KerasBatchGenerator(object):
    def __init__(self, input_data, output_data, num_steps, batch_size, output_range, skip_step=5):
        self.input_data = input_data;
        self.output_data = output_data;
        self.num_steps = num_steps;
        self.batch_size = batch_size;
        self.output_range = output_range;
        self.current_idx = 0;
        self.skip_step = skip_step;
        
    def generate(self):
        x = np.zeros((self.batch_size, self.num_steps), dtype=int);
        y = np.zeros((self.batch_size, self.num_steps, self.output_range), dtype=int);
        while True:
            for i in range(self.batch_size):
                if self.current_idx + self.num_steps >= len(self.input_data):
                    self.current_idx = 0;
                x[i, :] = self.input_data[self.current_idx:self.current_idx + self.num_steps]
                temp_y = self.output_data[self.current_idx:self.current_idx + self.num_steps]
                for j in range(self.num_steps):
                    y[i, j, temp_y[j]] = 1;
                self.current_idx += self.skip_step;
            yield x, y
  
#train_data_generator = KerasBatchGenerator(midi_files[16], patt_files[16], num_steps, batch_size, output_range, skip_step=num_steps);
#valid_data_generator = KerasBatchGenerator(midi_files[16], patt_files[16], num_steps, batch_size, output_range, skip_step=num_steps);

# =============================================================================
# print(midi_files[45])
# print(patt_files[45])
# print(len(midi_files[45]))
# print(len(patt_files[45]))
# =============================================================================