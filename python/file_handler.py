# -*- coding: utf-8 -*-
"""
Created on Mon Mar 18 09:16:29 2019

@author: Erik
"""
import os;
import music21 as m21;
import numpy as np;
import csv;
import random;
import collections as co;
import math;
from fractions import Fraction;
from collections import Counter;

def alphanum_key(s):
    return int(os.path.splitext(s)[0])

def load_simple_with_class(patt_len, patt_num, file_num, patt_dict, save=False):
    midi_files = np.empty(file_num, dtype=object);
    patt_files = np.empty(file_num, dtype=object);
    
    patt_dict["pat1"] = 0;
    patt_dict["pat2"] = 1;
    patt_dict["pat3"] = 2;
    
    for f in range(file_num):
        midi_array = np.zeros([patt_len * patt_num], dtype=int);
        patt_array = np.zeros([patt_len * patt_num], dtype=int);
        
        for i in range(patt_num):
            rand_pick = random.randint(1, len(patt_dict));
            
            patt_name = "pat" + str(rand_pick);        
            pitch = 60 + (patt_dict[patt_name] * 4);
            
            patt_start = i * patt_len;
            
            midi_array[patt_start:patt_start + patt_len] = pitch;
            patt_array[patt_start:patt_start + patt_len] = patt_dict[patt_name];
            
        midi_files[f] = midi_array;
        patt_files[f] = patt_array;
    
    return midi_files, patt_files

def save_simple(midi_files, patt_files, patt_len, patt_num, patt_dict, midi_path, patt_path):
    for i in range(len(midi_files)):
        print('Saving File ' + str(i) + '...', end='')
        stream = m21.stream.Stream();        
        for n in midi_files[i]:
            stream.append(m21.note.Note(n));        
        
        csv_list = [];
        for j in range(patt_num):
            patt_name = list(patt_dict.keys())[list(patt_dict.values()).index(patt_files[i][j * patt_len])]
            csv_list.append(co.OrderedDict([('Patt', patt_name), ('Len', patt_len)]));
            
        save_generated(stream, csv_list, midi_path, patt_path, str(i));
        print('DONE');
        

def load_simple_with_bounds(patt_len, patt_num, file_num, patt_dict):
    PATT_START = 1;
    PATT_END = 2;
    
    midi_files = np.empty(file_num, dtype=object);
    patt_files = np.empty(file_num, dtype=object);

    patt_dict["pat1"] = 0;
    patt_dict["pat2"] = 1;
    patt_dict["pat3"] = 2;
    
    for f in range(file_num):
        midi_array = np.zeros([patt_len * patt_num], dtype=int);
        patt_array = np.zeros([patt_len * patt_num], dtype=int);
        
        for i in range(patt_num):
            rand_pick = random.randint(1, len(patt_dict));
            
            patt_name = "pat" + str(rand_pick);        
            pitch = 60 + (patt_dict[patt_name] * 4);
            
            patt_start = i * patt_len;
            
            midi_array[patt_start:patt_start + patt_len] = pitch;
            patt_array[patt_start] = PATT_START;
            patt_array[patt_start + patt_len - 1] = PATT_END;
            
        midi_files[f] = midi_array;
        patt_files[f] = patt_array;
    
    return midi_files, patt_files
        

def load_midi_with_bounds(midi_path, patt_path, num_steps=4, sort=True, spq=1):
    PATT_START = 1;
    PATT_END = 2;
    
    path, dirs, files = next(os.walk(midi_path))

    if sort is True: files.sort(key=alphanum_key) 

    midi_files = np.empty([len(files)], dtype=object);
    patt_files = np.empty([len(files)], dtype=object);    

    file_count = 0;
    
    for f in files:
        print('Loading File ' + f + '...', end='')
        midi_stream = m21.midi.translate.midiFilePathToStream(midi_path + f);
        
        file_duration = int(midi_stream.duration.quarterLength) * spq;
        if (file_duration < num_steps):
            print('File under minimum length...SKIPPED')
            midi_files = np.delete(midi_files, -1)
            patt_files = np.delete(patt_files, -1)
            continue
            
        midi_array = np.zeros([file_duration], dtype=int);
        patt_array = np.zeros([file_duration], dtype=int);
        
        print('Parsing MIDI...', end='')
        
        i = 0;
        for n in midi_stream.flat.notesAndRests:
            note_len = int(float(n.duration.quarterLength) * spq);
            if (isinstance(n, m21.note.Note)):
                midi_array[i:i + note_len] = int(n.pitch.midi);
            elif (isinstance(n, m21.note.Rest)):
                midi_array[i:i + note_len] = 0;            
            i += note_len;        
            
        midi_files[file_count] = midi_array;
        
        print('Parsing CSV...', end='')
        
        patt_array[0] = PATT_START;
        patt_array[len(patt_array) - 1] = PATT_END;
         
        curr_idx = 0;
        with open(patt_path + 'gt' + os.path.splitext(f)[0] + '.csv') as csv_file:
            reader = csv.DictReader(csv_file, ['Patt', 'Len'])
            for row in reader:
                curr_len = int(convert_to_float(row['Len']) * spq);    
                curr_idx += curr_len;
                if(curr_idx > len(patt_array)): break;
                patt_array[curr_idx - 1] = PATT_END;
                if (curr_idx < len(patt_array)):
                    patt_array[curr_idx] = PATT_START;
         
        patt_files[file_count] = patt_array;
               
        print('DONE');
        file_count += 1;
    
    return midi_files, patt_files;

def load_midi_with_class(midi_path, patt_path, patt_dict, num_steps=4, sort=True, spq=1):
    patt_count = 0;
    
    path, dirs, files = next(os.walk(midi_path))

    if sort is True: files.sort(key=alphanum_key) 

    midi_files = np.empty([len(files)], dtype=object);
    patt_files = np.empty([len(files)], dtype=object);    

    file_count = 0;
    
    for f in files:
        print('Loading File ' + f + '...', end='')
        midi_stream = m21.midi.translate.midiFilePathToStream(midi_path + f);
        
        file_duration = int(midi_stream.duration.quarterLength) * spq;
        if (file_duration < num_steps):
            print('File under minimum length...SKIPPED')
            midi_files = np.delete(midi_files, -1)
            patt_files = np.delete(patt_files, -1)
            continue
            
        midi_array = np.zeros([file_duration], dtype=int);
        patt_array = np.zeros([file_duration], dtype=int);
        
        print('Parsing MIDI...', end='')
        
        i = 0;
        for n in midi_stream.flat.notesAndRests:
            note_len = int(float(n.duration.quarterLength) * spq);
            if (isinstance(n, m21.note.Note)):
                midi_array[i:i + note_len] = int(n.pitch.midi);
            elif (isinstance(n, m21.note.Rest)):
                midi_array[i:i + note_len] = 0;            
            i += note_len;        
            
        midi_files[file_count] = midi_array;
        
        print('Parsing CSV...', end='')
         
        curr_idx = 0;
        with open(patt_path + 'gt' + os.path.splitext(f)[0] + '.csv') as csv_file:
            reader = csv.DictReader(csv_file, ['Patt', 'Len'])
            for row in reader:
                curr_patt = row['Patt'];
                curr_len = int(convert_to_float(row['Len']) * spq);
                
                if curr_patt not in patt_dict:
                    patt_dict[curr_patt] = patt_count;                
                    patt_count += 1;
                    
                patt_array[curr_idx:curr_idx + curr_len] = patt_dict[curr_patt];
                curr_idx += curr_len;
        
        patt_files[file_count] = patt_array;
               
        print('DONE');
        file_count += 1;
    
    return midi_files, patt_files;

def load_midi(midi_path, num_steps=4, sort=True, spq=1):    
    path, dirs, files = next(os.walk(midi_path))

    if sort is True: files.sort(key=alphanum_key) 

    midi_files = np.empty([len(files)], dtype=object);
    patt_files = np.empty([len(files)], dtype=object);    

    file_count = 0;
    
    for f in files:
        print('Loading File ' + f + '...', end='')
        midi_stream = m21.midi.translate.midiFilePathToStream(midi_path + f);
        
        file_duration = int(midi_stream.duration.quarterLength) * spq;
        #if (file_duration < num_steps): file_duration = num_steps
        if (file_duration < num_steps):
            print('File under minimum length...SKIPPED')
            midi_files = np.delete(midi_files, -1)
            patt_files = np.delete(patt_files, -1)
            continue
            
        midi_array = np.zeros([file_duration], dtype=int);
        patt_array = np.zeros([file_duration], dtype=int);
        
        print('Parsing MIDI...', end='')
        
        i = 0;
        for n in midi_stream.flat.notesAndRests:
            note_len = int(float(n.duration.quarterLength) * spq);
            if (isinstance(n, m21.note.Note)):
                midi_array[i:i + note_len] = int(n.pitch.midi);
            elif (isinstance(n, m21.note.Rest)):
                midi_array[i:i + note_len] = 0;            
            i += note_len;        
            
        midi_files[file_count] = midi_array;        
        patt_files[file_count] = patt_array;
               
        print('DONE');
        file_count += 1;
    
    return midi_files, patt_files;

def convert_pitch_to_interval(midi_files):        
    for i in range(len(midi_files)):
        midi_array = np.zeros([len(midi_files[i])], dtype=int);
        
        for j in range(len(midi_files[i])):
            if j + 1 >= len(midi_files[i]): break;
            midi_array[j + 1] = midi_files[i][j + 1] - midi_files[i][j];         
        
        midi_files[i] = [x+128 for x in midi_array];
        
def parse_mtcann(motifs_path, midi_path, patt_path, spq=12):
    patt_dict = dict();
    patt_dict['None'] = 0;
    patt_count = 1;
    
    path, dirs, files = next(os.walk(midi_path));
    
    for f in files:            
        print('Parsing File ' + f + '...')
        
        midi_stream = m21.midi.translate.midiFilePathToStream(midi_path + f);
        
        file_duration = int(midi_stream.duration.quarterLength) * spq;
        
        patt_array = np.zeros([file_duration], dtype=int);
        
        song_id = os.path.splitext(f)[0];
    
        with open(motifs_path + 'motifs.csv', encoding="utf-8-sig") as csv_file:
            reader = csv.DictReader(csv_file, ['songid','begintime','endtime','duration','startindex','endindex','numberofnotes','motifclass'])
            for row in reader:
                curr_id = row['songid']
                if (song_id != curr_id): continue 
                curr_beg = int(convert_to_float(row['begintime']) * spq);
                curr_end = int(convert_to_float(row['endtime']) * spq);
                curr_patt = row['motifclass'];
                
                if curr_patt not in patt_dict:
                    patt_dict[curr_patt] = patt_count;                
                    patt_count += 1;
                
                patt_array[curr_beg:curr_end] = patt_dict[curr_patt];
                
        patt_split = np.split(patt_array, np.where(np.diff(patt_array))[0]+1)
        
        with open(patt_path + 'gt' + os.path.splitext(f)[0] + '.csv', mode='w') as csv_file:
            patt_writer = csv.writer(csv_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL, lineterminator = '\n')
            for arr in patt_split:
                patt_len = str(Fraction(len(arr)/spq));
                patt_idx = list(patt_dict.keys())[list(patt_dict.values()).index(arr[0])];
                patt_writer.writerow([patt_idx, patt_len]);

def load_midi_vocabulary_old(midi_path, patt_path, sort=False):
    
    path, dirs, files = next(os.walk(midi_path))

    if sort is True: files.sort(key=alphanum_key)
    
    midi_vocab = [];
    patt_vocab = [];
    
    for f in files:
        print('Loading File ' + f + '...', end='')
        midi_stream = m21.midi.translate.midiFilePathToStream(midi_path + f);
            
        midi_array = [];
        
        print('Parsing MIDI...', end='')
        
        for n in midi_stream.flat.notesAndRests:
            note_len = int(n.duration.quarterLength);
            for i in range(note_len):
                if (isinstance(n, m21.note.Note)):
                    midi_array.append(m21.note.Note(n.pitch, quarterLength=1));
                elif (isinstance(n, m21.note.Rest)):
                    midi_array.append(m21.note.Rest());   
        
        print('Parsing CSV...', end='')
         
        curr_idx = 0;
        with open(patt_path + 'gt' + os.path.splitext(f)[0] + '.csv') as csv_file:
            reader = csv.DictReader(csv_file, ['Patt', 'Len'])
            for row in reader:
                curr_patt = row['Patt'];
                curr_len = int(row['Len']);
                
                midi_vocab.append(midi_array[curr_idx:curr_idx + curr_len]);
                patt_vocab.append(curr_patt);
                curr_idx += curr_len;
               
        print('DONE');
    
    return midi_vocab, patt_vocab;

def load_midi_vocabulary(midi_path, patt_path, sort=False):
    
    path, dirs, files = next(os.walk(midi_path))

    if sort is True: files.sort(key=alphanum_key)
    
    midi_vocab = [];
    patt_vocab = [];
    
    for f in files:
        print('Loading File ' + f + '...', end='')
        midi_stream = m21.midi.translate.midiFilePathToStream(midi_path + f);
            
        midi_array = [];
        
        print('Parsing MIDI...', end='')
        
        for n in midi_stream.flat.notesAndRests:
            note_dict = dict();
            note_len = Fraction(n.duration.quarterLength)
            note_dict['Note'] = n;
            note_dict['Len'] = note_len;
            midi_array.append(note_dict);
            
        print('Parsing CSV...', end='')
        
        curr_idx = 0;
        with open(patt_path + 'gt' + os.path.splitext(f)[0] + '.csv') as csv_file:
            reader = csv.DictReader(csv_file, ['Patt', 'Len'])
            for row in reader:
                curr_patt = row['Patt'];
                if (curr_patt == 'None'): continue
                curr_len = Fraction(row['Len']);
                
                temp_idx = curr_idx
                temp_len = curr_len
                while temp_len > 0:
                    temp_len -= midi_array[temp_idx]['Len']
                    temp_idx += 1
                
                midi_vocab.append(midi_array[curr_idx:temp_idx - 1]);
                patt_vocab.append(curr_patt);                                    
                curr_idx = temp_idx
               
        print('DONE');
    
    return midi_vocab, patt_vocab;

def clean_vocabulary(midi_vocab, patt_vocab, keep):
    ctr = Counter(patt_vocab);
    
    print(ctr)
    
    mc = ctr.most_common(keep);
    
    keep_vocab = [];
    for c in mc:
        keep_vocab.append(c[0]);
        
    #print(keep_vocab)
    
    new_patt_vocab = [];
    new_midi_vocab = [];
    
    for i in range(len(patt_vocab)):
        if patt_vocab[i] in keep_vocab:
            new_patt_vocab.append(patt_vocab[i])
            new_midi_vocab.append(midi_vocab[i])
    
    return new_midi_vocab, new_patt_vocab    

def create_midi_from_vocabulary_old(midi_vocab, patt_vocab, amount_min, amount_max):
    create_len = random.randint(amount_min, amount_max);
    stream = m21.stream.Stream();
    csv_list = [];
    
    print('Pattern Length: ' + str(create_len));    
    print('Vocab Length: ' + str(len(midi_vocab)))
    
    pick_log = []
    
    i = 0;
    while i < create_len:
        rand_pick = random.randint(0, len(midi_vocab) - 1);        
        print('Pick: ' + str(rand_pick))
        if rand_pick not in pick_log:  
            pick_log.append(rand_pick);      
            stream.append(midi_vocab[rand_pick]);
            csv_list.append(co.OrderedDict([('Patt', patt_vocab[rand_pick]), ('Len', len(midi_vocab[rand_pick]))]));
            i += 1;
        else:
            print('Repeated Skip');
            continue;            
    
    print(stream.duration.quarterLength)
    return stream, csv_list;

def create_midi_from_vocabulary(midi_vocab, patt_vocab, amount_min, amount_max):
    create_len = random.randint(amount_min, amount_max);
    stream = m21.stream.Stream();
    csv_list = [];
    
    print('Pattern Length: ' + str(create_len));    
    print('Vocab Length: ' + str(len(midi_vocab)))
    
    pick_log = []
    
    i = 0;
    while i < create_len:
        rand_pick = random.randint(0, len(midi_vocab) - 1);        
        print('Pick: ' + str(rand_pick))
        if rand_pick not in pick_log:  
            pick_log.append(rand_pick);            
            patt_len = 0;
            
            for n in midi_vocab[rand_pick]:
                stream.append(n['Note']);
                patt_len += n['Len']
            csv_list.append(co.OrderedDict([('Patt', patt_vocab[rand_pick]), ('Len', patt_len)]));
            i += 1;
        else:
            print('Repeated Skip');
            continue;            
    
    print(stream.duration.quarterLength)
    return stream, csv_list;

def generate_midis(midi_vocab, patt_vocab, midi_path, patt_path, gen_num, gen_base=None):
    AMT_MIN = 8;
    AMT_MAX = 16;
    
    path, dirs, files = next(os.walk(midi_path))

    files.sort(key=alphanum_key)
    
    for i in range(gen_num):
        stream, csv_list = create_midi_from_vocabulary(midi_vocab, patt_vocab, AMT_MIN, AMT_MAX);
        print('Saving File ' + str(i) + '...', end='')
        if gen_base is not None:
            save_generated(stream, csv_list, midi_path, patt_path, gen_base + "_" + str(i))
        else:    
            base = int(os.path.splitext(files[-1])[0]) + 1;
            save_generated(stream, csv_list, midi_path, patt_path, str(base + i));
        print('DONE');
        

def save_generated(stream, csv_list, midi_path, patt_path, file_name):
    midi_file = m21.midi.translate.streamToMidiFile(stream);
    midi_file.open(midi_path + file_name + '.mid', 'wb');
    midi_file.write();
    midi_file.close();
    
    with open(patt_path + 'gt' + file_name + '.csv', 'w', newline='') as csv_file:
        writer = csv.DictWriter(csv_file, ['Patt', 'Len'])
        
        for row in csv_list:
            writer.writerow(row);
            
def convert_to_float(frac_str):
    try:
        return float(frac_str)
    except ValueError:
        num, denom = frac_str.split('/')
        try:
            leading, num = num.split(' ')
            whole = float(leading)
        except ValueError:
            whole = 0
        frac = float(num) / float(denom)
        return whole - frac if whole < 0 else whole + frac
    
def shuffle_together(array_a, array_b):
    rng_state = np.random.get_state()
    np.random.shuffle(array_a)
    np.random.set_state(rng_state)
    np.random.shuffle(array_b)
    
def ceil(number, bound=1):
    return bound * math.ceil(number / bound)
    
        
        
    