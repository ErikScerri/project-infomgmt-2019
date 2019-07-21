# -*- coding: utf-8 -*-
"""
Created on Mon Jan 21 09:22:53 2019

@author: Erik
"""
import sys;
import numpy as np
import keras as kr
import file_handler as fh
import keras_batch_generator as kgb
import datetime
import json

np.set_printoptions(threshold=sys.maxsize)

data_path = "data/";
cij115_orig_path = "data/symbolic/CIJ115/orig/";
cij115_midi_path = "data/symbolic/CIJ115/mid/";
cij115_patt_path = "data/symbolic/CIJ115/csv/";
mtcann_path = "data/symbolic/MTCANN/";
mtcann_midi_path = "data/symbolic/MTCANN/mid/";
mtcann_patt_path = "data/symbolic/MTCANN/csv/";
mtcgen_midi_path = "data/symbolic/MTCGEN/mid/";
mtcgen_patt_path = "data/symbolic/MTCGEN/csv/";
mtcred_test_path = "data/symbolic/MTCRED/tst/";
mtcred_midi_path = "data/symbolic/MTCRED/mid/";
mtcred_patt_path = "data/symbolic/MTCRED/csv/";
gencls_midi_path = "data/symbolic/gencls/mid/";
gencls_patt_path = "data/symbolic/gencls/csv/";
logs_path = "data/logs/";
chkp_path = "data/checkpoints/";

num_steps = 8;        
batch_size = 3;
skip_step = 8; 

midi_pitch_range = 128;
interval_range = 256;
output_range = 3;
    
hidden_size = 128;
use_dropout = True;
kfold_splits = 10;   

patt_dict = dict();

# Loading of different datasets with Bounds
#midi_files, patt_files = fh.load_simple_with_bounds(8, 16, 50, patt_dict);
#midi_files, patt_files = fh.load_midi_with_bounds(cij115_midi_path, cij115_patt_path, num_steps);
#midi_files, patt_files = fh.load_midi_with_bounds(cij115_orig_path, cij115_patt_path, num_steps);
#midi_files, patt_files = fh.load_midi_with_bounds(gencls_midi_path, gencls_patt_path, num_steps);
#midi_files, patt_files = fh.load_midi_with_bounds(mtcgen_midi_path, mtcgen_patt_path, num_steps, sort=False, spq=12);
#midi_files, patt_files = fh.load_midi_with_bounds(mtcred_midi_path, mtcred_patt_path, num_steps, sort=False, spq=12);
#midi_files, patt_files = fh.load_midi_with_bounds(mtcann_midi_path, mtcann_patt_path, num_steps, sort=False, spq=12);

# Loading of different datasets with Class
#midi_files, patt_files = fh.load_simple_with_class(24, 4, 50, patt_dict);
midi_files, patt_files = fh.load_midi_with_class(cij115_midi_path, cij115_patt_path, patt_dict, num_steps);
#midi_files, patt_files = fh.load_midi_with_class(cij115_orig_path, cij115_patt_path, patt_dict, num_steps);
#midi_files, patt_files = fh.load_midi_with_class(gencls_midi_path, gencls_patt_path, patt_dict, num_steps);
#midi_files, patt_files = fh.load_midi_with_class(mtcred_midi_path, mtcred_patt_path, patt_dict, num_steps, sort=False, spq=12);
#midi_files, patt_files = fh.load_midi_with_class(mtcgen_midi_path, mtcgen_patt_path, patt_dict, num_steps, sort=False, spq=12);
#midi_files, patt_files = fh.load_midi_with_class(mtcann_midi_path, mtcann_patt_path, patt_dict, num_steps, sort=False, spq=12);

#midi_vocab, patt_vocab = fh.load_midi_vocabulary(cij115_orig_path, cij115_patt_path);
#midi_vocab, patt_vocab = fh.load_midi_vocabulary(mtcann_midi_path, mtcann_patt_path);
#midi_vocab, patt_vocab = fh.clean_vocabulary(midi_vocab, patt_vocab, 5);

#midi_files, patt_files = fh.load_midi(mtcred_test_path, num_steps, sort=False, spq=12)

#fh.convert_pitch_to_interval(midi_files);

#fh.generate_midis(midi_vocab, patt_vocab, mtcred_midi_path, mtcred_patt_path, 100, gen_base='MTC');
#fh.generate_midis(midi_vocab, patt_vocab, mtcgen_midi_path, mtcgen_patt_path, 100, gen_base='MTC');
#fh.save_simple(midi_files, patt_files, 24, 4, patt_dict, gencls_midi_path, gencls_patt_path);  

#fh.parse_mtcann(mtcann_path, mtcann_midi_path, mtcann_patt_path);

model = kr.models.Sequential();
model.add(kr.layers.Embedding(midi_pitch_range, hidden_size, input_length=num_steps));
model.add(kr.layers.LSTM(hidden_size, return_sequences=True))
model.add(kr.layers.LSTM(hidden_size, return_sequences=True))
if use_dropout:
    model.add(kr.layers.Dropout(0.5));
    
model.add(kr.layers.TimeDistributed(kr.layers.Dense(output_range)));
model.add(kr.layers.Activation('softmax'));

adam = kr.optimizers.Adam();
sgd = kr.optimizers.SGD(lr=0.01,clipvalue=0.5);
model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['categorical_accuracy']);
print(model.summary());

log_date = datetime.datetime.now().strftime('%Y-%m-%d');
print(log_date)

model.save(data_path + "init_weights.h5");

checkpointer = kr.callbacks.ModelCheckpoint(filepath=chkp_path + 'model-{epoch:02d}.hdf5', verbose=1)
num_epochs = 25;

ARG = 2;

if ARG == 1:
    for i in range(len(midi_files)):        
        print('Training File ' + str(i) + '...')
        
        tmp_batch_size = batch_size
        if (len(midi_files[i]) < ((batch_size - 1) * num_steps)): tmp_batch_size = 1
        
        train_data_generator = kgb.KerasBatchGenerator(midi_files[i], patt_files[i], num_steps, tmp_batch_size, output_range, skip_step=skip_step);
        valid_data_generator = kgb.KerasBatchGenerator(midi_files[i], patt_files[i], num_steps, tmp_batch_size, output_range, skip_step=skip_step);

    model.save(data_path + "final_model.hdf5");
elif ARG == 2:
    test_num = "s3tRanB";
    
    test_dict = dict();
    test_dict["PATT_CONT"] = 0;
    test_dict["PATT_START"] = 1;
    test_dict["PATT_END"] = 2;
    #test_dict["pat1"] = 0;
    #test_dict["pat2"] = 1;
    #test_dict["pat3"] = 2;
    
    split_precision = np.empty((kfold_splits,len(test_dict)));
    split_recall = np.empty((kfold_splits,len(test_dict)));
    split_output = np.empty((kfold_splits,2));
    
    for k in range(kfold_splits):
        print(test_num + "k" + str(k)); 
        
        conf_mtx = np.zeros((len(test_dict),len(test_dict)), dtype=int);
        total_predict = np.zeros(len(test_dict), dtype=int);
        total_actual = np.zeros(len(test_dict), dtype=int);    
        
        model = kr.models.load_model(data_path + "archive/" + test_num + "/final_model_" + str(k) + ".hdf5");
        
        file = open(data_path + "archive/" + test_num + "/output_fold" + str(k) + ".txt","w+")  
        
        for i in range(len(midi_files)):
            print("Testing File " + str(i) + "...");
            example_training_generator = kgb.KerasBatchGenerator(midi_files[i], patt_files[i], num_steps, 1, output_range, skip_step=1);

            true_print_out = "Actual words: "
            pred_print_out = "Argmax words: "
            
            num_predict = len(midi_files[i]) - num_steps;
            for j in range(num_predict):
                data = next(example_training_generator.generate())
                prediction = model.predict(data[0])
                predict_word = np.argmax(prediction[:, num_steps-1, :])
                #predict_word = random.randint(0, len(test_dict) - 1);                
                actual_word = patt_files[i][j + num_steps]                
                conf_mtx[predict_word][actual_word] += 1
                total_predict[predict_word] += 1;
                total_actual[actual_word] += 1;
                
                true_print_out += str(actual_word) + " "
                pred_print_out += str(predict_word) + " "
            
            #print(midi_files[i])
            #print(patt_files[i])
            #print(true_print_out)
            #print(pred_print_out)
            
            file.write(true_print_out);
            file.write('\n');
            file.write(pred_print_out);
            file.write('\n\n');
    
        for f in range(len(test_dict)):
            split_recall[k][f] = conf_mtx[f][f]/total_actual[f];
            split_precision[k][f] = conf_mtx[f][f]/total_predict[f];
            
        file.close();
            
    file = open(data_path + "archive/" + test_num + "/recall.csv","w+")    
    for r in split_recall:
        file.write(str(r[0]) + "," + str(r[1]) + "," + str(r[2]))
        file.write('\n');
    file.close();
    
    file = open(data_path + "archive/" + test_num + "/precision.csv","w+")    
    for r in split_precision:
        file.write(str(r[0]) + "," + str(r[1]) + "," + str(r[2]))
        file.write('\n');
    file.close();
elif ARG == 4:
    test_num = "s2tRealC";
    
    test_dict = patt_dict;
    
    for k in range(kfold_splits):
        print(test_num + "k" + str(k)); 
        
        model = kr.models.load_model(data_path + "archive/" + test_num + "/final_model_" + str(k) + ".hdf5");
        
        file = open(data_path + "archive/" + test_num + "/output_fold" + str(k) + ".txt","w+")  
        
        for i in range(len(midi_files)):
            print("Testing File " + str(i) + "...");
            example_training_generator = kgb.KerasBatchGenerator(midi_files[i], patt_files[i], num_steps, 1, output_range, skip_step=1);

            pred_print_out = "Argmax words: "
            
            num_predict = len(midi_files[i]) - num_steps;
            for j in range(num_predict):
                data = next(example_training_generator.generate())
                prediction = model.predict(data[0])
                predict_word = np.argmax(prediction[:, num_steps-1, :])
                #predict_word = random.randint(0, len(test_dict) - 1);                
                actual_word = patt_files[i][j + num_steps]
                
                pred_print_out += str(predict_word) + " "
            
            #print(midi_files[i])
            #print(patt_files[i])
            #print(true_print_out)
            #print(pred_print_out)
            
            file.write(pred_print_out);
            file.write('\n\n');
            
        file.close();                
elif ARG == 3:
    cvscores = []
    fh.shuffle_together(midi_files, patt_files)
    
    midi_files_split = np.array_split(midi_files, kfold_splits)
    patt_files_split = np.array_split(patt_files, kfold_splits)
    
    #print(midi_files_split[0])
    #print(patt_files_split[0])
    
    for k in range(kfold_splits):
        logs_name = logs_path + 'logs' + log_date + '_fold' + str(k);        
        csv_logger = kr.callbacks.CSVLogger(filename=logs_name + '.csv', separator=',', append=True)    
        
        file = open(logs_name + ".txt","w+")
        
        model = kr.models.load_model(data_path + "init_weights.h5");
        
        midi_temp = midi_files_split.copy()
        patt_temp = patt_files_split.copy()
        
        midi_valid = np.concatenate(midi_temp.pop(k))
        patt_valid = np.concatenate(patt_temp.pop(k))
        
        midi_train = np.concatenate(np.concatenate(midi_temp))
        patt_train = np.concatenate(np.concatenate(patt_temp))
        
        #rint(len(midi_temp))
        #print(len(midi_test))
        #print(midi_train)
        print(len(midi_train))
        
#        for i in range(len(midi_train)):        
#            print('Training File ' + str(i) + '...')            
#            tmp_batch_size = batch_size
#            if (len(midi_train) < ((batch_size - 1) * num_steps)): tmp_batch_size = 1
            
        train_data_generator = kgb.KerasBatchGenerator(midi_train, patt_train, num_steps, batch_size, output_range, skip_step=skip_step);
        valid_data_generator = kgb.KerasBatchGenerator(midi_valid, patt_valid, num_steps, batch_size, output_range, skip_step=skip_step);
        
        
        epoch_length_train = len(midi_train)//(skip_step * batch_size)
        epoch_length_valid = len(midi_valid)//(skip_step * batch_size)
        
        model.fit_generator(train_data_generator.generate(), epoch_length_train, num_epochs, 
                            validation_data=valid_data_generator.generate(), validation_steps=epoch_length_valid,
                            callbacks=[csv_logger, checkpointer]);
                                
            #print('Weights After')                    
            #print(model.layers[0].get_weights()[0])
            
        for layer in model.layers:
            file.write(json.dumps(layer.get_config()))
            file.write('\n')    
            weights = [str(i) for i in layer.get_weights()] 
            file.write("".join(weights))
            file.write('\n')    
        file.close();
        model.save(data_path + "final_model_" + str(k) + ".hdf5");
    
    
                    

