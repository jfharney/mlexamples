import os
import sys

import pandas as pd
import time
import dateutil.parser
import datetime

from utils import september_filter_df,september_filter_file


divisible_by = 10
step = 3000000

decomponsed_clicks_sources_dir = 'clicks_sources/'
decomponsed_buys_sources_dir = 'buys_sources/'
sourcefile_dir = '/Users/8xo/tutorials/ml-scikit/mlexamples/yoochoose-dataFull/recsys/'
sourcefile_filename = 'yoochoose-clicks.dat-old' 
header_text = 'SessID,Timestamp,ItemID,Category\n'

def decorate_files():
    
    header = 'clicks-header.dat'
    source_file = 'yoochoose-clicks.dat-old'
    output_file = 'yoochoose-clicks-decorated.dat'
    
    file_names = [header,source_file]
    
    with open(sourcefile_dir+output_file, 'w') as outfile:
        for fname in file_names:
            with open(fname) as infile:
                for line in infile:
                    outfile.write(line)
        
    #print 'in decorate'


#creates four new files 
#-sept_buys_test.csv
#-sept_clicks_test.csv
#-sept_buys_train.csv
#-sept_clicks_train.csv
def september_filter():
    
    print 'september_filter'
    
    
    clicks_train_dir = '/Users/8xo/tutorials/ml-scikit/mlexamples/yoochoose-dataFull/recsys/'

    #this is a small sample
    clicks_train_file = '9500095-9600095.txt'
    
    #this is the full file - takes ~30 secs
    clicks_train_file = 'yoochoose-clicks-decorated.dat'
    
    #this is a medium sample - takes 2 secs
    #clicks_train_file = 'clicks_sources/6000002-9000002.txt'
    
    buys_train_dir = '/Users/8xo/tutorials/ml-scikit/mlexamples/yoochoose-dataFull/recsys/'
    buys_train_file = 'yoochoose-buys-decorated.dat'
    
    clicks = True
    train = False
    begin = time.time()
    file = september_filter_file(clicks_train_dir,clicks_train_file,train,clicks)
    
    
    clicks = False
    train = False
    file = september_filter_file(buys_train_dir,buys_train_file,train,clicks)
    
    clicks = True
    train = True
    file = september_filter_file(clicks_train_dir,clicks_train_file,train,clicks)
    
    clicks = False
    train = True
    file = september_filter_file(buys_train_dir,buys_train_file,train,clicks)
    
    
    end = time.time()
    print str((end-begin))
    
    
    



def main():
    #decorate_files()
    decompose_files()
    
if __name__ == "__main__":
    main()
    
   
