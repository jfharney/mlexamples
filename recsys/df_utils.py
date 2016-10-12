import pandas as pd
import time
import sys

from file_utils import september_filter
from utils import count,func2,func3,func5,isBought
#from df_utils import make_category_dict,make_category_df,make_list_item_numclicks_df,make_ind_item_numclicks_df,make_item_numclicks_df,make_category_numclicks



def make_category_dict(df):
    
    grouped = df.groupby('Category')
    newdf = pd.Series.to_frame(grouped['ItemID'].apply(func2))
    
    newdf['Category'] = newdf.index
    
    print newdf
    
    print newdf['ItemID'][newdf['Category'] == 'c2']
    
    category_dict = newdf.set_index('Category').to_dict()

    return category_dict

def make_category_df(df):
    
    grouped = df.groupby('Category')
    newdf = pd.Series.to_frame(grouped['ItemID'].apply(func2))
    
    newdf['Category'] = newdf.index
    
    return newdf

def make_list_item_numclicks_df(df,lst):
    
    begin = time.time()
    
    #print str(time.time())
    
    #print "executing grouped = df.groupby('SessID')"
    grouped = df.groupby('SessID')
    
    #print str(time.time())
    
    #print "executing s1 = grouped['ItemID'].apply(count)"
    s1 = grouped['ItemID'].apply(count)
    
    #print str(time.time())
    #print s1
    
    #print 'executing the rest'
    #item_column = df['ItemID']
    
    #for item in item_column:
    #    print item
    #s2 = grouped['ItemID'].apply(func5,item)
    
    #print s2
    
    feature_list = []
    feature_list_headers = []
    
    
    for item in lst:
        #print 'item: ' + str(item)
        #print type(item)
        s2 = grouped['ItemID'].apply(func5,item)
        feature_list.append(s2)
        feature_list_headers.append('numclicks_' + str(item))
        #print s2
        
    #print str(feature_list_headers)
    
    
    #feature_list.append(s2)
    #feature_list_headers.append('numclicks_' + str(item))
    
    newdf = pd.concat(feature_list, axis=1)
    newdf['SessID'] = newdf.index
    feature_list_headers.append('SessID')
    newdf.columns = feature_list_headers
    
    #print str(time.time())
    
    end = time.time()
    
    print 'totaltime: ' + str(end-begin)
    return newdf




def make_ind_item_numclicks_df(df,item):
    
    print 'in make_ind_item_numclicks_df for item: ' + str(item)
    begin = time.time()
    
    #print str(time.time())
    
    #print "executing grouped = df.groupby('SessID')"
    grouped = df.groupby('SessID')
    
    #print str(time.time())
    
    #print "executing s1 = grouped['ItemID'].apply(count)"
    #s1 = grouped['ItemID'].apply(count)
    
    #print str(time.time())
    #print 's1'
    #print s1
    
    #print 'executing the rest'
    item_column = df['ItemID']
    
    #for item in item_column:
    #    print item
    s2 = grouped['ItemID'].apply(func5,item)
    
    #print 's2'
    #print s2
    
    
    feature_list = []
    feature_list_headers = []
    
    feature_list.append(s2)
    feature_list_headers.append('numclicks_' + str(item))
    
    newdf = pd.concat(feature_list, axis=1)
    newdf['SessID'] = newdf.index
    feature_list_headers.append('SessID')
    newdf.columns = feature_list_headers
    
    #print newdf[(newdf['numclicks_' + str(item)] > 0)].head()
    #print str(time.time())
    
    end = time.time()
    
    print 'totaltime: ' + str(end-begin)
    
    return newdf
    
    #item_column = item_column.drop_duplicates()
    

def make_item_numclicks_df(df):
    grouped = df.groupby('SessID')
    s1 = grouped['ItemID'].apply(count)
    
    #print s1
    
    feature_list = []
    feature_list_headers = [] 
    
    item_column = df['ItemID']
    item_column = item_column.drop_duplicates()
    #print item_column
    for item in item_column:
        #print 'item: ' + str(item)
        #print type(item)
        s2 = grouped['ItemID'].apply(func5,item)
        feature_list.append(s2)
        feature_list_headers.append('numclicks_' + item)
        #print s2
        
    #print str(feature_list_headers)
    
    newdf = pd.concat(feature_list, axis=1)
    newdf['SessID'] = newdf.index
    feature_list_headers.append('SessID')
    newdf.columns = feature_list_headers
    
    #print newdf
    return newdf
    
    
def make_category_numclicks(df):
    
    return df   

