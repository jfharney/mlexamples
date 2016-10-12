import pandas as pd
import time
import sys
import os
#from utils import func,func2,func3,count,date_filter,timestamp_filter
#from file_utils import september_filter

from file_utils import september_filter
from utils import count,func2,func3,func5,func7,isBought
#from df_utils import make_category_dict,make_category_df,make_list_item_numclicks_df,make_ind_item_numclicks_df,make_item_numclicks_df,make_category_numclicks


def make_item_numclicks_df_wrapper(clicks_df,item_name,cache_name='cache_train/'):
    #item_name = str(item_name)
    #Make the item numclicks for item
    item_numclicks_df_filename = cache_name + 'item_numclicks_' + str(item_name) + '_df.csv'
    
    if os.path.exists(item_numclicks_df_filename):
        print 'load the file'
        item_numclicks_df = pd.read_csv(item_numclicks_df_filename)
    else:
        print 'dyanmically create df'
        item_numclicks_df = make_ind_item_numclicks_df(clicks_df,item_name)
        #print 'item_numclicks_df time: ' + str(item_numclicks_df_end-item_numclicks_df_begin)
        
        #print item_numclicks_df.head()
        #print item_numclicks_df
        item_numclicks_df[['SessID','numclicks_' + str(item_name)]].to_csv(item_numclicks_df_filename)
    return item_numclicks_df


def make_ind_item_numclicks_df(df,item):
    
    print 'in make_ind_item_numclicks_df for item: ' + str(item)
    begin = time.time()
    
    #print str(time.time())
    
    #print "executing grouped = df.groupby('SessID')"
    grouped = df.groupby(['SessID','ItemID'], as_index=False).count()
    
    #print grouped.head()
    
    
    #print grouped[(grouped['ItemID']=='i1')].head()
    #print 'item: ' +  str(item) + ' type: ' + str(type(item))
    
    new_grouped = grouped[(grouped['ItemID']==item)][['SessID','Timestamp']]
    
    #print new_grouped
    
    new_grouped.columns = ['SessID','numclicks_' + str(item)]
    
    #print new_grouped
    #print str(type(new_grouped))
    
    
    end = time.time()
    
    #print 'totaltime: ' + str(end-begin)
    
    #print str(time.time())
    
    #print "executing s1 = grouped['ItemID'].apply(count)"
    #s1 = grouped['ItemID'].apply(count)
    
    #print str(time.time())
    #print 's1'
    #print s1
    
    #print new_grouped.head()
    
    return new_grouped
    
    sys.exit()
    
    
    print 'len: ' + str(len(grouped))
    print str(type(grouped))
    #print str(grouped.describe())
    
    print grouped['ItemID']
    '''
    s1 = grouped['ItemID'].apply(count)
    print s1
    
    #print 'executing the rest'
    item_column = df['ItemID']
    
    #for item in item_column:
    #    print item
    print 'buidling s2...'
    s2 = grouped['ItemID'].apply(func7,item)
    print 'end buidling s2...'
    
    #print 's2'
    print s2
    '''
    
    
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
    

def test():
    clicks_df = pd.DataFrame({'SessID' : ['11', '2', '3', '11','2', '4', '2', '4','11','3','6'],
                       'ItemID' : ['i1', 'i2', 'i3', 'i4','i5', 'i6', 'i7', 'i8','i1','i4','i6'],
                       'Timestamp' : [
                                      "2014-09-01T18:07:58.937Z", #,2014-09-02T18:07:58.937Z \
                                      "2014-09-03T18:07:58.937Z",
                                      "2014-09-04T18:07:58.937Z",
                                      "2014-09-05T18:07:58.937Z",
                                      "2014-09-06T18:07:58.937Z",
                                      "2014-09-07T18:07:58.937Z",
                                      "2014-09-08T18:07:58.937Z",
                                      "2014-09-09T18:07:58.937Z",
                                      "2014-09-10T18:07:58.937Z",
                                      "2014-09-13T18:07:58.937Z",
                                      "2014-09-11T18:07:58.937Z"
                                      ],
                       'Category': ['c1','c2','c3','c3','c3','c4','c5','c1','c1','c10','c10']
                       #,
                       #'list' : [['2','3'],['2','3'],['2','3'],['2','3'],['2','3'],['2','3'],['2','3'],['2','3']]
                       })
    
    
    buys_df = pd.DataFrame(
                           {
                            'SessID' : ['11', '2', '3','11'],
                            'Timestamp' : [
                                            "2014-09-05T18:07:58.937Z",
                                            "2014-09-06T18:07:58.937Z",
                                            "2014-09-07T18:07:58.937Z",
                                            "2014-09-08T18:07:58.937Z"
                                           ],
                            'ItemID' : ['i1', 'i2', 'i3','i2'],
                            'Price' : ['30', '5', '31','5'],
                            'Quantity' : ['1', '2', '3','4'],
                           #,
                           #'list' : [['2','3'],['2','3'],['2','3'],['2','3'],['2','3'],['2','3'],['2','3'],['2','3']]
                           })
    
    
    
    print 'in test'
    
    clicks_train_dir = '/Users/8xo/tutorials/ml-scikit/mlexamples/yoochoose-dataFull/recsys/'
    clicks_train_file = 'sept_clicks_train.csv'
    buys_train_dir = '/Users/8xo/tutorials/ml-scikit/mlexamples/yoochoose-dataFull/recsys/'
    buys_train_file = 'sept_buys_train.csv'
    
    clicks_test_dir = '/Users/8xo/tutorials/ml-scikit/mlexamples/yoochoose-dataFull/recsys/'
    clicks_test_file = 'sept_clicks_test.csv'
    buys_test_dir = '/Users/8xo/tutorials/ml-scikit/mlexamples/yoochoose-dataFull/recsys/'
    buys_test_file = 'sept_buys_test.csv'
    
    clicks_train_df = pd.read_csv(clicks_train_dir+clicks_train_file)
    buys_train_df = pd.read_csv(buys_train_dir+buys_train_file)
    
    
    cache_name = 'cache_train_views/'
    dir = os.path.dirname(cache_name)
    try:
        os.stat(dir)
    except:
        os.mkdir(dir)  
    
    item_name = 'i1'
    
    clicks_df = clicks_train_df
    buys_df = buys_train_df
    item_name = 214853657

    
    print clicks_df['SessID'].head()
    
    #example item
    
    
    item_numclicks_df_begin = time.time()
    #item_numclicks_df = make_item_numclicks_df_wrapper(clicks_df,item_name,cache_name)
    
    
    
    print str(clicks_df['ItemID'][0]) + ' ' + str(type(clicks_df['ItemID'][0]))
    print str(len(clicks_df['ItemID']))
    print len(clicks_df['ItemID'].unique())
    for i in range(0,len(clicks_df['ItemID'].unique())):
        print 'i: ' + str(i) + ' ' + str(len(clicks_df['ItemID'].unique()))
        if (i > 100):
            item_name = clicks_df['ItemID'][i]
            item_numclicks_df = make_item_numclicks_df_wrapper(clicks_df,item_name,cache_name)
    
    '''
    for i in range(0,len(clicks_df['ItemID'])):
        if (i >= 1000):
            print 'create cache entry for... ' + str(item_name) + ' ' + str(type(item_name))
            item_name = clicks_df['ItemID'][i]
            item_numclicks_df = make_item_numclicks_df_wrapper(clicks_df,item_name,cache_name)
    
    '''
    item_numclicks_df_end = time.time()
    
    print 'totaltime: ' + str(item_numclicks_df_end-item_numclicks_df_begin)
    
    
    sys.exit()
    
    

def main():
    
    
    test()
    
    #large_sample()
    
    #small_sample()
    
    #decorate the given source data with headers
    #decorate_files()
    
    #filter the input into september only records
    #september_filter()
    #print 'after september_filter'
    
    
    
    
    
if __name__ == "__main__":
    main()
    