import pandas as pd
import time
import sys
#from utils import func,func2,func3,count,date_filter,timestamp_filter
#from file_utils import september_filter

from file_utils import september_filter
from utils import count,func2,func3,func5,isBought
from df_utils import make_category_dict,make_category_df,make_list_item_numclicks_df,make_ind_item_numclicks_df,make_item_numclicks_df,make_category_numclicks

from mods import small_sample




def large_sample():
    clicks_df = pd.DataFrame({'SessID' : ['11', '2', '3', '11','2', '4', '2', '4','11'],
                       'ItemID' : ['i1', 'i2', 'i3', 'i4','i5', 'i6', 'i7', 'i8','i1'],
                       'Category': ['c1','c2','c3','c3','c3','c4','c5','c1','c1']
                       #,
                       #'list' : [['2','3'],['2','3'],['2','3'],['2','3'],['2','3'],['2','3'],['2','3'],['2','3']]
                       })
    #2014-09-01T18:07:58.937Z
    #2014-09-02T18:07:58.937Z
    #2014-09-03T18:07:58.937Z
    #2014-09-04T18:07:58.937Z
    #2014-09-05T18:07:58.937Z
    #2014-09-06T18:07:58.937Z
    #2014-09-07T18:07:58.937Z
    #2014-09-08T18:07:58.937Z
    #2014-09-09T18:07:58.937Z
    
    buys_df = pd.DataFrame(
                           {'SessID' : ['11', '2', '3','11'],
                            'ItemID' : ['i1', 'i2', 'i3','i2']
                           #,
                           #'list' : [['2','3'],['2','3'],['2','3'],['2','3'],['2','3'],['2','3'],['2','3'],['2','3']]
                           })
    
    clicks_train_dir = '/Users/8xo/tutorials/ml-scikit/mlexamples/yoochoose-dataFull/recsys/'
    clicks_train_file = 'sept_clicks_train.csv'
    buys_train_dir = '/Users/8xo/tutorials/ml-scikit/mlexamples/yoochoose-dataFull/recsys/'
    buys_train_file = 'sept_buys_train.csv'
    
    clicks_train_df = pd.read_csv(clicks_train_dir+clicks_train_file)
    print len(clicks_train_df)
    
    #item -> category
    item_to_category_df = clicks_train_df[['ItemID','Category']].drop_duplicates()
    
    print (item_to_category_df.head())
    #(df['Timestamp'] >= '2014-09-01 00:00:00')
    cat = item_to_category_df[(item_to_category_df['ItemID'] == 214839911)]
    #cat = item_to_category_df['Category'][item_to_category_df['ItemID'] == 214839911]
    
    print (cat)
    
    sys.exit()
    #print first
    
    #category -> items
    
    category_df = make_category_df(clicks_train_df)
    print category_df.head()
    print len(category_df)
    
    item_numclicks_df = make_ind_item_numclicks_df(clicks_train_df,214839911)
    print 'len: ' + str(len(item_numclicks_df))
    
    '''
    #buys_train_df = pd.read_csv(buys_train_dir+buys_train_file)
    
    #item_numclicks_df = make_ind_item_numclicks_df(clicks_df,214839911)
    item_list = [214834871,214839911,214701787]
    #item_ind_numclicks_df = make_ind_item_numclicks_df(clicks_train_df,214839911)
    
    #print item_ind_numclicks_df.head()
    
    item_list_numclicks_df = make_list_item_numclicks_df(clicks_train_df,item_list)
    
    print item_list_numclicks_df.head()
    #print item_numclicks_df.head(2)
    
    #print(clicks_train_df[0:4])
    '''
    
    
    category_df = make_category_df(buys_df)
    print category_df
    
    item_numclicks_df = make_item_numclicks_df(clicks_df)
    
    print item_numclicks_df
    
    sys.exit()
    
    category_column = category_df['Category']
    category_column = category_column.drop_duplicates()
    
    
    
    for category in category_column:
        
        itemlist = category_df['ItemID'][category_df['Category'] == category]
        #hack
        itemlist = itemlist.tolist()[0]
        print 'category' + str(category)
        print itemlist
        print type(itemlist)
        
        #for each item in the list, find the number of clicks
        for item in itemlist:
            
            #find number of clicks
            
            print item
            
        #create a series for a specific column indicating the number of clicks
        
        #add
        
        
    
    #print category_df
    
    '''
    print category_df['ItemID'][category_df['Category'] == 'c2'][0]
    print type(category_df['ItemID'][category_df['Category'] == 'c2'][0])
    
    lst = category_df['ItemID'][category_df['Category'] == 'c2'][0]
    
    print 'making numclicks'
    item_numclicks_df = make_item_numclicks_df(df)
    
    print item_numclicks_df
    '''
    
    
    print '----original----'
    
    #print (len(df['ItemID'].value_counts()))
    #print (buys_df)
    
    print '----------------\n'
    
    
    print '----derived-----'
    grouped = df.groupby('SessID')
    
    s1 = grouped['ItemID'].apply(count)
    
    print s1
    
    feature_list = []
    feature_list_headers = [] 
    
    item_column = df['ItemID']
    item_column = item_column.drop_duplicates()
    print item_column
    for item in item_column:
        print 'item: ' + str(item)
        #print type(item)
        s2 = grouped['ItemID'].apply(func5,item)
        feature_list.append(s2)
        feature_list_headers.append('numclicks_' + item)
        print s2
        
    print str(feature_list_headers)
    
    newdf = pd.concat(feature_list, axis=1)
    newdf['SessID'] = newdf.index
    feature_list_headers.append('SessID')
    newdf.columns = feature_list_headers
    
    print newdf
    
    
    
    
    
    
    
    
    newdf = pd.concat([s1], axis=1)
    newdf['SessID'] = newdf.index
    
    newdf.columns = ['NumClicks','SessID']
    print(newdf)
    
    
    grouped = buys_df.groupby('SessID')
    s2 = grouped['ItemID'].apply(func2)
    
    new_buys_df = pd.concat([s2], axis=1)
    new_buys_df['SessID'] = new_buys_df.index
    
    
    print(new_buys_df)
    
    #newdf['ItemsBought'] = new_buys_df['SessID'].apply(func3)
    
    merged = pd.merge(newdf,new_buys_df,on='SessID',how='outer')
    
    merged.ItemID.fillna(0,inplace=True)
    
    print(merged)
    
    merged['Item1'] = merged['ItemID'].apply(isBought,args=('i1',))
    
    
    '''
    print merged
    
    newmerged = merged[['SessID','NumClicks','Item1']]
    
    print newmerged
    '''
    
    newmerged = pd.merge(merged,item_numclicks_df,on='SessID',how='outer')
    
    print newmerged
    
    
    
    import numpy as np
    #X = np.asarray(newmerged['NumClicks'])
    
    print 'partial newmerged'
    columns = newmerged.columns[4:]
    
    X = np.asarray(newmerged[columns])
    
    print X
    print len(X)
    y = np.asarray(newmerged['Item1'].transpose())
    
    print y
    print len(y)
    
    from sklearn.naive_bayes import MultinomialNB
    nb = MultinomialNB()
    
    
    nb.fit(X, y)
    
    print (nb.predict(np.asarray(newmerged[columns])))
    
    print (np.mean(nb.predict(np.asarray(newmerged[columns])) == newmerged.Item1))
    
    print '----decision tree----'
    
    
    from sklearn import tree
    
    # Create the target and features numpy arrays: target, features_one
    target = newmerged["Item1"].values
    features_one = newmerged[["NumClicks", "numclicks_i6"]].values
    
    
    # Fit your first decision tree: my_tree_one
    my_tree_one = tree.DecisionTreeClassifier()
    my_tree_one = my_tree_one.fit(features_one, target)
    
    # Look at the importance and score of the included features
    print(my_tree_one.feature_importances_)
    print(my_tree_one.score(features_one, target))
    
    sys.exit()
    
    
    '''KEEP'''
    '''
    september_clicks_train_dir = '/Users/8xo/tutorials/ml-scikit/mlexamples/yoochoose-dataFull/recsys/'
    september_clicks_train_file = 'sept_clicks_train.csv'
    september_clicks_test_file = 'sept_clicks_test.csv'
    
    september_buys_train_dir = '/Users/8xo/tutorials/ml-scikit/mlexamples/yoochoose-dataFull/recsys/'
    september_buys_train_file = 'sept_buys_train.csv'
    september_buys_test_file = 'sept_buys_test.csv'
    
    
    september_buys_train = pd.read_csv(september_buys_train_dir+september_buys_train_file)
    september_clicks_train = pd.read_csv(september_clicks_train_dir+september_clicks_train_file)
    
    import time
    
    first = time.time()
    september_clicks_train = september_clicks_train[['SessID','Timestamp','ItemID','Category']]
    september_buys_train = september_buys_train[['SessID','Timestamp','ItemID','Price','Quantity']]
    second = time.time()
    
    #print (september_clicks_train['ItemID'].value_counts())
    #print (len(september_clicks_train['ItemID'].value_counts()))
    print (september_clicks_train['Category'].value_counts())
    print (len(september_clicks_train['Category'].value_counts()))
    '''
    
    '''
    
    category_df = make_category_df(september_clicks_train[['ItemID','Category']])
    third = time.time()
    
    print 'first: ' + str(first)
    print 'second: ' + str(second)
    print 'third: ' + str(third)
    
    #print(category_df.head())
    #print(len(category_df))
    
    
    
    
    grouped = september_clicks_train.groupby('SessID')
    
    s2 = grouped['ItemID'].apply(count)
    
    print(s2[0:5])
    print(len(s2))
    '''
    
    
    #group_by_example()
    
    
    sys.exit()
    #timestamp_filter()


    

def main():
    
    
    #large_sample()
    
    small_sample()
    
    #decorate the given source data with headers
    #decorate_files()
    
    #filter the input into september only records
    #september_filter()
    #print 'after september_filter'
    
    
    
    
    
if __name__ == "__main__":
    main()
    


#decorate_files()
#timestamp_filter()
#group_by_example2()