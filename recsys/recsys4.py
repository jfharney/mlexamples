import pandas as pd
import time
import sys
import os


from sklearn.externals import joblib
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
    
    print 'totaltime: ' + str(end-begin)
    
    #print str(time.time())
    
    #print "executing s1 = grouped['ItemID'].apply(count)"
    #s1 = grouped['ItemID'].apply(count)
    
    #print str(time.time())
    #print 's1'
    #print s1
    
    print new_grouped.head()
    
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

def make_total_clicks_df_wrapper(clicks_df,cache_name='cache_train/',filename_prefix=''):
    
    total_clicks_df_filename = cache_name + str(filename_prefix) + 'total_clicks_df.csv'
    print 'total_clicks_df'
    
    if os.path.exists(total_clicks_df_filename): 
        #print 'cache_traind'
        #total_clicks_df = pd.read_csv(total_clicks_df_filename)
        total_clicks_df = make_total_clicks_df(clicks_df)
    else:
        total_clicks_df = make_total_clicks_df(clicks_df)
        #total_clicks_df.to_csv(total_clicks_df_filename)
    return total_clicks_df

def make_total_clicks_df(df):
    
    grouped = df.groupby('SessID')
    
    s1 = grouped['ItemID'].apply(count)
    
    newdf = pd.concat([s1], axis=1)
    newdf['SessID'] = newdf.index
    feature_headers = ['NumClicks','SessID']
    newdf.columns = feature_headers
    return newdf   

def make_items_bought_per_session_df(df):
    grouped = df.groupby('SessID')
    s2 = grouped['ItemID'].apply(func2)
    
    new_buys_df = pd.concat([s2], axis=1)
    new_buys_df['SessID'] = new_buys_df.index

    return new_buys_df

def build_feature_map(clicks_df,features_list,item_name,cache_name):
    #print str(features_list)
    
    item_numclicks_df = None
    if 'numclicks_'+str(item_name) in features_list:
        #print 'building' + 'numclicks_'+str(item_name)
        item_numclicks_df = make_item_numclicks_df_wrapper(clicks_df,item_name,cache_name)
        item_numclicks_df = item_numclicks_df[['SessID','numclicks_' + str(item_name)]]
        print item_numclicks_df
        
    
    if 'NumClicks' in features_list:
        #print 'building NumClicks'
        total_clicks_df = make_total_clicks_df_wrapper(clicks_df,cache_name)
        print total_clicks_df
    
    merged_df = pd.merge(item_numclicks_df,total_clicks_df,on='SessID',how='outer')
    
    merged_df = merged_df[['SessID','NumClicks','numclicks_'+str(item_name)]]
    
    #merged_df = merged_df[(merged_df[])]
    
    print 'merged_df'
    print merged_df
    
    #sys.exit()
    #clean
    for feature in features_list:
        merged_df = merged_df.fillna({feature: 0})
    
    
    return merged_df

def build_trainer_df(buys_df,feature_map_df,item_name):
    #get a bought_items_per_session_df
    bought_items_per_session_df = make_items_bought_per_session_df(buys_df)
    
    #print 'bought_items_per_session_df'
    #print bought_items_per_session_df.head()
    #print len(bought_items_per_session_df)
    
    
    new_merged_df = pd.merge(feature_map_df,bought_items_per_session_df,on='SessID',how='outer')
    new_merged_df.ItemID.fillna(0,inplace=True)
    new_merged_df['Item_Bought_'+str(item_name)] = new_merged_df['ItemID'].apply(isBought,args=(item_name,))
    trainer_df = new_merged_df[['SessID',str('numclicks_'+str(item_name)),'NumClicks','Item_Bought_'+str(item_name)]]
    
    return trainer_df


def make_decision_tree(features_list,trainer_df,item_name):
    import numpy as np
    from sklearn import tree
    from sklearn.naive_bayes import MultinomialNB
    
    #Decision Tree
    target = trainer_df['Item_Bought_'+str(item_name)].values
    features = trainer_df[features_list].values
    
    # Fit  decision tree: my_tree_one
    my_tree = tree.DecisionTreeClassifier()
    my_tree = my_tree.fit(features, target)
    my_prediction = my_tree.predict(features)
    
    # Print initial stats
    #print(my_tree.feature_importances_)
    #print(my_tree.score(features, target))
    
    #quick test - take out
    print 'target length: ' + str(len(target))
    print 'my_prediction length: ' + str(len(my_prediction))
    
    counter = 0
    for value in target:
        if value == 1:
            print 'there is a 1 in target at index ' + str(counter)
        counter = counter + 1
    counter = 0
    for value in my_prediction:
        if value == 1:
            print 'there is a 1 in my_prediction at index ' + str(counter)
        counter = counter + 1
    
    #print (np.mean(my_tree.predict(features) == target))
    
    # Pickle and save to file
    from sklearn.externals import joblib
    cache_models = 'cache_models/'
    dir = os.path.dirname(cache_models)
    try:
        os.stat(dir)
    except:
        os.mkdir(dir) 
    dt_file = cache_models + 'dt_' + str(item_name) + '.pkl'
    joblib.dump(my_tree, dt_file) 
    print 'writing to ' + str(dt_file)
    
    
def get_decision_tree(item_name):
    dt_file = str(item_name) + '.pkl'
    #Create the decision tree model
    #make_decision_tree(features_list,trainer_df,item_name)
    cache_models = 'cache_models/'
    dir = os.path.dirname(cache_models)
    try:
        os.stat(dir)
    except:
        os.mkdir(dir) 
    dt_file = cache_models + 'dt_' + str(item_name) + '.pkl'
    
        
    #use the same model as before
    import numpy as np
    from sklearn import tree
    from sklearn.naive_bayes import MultinomialNB
    
    
    my_tree = joblib.load(dt_file) 
    
    return my_tree

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
    
    
    clicks_train_df = pd.DataFrame({'SessID' : ['11', '2', '3', '11','2', '4', '2', '4','11','3','6'],
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
    clicks_test_df = pd.read_csv(clicks_test_dir+clicks_test_file)
    buys_test_df = pd.read_csv(buys_test_dir+buys_test_file)
    
    
    cache_name = 'cache_train_views/'
    dir = os.path.dirname(cache_name)
    try:
        os.stat(dir)
    except:
        os.mkdir(dir)  
    
    item_name = 'i1'
    
    clicks_df = clicks_train_df
    buys_df = buys_train_df
    item_name = 214712244
    
    
    features_list = ['NumClicks','numclicks_'+str(item_name)]
    
    #clicks_df = clicks_train_df
    #buys_df = buys_train_df
    #item_name = 214712244#214853657#214695549#
    item_name_list = [214712244,214853657,214695549]
    #for item_name in item_name_list:
    
    import numpy as np
    
    session_only_df = clicks_df['SessID'].unique()
    #session_only_df['itemsbought'] = session_only_df['SessID'].apply(func3) 
    #session_only_df['items_bought'] = 0
    
    #prep the feature map
    feature_map_df = build_feature_map(clicks_df,features_list,item_name,cache_name)
    print feature_map_df.head(10)

    print str(len(feature_map_df))
    
    #prep the target values and combine with feature map
    trainer_df = build_trainer_df(buys_df,feature_map_df,item_name)

    prepare_end = time.time()
    #print 'total prepare time: ' + str(prepare_end-prepare_begin)

    #print 'trainer_df'
    
    #clean the trainer_df
    for feature in features_list:
        trainer_df = trainer_df.fillna({feature: 0})
    
        #print trainer_df

    #Create the decision tree model
    my_tree = make_decision_tree(features_list,trainer_df,item_name)
        
    
    
    print 'get_decision_tree given the unseen samples'
    
    #prepare the unseen sample feature_map
    clicks_df = clicks_test_df
    feature_map_df = build_feature_map(clicks_df,features_list,item_name,cache_name)
    print 'lennnn of feature map: ' + str(len(feature_map_df))
    
    #get the decision tree model from item_name
    my_tree = get_decision_tree(item_name)
    #prepare the features for the predictions
    features = feature_map_df[['numclicks_'+str(item_name), 'NumClicks']].values
    
    #predict
    
    my_prediction = my_tree.predict(features)
    
    # Print initial stats
    #print(my_tree.feature_importances_)
    #print(my_tree.score(features, target))
    
    #quick test - take out
    
    #print (np.mean(my_tree.predict(features) == my_prediction))
    print str(len(my_tree.predict(features)))
    
    
    
    
    prediction_df = pd.DataFrame(my_prediction)
    print str(len(session_only_df))
    print session_only_df
    print str(len(prediction_df))
    print prediction_df
    
    #joined_df = pd.concat([session_only_df, prediction_df], axis=1)
    
    #print joined_df.head()
    #final_df = pd.concat([session_only_df,prediction_df], axis=1)
    
    #print final_df
    merged = pd.merge(feature_map_df,session_only_df,on='SessID',how='outer')
    merged = pd.merge(session_only_df,prediction_df,on='SessID',how='outer')
    
    
    sys.exit()
    
    
    
    
    print '----predict----'
    
    
    dt_file = str(item_name) + '.pkl'
    #Create the decision tree model
    #make_decision_tree(features_list,trainer_df,item_name)
    cache_models = 'cache_models/'
    dir = os.path.dirname(cache_models)
    try:
        os.stat(dir)
    except:
        os.mkdir(dir) 
    dt_file = cache_models + 'dt_' + str(item_name) + '.pkl'
    
        
    #use the same model as before
    import numpy as np
    from sklearn import tree
    from sklearn.naive_bayes import MultinomialNB
    
    
    my_tree = joblib.load(dt_file) 
    
    #Decision Tree
    # Create the target and features numpy arrays: target, features_one
    target = trainer_df['Item_Bought_'+str(item_name)].values
    features = trainer_df[['numclicks_'+str(item_name), 'NumClicks']].values
    
    
    print 'prediction\n'
    my_prediction = my_tree.predict(features)
    print my_prediction
    print 'len my prediction: ' + str(len(my_tree.predict(features)))
    print 'len target: ' + str(len(target))
    print target
    #value = 1
    counter = 0
    for value in target:
        if value == 1:
            print 'there is a 1 in target at index ' + str(counter)
        counter = counter + 1
    counter = 0
    for value in my_prediction:
        if value == 1:
            print 'there is a 1 in my_prediction at index ' + str(counter)
        counter = counter + 1


    print (np.mean(my_tree.predict(features) == target))

    
    print 'end need to read the different decision trees'
    
    
    sys.exit()
    
    
    '''
    
    # train
    
    for i in range(0,len(clicks_df['ItemID'].unique())):
    #for i in range(0,1):    
        
        item_name = clicks_df['ItemID'].unique()[i]
        print 'i: ' + str(i) + ' ' + str(item_name)
        
        #item_name = item_name_list[1]
        
        if (i < 2000):
            features_list = ['NumClicks','numclicks_'+str(item_name)]
        
        
            prepare_begin = time.time()
        
        
            #prep the feature map
            feature_map_df = build_feature_map(clicks_df,features_list,item_name,cache_name)
            print feature_map_df.head()
        
            #prep the target values and combine with feature map
            trainer_df = build_trainer_df(buys_df,feature_map_df,item_name)
        
            prepare_end = time.time()
            #print 'total prepare time: ' + str(prepare_end-prepare_begin)
        
            #print 'trainer_df'
            
            #clean the trainer_df
            for feature in features_list:
                trainer_df = trainer_df.fillna({feature: 0})
            
                #print trainer_df
        
            #Create the decision tree model
            make_decision_tree(features_list,trainer_df,item_name)
        
        else: 
            sys.exit()
    
    
    
    cache_name = 'cache_test_views/'
    dir = os.path.dirname(cache_name)
    try:
        os.stat(dir)
    except:
        os.mkdir(dir)  
    
    
    clicks_test_df = pd.read_csv(clicks_test_dir+clicks_test_file)
    buys_test_df = pd.read_csv(buys_test_dir+buys_test_file)
    
    clicks_df = clicks_test_df
    buys_df = buys_test_df
    
    # validate/evaluate over test set
    #for i in range(0,len(clicks_df['ItemID'].unique())):
    for i in range(0,1):    
        item_name = clicks_df['ItemID'].unique()[i]
        print 'i: ' + str(i) + ' ' + str(item_name)

        item_name = item_name_list[1]
        if i < 1:
            features_list = ['NumClicks','numclicks_'+str(item_name)]
        
        
            prepare_begin = time.time()
        
        
            #prep the feature map
            feature_map_df = build_feature_map(clicks_df,features_list,item_name,cache_name)
            #print feature_map_df.head()
        
            #prep the target values and combine with feature map
            trainer_df = build_trainer_df(buys_df,feature_map_df,item_name)
        
            prepare_end = time.time()
            print 'total prepare time: ' + str(prepare_end-prepare_begin)
        
            #print 'trainer_df'
            
            #clean the trainer_df
            for feature in features_list:
                trainer_df = trainer_df.fillna({feature: 0})
            
                #print trainer_df
        
            print 'need to read the different decision trees'
            
            dt_file = str(item_name) + '.pkl'
            #Create the decision tree model
            #make_decision_tree(features_list,trainer_df,item_name)
            cache_models = 'cache_models/'
            dir = os.path.dirname(cache_models)
            try:
                os.stat(dir)
            except:
                os.mkdir(dir) 
            dt_file = cache_models + 'dt_' + str(item_name) + '.pkl'
            
                
            #use the same model as before
            import numpy as np
            from sklearn import tree
            from sklearn.naive_bayes import MultinomialNB
            
            
            my_tree = joblib.load(dt_file) 
            
            #Decision Tree
            # Create the target and features numpy arrays: target, features_one
            target = trainer_df['Item_Bought_'+str(item_name)].values
            features = trainer_df[['numclicks_'+str(item_name), 'NumClicks']].values
            
            
            print 'prediction\n'
            my_prediction = my_tree.predict(features)
            print my_prediction
            print 'len my prediction: ' + str(len(my_tree.predict(features)))
            print 'len target: ' + str(len(target))
            print target
            #value = 1
            counter = 0
            for value in target:
                if value == 1:
                    print 'there is a 1 in target at index ' + str(counter)
                counter = counter + 1
            counter = 0
            for value in my_prediction:
                if value == 1:
                    print 'there is a 1 in my_prediction at index ' + str(counter)
                counter = counter + 1
        
    
            print (np.mean(my_tree.predict(features) == target))
    
            
            print 'end need to read the different decision trees'
            #sys.exit()
            
    
        else: 
            sys.exit()
    
    
    
    sys.exit()
    
    '''
    
    
    
    '''
    #Naive Bayes
    columns = trainer_df.columns[1:3]
    print str(columns)
    
    
    X = np.asarray(trainer_df[columns])
    y = np.asarray(trainer_df['Item_Bought_'+str(item_name)].transpose())
    nb = MultinomialNB()
    nb.fit(X, y)
    print (nb.predict(np.asarray(trainer_df[columns])))
    print (np.mean(nb.predict(np.asarray(trainer_df[columns])) == trainer_df['Item_Bought_'+str(item_name)]))
    
    
    #dump the model to file
    from sklearn.externals import joblib
    nb_file = str(item_name) + '.pkl'
    joblib.dump(nb, nb_file) 
    
    clf = joblib.load(nb_file) 
    clf.fit(X, y)
    print (nb.predict(np.asarray(trainer_df[columns])))
    print (np.mean(nb.predict(np.asarray(trainer_df[columns])) == trainer_df['Item_Bought_'+str(item_name)]))
    '''
    
    print '\n\n\n-----test-----'
    
    
    clicks_test_df = pd.read_csv(clicks_test_dir+clicks_test_file)
    buys_test_df = pd.read_csv(buys_test_dir+buys_test_file)
    
    
    cache_name = 'cache_test_views/'
    dir = os.path.dirname(cache_name)
    try:
        os.stat(dir)
    except:
        os.mkdir(dir)  
    
    
    clicks_df = clicks_test_df
    buys_df = buys_test_df
    #item_name = 214853657
    
    
    print clicks_df['SessID'].head()
    
    #print clicks_df[(clicks_df['ItemID']==item_name)].head()
    
    
    item_numclicks_df = make_item_numclicks_df_wrapper(clicks_df,item_name,cache_name)
    
    print item_numclicks_df.head()
    print len(item_numclicks_df)
    
    
    total_clicks_df_begin = time.time()
    total_clicks_df = make_total_clicks_df_wrapper(clicks_df,cache_name)
    total_clicks_df_end = time.time()
    
    print total_clicks_df.head()
    print len(total_clicks_df)
    
    merged_df = pd.merge(item_numclicks_df,total_clicks_df,on='SessID',how='outer')
    
    merged_df = merged_df[['SessID','NumClicks','numclicks_'+str(item_name)]]
    #new_merged_df.'numclicks_'+str(item_name).fillna(0,inplace=True)
    
    
    print 'merged_df'
    print merged_df.head()
    print len(merged_df)
    
    #prep the target values
    #get a bought_items_per_session_df
    bought_items_per_session_df = make_items_bought_per_session_df(buys_df)
    
    print 'bought_items_per_session_df'
    print bought_items_per_session_df.head()
    print len(bought_items_per_session_df)
    
    
    new_merged_df = pd.merge(merged_df,bought_items_per_session_df,on='SessID',how='outer')
    new_merged_df.ItemID.fillna(0,inplace=True)
    
    new_merged_df['Item_Bought_'+str(item_name)] = new_merged_df['ItemID'].apply(isBought,args=(item_name,))
    
    
    trainer_df = new_merged_df[['SessID',str('numclicks_'+str(item_name)),'NumClicks','Item_Bought_'+str(item_name)]]
    
    
    trainer_df = trainer_df.fillna({'NumClicks': 0})
    trainer_df = trainer_df.fillna({'numclicks_'+str(item_name): 0})
    print len(trainer_df)
    
    print 'trainer_df'
    
    print trainer_df['Item_Bought_'+str(item_name)].value_counts()
    
    trainer_df[['SessID']] = trainer_df[['SessID']].astype(int)
    print 'lennnL ' + str(len(trainer_df[(trainer_df['Item_Bought_'+str(item_name)] > 0)]))
    
    
    
    #use the same model as before
    import numpy as np
    from sklearn import tree
    from sklearn.naive_bayes import MultinomialNB
    
    
    my_tree_one = joblib.load(dt_file) 
    
    #Decision Tree
    # Create the target and features numpy arrays: target, features_one
    target = trainer_df['Item_Bought_'+str(item_name)].values
    features_one = trainer_df[['numclicks_'+str(item_name), 'NumClicks']].values
    
    print 'features_one type: ' + str(type(features_one))
    
    print 'prediction\n'
    my_prediction = my_tree_one.predict(features_one)
    print my_prediction
    print 'len my prediction: ' + str(len(my_tree_one.predict(features_one)))
    print 'len target: ' + str(len(target))
    print target
    #value = 1
    counter = 0
    for value in target:
        if value == 1:
            print 'there is a 1 in target at index ' + str(counter)
        counter = counter + 1
    counter = 0
    for value in my_prediction:
        if value == 1:
            print 'there is a 1 in my_prediction at index ' + str(counter)
        counter = counter + 1
        
    
    print (np.mean(my_tree_one.predict(features_one) == target))
    '''
    if value in target[:, 0]:
        print 'there is a one in target'
    if value in my_prediction[:, 0]:
        print 'there is a one in my [rediction'
    '''
    # Look at the importance and score of the included features
    #print(my_tree_one.feature_importances_)
    #print(my_tree_one.score(features_one, target))
    
    '''
    #Naive Bayes
    columns = trainer_df.columns[1:3]
    print str(columns)
    
    
    X = np.asarray(trainer_df[columns])
    y = np.asarray(trainer_df['Item_Bought_'+str(item_name)].transpose())
    
    nb = joblib.load(nb_file) 
    print (nb.predict(np.asarray(trainer_df[columns])))
    print trainer_df['Item_Bought_'+str(item_name)].head()
    print (np.mean(nb.predict(np.asarray(trainer_df[columns])) == trainer_df['Item_Bought_'+str(item_name)]))
    #nb = MultinomialNB()
    #nb.fit(X, y)
    #print (nb.predict(np.asarray(trainer_df[columns])))
    #print (np.mean(nb.predict(np.asarray(trainer_df[columns])) == trainer_df['Item_Bought_'+str(item_name)]))
    
    
    '''
    
    
    
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
    
'''
    item_numclicks_df = make_item_numclicks_df_wrapper(clicks_df,item_name,cache_name)
    
    
    
    print item_numclicks_df.head()
    print len(item_numclicks_df)
    
    
    total_clicks_df_begin = time.time()
    total_clicks_df = make_total_clicks_df_wrapper(clicks_df,cache_name)
    total_clicks_df_end = time.time()
    
    print total_clicks_df.head()
    print len(total_clicks_df)
    
    merged_df = pd.merge(item_numclicks_df,total_clicks_df,on='SessID',how='outer')
    
    merged_df = merged_df[['SessID','NumClicks','numclicks_'+str(item_name)]]
    #new_merged_df.'numclicks_'+str(item_name).fillna(0,inplace=True)
    
    
    print 'merged_df'
    print merged_df.head()
    print len(merged_df)
'''
    

    
'''
    #get a bought_items_per_session_df
    bought_items_per_session_df = make_items_bought_per_session_df(buys_df)
    
    print 'bought_items_per_session_df'
    print bought_items_per_session_df.head()
    print len(bought_items_per_session_df)
    
    
    new_merged_df = pd.merge(feature_map_df,bought_items_per_session_df,on='SessID',how='outer')
    new_merged_df.ItemID.fillna(0,inplace=True)
    new_merged_df['Item_Bought_'+str(item_name)] = new_merged_df['ItemID'].apply(isBought,args=(item_name,))
    trainer_df = new_merged_df[['SessID',str('numclicks_'+str(item_name)),'NumClicks','Item_Bought_'+str(item_name)]]
    
    
    trainer_df = trainer_df.fillna({'NumClicks': 0})
    trainer_df = trainer_df.fillna({'numclicks_'+str(item_name): 0})
    print len(trainer_df)
'''
    
    
    
#trainer_df[['SessID']] = trainer_df[['SessID']].astype(int)
#print 'lennnL ' + str(len(trainer_df[(trainer_df['Item_Bought_'+str(item_name)] > 0)]))

#trainer_df_filename = cache_name + 'trainer_df.csv'
#trainer_df.head().to_csv(trainer_df_filename)    
    
    