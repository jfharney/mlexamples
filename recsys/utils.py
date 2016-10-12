import pandas as pd
import time
import sys
import dateutil.parser
import datetime



def func(word):
    return len(word)-1


def func2(nums):
    text = ''
    text_list = []
    for num in nums:
        #print '\tin nums ' + str(num)
        text += str(num) + ','
        text_list.append(num)
    #print 'text: ' + str(text)
    #print 'text_list: ' + str(text_list)
    #return nums.count()
    return text_list

def isBought(nums,item):
    
    #print 'item: ' + item
    
    isBought = 0
    
    #print nums
    if type(nums) is list:
        #print len(nums)
        if item in nums:
            #print 'in list'
            isBought = 1
        
    
    return isBought

def func3(nums):
    #for num in nums:
    #    print '\tin nums ' + str(num)
    return []

def func5(nums,var):
    count = 0
    for num in nums:
        #print '\tin num ' + str(num) + ' var: ' + str(var) + ' ' + str(type(num)) + ' ' + str(type(var))
        #if num == var:
        #print str(str(num) == str(var))
        
        if str(num) == str(var):
            count = count + 1
    return count


def func7(nums,var):
    count = 0
    #print '---in func7 for ' + str(type(nums)) + '---'
    #print str(nums.value_counts().keys())
    #print str(type(nums.value_counts().keys()))
    if var in nums.value_counts().keys():
    #    print 'in keys'
        count = count + nums.value_counts()[var]#1
        #print 'count is: ' + str(count)
    ##else:
    #    print 'not in keys'
    old_count = count
    '''
    if nums.value_counts()['i1'] != None:
    #s1 = nums.value_counts()['i1']
    #if s1 != None:
        s1 = nums.value_counts()['i1']
        print 'type: ' + str(type(s1))
        print 's1: ' + str(s1)
    '''   
    '''
    count = 0
    for num in nums:
        #print '\tin num ' + str(num) + ' var: ' + str(var) + ' ' + str(type(num)) + ' ' + str(type(var))
        #if num == var:
        #print str(str(num) == str(var))
        
        if str(num) == str(var):
            count = count + 1
    
    if (old_count != count):
        print 'old_count: ' + str(old_count) + ' new_Count' + str(count) + ' ' + str((old_count == count))
    '''
    
    return count


def count(nums):
    return nums.count()

def date_filter(timestamp):
    
    d = dateutil.parser.parse(timestamp)
    if d.month == 9:
        return 1
    else:
        return 0


def make_category_df(df):
    
    grouped = df.groupby('Category')
    newdf = pd.Series.to_frame(grouped['ItemID'].apply(func2))
    
    newdf['Category'] = newdf.index
    #category_dict = newdf.set_index('Category').to_dict()
    
    #print str(new_dict)
    return newdf
    

def make_category_dict(df):
    
    grouped = df.groupby('Category')
    newdf = pd.Series.to_frame(grouped['ItemID'].apply(func2))
    
    newdf['Category'] = newdf.index
    category_dict = newdf.set_index('Category').to_dict()

    return category_dict
    

def timestamp_filter():
    
    september_clicks_train_dir = '/Users/8xo/tutorials/ml-scikit/mlexamples/yoochoose-dataFull/recsys/'
    september_clicks_train_file = 'september_clicks_train.csv'
    september_clicks_test_file = 'september_clicks_test.csv'
    
    september_buys_train_dir = '/Users/8xo/tutorials/ml-scikit/mlexamples/yoochoose-dataFull/recsys/'
    september_buys_train_file = 'september_buys_train.csv'
    september_buys_test_file = 'september_buys_test.csv'
    
    september_buys_train = pd.read_csv(september_buys_train_dir+september_buys_train_file)
    september_clicks_train = pd.read_csv(september_clicks_train_dir+september_clicks_train_file)
    
    #print len(september_clicks_train['Category'].value_counts())
    september_clicks_train = september_clicks_train[['SessID','Timestamp','ItemID','Category']]
    #print(september_clicks_train.head())
    #new_clicks.to_csv('new_clicks.csv')
    
    category_dict = make_category_dict(september_clicks_train[['ItemID','Category']])
    
    print str(category_dict)
    
    category_df = make_category_df(september_clicks_train[['ItemID','Category']])
    
    print str(category_df)

def september_filter_df(dir,file, train=True, clicks=True):
    
    print 'september_filter_df'
    
    df = pd.read_csv(dir+file)
    
    if train:
        df = df[(df['Timestamp'] >= '2014-09-01 00:00:00') & (df['Timestamp'] < '2014-09-24 00:00:00')]
    else:
        df = df[(df['Timestamp'] >= '2014-09-24 00:00:00')]
    
    
    return df
    
    

def september_filter_file(dir, file, train=True, clicks=True):
    
    print 'september_filter_file'
    
    df = pd.read_csv(dir+file)
    
    if train:
        df = df[(df['Timestamp'] >= '2014-09-01 00:00:00') & (df['Timestamp'] < '2014-09-24 00:00:00')]
    else:
        df = df[(df['Timestamp'] >= '2014-09-24 00:00:00')]
    
    
    outfilename = 'sept_'
    
    if clicks:
        outfilename += 'clicks_'
    else:
        outfilename += 'buys_'
    
    if train:
        outfilename += 'train.csv'
    else:
        outfilename += 'test.csv'
    
    
    
    df.to_csv(outfilename)
    
    



    
    
    
    
    
    
    
'''
    #print buys_train.head()
    
    #clicks_train = pd.read_csv(clicks_train_dir+clicks_train_file)
    buys_train = pd.read_csv(buys_train_dir+buys_train_file)
    clicks_train = pd.read_csv(buys_train_dir+buys_train_file)

    

    print('-----buys-----')
    
    
    buys_train['Timestamp'] = pd.to_datetime(buys_train['Timestamp'])
    
    print 'MIN: \n' + str(buys_train['Timestamp'].min())
    
    september_buys_train = buys_train[(buys_train['Timestamp'] >= '2014-09-01 00:00:00') & (buys_train['Timestamp'] < '2014-09-24 00:00:00')]
    september_buys_test = buys_train[(buys_train['Timestamp'] >= '2014-09-24 00:00:00')]
    
    print (september_buys_train.head(2))
    print (september_buys_test.head(2))
    
    print(september_buys_train.iloc[1])
    
    print(september_buys_train['Price'].max())
    
    
    clicks_train = pd.read_csv(clicks_train_dir+clicks_train_file)
    september_clicks_train = clicks_train[(clicks_train['Timestamp'] >= '2014-09-01 00:00:00') & (clicks_train['Timestamp'] < '2014-09-24 00:00:00')]
    september_clicks_test = clicks_train[(clicks_train['Timestamp'] >= '2014-09-24 00:00:00')]
    
    #print(september_clicks_train['Category'].value_counts())
    
    september_buys_train.to_csv('september_buys_train.csv')
    september_buys_test.to_csv('september_buys_test.csv')
    september_clicks_train.to_csv('september_clicks_train.csv')
    september_clicks_test.to_csv('september_clicks_test.csv')
'''
    
    
    
   

#september_buys_train = buys_train[(buys_train['Timestamp'] > '2014-07-23 07:30:00') & (buys_train['Timestamp'] < '2014-07-23 09:00:00')]

#print(buys_train.iloc[1])

#print(buys_train[date_filter(buys_train['Timestamp'] == 1)])


'''
    select SessID, count(itemID)
    from t1
    groupby SessID
'''
    
    

'''
train = True
clicks = True
#df = september_filter_df(clicks_train_dir,clicks_train_file,train,clicks)

file = september_filter_file(clicks_train_dir,clicks_train_file,train,clicks)

clicks = False
train = True

file = september_filter_file(buys_train_dir,buys_train_file,train,clicks)
'''

'''
    print('---------------')
    
    #print(buys_train['Timestamp'])
    old = time.time()
    #buys_train['september_purchase'] = buys_train.Timestamp.apply(date_filter)
    new = time.time()
'''   
