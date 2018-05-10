# preprocessing(get access daily return)
import pickle
import numpy as np
import random

pkl_file = open('Index_close.pkl','rb')
zz500_close = pickle.load(pkl_file)[['000905']].loc[20150105:20180104]
pkl_file = open('close.pkl','rb')
close = pickle.load(pkl_file).loc[20150105:20180104]
close = close.join(zz500_close)
del(zz500_close)
daily_return = close.pct_change(1).iloc[1:-1,:]
zz500_return = daily_return.iloc[:,-1]
access = daily_return.iloc[:,0:-1].sub(zz500_return,axis=0)
del(close)
del(daily_return)
pkl_file = open('000905.pkl','rb')
comp = pickle.load(pkl_file)

n = 1000
day = 100
stock = 100
day_after = 10
x = np.zeros([n,day,stock])
y = np.zeros([n,day_after,stock])
z = np.zeros([n,day_after])
i = 0
while i < n:
    print(i)
    samples = np.random.randint(0,len(list(access.index[0:-day-day_after])),1)
    current_comp_index = len(comp[0][comp[0]<access.index[samples][0]])-1
    stocks = random.sample(comp[1][current_comp_index],stock)
    x[i] = access.loc[access.index[samples][0]:access.index[samples+day-1][0],stocks]
    y[i] = access.loc[access.index[samples+day][0]:access.index[samples+day+day_after-1][0],stocks]
    z[i] = zz500_return.loc[access.index[samples+day][0]:access.index[samples+day+day_after-1][0]]
    if sum(sum(np.isnan(x[i])))>0 or sum(sum(np.isnan(y[i])))>0:
        i = i-1
    i = i+1

clean_data = [x,y,z]

output = open('clean_data.pkl', 'wb')
pickle.dump(clean_data, output, -1)
output.close()