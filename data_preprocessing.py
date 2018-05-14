# preprocessing(get access daily return)
import pickle
import numpy as np

pkl_file = open('Index_close.pkl','rb')
zz500_close = pickle.load(pkl_file)[['000905']].loc[20050105:20180104]
pkl_file = open('close.pkl','rb')
close = pickle.load(pkl_file).loc[20050105:20180104]
close = close.join(zz500_close)
del(zz500_close)
daily_return = close.pct_change(1).iloc[1:-1,:]
zz500_return = daily_return.iloc[:,-1]
access = daily_return.iloc[:,0:-1].sub(zz500_return,axis=0)
del(close)
del(daily_return)

clean_data = [access,zz500_return]
output = open('clean_data.pkl', 'wb')
pickle.dump(clean_data, output, -1)
output.close()