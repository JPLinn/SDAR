import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
import matplotlib
import matplotlib.pyplot as plt
# matplotlib.use('TkAgg')
import properscoring as ps
from scipy.stats import norm
from statsmodels.distributions.empirical_distribution import ECDF
from math import sqrt
import pickle

def CRPS(q, ob, step):
    qbin_area = np.zeros(ob.size)
    length = np.zeros([q.shape[0], q.shape[1]+1])
    length[:,1:-1] = np.diff(q)
    length[:,0], length[:,q.shape[1]] = q[:,0], 1-q[:,q.shape[1]-1]
    height = np.power(np.arange(step, 1+0.1*step, step),2)
    qbin2_area = np.matmul(length, height)
    for i in range(ob.size):
        temp = np.where(q[i,:] <= ob[i])[0]
        if temp.size != 0:
            idx = temp.max()
            length[i,idx + 1] -= ob[i] - q[i,idx]
        else:
            idx = -1
            length[i,0] -= ob[i]
        height = np.arange((idx+2)*step, 1+0.1*step, step)
        qbin_area[i] = length[i,-height.size:].dot(height)
    ob_area = (1.-ob)
    return qbin2_area - 2*qbin_area + ob_area

def PeEn(test_len, dataset, n_sample, period):
    pe_en = np.zeros([test_len, n_sample])
    idx = dataset.size - test_len
    count = 0
    for i in range(idx,dataset.size):
        for j in range(n_sample):
            pe_en[count, j] = dataset[i - (j+1)*period]
        count +=1
    return pe_en

def relia_curve_qr(y_and_yh, bins=10):
    norm_prob = np.arange(0.1, 0.99, 0.1)
    empi_prob = np.zeros(bins-1)
    for yi in y_and_yh:
        occur = np.array(np.where(yi[1:]>=yi[0]))
        if occur.size != 0:
            if occur.min() > 49:
                empi_prob[((occur.min()-50)//5):] += 1
            else:
                occur = np.array(np.where(yi[1:]<=yi[0]))
                if occur.size != 0:
                    empi_prob[((48-occur.max())//5):] += 1
    empi_prob /= y_and_yh.shape[0]
    return np.vstack((norm_prob, empi_prob))

def relia_curve_qr(y_and_yh, bins=10):
    norm_prob = np.arange(0.1, 0.99, 0.1)
    empi_prob = np.zeros(bins-1)
    for yi in y_and_yh:
        occur = np.array(np.where(yi[1:]>=yi[0]))
        if occur.size != 0:
            empi_prob[(occur.min()//10):] += 1
    empi_prob /= y_and_yh.shape[0]
    return np.vstack((norm_prob, empi_prob))

def relia_curve_ensemble(y, samples, bins=10):
    norm_prob = np.arange(0.1, 0.99, 0.1)
    empi_prob = np.zeros(bins-1)
    for i in range(y.size):
        ecdf = ECDF(samples[i,:])
        prob_ob = ecdf(y[i])
        if prob_ob < 0.5:
            empi_prob[int(100*(0.5-prob_ob)//5):] += 1
        else:
            empi_prob[int(100*(prob_ob-0.5)//5):] += 1
    empi_prob /= y.size
    return np.vstack((norm_prob, empi_prob))

def relia_curve_ensemble(y, samples, bins=10):
    norm_prob = np.arange(0.1, 0.99, 0.1)
    empi_prob = np.zeros(bins-1)
    for i in range(y.size):
        ecdf = ECDF(samples[i,:])
        prob_ob = ecdf(y[i])
        if prob_ob < 0.9:
            empi_prob[int(100*prob_ob//10):] += 1
    empi_prob /= y.size
    return np.vstack((norm_prob, empi_prob))

train_start = '2012-04-01 01:00:00'
train_end = '2014-02-01 00:00:00'
train_start_qr = '2013-02-01 01:00:00'
test_start = '2014-02-01 01:00:00'  # need additional 10 days as given info
test_end = '2014-07-01 00:00:00'
test_len = 1200

data_path = './data/solar/Zone1.csv'

data_frame = pd.read_csv(data_path, sep=",", index_col=0, parse_dates=True)
data_frame['L1'] = data_frame['power'].shift()
data_frame['L2'] = data_frame['power'].shift(2)
data_frame['L3'] = data_frame['power'].shift(3)
data_frame['L4'] = data_frame['power'].shift(4)
data_frame['L5'] = data_frame['power'].shift(5)
data_frame['Hour'] = data_frame.index.hour

train_set = data_frame[train_start:train_end]
test_set = data_frame[test_start:test_end]
test_set.insert(1,'unit', 1)

np.save('./data/temp/test',test_set.values)
np.save('./data/temp/train', train_set.values)

##### QR #####
quantiles = np.arange(.01, .999, .01)

mod_h1 = smf.quantreg('power ~ ssrd + L1 + L2 + Hour', data_frame[train_start_qr:train_end])
mod_h2 = smf.quantreg('power ~ ssrd + L2 + L3 + Hour', data_frame[train_start_qr:train_end])
mod_h3 = smf.quantreg('power ~ ssrd + L3 + L4 + Hour', data_frame[train_start_qr:train_end])
mod_h4 = smf.quantreg('power ~ ssrd + L4 + L5 + Hour', data_frame[train_start_qr:train_end])

def fit_model_h1(q):
    res = mod_h1.fit(q=q, max_iter=1000)
    return [q, *res.params.values.tolist()]

def fit_model_h2(q):
    res = mod_h2.fit(q=q, max_iter=1000)
    return [q, *res.params.values.tolist()]

def fit_model_h3(q):
    res = mod_h3.fit(q=q, max_iter=1000)
    return [q, *res.params.values.tolist()]

def fit_model_h4(q):
    res = mod_h4.fit(q=q, max_iter=1000)
    return [q, *res.params.values.tolist()]

models_h1 = [fit_model_h1(x) for x in quantiles]
models_h2 = [fit_model_h2(x) for x in quantiles]
models_h3 = [fit_model_h3(x) for x in quantiles]
models_h4 = [fit_model_h4(x) for x in quantiles]

models_h1 = pd.DataFrame(models_h1, columns=['q', 'a', 'b_ssrd', 'b_L1', 'b_L2', 'b_Hour'])
models_h2 = pd.DataFrame(models_h2, columns=['q', 'a', 'b_ssrd', 'b_L1', 'b_L2', 'b_Hour'])
models_h3 = pd.DataFrame(models_h3, columns=['q', 'a', 'b_ssrd', 'b_L1', 'b_L2', 'b_Hour'])
models_h4 = pd.DataFrame(models_h4, columns=['q', 'a', 'b_ssrd', 'b_L1', 'b_L2', 'b_Hour'])

Yh_h1 = np.matmul(test_set[['unit', 'ssrd', 'L1', 'L2', 'Hour']].values,models_h1[['a', 'b_ssrd', 'b_L1', 'b_L2', 'b_Hour']].values.T)
Yh_h2 = np.matmul(test_set[['unit', 'ssrd', 'L2', 'L3', 'Hour']].values,models_h2[['a', 'b_ssrd', 'b_L1', 'b_L2', 'b_Hour']].values.T)
Yh_h3 = np.matmul(test_set[['unit', 'ssrd', 'L3', 'L4', 'Hour']].values,models_h3[['a', 'b_ssrd', 'b_L1', 'b_L2', 'b_Hour']].values.T)
Yh_h4 = np.matmul(test_set[['unit', 'ssrd', 'L4', 'L5', 'Hour']].values,models_h4[['a', 'b_ssrd', 'b_L1', 'b_L2', 'b_Hour']].values.T)

Yh_h1.sort()
Yh_h1[Yh_h1<0.] = 0.
Yh_h1[Yh_h1>1.] = 1.

Yh_h2.sort()
Yh_h2[Yh_h2<0.] = 0.
Yh_h2[Yh_h2>1.] = 1.

Yh_h3.sort()
Yh_h3[Yh_h3<0.] = 0.
Yh_h3[Yh_h3>1.] = 1.

Yh_h4.sort()
Yh_h4[Yh_h4<0.] = 0.
Yh_h4[Yh_h4>1.] = 1.

crps_qr_h1 = CRPS(Yh_h1,test_set['power'].values,0.01).mean()
crps_qr_h2 = CRPS(Yh_h2,test_set['power'].values,0.01).mean()
crps_qr_h3 = CRPS(Yh_h3,test_set['power'].values,0.01).mean()
crps_qr_h4 = CRPS(Yh_h4,test_set['power'].values,0.01).mean()

sharp50_qr_h1 = (Yh_h1[:,74] - Yh_h1[:,24]).mean()
sharp90_qr_h1 = (Yh_h1[:,94] - Yh_h1[:,4]).mean()

sharp50_qr_h2 = (Yh_h2[:,74] - Yh_h2[:,24]).mean()
sharp90_qr_h2 = (Yh_h2[:,94] - Yh_h2[:,4]).mean()

sharp50_qr_h3 = (Yh_h3[:,74] - Yh_h3[:,24]).mean()
sharp90_qr_h3 = (Yh_h3[:,94] - Yh_h3[:,4]).mean()

sharp50_qr_h4 = (Yh_h4[:,74] - Yh_h4[:,24]).mean()
sharp90_qr_h4 = (Yh_h4[:,94] - Yh_h4[:,4]).mean()

output_str = 'power ~  ssrd + L1:\n'

#print('crps_qr_h1:{:.3f}\ncrps_qr_h2:{:.3f}\ncrps_qr_h3:{:.3f}\ncrps_qr_h4:{:.3f}\n'.format(crps_qr_h1,crps_qr_h2,crps_qr_h3,crps_qr_h4))
output_str += 'crps_qr_h1:{:.3f}\ncrps_qr_h2:{:.3f}\ncrps_qr_h3:{:.3f}\ncrps_qr_h4:{:.3f}\n'.format(crps_qr_h1,crps_qr_h2,crps_qr_h3,crps_qr_h4)

rc_qr_h1 = relia_curve_qr(np.hstack((test_set.values[:,0][:,np.newaxis],Yh_h1)))
rcs_qr_h1 = np.abs(np.diff(rc_qr_h1, axis=0)).mean()
output_str += 'rc_h1:' + ','.join(str(i) for i in rc_qr_h1[1,:]) + '\n' + 'rcs_h1:{:.4f}'.format(rcs_qr_h1) + '\n'

rc_qr_h2 = relia_curve_qr(np.hstack((test_set.values[:,0][:,np.newaxis],Yh_h2)))
rcs_qr_h2 = np.abs(np.diff(rc_qr_h2, axis=0)).mean()
output_str += 'rc_h2:' + ','.join(str(i) for i in rc_qr_h2[1,:]) + '\n' + 'rcs_h2:{:.4f}'.format(rcs_qr_h2) + '\n'

rc_qr_h3 = relia_curve_qr(np.hstack((test_set.values[:,0][:,np.newaxis],Yh_h3)))
rcs_qr_h3 = np.abs(np.diff(rc_qr_h3, axis=0)).mean()
output_str += 'rc_h3:' + ','.join(str(i) for i in rc_qr_h3[1,:]) + '\n' + 'rcs_h3:{:.4f}'.format(rcs_qr_h3) + '\n'

rc_qr_h4 = relia_curve_qr(np.hstack((test_set.values[:,0][:,np.newaxis],Yh_h4)))
rcs_qr_h4 = np.abs(np.diff(rc_qr_h4, axis=0)).mean()
output_str += 'rc_h4:' + ','.join(str(i) for i in rc_qr_h4[1,:]) + '\n' + 'rcs_h4:{:.4f}'.format(rcs_qr_h4) + '\n'

bins = np.linspace(1,11,11)

pesu_s_qr_h1 = np.zeros((Yh_h1.shape[0],100))
pesu_s_qr_h1[:,0] = (Yh_h1[:,0])/2
pesu_s_qr_h1[:,-1] = (1+Yh_h1[:,-1])/2
pesu_s_qr_h1[:,1:-1] = (Yh_h1[:,:-1]+Yh_h1[:,1:])/2
rmse_qr_h1 = sqrt(np.square(pesu_s_qr_h1.mean(axis=1)-test_set['power'].values).mean())
rh_qr_h1 = np.sum(pesu_s_qr_h1.T<test_set['power'].values,axis=0)//10 + 1 #from 0~100 to 1~51
plt.hist(rh_qr_h1,bins,alpha=0.5)

pesu_s_qr_h2 = np.zeros((Yh_h2.shape[0],100))
pesu_s_qr_h2[:,0] = (Yh_h2[:,0])/2
pesu_s_qr_h2[:,-1] = (1+Yh_h2[:,-1])/2
pesu_s_qr_h2[:,1:-1] = (Yh_h2[:,:-1]+Yh_h2[:,1:])/2
rmse_qr_h2 = sqrt(np.square(pesu_s_qr_h2.mean(axis=1)-test_set['power'].values).mean())
rh_qr_h2 = np.sum(pesu_s_qr_h2.T<test_set['power'].values,axis=0)//10 + 1 #from 0~100 to 1~51
plt.hist(rh_qr_h2,bins,alpha=0.5)

pesu_s_qr_h3 = np.zeros((Yh_h3.shape[0],100))
pesu_s_qr_h3[:,0] = (Yh_h3[:,0])/2
pesu_s_qr_h3[:,-1] = (1+Yh_h3[:,-1])/2
pesu_s_qr_h3[:,1:-1] = (Yh_h3[:,:-1]+Yh_h3[:,1:])/2
rmse_qr_h3 = sqrt(np.square(pesu_s_qr_h3.mean(axis=1)-test_set['power'].values).mean())
rh_qr_h3 = np.sum(pesu_s_qr_h3.T<test_set['power'].values,axis=0)//10 + 1 #from 0~100 to 1~51
plt.hist(rh_qr_h3,bins,alpha=0.5)

pesu_s_qr_h4 = np.zeros((Yh_h4.shape[0],100))
pesu_s_qr_h4[:,0] = (Yh_h4[:,0])/2
pesu_s_qr_h4[:,-1] = (1+Yh_h4[:,-1])/2
pesu_s_qr_h4[:,1:-1] = (Yh_h4[:,:-1]+Yh_h4[:,1:])/2
rmse_qr_h4 = sqrt(np.square(pesu_s_qr_h4.mean(axis=1)-test_set['power'].values).mean())
rh_qr_h4 = np.sum(pesu_s_qr_h4.T<test_set['power'].values,axis=0)//10 + 1 #from 0~100 to 1~51
plt.hist(rh_qr_h4,bins,alpha=0.5)

plt.plot(rc_qr_h1[0,:],rc_qr_h1[1,:], label='QR_h1')
plt.plot(rc_qr_h2[0,:],rc_qr_h2[1,:], label='QR_h2')
plt.plot(rc_qr_h3[0,:],rc_qr_h3[1,:], label='QR_h3')
plt.plot(rc_qr_h4[0,:],rc_qr_h4[1,:], label='QR_h4')
plt.plot([0,0],[1,1])
plt.show()

qr_s2 = {}
qr_s2['crps'] = [crps_qr_h1, crps_qr_h2, crps_qr_h3, crps_qr_h4]
qr_s2['sh50'] = [sharp50_qr_h1, sharp50_qr_h2, sharp50_qr_h3, sharp50_qr_h4]
qr_s2['sh90'] = [sharp90_qr_h1, sharp90_qr_h2, sharp90_qr_h3, sharp90_qr_h4]
qr_s2['rmse'] = [rmse_qr_h1, rmse_qr_h2, rmse_qr_h3, rmse_qr_h4]
qr_s2['rc'] = (rc_qr_h1, rc_qr_h2, rc_qr_h3, rc_qr_h4)
qr_s2['rh'] = (rh_qr_h1, rh_qr_h2, rh_qr_h3, rh_qr_h4)
print(output_str)

# with open("qr_tune.txt","a+") as f:
#     f.write(output_str)

pe_en = PeEn(test_len, data_frame['power'].values, 10, 8)
peEn_crps = ps.crps_ensemble(test_set['power'].values,pe_en).mean()
peEn_rmse = sqrt(np.square(pe_en.mean(axis=1)-test_set['power'].values).mean())
peEn_rc = relia_curve_ensemble(test_set['power'].values,pe_en)
q5_pe_en = np.percentile(pe_en, 5, axis = 1)
q25_pe_en = np.percentile(pe_en, 25, axis = 1)
q75_pe_en = np.percentile(pe_en, 75, axis = 1)
q95_pe_en = np.percentile(pe_en, 95, axis = 1)
sharp50_pe_en = (q75_pe_en - q25_pe_en).mean()
sharp90_pe_en = (q95_pe_en - q5_pe_en).mean()
rh_pe_en = np.sum((pe_en.T<test_set['power'].values),axis=0) + 1
plt.hist(rh_pe_en,bins)

pe_en_s2 = {}
pe_en_s2['crps'] = peEn_crps
pe_en_s2['rmse'] = peEn_rmse
pe_en_s2['sh50'] = sharp50_pe_en
pe_en_s2['sh90'] = sharp90_pe_en
pe_en_s2['rc'] = peEn_rc
pe_en_s2['rh'] = rh_pe_en

s2 = {}
s2['qr'] = qr_s2
s2['pe_en'] = pe_en_s2

with open('s2.pickle', 'wb') as f:
    pickle.dump(s2, f)

'''
np.save('./data/temp/test',test_set.values)
np.save('./data/temp/Yh', Yh)
np.save('./data/temp/pe_en', pe_en)
np.save('./data/temp/qr_rc', qr_rc)
'''



