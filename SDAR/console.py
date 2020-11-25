x_smooth = np.linspace(0,1,300)
x = np.linspace(0,1,11)
qr_rc_smooth = make_interp_spline(x,qr_rc)(x_smooth)
pe_rc_smooth = make_interp_spline(x,pe_rc)(x_smooth)
da_rc = np.array([0, 0.11194653, 0.22807018, 0.33166249, 0.42857143, 0.53884712, 0.65580618, 0.74018379, 0.81453634, 0.88137009, 1])
da_rc_smooth = make_interp_spline(x,da_rc)(x_smooth)

plt.plot([0,1],[0,1],'k--', label='Perfect', linewidth=1)
plt.plot(x_smooth, pe_rc_smooth, label='PeEn', linewidth=2)
plt.plot(x_smooth, qr_rc_smooth, label='QR', linewidth=2)
plt.plot(x_smooth, da_rc_smooth, label='DeepAR', linewidth=2)
plt.rcParams['xtick.direction'] = 'in'
ul = x + 0.05
ll = x - 0.05
plt.xlim((0,1))
plt.ylim((0,1))
# plt.fill_between(x, ll, ul, color='blue', alpha=0.2)
plt.legend(loc=0)
plt.xlabel('Nominal proportion')
plt.ylabel('Observed proportion')

plt.hist([pe_rh, qr_rh, da_rh], density=True, rwidth=0.5, label=['PeEn','QR','DeepAR'])
ax = plt.axes()
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
plt.rcParams['ytick.direction'] = 'in'
x_lim_s = plt.xlim()
t = np.arange(0,12,0.1)
y = np.ones(t.shape)*0.1
plt.plot(t,y,'k--')
plt.xlim(x_lim_s)
plt.legend(loc=(0.6,0.7))
plt.xticks([])
plt.xlabel('Rank')
plt.ylabel('Relative frequency')

plt.savefig("data/rank_histogram.png",dpi=500,bbox_inches = 'tight')