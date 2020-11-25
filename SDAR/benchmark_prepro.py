import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('TkAgg')
import numpy as np

mean, cov = [0, 0], [(1, .6), (.6, 1)]
x, y = np.random.multivariate_normal(mean, cov, 100).T
y += x + 1
f, ax = plt.subplots(figsize=(6, 6))
ax.scatter(x, y, c=".3")
ax.plot([-3, 3], [-3, 3], ls="--", c=".3")
ax.set(xlim=(-3, 3), ylim=(-3, 3))
plt.show()

def prep_data(dataset, covariates_set, data_start, train = True):
    #print("train: ", train)
    time_len = dataset.shape[1]
    input_size = window_size - stride_size
    windows = (time_len - window_size + 1)
    for i in range(dataset.shape[0]):
        data = dataset[i,:,:]
        covariates = covariates_set
        if train:
            data[data > 1] = 0.99
        # print("time_len: ", time_len)
        # print("windows pre: ", windows_per_series.shape)
        # if train: windows -= (data_start+stride_size-1) // stride_size
        # print("data_start: ", data_start.shape)
        # print(data_start)
        # print("windows: ", windows_per_series.shape)
        # print(windows_per_series)
        # total_windows = np.sum(windows_per_series)
        x_input_temp = np.zeros((windows, window_size, 1 + num_covariates + 1), dtype='float32')
        label_temp = np.zeros((windows, window_size), dtype='float32')
        v_input_temp = np.zeros((windows, 2), dtype='float32')
        # cov = 3: ground truth + age + day_of_week + hour_of_day + num_series
        # cov = 4: ground truth + age + day_of_week + hour_of_day + month_of_year + num_series
        count = 0
        if not train:
            covariates = covariates[-time_len:]
        for i in range(windows):
            if train:
                window_start = i + data_start
            else:
                window_start = i
            window_end = window_start + window_size
            '''
            print("x: ", x_input[count, 1:, 0].shape)
            print("window start: ", window_start)
            print("window end: ", window_end)
            print("data: ", data.shape)
            print("d: ", data[window_start:window_end-1, series].shape)
            '''
            x_input_temp[count, 1:, 0] = data[window_start:window_end - 1]
            x_input_temp[count, :, 1:1 + num_covariates] = covariates[window_start:window_end, :]
            # x_input[count, :, -1] = series
            label_temp[count, :] = data[window_start:window_end]
            nonzero_sum = (x_input_temp[count, 1:input_size, 0] != 0).sum()
            if nonzero_sum == 0:
                v_input_temp[count, 0] = 0
            else:
                v_input_temp[count, 0] = 1
                x_input_temp[count, :, 0] = x_input_temp[count, :, 0] / v_input_temp[count, 0]
                if train:
                    label_temp[count, :] = label_temp[count, :] / v_input_temp[count, 0]
            count += 1
    prefix = os.path.join(save_path, 'train_' if train else 'test_')
    np.save(prefix+'data_'+save_name, x_input)
    np.save(prefix+'v_'+save_name, v_input)
    np.save(prefix+'label_'+save_name, label)