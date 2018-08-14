import os
import datetime
import tensorflow as tf
from tffm import TFFMRegressor
from data import Data
from save import Save
from featurelize import Featurelize
from utils import path_create,get_date
from sklearn import metrics
import pickle

def get_export_path():
    path = os.path.join(os.getcwd(),'models')
    dirs = os.listdir(path)
    if not dirs:
        version = '1'.zfill(5)
    else:
        version  = str(int(max(dirs)) + 1).zfill(5)
    export_path = os.path.join(path,version)
    return export_path,version

def data_load(args):
    train_file_path = 'data/feature_dict_train.pkl'
    test_file_path = 'data/feature_dict_test.pkl'
    if not args.raw:
        feature_dict_train = pickle.load(open(train_file_path, 'rb'))
        feature_dict_test = pickle.load(open(test_file_path, 'rb'))
    else:
        ## determine the train date
        if args.date:
            train_date = args.date
        else:
            train_date = get_date(1)
        ## load date
        print('train_date is {}'.format(train_date))
        data = Data(train_date,args.gap,args.sample,args.reserve)
        featurelize = Featurelize(args.feature)
        df, test_date = data.load()
        train = df[df['tdate'] != test_date]
        test = df[df['tdate'] == test_date]
        feature_dict_train = featurelize.features_dict_prepare(train)
        feature_dict_test = featurelize.features_dict_prepare(test)
        ## dump data
        pickle.dump(feature_dict_train, open(train_file_path, 'wb'))
        pickle.dump(feature_dict_test, open(test_file_path, 'wb'))
    x_train, y_train = featurelize.input_data_prepare(feature_dict_train)
    x_test, y_test = featurelize.input_data_prepare(feature_dict_test)
    return x_train, y_train, x_test, y_test

def train(export_path, data, version, args):
    x_train, y_train, x_test, y_test = data
    y_train = y_train * 2 - 1
    print("train date shape is {}".format(x_train.shape))

    if args.log:
        log_dir = './log'
    else:
        log_dir = None

    model = TFFMRegressor(
        order=2,
        rank=args.rank,
        optimizer=tf.train.AdamOptimizer(learning_rate=0.0001),
        n_epochs=1,
        batch_size=128,
        log_dir = log_dir,
        init_std=0.01,
        reg=0.5,
        input_type='sparse'
        )

    base_path = 'ckpt/{}'.format(version)
    path_create(base_path)
    model_path = os.path.join(base_path,'state.tf')
    print('model path is {}'.format(model_path))
    model.core.set_num_features(x_train.shape[1])
    model.fit(x_train, y_train, show_progress=True)
    print('train the model successfully')
    model.save_state(model_path)
    print('checkpoint save successfully')

    if args.save:
        save=Save(model,export_path)
        save.save()
    return model


def eval(model,data):
    x_train, y_train, x_test, y_test = data
    predictions = model.predict(x_train)
    fpr, tpr, thresholds = metrics.roc_curve(y_train, (predictions + 1)/2, pos_label=1)
    auc = metrics.auc(fpr, tpr)
    print('train auc :{}'.format(auc))

    predictions = model.predict(x_test)
    fpr, tpr, thresholds = metrics.roc_curve(y_test, (predictions + 1)/2, pos_label=1)
    auc = metrics.auc(fpr, tpr)
    print('auc :{}'.format(auc))

def train_main(args):
    ## determin the model version and path
    export_path, version = get_export_path()
    ## data load
    data = data_load(args)
    ## model train
    model = train(export_path, data, version, args)
    ## model eval
    eval(model,data)
    ## model destory
    model.destroy()


