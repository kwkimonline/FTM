import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader
from aif360.datasets import AdultDataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler


class WithIndexDataset(TensorDataset): 
    def __init__(self, x, y, s=None):
        self.x, self.y, self.s = x, y, s
    def __len__(self):
        return len(self.x)
    def __getitem__(self, index):
        if self.s != None:
            return self.x[index], self.y[index], self.s[index], index
        else:
            return self.x[index], self.y[index], index


def load_data(seed, source, target, batch_size):
    
    print('[Info] Loading data')
    traindset_source, trainloader_source = TableData(seed, source, 'train', batch_size)
    trainevaldset_source, trainevalloader_source = TableData(seed, source, 'traineval', batch_size)
    valdset_source, valloader_source = TableData(seed, source, 'val', batch_size)
    testdset_source, testloader_source = TableData(seed, source, 'test', batch_size)

    traindset_target, trainloader_target = TableData(seed, target, 'train', batch_size)
    trainevaldset_target, trainevalloader_target = TableData(seed, target, 'traineval', batch_size)
    valdset_target, valloader_target = TableData(seed, target, 'val', batch_size)
    testdset_target, testloader_target = TableData(seed, target, 'test', batch_size)

    if 'Adult' in source:
        input_dim = 101

    source_dsets = (traindset_source, trainevaldset_source, valdset_source, testdset_source)
    source_dloaders = (trainloader_source, trainevalloader_source, valloader_source, testloader_source)
    target_dsets = (traindset_target, trainevaldset_target, valdset_target, testdset_target)
    target_dloaders = (trainloader_target, trainevalloader_target, valloader_target, testloader_target)
    train_source_size, train_target_size = len(traindset_source), len(traindset_target)

    return source_dsets, source_dloaders, target_dsets, target_dloaders, (input_dim, train_source_size, train_target_size)


def TableData(seed, name, mode, batch_size):

    dataset_name = name.split('_')[0]


    if dataset_name == 'Adult': # Adult
    
        label_map = {1.0: '>50K', 0.0: '<=50K'}
        protected_attribute_maps = [{1.0: 'Male', 0.0: 'Female'}]
        categorical_features = ['workclass', 'education', 'marital-status',
                                'occupation', 'relationship', 'native-country', 'race']
        raw_data = AdultDataset(protected_attribute_names=['sex'],
                                categorical_features=categorical_features,
                                privileged_classes=[['Male']], metadata={'label_map': label_map,
                                                                         'protected_attribute_maps': protected_attribute_maps})

        all_features = np.concatenate([raw_data.features[:, :2], raw_data.features[:, 3:]], axis=1)
        all_sensitives = raw_data.features[:, 2]
        all_labels = raw_data.labels.flatten()
        feature_names = np.concatenate([raw_data.feature_names[:2], raw_data.feature_names[3:]], axis=0)

        # train/test split
        np.random.seed(seed)
        all_ids = np.random.permutation(all_features.shape[0])
        train_ids = all_ids[:32561]
        test_ids = all_ids[32561:]
                
        train_features, train_sensitives, train_labels = all_features[train_ids], all_sensitives[train_ids], all_labels[train_ids]
        test_features, test_sensitives, test_labels = all_features[test_ids], all_sensitives[test_ids], all_labels[test_ids]
        sub_train_features, val_features, sub_train_sensitives, val_sensitives, sub_train_labels, val_labels = train_test_split(train_features, train_sensitives, train_labels, test_size=0.2, random_state=2022)
        
        # scaling
        scaler = MinMaxScaler()
        scaler.fit(train_features)
        train_features = scaler.transform(train_features)
        val_features = scaler.transform(val_features)
        test_features = scaler.transform(test_features)
        
        # to torch
        train_features, train_sensitives, train_labels = torch.from_numpy(train_features).float(), torch.from_numpy(train_sensitives), torch.from_numpy(train_labels)
        val_features, val_sensitives, val_labels = torch.from_numpy(val_features).float(), torch.from_numpy(val_sensitives), torch.from_numpy(val_labels)
        test_features, test_sensitives, test_labels = torch.from_numpy(test_features).float(), torch.from_numpy(test_sensitives), torch.from_numpy(test_labels)

        assert train_features.size(0) == train_labels.size(0)

        # divide by sensitive attribute
        sensitive_id = int(name.split('_')[1])
        if (mode == 'train') or (mode == 'traineval'):
            dset = WithIndexDataset(train_features[train_sensitives == sensitive_id], train_labels[train_sensitives == sensitive_id])
        elif mode == 'val':
            dset = WithIndexDataset(val_features[val_sensitives == sensitive_id], val_labels[val_sensitives == sensitive_id])
        elif mode == 'test':
            dset = WithIndexDataset(test_features[test_sensitives == sensitive_id], test_labels[test_sensitives == sensitive_id])
        dloader = DataLoader(dset, batch_size, num_workers=4, 
                            shuffle=True if mode == 'train' else False, drop_last=True if mode == 'train' else False)


    return dset, dloader

