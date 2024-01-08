import pytest
import os
from day_2.data.make_dataset import load_data
from paths import _PATH_DATA, _PATH_DATA_RAW, _PATH_DATA_PROCESSED
import torch

@pytest.mark.skipif(not os.path.exists(_PATH_DATA_PROCESSED), reason="Data folder not found")
def test_data():       
    dataset_train = torch.load(os.path.join(_PATH_DATA_PROCESSED,"train_images.pt"))
    assert dataset_train.shape == torch.Size([50000,1,28,28]), "Data set did not have correct size"
    # check that all labels are represented
    train_labels = torch.load(os.path.join(_PATH_DATA_PROCESSED,"train_target.pt"))
    required_labels = list(range(10))
    assert set(required_labels) == set(train_labels.tolist())   
    
    # do the same for test data
    dataset_test = torch.load(os.path.join(_PATH_DATA_PROCESSED,"test_images.pt"))  
    assert dataset_test.shape == torch.Size([5000,1,28,28])
    test_labels = torch.load(os.path.join(_PATH_DATA_PROCESSED,"test_target.pt"))
    assert set(required_labels) == set(test_labels.tolist())

    # ligegyldig test, just to see if coverage changed
    # train,test = load_data(batch_size=64, shuffle=True, path=_PATH_DATA_PROCESSED)
    # iter_train = iter(train)
    # assert len(iter_train) == 782

test_data()