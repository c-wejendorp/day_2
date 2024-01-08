import pytest
from day_2.models.model import MyAwesomeModel
import torch

def test_error_on_wrong_shape():
    model = MyAwesomeModel()
    with pytest.raises(ValueError, match='Expected input to a 4D tensor'):        
        model(torch.randn(1,2,3))
        
        






