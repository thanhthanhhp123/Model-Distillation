from students import *
import torch

pdn = get_pdn_small(padding = True)
test_tensor = torch.randn(1, 3, 224, 224)
output = pdn(test_tensor)
print(output.shape)