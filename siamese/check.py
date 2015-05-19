from siamese import SiameseNet
from numpy import * 

random.seed(10)

W_dummy = random.randn(10, 50)
L_dummy = random.randn(20, 50)
model = SiameseNet(L0=L_dummy, W=W_dummy, rseed=10)
model.grad_check(array([[1, 1, 1], [4, 9, 6]]), array(([],[9, 11, 11])))
