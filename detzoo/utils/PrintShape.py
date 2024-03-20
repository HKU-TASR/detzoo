from torch.nn import Module

class PrintShape(Module):
    def __init__(self, message="Shape"):
        super(PrintShape, self).__init__()
        self.message = message

    def forward(self, x):
        print(f"{self.message}: {x.shape}")
        return x