from torch import nn

class AddNorm(nn.Module):
    def __init__(self, d_model=512):
        super().__init__()

        self.layer_norm = nn.LayerNorm(d_model)

    def forward(self, x, sublayer_output):
        x = x + sublayer_output
        x = self.layer_norm(x)
        
        return x
    
print("add_norm.py loaded")
