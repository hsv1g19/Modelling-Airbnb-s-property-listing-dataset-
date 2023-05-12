# Modelling-Airbnb-s-property-listing-dataset-

```class FcNet(nn.Module):
    
    def __init__(self, config , input_dim, output_dim):
        super().__init__()
        
        width = config['hidden_layer_width']
        depth = config['depth']


        # input layer
        layers = [nn.Linear(input_dim, width), nn.ReLU()]
        
        # hidden layers
        for hidden_layer in range(depth - 1):
            layers.append(nn.Linear(width, width))
            layers.append(nn.ReLU())
        
        # output layer
        layers.append(nn.Linear(width, output_dim))
        
        # create a sequential model
        self.layers = nn.Sequential(*layers)
        
    def forward(self, features):
        x = self.layers(features)
        return x