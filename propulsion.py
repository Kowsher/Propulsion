
import torch
import torch.nn as nn
from transformers.activations import ACT2FN

# Define the custom linear layer
class PropulsionLinear(nn.Module):
    def __init__(self, input_features, output_features, bias=True, degree=15, **kwargs):
        super(PropulsionLinear, self).__init__()
        # Initialize the underlying nn.Linear with both the specified arguments and any additional kwargs
        self.linear = nn.Linear(input_features, output_features, bias=bias, **kwargs)
        self.propulsion = nn.Parameter(torch.ones(output_features))
        self.degree = degree
 
    def forward(self, x):
        self.push = torch.pow(self.propulsion, self.degree)
        return self.linear(x) * self.push
    
class PropulsionEmbedding(nn.Module):
    def __init__(self, degree=15, **kwargs):
        super(PropulsionEmbedding, self).__init__()
        # Initialize the embedding layer with kwargs passed to the constructor
        self.embeddings = nn.Embedding(**kwargs)
        # Assuming embedding_dim is one of the kwargs, use it to initialize propulsion
        self.propulsion = nn.Parameter(torch.ones(kwargs['embedding_dim']))
        self.degree = degree
        
    @property
    def weight(self):
        return self.embeddings.weight
    
    def forward(self, x):
        self.push = torch.pow(self.propulsion, self.degree)
        return self.embeddings(x)* self.push




class PropulsionLayerNorm(nn.Module):
    def __init__(self, normalized_shape, degree=1, **kwargs):
        super(PropulsionLayerNorm, self).__init__()
        self.layer_norm = nn.LayerNorm(normalized_shape, **kwargs)
        self.propulsion = nn.Parameter(torch.ones(normalized_shape))
        self.degree = degree


    def forward(self, x):
        self.push = torch.pow(self.propulsion, self.degree)
        return self.layer_norm(x)* self.push
'''
def replace_layers_with_custom(model, linear_degree=5, embedding_degree=5):
    """
    Recursively replaces nn.Linear and nn.Embedding layers with CustomLinear
    and CustomEmbedding layers, copying the weights and setting the degrees.
    """
    for name, module in model.named_children():
        # Replace nn.Linear with CustomLinear
        if isinstance(module, nn.Linear):
            custom_linear = custom_linear = PropulsionLinear(module.in_features, module.out_features, module.bias is not None, degree=linear_degree)
            custom_linear.linear.weight = nn.Parameter(module.weight.data.clone())
            if module.bias is not None:
                custom_linear.linear.bias = nn.Parameter(module.bias.data.clone())
            setattr(model, name, custom_linear)
        # Replace nn.Embedding with CustomEmbedding
        elif isinstance(module, nn.Embedding):
            custom_embedding = PropulsionEmbedding(num_embeddings=module.num_embeddings, embedding_dim=module.embedding_dim, padding_idx=module.padding_idx, degree=embedding_degree)
            custom_embedding.embeddings.weight = nn.Parameter(module.weight.data.clone())
            setattr(model, name, custom_embedding)

        else:
            # Recursively apply this function to children modules
            replace_layers_with_custom(module, linear_degree=linear_degree, embedding_degree=embedding_degree)
'''
def replace_layers_with_custom(model, degree, targets, bias):
    """
    Recursively replaces nn.Linear layers with a custom layer.
    """
    #print(targets, len(targets))

    
        
    for name, module in model.named_children():
        #print(name, module)

        # Replace nn.Linear with CustomLinear_colum
        if isinstance(module, nn.Linear):
            #print(name)
            if len(targets)>0:
                if name in targets:#== 'value' or  name == 'query' or  name == 'key': #( name == 'value' or  name == 'query' or  name == 'key')    
                    custom_linear = custom_linear = PropulsionLinear(module.in_features, module.out_features, module.bias is not None, degree=degree)
                    custom_linear.linear.weight = nn.Parameter(module.weight.data.clone())
                    custom_linear.linear.weight.requires_grad=False
                    if module.bias is not None:
                        custom_linear.linear.bias = nn.Parameter(module.bias.data.clone())
                        if bias:
                            custom_linear.linear.bias.requires_grad = True
                        else:
                            custom_linear.linear.bias.requires_grad = False  # Explicitly set this to False when bias is False
                            
                    setattr(model, name, custom_linear)
            else:
                custom_linear = custom_linear = PropulsionLinear(module.in_features, module.out_features, module.bias is not None, degree=degree)
                custom_linear.linear.weight = nn.Parameter(module.weight.data.clone())
                custom_linear.linear.weight.requires_grad=False
                if module.bias is not None:
                    custom_linear.linear.bias = nn.Parameter(module.bias.data.clone())
                    if bias:
                        custom_linear.linear.bias.requires_grad = True
                    else:
                        custom_linear.linear.bias.requires_grad = False  # Explicitly set this to False when bias is False
                        
                setattr(model, name, custom_linear)
            
        elif isinstance(module, nn.Embedding):
            if (len(targets)>0 and name in targets) or len(targets)==0:
                custom_embedding = PropulsionEmbedding(num_embeddings=module.num_embeddings, embedding_dim=module.embedding_dim, padding_idx=module.padding_idx, degree=degree)
                custom_embedding.embeddings.weight = nn.Parameter(module.weight.data.clone())
                custom_embedding.embeddings.weight.requires_grad=False
                setattr(model, name, custom_embedding)
        else:
            # Recursively apply this function to children modules
            replace_layers_with_custom(module, degree, targets, bias)

def PEFT(model,degree=5, targets=[], bias=True):
    """
    Recursively replaces nn.Linear layers with a custom layer.
    """
        # Freeze all layers by default
    
    for name, param in model.named_parameters():
        param.requires_grad = False

    replace_layers_with_custom(model, degree, targets, bias)