from keras import regularizers
from keras import optimizers

l2_decay=1e-3

quick_graph={
    'meta': {
        'network_type': 'TIRAMISU'
    },
    'compile': {
        'metrics': ['accuracy']
    },
    'inputs': {
        'batch_shape': (None,256,256,4),
    },
    'input_conv': {
        'filters':48,
        'kernel_size':3,
        'padding':'same'
    },
    'down_path':{
        'layers_list': [2,2,2,2,2]
    },
    'bottleneck': 2,
    'output_layers': [
        {
            'type':'Flatten'
        },
        { 
            'type':'fc',
            'units': 4096
        },
        { 
            'type':'fc',
            'units': 2048
        },
        {
            'type':'Dense',
            'units': 17
        }
    ],
    'output': { 
        'activation': 'sigmoid' 
    }
}

graph_b={
    'meta': {
        'network_type': 'TIRAMISU'
    },
    'compile': {
        'metrics': ['accuracy']
    },
    'inputs': {
        'batch_shape': (None,256,256,4),
    },
    'input_conv': {
        'filters':48,
        'kernel_size':3,
        'padding':'same'
    },
    'down_path':{
        'layers_list': [2,2,2,2,2]
    },
    'bottleneck': 4,
    'output_layers': [
        {
            'type':'Flatten'
        },
        { 
            'type':'fc',
            'units': 4096
        },
        { 
            'type':'fc',
            'units': 2048
        },
        {
            'type':'Dense',
            'units': 17
        }
    ],
    'output': { 
        'activation': 'sigmoid' 
    }
}

graph={
    'meta': {
        'network_type': 'TIRAMISU'
    },
    'compile': {
        'metrics': ['accuracy'],
        'keras.optimizers':optimizers.RMSprop(decay=0.995)
    },
    'inputs': {
        'batch_shape': (None,256,256,4),
    },
    'input_conv': {
        'filters':48,
        'kernel_size':3,
        'padding':'same'
    },
    'down_path':{
        'layers_list': [4,5,7,10,12]
    },
    'bottleneck': 4,
    'output_layers': [
        {
            'type':'Flatten'
        },
        { 
            'type':'fc',
            'units': 4096,
            'kernel_regularizer':regularizers.l2(l2_decay)
        },
        { 
            'type':'fc',
            'units': 2048,
            'kernel_regularizer':regularizers.l2(l2_decay) 
        },
        {
            'type':'Dense',
            'units': 17
        }
    ],
    'output': { 
        'activation': 'sigmoid' 
    }
}
