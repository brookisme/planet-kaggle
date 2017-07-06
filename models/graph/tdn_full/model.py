graph={
    'meta': {
        'network_type': 'TIRAMISU'
    },
    'compile': True,
    'inputs': {
        'batch_shape': (None,256,256,4),
    },
    'input_conv': {
        'filters':48,
        'kernel_size':3,
        'padding':'same'
    },
    'down_path':{
        'layers_list': [1,1,1,1,1]
    },
    'bottleneck': 1,
    'output_layers': [
        {
            'type':'Flatten'
        },
        { 
            'type':'fc',
            'units': 2560 
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
