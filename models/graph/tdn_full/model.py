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
        'layers_list': [4,5,7,10]
    },
    'bottleneck': False,
    'output_layers': [
        {
            'type':'Flatten'
        },
        { 
            'type':'fc',
            'units': 5248
        },
        { 
            'type':'fc',
            'units': 2624 
        },
        { 
            'type':'fc',
            'units': 1312 
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
