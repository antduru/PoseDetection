import numpy as np 

translate = 'abcdefghijklmnoprstuvyzxqw.,'
in_features = len(translate)

x = np.random.random((1, in_features))

model = {
    'h': np.random.random((1, in_features)),
    'w1': np.random.random((2 * in_features, 2 * in_features)),
    'b1': np.random.random((1, 2 * in_features)),
}

def forward(model, x):
    i_xh = np.concatenate([x, model['h']], axis=1)
    o_yh = (np.matmul(i_xh, model['w1']) + model['b1'])

    model['h'] = np.array(o_yh[:, :in_features])
    return np.array(o_yh[:, in_features:])

y = x
for i in range(100):
    argmax = np.argmax(y)
    print(translate[argmax], end='')

    y = forward(model, y)

print()