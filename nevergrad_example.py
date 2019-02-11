from nevergrad.optimization import optimizerlib

def square(x):
    return (x - .5)**2

optimizer = optimizerlib.OnePlusOne(dimension=1, budget=100)
# alternatively, you can use optimizerlib.registry which is a dict containing all optimizer classes
recommendation = optimizer.optimize(square, executor=None, batch_mode=True)
