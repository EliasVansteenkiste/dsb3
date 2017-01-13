from multiprocessing.dummy import Process, Pipe, Pool
from itertools import izip

def spawn(f):
    def fun(pipe, *args, **kwargs):
        pipe.send(f(*args,**kwargs))
        pipe.close()
    return fun

def parmap(f,X):
    pipe=[Pipe() for x in X]
    proc=[Process(target=spawn(f),args=(c,x)) for x,(p,c) in izip(X,pipe)]
    [p.start() for p in proc]
    [p.join() for p in proc]
    return [p.recv() for (p,c) in pipe]



def wrap(f):
    def fun(*args, **kwargs):
        return f(*args,**kwargs)
    return fun


class AsyncPool(object):
    def __init__(self, size):
        self.pool = Pool(processes=4)
        self.size = size
        self.processes = [None]*self.size

    def __del__(self):
        self.pool.close()

    def add_task(self, index, task, *args, **kwargs):
        self.processes[index] = self.pool.apply_async(wrap(task), args=args, kwds=kwargs)

    def get_results(self):
        for p in self.processes:
            if p is not None:
                p.wait()

        return [p.get() if p else None for p in self.processes]



if __name__ == '__main__':
    print parmap(lambda x:x**x,range(1,5))