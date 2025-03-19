import time 

def fn_timer(func, *args, **kwargs): 
    st_time = time.time()
    func_result = func(*args, **kwargs)
    en_time = time.time()
    return func_result, en_time - st_time 
