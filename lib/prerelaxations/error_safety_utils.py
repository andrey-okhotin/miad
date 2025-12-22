import resource




# memory constraints

def limit_memory(percent_of_max_memory):
    max_memory = get_max_memory()
    bound = int(max_memory * (percent_of_max_memory / 100))
    resource.setrlimit(resource.RLIMIT_AS, (bound, bound))

def get_max_memory():
    with open('/proc/meminfo', 'r') as mem:
        free_memory = 0
        for i in mem:
            sline = i.split()
            if str(sline[0]) in ('MemFree:', 'Buffers:', 'Cached:'):
                free_memory += int(sline[1])
    return free_memory * 1024  # Bytes