



def steps_printer(
    index, num_iterations, 
    time_point, num_time_points,
    gpu=0,
    fprint=print
):
    if (time_point+1) % steps_printer.print_freq == 0:
        s = (
            f"-- gpu: {gpu}    iter: {int(index+1):4d}/{int(num_iterations):4d}"+
            f"   t={int(time_point+1):4d}/{int(num_time_points):4d}"
        )
        fprint(s)
    pass




def trainer_printer(
    rank,
    epoch, num_epochs, 
    batch_index, num_batches,
    fprint=print
):
    if rank == 0:
        s = f"\r-- epoch: {epoch:4d}/{num_epochs:4d},    batch: {batch_index:4d}/{num_batches:4d}"
        fprint(s, end="")
    pass    