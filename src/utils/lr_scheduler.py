import math
import matplotlib.pyplot as plt

def lr_time_based_decay(epoch, lr, nb_epoch=0):
    # decay = lr / nb_epoch
    decay = 0.0045
    return lr * 1 / (1 + decay * epoch)


def lr_step_decay(epoch, lr, nb_epoch=0):
    drop_rate = .99
    epochs_drop = 8

    if epoch % epochs_drop == 0:
        return lr * math.pow(drop_rate, math.floor(epoch/epochs_drop))

    return lr


def lr_asc_desc_decay(epoch, lr, nb_epoch=0):
    lr_max = 1e-3
    lr_min = 1e-5
    lr_ascending_ep = 15
    lr_sus_ep = 0
    decay = 0.85
    ascending_penalty = 0.85

    if epoch < lr_ascending_ep:
        lr = (lr_max - lr) / lr_ascending_ep * epoch + (lr*ascending_penalty)

    elif epoch < lr_ascending_ep + lr_sus_ep:
        lr = lr_max

    else:
        lr = (lr_max - lr_min) * decay**(epoch -
                                         lr_ascending_ep - lr_sus_ep) + lr_min

    return lr


def plot_lr_decay(lr_function, lr, epoch, nb_epoch):
    if epoch == nb_epoch:
        return lr
    else:
        lr.append(lr_function(epoch, lr[-1], nb_epoch))
        return plot_lr_decay(lr_function, lr, epoch+1, nb_epoch)

if __name__ == "__main__":
    nb_epoch = 40
    returned_lr = plot_lr_decay(lr_time_based_decay, [1e-3], 0, nb_epoch)
    print(returned_lr)

    plt.plot(list(range(0, (nb_epoch+1))), returned_lr)
    plt.show()
