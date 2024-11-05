
# string to search in file
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd 
import argparse

def get_parser():
    
    parser = argparse.ArgumentParser(description='Plot Accuracy')

    parser.add_argument(
        '--dir',
        type=str,
        required=True,
        help='the work folder for storing results')


if __name__ == '__main__':    
    
    parser = argparse.ArgumentParser(description='Plot Accuracy')
    parser.add_argument(
        '--dir',
        type=str,
        required=True,
        help='the work folder for storing results')
    
    p = parser.parse_args()
    print(p.dir)
    lst = []
    loss_lst = []
    loss_test = []
    
    
    with open(p.dir, 'r') as fp:
        # read all lines using readline()
        lines = fp.readlines()
        for row in lines:
            word = 'Top 1:'
            loss = 'Mean training loss: '
            test_loss = 'Mean test loss of '
            if row.find(word) != -1:
                lst.append(float(row.split(": ")[1].split("%")[0]))
                
            if row.find(loss) != -1:
                loss_lst.append(float(row.split(": ")[1].split(" ")[0]))
                # print()
            if row.find(test_loss) != -1:
                loss_test.append(float(row.split(": ")[1].rstrip('.\n')))
                # print(max(loss_test))
    acr = np.array(lst)
    loss = np.array(loss_lst)
    epoch_num = np.array([i for i in range(len(acr))])

    figure, axis = plt.subplots(1,2,dpi = 180) 
    axis[0].plot(epoch_num, acr, label = "Training Accuracy")
    
    if (len(loss) > len(epoch_num)):
        axis[1].plot(epoch_num, loss[:-1], color="red", label = "Training Loss")
        axis[1].plot(epoch_num, loss_test[:-1], color="blue", label = "Testting Loss")
    else:
        axis[1].plot(epoch_num, loss[:], color="red", label = "Mean Loss")
        axis[1].plot(epoch_num, loss_test[:], color="blue", label = "Testting Loss")
        
    
    
    

    

    axis[0].title.set_text("Accuracy")
    axis[1].title.set_text("Loss")

    axis[0].legend()
    axis[1].legend()

    axis[0].grid()
    axis[1].grid()   
    plt.show()