import pickle
import matplotlib.pyplot as plt
import numpy as np

for x in range(0,1):
    filename = 'objs'+str(x)+'.pickle'
    with open(filename,'rb') as f:
        fig = plt.figure()
        eval_indices,output_indices,train_loss, testing_loss, test_accuracy, validation_accuracy = pickle.load(f)
        test_accuracy = 100 - test_accuracy
        validation_accuracy = 100 - validation_accuracy
        plt.plot(output_indices, train_loss, 'k-', label='validation')
        plt.title('Softmax Loss per Generation'+str(x))
        plt.xlabel('Generation')
        plt.ylabel('Softmax Loss')
        fig.show()
        fig.savefig("TrainLoss"+str(x))

        fig1 = plt.figure()
        plt.plot(output_indices, testing_loss,'k-', label='test')
        plt.title('Test Loss ')
        plt.xlabel('Generation')
        plt.ylabel('Softmax Loss')
        fig1.show()
        fig1.savefig("TestLoss"+str(x))

        # Plot accuracy over time
        fig2 = plt.figure()
        plt.plot(eval_indices, test_accuracy, 'r',label='test')
        plt.plot(eval_indices,validation_accuracy,'b',label='train')
        plt.legend()
        plt.title('Accuracy')
        plt.xlabel('Generation')
        plt.ylabel('Accuracy')
        fig2.show()
        fig2.savefig("Accuracy"+str(x))
        print("Mean train accuracy : " , np.mean(validation_accuracy) , "Variance train accuracy : ", np.var(validation_accuracy))
        print("Mean test accuracy : ", np.mean(test_accuracy), "Variance test accuracy : ",
              np.var(test_accuracy))