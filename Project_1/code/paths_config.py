
class Config:

    # Define a path for storing the plot results and a path for loading the cifar10 dataset
    RESULTS_PATH = '/home/k/karatisd/nn_project_1/results/'
    DATASET_PATH = '/home/k/karatisd/cifar-10/' 

    # Define custom test set for cifar10 testing
    image_paths_cifar10 = [  
        "/home/k/karatisd/nn_project_1/cifar10Testbench/horse2.png",
        "/home/k/karatisd/nn_project_1/cifar10Testbench/car1.png",
        "/home/k/karatisd/nn_project_1/cifar10Testbench/bird1.png",
        "/home/k/karatisd/nn_project_1/cifar10Testbench/horse3.png",
        "/home/k/karatisd/nn_project_1/cifar10Testbench/car2.png",
        "/home/k/karatisd/nn_project_1/cifar10Testbench/ship1.png",
        "/home/k/karatisd/nn_project_1/cifar10Testbench/airplane1.png",
        "/home/k/karatisd/nn_project_1/cifar10Testbench/dog1.png",
        "/home/k/karatisd/nn_project_1/cifar10Testbench/frog1.png",
        "/home/k/karatisd/nn_project_1/cifar10Testbench/deer1.png"
    ]
    
    # Define custom test set for mnist testing
    image_paths_mnist = [  
        "/home/k/karatisd/nn_project_1/mnistTestbench/five1.png",
        "/home/k/karatisd/nn_project_1/mnistTestbench/seven1.png",
        "/home/k/karatisd/nn_project_1/mnistTestbench/seven2.png",
        "/home/k/karatisd/nn_project_1/mnistTestbench/six1.png",
        "/home/k/karatisd/nn_project_1/mnistTestbench/two1.png"
    ]