

# NYU // AI // Project 03 // CodeSearchNet

Machine learning is often a computation heavy exercise. To take advantage of large datasets, deep learning, and to install the best contemporary frameworks such as TensorFlow-GPU you need to have a machine with substantial GPU stats.

Fortunately, you don't have to spend thousands of dollars to buy your own machine. You can rent a Virtual Machine with the stats you need on the cloud. Virtual Machines are great to have in your tool box. Do you need a new setup or more powerful machine? No problem! 

This guide will take you through the process of setting up a Virtual Machine. In addition, we will walk though the steps required to setup the CodeSearchNet competition on your virtual machine. 

Even if you have no experience with Linux, cloud computing, or GPU drivers you will be able to easily follow this step by step guide. 

There isn't a fixed price, but I would estimate the process will cost you around $30. 

If you have $30 and a browser you are ready to go! 

# 1) Setup an account

There are many cloud computing providers available. It can be overwhelming to choose. We are going to use [Paperspace.com](https://www.paperspace.com/). They arguable have the simplest process to get your machine up and running. So lets be lazy! 

![homepage](https://github.com/aobject/NYU-AI-Project-03/blob/master/media/001_paperspace_home.png?raw=true)

  

First, create a login using your Google or GitHub account. You will get $10 free if you use my referral code NN6C240. The referral link is [here](https://paperspace.io/&R=NN6C24O). 

![account](https://github.com/aobject/NYU-AI-Project-03/blob/master/media/002-signin.png?raw=true)



![firstPage](https://github.com/aobject/NYU-AI-Project-03/blob/master/media/003.png?raw=true)



![secondPage](https://github.com/aobject/NYU-AI-Project-03/blob/master/media/004.png?raw=true)



![addCard](https://github.com/aobject/NYU-AI-Project-03/blob/master/media/005_add_card.png?raw=true)



![subscription](https://github.com/aobject/NYU-AI-Project-03/blob/master/media/005_Update_Subscription.png?raw=true)



![upgrade](https://github.com/aobject/NYU-AI-Project-03/blob/master/media/006_Upgrade_to_G1.png?raw=true)



![console](https://github.com/aobject/NYU-AI-Project-03/blob/master/media/007_click_home_to_console.png?raw=true)



![clickCore](https://github.com/aobject/NYU-AI-Project-03/blob/master/media/008_click_core.png?raw=true)



![newMachine](https://github.com/aobject/NYU-AI-Project-03/blob/master/media/009_New_Machine.png?raw=true)



![selectRegion](https://github.com/aobject/NYU-AI-Project-03/blob/master/media/010_Select_region.png?raw=true)



![selectOS](https://github.com/aobject/NYU-AI-Project-03/blob/master/media/011_Choose_OS_ML.png?raw=true)



![applyForGPU](https://github.com/aobject/NYU-AI-Project-03/blob/master/media/012_0_ApplyforGPU.png?raw=true)



![chooseMachine](https://github.com/aobject/NYU-AI-Project-03/blob/master/media/012_ChooseMachineP4000.png?raw=true)



![ChooseStorage100](https://github.com/aobject/NYU-AI-Project-03/blob/master/media/013_ChooseStorage100GB.png?raw=true)



![defaults](https://github.com/aobject/NYU-AI-Project-03/blob/master/media/014_LeaveDefaults.png?raw=true)



![createPaperspace](https://github.com/aobject/NYU-AI-Project-03/blob/master/media/015_CreatePaperspace.png?raw=true)

![getPass](https://github.com/aobject/NYU-AI-Project-03/blob/master/media/016_check_email.png?raw=true)



![clickOnGear](https://github.com/aobject/NYU-AI-Project-03/blob/master/media/017_0_clickOnGear.png?raw=true)



![openTerminal](https://github.com/aobject/NYU-AI-Project-03/blob/master/media/022_OpenTerminal.png?raw=true)



![changePass](https://github.com/aobject/NYU-AI-Project-03/blob/master/media/021_changePassword.png?raw=true)



Check out our new GPU: 

**INPUT**

```
$ lspci | grep -i nvidia
```



**OUTPUT**

```
00:05.0 VGA compatible controller: NVIDIA Corporation GP104GL [Quadro P4000] (rev a1)
00:06.0 Audio device: NVIDIA Corporation GP104 High Definition Audio Controller (rev a1)
```



Check the Nvida Driver in use:

**INPUT**

```
$ nvidia-smi
```

**OUTPUT**

```
Thu Apr 16 15:08:11 2020       
+-----------------------------------------------------------------------------+
| NVIDIA-SMI 418.67       Driver Version: 418.67       CUDA Version: 10.1     |
|-------------------------------+----------------------+----------------------+
| GPU  Name        Persistence-M| Bus-Id        Disp.A | Volatile Uncorr. ECC |
| Fan  Temp  Perf  Pwr:Usage/Cap|         Memory-Usage | GPU-Util  Compute M. |
|===============================+======================+======================|
|   0  Quadro P4000        On   | 00000000:00:05.0  On |                  N/A |
| 46%   28C    P8     6W / 105W |    309MiB /  8119MiB |      0%      Default |
+-------------------------------+----------------------+----------------------+
                                                                               
+-----------------------------------------------------------------------------+
| Processes:                                                       GPU Memory |
|  GPU       PID   Type   Process name                             Usage      |
|=============================================================================|
|    0      1393      G   /usr/lib/xorg/Xorg                           175MiB |
|    0      1589      G   /usr/bin/gnome-shell                         131MiB |
+-----------------------------------------------------------------------------+
WARNING: infoROM is corrupted at gpu 0000:00:05.0
```



Check the Nvidia Cuda Version:

**INPUT**

```
$ nvcc --version
```

**OUTPUT**

```
nvcc: NVIDIA (R) Cuda compiler driver
Copyright (c) 2005-2018 NVIDIA Corporation
Built on Sat_Aug_25_21:08:01_CDT_2018
Cuda compilation tools, release 10.0, V10.0.130
```

Check the Docker version:

**INPUT**

```
paperspace@psyuowt9y:~$ docker -v
```

**OUTPUT**

```
Docker version 19.03.1, build 74b1e89
```



Verify nvidia-docker-toolkit is installed:

**INPUT**

```
paperspace@psyuowt9y:~$ sudo apt list nvidia-container*
```

**OUTPUT**

```
nvidia-container-runtime/bionic,now 3.1.0-1 amd64 [installed]
nvidia-container-runtime-hook/bionic,now 1.4.0-1 amd64 [residual-config]
nvidia-container-toolkit/bionic,now 1.0.1-1 amd64 [installed]
```



We are ready to setup the CodeSearchNet repository on our machine. 

First lets make a folder on our virtual machine to put the project and move inside. 

```
~$ cd ~/Documents/
~/Documents$ mkdir ai-project-03
~/Documents$ cd ai-project-03/
```

Now we clone the competition repository:

```
~/Documents/ai-project-03$ git clone https://github.com/github/CodeSearchNet.git
...
Cloning into 'CodeSearchNet'...
remote: Enumerating objects: 1, done.
remote: Counting objects: 100% (1/1), done.
remote: Total 599 (delta 0), reused 0 (delta 0), pack-reused 598
Receiving objects: 100% (599/599), 29.01 MiB | 22.12 MiB/s, done.
Resolving deltas: 100% (262/262), done.
...
paperspace@psyuowt9y:~/Documents/ai-project-03$ ls
CodeSearchNet
```



Verify the new folder exists. 

**Input**

```
~/Documents/ai-project-03$ ls
CodeSearchNet
```

**Output**

```
CodeSearchNet
```



Move into the folder and run the setup script. This will download all the data used, about 3.5GB, from AWS S3. Go get some coffee this will take a few minutes. 

**Input**

```
~/Documents/ai-project-03$ cd CodeSearchNet/
~/Documents/ai-project-03/CodeSearchNet$ sudo script/setup
```

**Output**

```

...
Sending build context to Docker daemon  31.36MB
Step 1/5 : FROM python:3.7.3
3.7.3: Pulling from library/python
6f2f362378c5: Pull complete 
...
still downloading blah blah blah
...
Archive:  go.zip
   creating: go/
   creating: go/final/
   creating: go/final/jsonl/
   creating: go/final/jsonl/train/
...
  inflating: go/final/jsonl/test/go_test_0.jsonl.gz  
   creating: go/final/jsonl/valid/
  inflating: go/final/jsonl/valid/go_valid_0.jsonl.gz  
  inflating: go_dedupe_definitions_v2.pkl  
  inflating: go_licenses.pkl 
```



We are going to run a shell within Docker were we can issue commands to train models and make predictions. 

**Input**

```
paperspace@psyuowt9y:~/Documents/ai-project-03/CodeSearchNet$ sudo script/console
```

**Output**

```
Thu Apr 16 20:02:54 2020       
+-----------------------------------------------------------------------------+
| NVIDIA-SMI 418.67       Driver Version: 418.67       CUDA Version: 10.1     |
|-------------------------------+----------------------+----------------------+
| GPU  Name        Persistence-M| Bus-Id        Disp.A | Volatile Uncorr. ECC |
| Fan  Temp  Perf  Pwr:Usage/Cap|         Memory-Usage | GPU-Util  Compute M. |
|===============================+======================+======================|
|   0  Quadro P4000        On   | 00000000:00:05.0  On |                  N/A |
| 46%   28C    P8     6W / 105W |    310MiB /  8119MiB |      0%      Default |
+-------------------------------+----------------------+----------------------+
                                                                               
+-----------------------------------------------------------------------------+
| Processes:                                                       GPU Memory |
|  GPU       PID   Type   Process name                             Usage      |
|=============================================================================|
+-----------------------------------------------------------------------------+
WARNING: infoROM is corrupted at gpu 0000:00:05.0
root@psyuowt9y:/home/dev/src# 
```

The last line is where we can issue shell commands to Docker.





```
root@psyuowt9y:/home/dev/src# wandb login
wandb: (1) Create a W&B account
wandb: (2) Use an existing W&B account
wandb: (3) Don't visualize my results
wandb: Enter your choice: 
```

Follow the prompts to setup your account with Weights and Biases on their website https://www.wandb.com.  After you go through the process of setting up an account cut and paste your new API Key into the terminal. Save this for later. You will need this key every time you want to login. If you loose it you can always look it up on the wandb website. When you are done you will see the following output. 

```
Successfully logged in to Weights & Biases!
```



**Input**

```
root@psyuowt9y:/home/dev/src# python train.py --testrun
```

**Output**

```
wandb: Started W&B process version 0.8.12 with PID 43
...
wandb: plus 9 W&B file(s) and 0 media file(s)
wandb:                                                                                
wandb: Synced neuralbowmodel-2020-04-16-20-09-48: https://app.wandb.ai/aobject/CodeSearchNet/runs/1yg7zqon
```



This will likely take about 15 minutes. Once the test run is complete we can go to the weights & Biases website to see a lot of great data about how we did. There is a dashboard that will show you loss, accuracy, and a variety of machine stats. 

Cut and past the URL in the last line of your output into the browser. This will take you to an amazing dashboard with more data about your test run than you had in your wildest data science dreams.



# picture of dash here



Now we are ready for the big time. To see all the command line options available to you with train.py type into your command line:

**Input**

```
# python train.py --help
Usage:
    train.py [options] SAVE_FOLDER TRAIN_DATA_PATH VALID_DATA_PATH TEST_DATA_PATH
    train.py [options] [SAVE_FOLDER]
```

**Output**

```
Usage:
    train.py [options] SAVE_FOLDER TRAIN_DATA_PATH VALID_DATA_PATH TEST_DATA_PATH
    train.py [options] [SAVE_FOLDER]
...
```





Now it is time to fully train a baseline model. 

**Input**

```
# python train.py --model neuralbow
```

This took 3 hours 11 minutes and 35 seconds on my virtual machine. So it should be about the same for you. 



Look at those stats! 



# Summary

Great work you have created your virtual machine for machine learning, implemented a GPU powered Docker container, and trained a Code Search Model! 