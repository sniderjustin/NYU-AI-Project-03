

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



Move into the folder and run the setup script.

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
```