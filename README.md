

# NYU // AI // Project 03 // CodeSearchNet Setup Guide

Machine learning is often a computation heavy exercise. To take advantage of large datasets, deep learning, and to take full advantage of the best contemporary frameworks such as TensorFlow you need to have a machine with a great GPU.

Fortunately, you don't have to spend thousands of dollars to buy your own machine. You can rent a Virtual Machine with the stats you need on the cloud. Knowing how to set up a virtual machine is an essential skill to have in your tool box. 

## Introduction

You don't need to have any experience with Linux, cloud computing, or GPU drivers to follow this tutorial. This guide is written with extensive descriptions of every step with lots of screenshots so you don't miss a single step. 

All you need to follow along is a web browser and about $30.  

In the end you will be running an Ubuntu virtual machine with a Nvidia P4000 GPU. The machine will be configure to meet the requirements for setting up and running the CodeSearchNet model. Which include the following requirements: 

* GPU (a good one)
* Nvidia driver (a new'ish one)
* Cuda toolkit (a new'ish one)

* Docker installed (a new'ish one)
* Nvidia-Docker installed (a new'ish one)

When the virtual machine is ready to go we will setup the CodeSearchNet repository on our machine and use Docker to run the baseline training model. 

Lets get started! 

## Part 1 // Create Your Virtual Machine

When the virtual machine is ready to go we will setup the CodeSearchNet repository on our machine and use Docker to run the baseline training model. 

There isn't a fixed price, but I would estimate the process will cost you around $30. 1.1) Setup an account

There are many cloud computing providers available. It can be overwhelming to choose. We are going to use [Paperspace.com](https://www.paperspace.com/). They arguable have the simplest process to get your machine up and running. So lets be lazy! 

![homepage](https://github.com/aobject/NYU-AI-Project-03/blob/master/media/001_paperspace_home.png?raw=true)

  

### 1.2) Create a login

First, create a login using your Google or GitHub account. You will get $10 free if you use my referral code NN6C240. The referral link is [here](https://paperspace.io/&R=NN6C24O). 

![account](https://github.com/aobject/NYU-AI-Project-03/blob/master/media/002-signin.png?raw=true)



### 1.3) Go to Billing

Now you are logged in! Lets setup our billing. 

![secondPage](https://github.com/aobject/NYU-AI-Project-03/blob/master/media/004.png?raw=true)



### 1.4) Click on "ADD CARD"

Input your payment information. 

![addCard](https://github.com/aobject/NYU-AI-Project-03/blob/master/media/005_add_card.png?raw=true)



### 1.5) Click on "UPDATE SUBSCRIPTION"

After you have the payment information setup you can pick your subscription type. 

![subscription](https://github.com/aobject/NYU-AI-Project-03/blob/master/media/005_Update_Subscription.png?raw=true)



### 1.6) Pick G1

The G1 Subscription plan is for ML/AI engineers. That is is us! Click upgrade to G1. 

![upgrade](https://github.com/aobject/NYU-AI-Project-03/blob/master/media/006_Upgrade_to_G1.png?raw=true)



### 1.7) Go to your dashboard by clicking on the house icon

The house icon takes you to your dashboard. 

![console](https://github.com/aobject/NYU-AI-Project-03/blob/master/media/007_click_home_to_console.png?raw=true)



### 1.8) Select "CORE"

Clicking CORE will take us to our machine menu. There we will be able to create a Cloud GPU machine with full remote access. 

![clickCore](https://github.com/aobject/NYU-AI-Project-03/blob/master/media/008_click_core.png?raw=true)



### 1.9) Click "NEW MACHINE"

The new machine starts the machine creation process. 

![newMachine](https://github.com/aobject/NYU-AI-Project-03/blob/master/media/009_New_Machine.png?raw=true)



### 1.10) Choose Region

I'm on team east coast. Pick your region. 

![selectRegion](https://github.com/aobject/NYU-AI-Project-03/blob/master/media/010_Select_region.png?raw=true)



### 1.11) Choose OS

The  CodeSearchNet setup script has a lot of requirements including:

* GPU (a good one)
* Nvidia driver
* Cuda toolkit
* Docker installed
* Nvidia-Docker installed



You are in luck. Paperspace.com has Public Templates that are preconfigured for different uses. One of those template is the famous "Ubuntu 18.04 ML-in-a-Box Desktop Edition". This template has all the requirements above pre-installed.

![selectOS](https://github.com/aobject/NYU-AI-Project-03/blob/master/media/011_Choose_OS_ML.png?raw=true)



### 1.12) Choose Machine

Next we select the hardware configuration and importantly the GPU. However, First we need to be approved. Click on the P4000 Machine. This machine has a great cost to power ratio. An unavailable window will pop up. Use the window to type in a request to use the machine. You will have to be approved before given access to any machines. 

The box looks small but don't be fooled. You want to make the best impression possible to ensure you don't get rejected. I recommend you write a letting the Paperspace team that you paste into the box for your request. Be polite, tell them about the project you are working on, and provide links to your other work. Let the bouncer know you are a human (not a robot) and you do very interesting work (maybe provide your GitHub link). 

People are rejected and approval can take time so do what you can to stand out in a good way. 

![applyForGPU](https://github.com/aobject/NYU-AI-Project-03/blob/master/media/012_0_ApplyforGPU.png?raw=true)



### 1.13) Choose GPU P4000

Congratulations, your GPU request has been approved! You are now part of the elite Paperspace GPU squad. You can now look down on those sad souls with no GPU access.  

![chooseMachine](https://github.com/aobject/NYU-AI-Project-03/blob/master/media/012_ChooseMachineP4000.png?raw=true)



### 1.14) Choose Storage

Select the 100 GB storage to do the Code Search Net competition. I was able to squeak by with 50 GB on the first setup, but I don't recommend it.



![ChooseStorage100](https://github.com/aobject/NYU-AI-Project-03/blob/master/media/013_ChooseStorage100GB.png?raw=true)



### 1.15) Use Default for the rest of the settings

Use the default settings for Machine Details, Choose Network, and Options. 

![defaults](https://github.com/aobject/NYU-AI-Project-03/blob/master/media/014_LeaveDefaults.png?raw=true)



### 1.16) Create Your Paperspace

Click on the "CREATE YOUR PAPERSPACE" button. This will create your new virtual machine. 

![createPaperspace](https://github.com/aobject/NYU-AI-Project-03/blob/master/media/015_CreatePaperspace.png?raw=true)



### 1.17) Check your email

Check your email. You will get an email telling you your machine is ready to go. Importantly the email contains your temporary password. Save this password, because you will need it later. 

![getPass](https://github.com/aobject/NYU-AI-Project-03/blob/master/media/016_check_email.png?raw=true)



## Part 2 // Access Your Virtual Machine



### 2.1) Go your CORE Machine Dashboard

You will now see the new machine your created. Click on the gear icon in the corner of the new machine. 

![clickOnGear](https://github.com/aobject/NYU-AI-Project-03/blob/master/media/017_0_clickOnGear.png?raw=true)



### 2.2) Click on "OPEN TERMINAL"

Click on "OPEN TERMINAL" button to open a browser based terminal to your new machine. 

![openTerminal](https://github.com/aobject/NYU-AI-Project-03/blob/master/media/022_OpenTerminal.png?raw=true)



### 2.3) Reset your default password

Safety first! Reset the default password using the command `passwd`

Then, provide the new password. If you want you can mistype the old password a few times for fun like I do here. 

![changePass](https://github.com/aobject/NYU-AI-Project-03/blob/master/media/021_changePassword.png?raw=true)



### 2.4) Verify Our GPU Meets our Requirements

 **Spoiler Alert** Our machine meet the minimum requirements. You can skip to part 3 if you are not interested in how to verify our machine meets the minimum requirements for the CodeSearchNet competition Docker container. **Spoiler Alert** 

Now just a refresher, the CodeSearchNet setup script has a lot of requirements including:

* GPU (a good one)
* Nvidia driver (a new'ish one)
* Cuda toolkit (a new'ish one)



What follows are a series of terminal bash command you you need to enter into your new virtual machine. Your output should be very similar. First, lets check out our new GPU we worked so hard to get: 

**INPUT**

```bash
$ lspci | grep -i nvidia
```

**OUTPUT**

```bash
00:05.0 VGA compatible controller: NVIDIA Corporation GP104GL [Quadro P4000] (rev a1)
00:06.0 Audio device: NVIDIA Corporation GP104 High Definition Audio Controller (rev a1)
```

Wow, a Quadro P4000, that is awesome. 

That is a good start. Hopefully we have a Nvida Driver to help us use that great GPU:

**INPUT**

```bash
$ nvidia-smi
```

**OUTPUT**

```bash
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

It is so exciting to have a 418 .XX driver. That is exactly what we need to use Cuda 10+. 

Cross your fingers and lets check the Nvidia Cuda Version:

**INPUT**

```bash
$ nvcc --version
```

**OUTPUT**

```bash
nvcc: NVIDIA (R) Cuda compiler driver
Copyright (c) 2005-2018 NVIDIA Corporation
Built on Sat_Aug_25_21:08:01_CDT_2018
Cuda compilation tools, release 10.0, V10.0.130
```

Fantastic, we have a GPU and we know how to use it. 



### 2.5) Verify we have Docker and Nvidia-Docker installed

Now just a refresher, the CodeSearchNet setup script requires the following:

* Docker installed (a new'ish one)
* Nvidia-Docker installed (a new'ish one)



To verify if Docker is installed we can check the Docker version:

**INPUT**

```bash
$ docker -v
```

**OUTPUT**

```bash
Docker version 19.03.1, build 74b1e89
```

We are good to go with Docker!

Finally, verify the Nvidia-Docker-toolkit is installed:

**INPUT**

```bash
paperspace@psyuowt9y:~$ sudo apt list nvidia-container*
```

**OUTPUT**

```bash
nvidia-container-runtime/bionic,now 3.1.0-1 amd64 [installed]
nvidia-container-runtime-hook/bionic,now 1.4.0-1 amd64 [residual-config]
nvidia-container-toolkit/bionic,now 1.0.1-1 amd64 [installed]
```

Our machine has all the required hardware, drivers, and packages. Nice work. 



### Part 3 // Setup the CodeSearchNet Docker Container

This is the moment you have been waiting for. We are finally ready to setup the CodeSearchNet repository on our machine. We are still in the terminal of our virtual machine. Below are the commands you will need to execute. 

First, lets make a folder on our virtual machine to put the project and move inside. 

```bash
~$ cd ~/Documents/
~/Documents$ mkdir ai-project-03
~/Documents$ cd ai-project-03/
```



Now we clone the competition repository from GitHub to our virtual machine:

**Input**

```bash
~/Documents/ai-project-03$ git clone https://github.com/github/CodeSearchNet.git
```

**Output**

```bash
...
Cloning into 'CodeSearchNet'...
remote: Enumerating objects: 1, done.
remote: Counting objects: 100% (1/1), done.
remote: Total 599 (delta 0), reused 0 (delta 0), pack-reused 598
Receiving objects: 100% (599/599), 29.01 MiB | 22.12 MiB/s, done.
Resolving deltas: 100% (262/262), done.
...
```



Once complete lets check to see if the CodeSearchNet directory is now present:

**Input**

```bash
$ ls
```

**Output**

```bash
CodeSearchNet
```



Move into the CodeSearchNet directory and run the setup script. This will download all the data used from AWS S3 directly onto your virtual machine. Go get some coffee this will take a few minutes. 

**Input**

```bash
$ cd CodeSearchNet/
$ sudo script/setup
```

**Output**

```bash
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

Your CodeSearchNet data is now downloaded to your virtual machine. 

We are ready to launch our Docker virtual machine container. The CodeSearchNet Docker container has its own operating system and is already preconfigured for us. By simply running the console script we launch Docker and enter a command line shell with control the CodeSearchNet container. 

**Input**

```bash
$ sudo script/console
```

**Output**

```bash
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

The last line is the shell prompt where we can issue shell commands to our Docker virtual machine.

First, we setup our wandb login. This will give us access to a fantastic interface with all kinds of data about our training results, test results, and much more. 

**Input**

```bash
# wandb login
```

**Output**

```bash
wandb: (1) Create a W&B account
wandb: (2) Use an existing W&B account
wandb: (3) Don't visualize my results
wandb: Enter your choice: 
```

Enter 1 and follow the prompts to setup your account with Weights and Biases on their website https://www.wandb.com.  After you go through the process of setting up an account cut and paste your new API Key into the terminal. Save this for later. You will need this key every time you want to login. If you loose it you can always look it up on the wandb website. When you are done you will see the following output. 

```
Successfully logged in to Weights & Biases!
```

## Part 4 // Train Your Model

We have a machine, we have a virtual machine, and we are ready to train our model. 

First, we do a simple training run with a small amount of data. This lets us know our setup is working. While it is running you will see the in process training run on your wandb dashboard at https://www.wandb.com. After the training run is finished you will see the results of your training run in the wandb dashboard. 

From the Docker shell prompt use python to run train.py with a --testrun flag. (For instructions on how to get to the Docker shell prompt see Part 3 above.)

**Input**

```
# python train.py --testrun
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

![machineStats](https://github.com/aobject/NYU-AI-Project-03/blob/master/media/024_test_train_stats.png?raw=true)



You can use the dash to compare different training runs. Just click the eyes to turn on and off the visibility of particular training runs. Here you can see a virtual machine in purple with a speedy and impressive completion of a training run. In blue, you can see my Ubuntu laptop starting to train... and crashing in a blaze of glory. 

![dashCompare](https://github.com/aobject/NYU-AI-Project-03/blob/master/media/023_04_dash.png?raw=true)



Now we are ready for the big time. To see all the command line options available to you with train.py type into your command line:

**Input**

```bash
# python train.py --help
```

**Output**

```bash
Usage:
    train.py [options] SAVE_FOLDER TRAIN_DATA_PATH VALID_DATA_PATH TEST_DATA_PATH
    train.py [options] [SAVE_FOLDER]
...
```



Brace yourself. Now it is time to fully train a baseline model. Here is the command to run the baseline training configuration. 

**Input**

```
# python train.py --model neuralbow
```

This took 3 hours 11 minutes and 35 seconds on my virtual machine. So it should be about the same for you. 



You can track your progress and see the final results of training at  https://www.wandb.com. 

Here you can see the loss on the baseline train model. 

![performance](https://github.com/aobject/NYU-AI-Project-03/blob/master/media/023_05_dash.png?raw=true)



Here you can see how our virtual machine's hardware does with the baseline model. 

![machinePerform](https://github.com/aobject/NYU-AI-Project-03/blob/master/media/023_06_dash.png?raw=true)

Look at those stats, so much great data! 



## Part 5 // Celebrate! 

Congratulation, you have accomplished quite a lot. You now know the following:

* How to set up a virtual machine
* How use the terminal to verify your GPU is accessible and has the correct drivers
* How to use Docker 
* How to setup the CodeSearchNet competition
* How to use Weights and Biases to track your deep learning models

## Resources

**Primary Resources**

1. [Paperspace](https://www.paperspace.com/)
   * Great Virtual Machine Provider we use for this tutorial 
2. [CodeSearchNet](https://github.com/github/CodeSearchNet)
   * Exciting AI Competition we setup on our virtual machine
3. [Weights and Biases](https://app.wandb.ai/)
   * Deep Learning Developer tools we use in this tutorial. 



**Drivers and Software**

1. [Ubuntu](https://ubuntu.com/)
   * The Linux OS we use for this tutorial. 
2. [Nvidia](https://www.nvidia.com/en-us/)
   * Source for all Nivida GPU requirements and installation procedures. 
3. [Nvidia-docker](https://github.com/NVIDIA/nvidia-docker)
   * GitHub repository with the Nividia-Docker install instructions. This allows the Docker container to use your GPU. 
4. [Docker](https://www.docker.com/)
   * Lightweight virtualization software that allows you to create, distribute, and use isolated containerized software environments. 