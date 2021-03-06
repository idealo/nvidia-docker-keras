# Train Neural Networks on Amazon EC2 with GPU support

Workflow that shows how to train neural networks on EC2 instances with GPU support. The goal is to present a simple and stable setup to train on GPU instances by using **Docker** and the NVIDIA Container Runtime **nvidia-docker**. A minimal example is given to train a small CNN built in Keras on MNIST. We achieve a 30-fold speedup in training time when training on GPU versus CPU.


## Getting started

1. Install [Docker](https://docs.docker.com/install/)

2. Install [Docker Machine](https://docs.docker.com/machine/install-machine/)

3. Install [AWS Command Line Interface](https://docs.aws.amazon.com/cli/latest/userguide/installing.html)


## Train locally on CPU

1. Build Docker image for CPU
```
docker build -t docker-keras . -f Dockerfile.cpu
```

2. Run training container (**NB:** you might have to increase the container resources [[link](https://docs.docker.com/config/containers/resource_constraints/)])
```
docker run docker-keras
```


## Train remote on GPU

1. Configure your AWS CLI. Ensure that your account has limits for GPU instances [[link](https://docs.aws.amazon.com/AWSEC2/latest/UserGuide/ec2-resource-limits.html)]

```
aws configure
```

2. Launch EC2 instance with **Docker Machine**. Choose an Ubuntu AMI based on your region (https://cloud-images.ubuntu.com/locator/ec2/).
For example, to launch a **p2.xlarge** EC2 instance named **ec2-p2** with a Tesla K80 GPU run
(**NB:** change region, VPC ID and AMI ID as per your setup)

```
docker-machine create --driver amazonec2 \
                      --amazonec2-region eu-west-1 \
                      --amazonec2-ami ami-58d7e821 \
                      --amazonec2-instance-type p2.xlarge \
                      --amazonec2-vpc-id vpc-abc \
                      ec2-p2
```
```
docker-machine create --driver amazonec2 --amazonec2-region eu-central-1 --amazonec2-ami ami-0932440befd74cdba --amazonec2-instance-type m5.2xlarge  --amazonec2-vpc-id vpc-b0ec4fda ec2-p2
```

3. ssh into instance

```
docker-machine ssh ec2-p2
```

4. Update NVIDIA drivers and install **nvidia-docker** (see this [blog post](https://towardsdatascience.com/using-docker-to-set-up-a-deep-learning-environment-on-aws-6af37a78c551) for more details)

```
# update NVIDIA drivers
sudo add-apt-repository ppa:graphics-drivers/ppa -y
sudo apt-get update
sudo apt-get install -y nvidia-375 nvidia-settings nvidia-modprobe

# install nvidia-docker
wget -P /tmp https://github.com/NVIDIA/nvidia-docker/releases/download/v1.0.1/nvidia-docker_1.0.1-1_amd64.deb
sudo dpkg -i /tmp/nvidia-docker_1.0.1-1_amd64.deb && rm /tmp/nvidia-docker_1.0.1-1_amd64.deb
```


5. Run training container on GPU instance

```
sudo nvidia-docker run idealo/nvidia-docker-keras
```

This will pull the Docker image `idealo/nvidia-docker-keras` from [DockerHub](https://hub.docker.com/r/idealo/nvidia-docker-keras) and start the training.
The corresponding Dockerfile can be found under `Dockerfile.gpu` for reference.



## Training time comparison

We trained MNIST for 3 epochs (~98% accuracy on validation set):

• MacBook Pro (2.8 GHz Intel Core i7, 16GB RAM): **620 seconds**

• p2.xlarge (Tesla K80): **41 seconds**

• p3.2xlarge (Tesla V100): **20 seconds**


## Copyright

See [LICENSE](LICENSE) for details.
