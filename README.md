# Adding a Conditional Step to Evaluate the Model Performance in AWS SageMaker Pipelines

## Table of Content
  * [Demo](#demo)
  * [Overview](#overview)
  * [Motivation](#motivation)
  * [Technical Aspect](#technical-aspect)
  * [Installation](#installation)
  * [Run](#run)
  * [Deployement on Heroku](#deployement-on-heroku)
  * [Directory Tree](#directory-tree)
  * [To Do](#to-do)
  * [Bug / Feature Request](#bug---feature-request)
  * [Technologies Used](#technologies-used)
  * [Contributing](#Contributing)
  * [License](#license)
  * [Credits](#credits)

## Demo
![full](https://github.com/Data-Fenix/aws-sagemaker-pipeline/blob/main/demo/full.gif)

## Overview

Previously we developed a customer churn prediction model using classification technique and already deploy that model in the production enviornment([To see that project, please click here](https://github.com/Data-Fenix/aws-sagemaker-pipeline)).

And also we created a seperate training pipeline for the project. Now datascientist wants to evaluate the model and he need to store the model versions based on the model performance matices.



##### More details: 
* Pipeline will run weekly and give the predictions base for each week
* This scheduling time will change according to the requirements of your organization.
A condition step requires a list of conditions, a list of steps to run if the condition evaluates to true, and a list of steps to run if the condition evaluates to false.

In here we are going to add that step to the pipeline and in here we are going to check the model performance exceed the accuracy level 0.8 or not. If the accuracy level exceed that treshold value, it will automatically going to the defined conditions. In here it's storing the model artifact in the model registry step. The following example shows how to create a Condition step definition.

According to the business requirements and considering the cost factor, two separate pipelines are developed for training and inference jobs. Following sections show how to deploy **“training pipelines”** in AWS Sagemaker and the uniqueness of this is, we use **“bring your own code concept to orchestrate the ML workflow”**.
 
## Dataset

This is data gathered from 7043 telco customers and dataset has 21 features (columns). Each row represents a customer, each column contains customer’s attributes described on the column Metadata.

The “Churn” column is our target variable and it has two outcomes. Therefore this is a binary classification problem and using below link,you can easily download the dataset. https://www.kaggle.com/blastchar/telco-customer-churn

## Motivation

When I was searching about AWS Sagemaker, I struggle a lot as lack of references in this feild. It has some references, but there are missing few things. Therefore I need to fullfil that gap. So now I have some experience in this feild and as a MLOps team memeber, I migrated a lot of projects into cloud. I saved data scientists' valuable time by automating and scheduling their ML projects. So now I need to share that experience and knowledge with you and this is my first step of that journey.
## Technical Aspects

This project highly relies on AWS cloud platform and in here we don’t train any model. We use previously developed projects from scratch and trying to deploy it in AWS.
1) Used s3 bucket to store the input/output data, models and model artifacts.
2) Used Sagemaker to implement and modify the existing scripts. And also automated the Python scripts.
3) Containerize all the scripts using Docker images and stored them in ECR repository
4) Used build your own container concept in AWS
5) Used AWS Sagemaker Studio to view the Sagemaker Pipelines, execution history, model performance matrices,etc.
6) Used Sagemaker Model Registry to save the model

## Installation

#### Requirements

1. An AWS account
2. Python 3.5+
3. Docker (optional)


Only thing you need to satisfy in this list is you must have an AWS account. If you don't have an account you can create it free, using below link:
https://aws.amazon.com/free/
    
## Run
1) Upload/push/clone previously implemented mobile price prediction project into AWS Sagemaker instance or JupyterLab environment.
2) Run build_docker.ipynb
3) Execute training_pipeline.ipynb
4) See the workflow using AWS Sagemaker stuido
* (i) Open Sagemaker Studio
* (ii) Choose Pipeline option then you can see the execution list
* (iii) Select the executing one, see the visualization
* (iv) If you need to see more details about each stage, click on the each bubble and right hand side you can see all the details of each stage
* (v) To see the model registry, click the Model Registry option in the left hand side

## Deployment on AWS Sagemaker

Will disscuss more details in the pipeline_train.ipynb

## Directory Tree

```
├── data
│   ├── input_dataset
|       └── telco_cutomer_churn.csv
|   └──data_inference.csv
├── customer_churn_training_preprocessing
│   ├── Dockerfile
|   └── preprocessing.py
├── customer_churn_training_evaluation
│   ├── Dockerfile
|   └── evaluation.py
├── customer_churn_training
│   ├── model
|       └── train.py
|   ├── Dockerfile
├── images
├── build_docker.ipynb
├── aws_helper.py
├── CONTRIBUTING.md
├── pipeline_training.ipynb
├── LICENSE
├── setup.py
└── tox.ini
```

## To Do

Need to add,
1) Need to add evalution step as a seperate component
2) Need to add accuracy condition to the workflow

Don't worry, we will discss above topics and many more in the future sections.

## Bug / Feature Requests
If you find a bug (the website couldn't handle the query and / or gave undesired results), kindly open an issue [here](https://github.com/Data-Fenixcondition-step-aws-sagemaker-training-pipeline/issues/new) by including your search query and the expected result.

If you'd like to request a new function, feel free to do so by opening an issue [here](https://github.com/Data-Fenix/acondition-step-aws-sagemaker-training-pipeline/issues/new). Please include sample queries and their corresponding results.

## Technologies Used
[<img target="_blank" src="https://venturebeat.com/wp-content/uploads/2021/02/SageMaker.jpg?fit=1292%2C664&strip=all" width=200>](https://venturebeat.com/wp-content/uploads/2021/02/SageMaker.jpg?fit=1292%2C664&strip=all) [<img target="_blank" src="https://www.cloudsavvyit.com/p/uploads/2019/06/55634f08.png?width=1198&trim=1,1&bg-color=000&pad=1,1" width = 200>](https://www.cloudsavvyit.com/p/uploads/2019/06/55634f08.png?width=1198&trim=1,1&bg-color=000&pad=1,1) [<img target="_blank" src="https://jfrog.com/connect/images/6053d4dc2f6c53160a53d407_linux-container-updates-iot.png" width = 200>](https://jfrog.com/connect/images/6053d4dc2f6c53160a53d407_linux-container-updates-iot.png) [<img target="_blank" src="https://logos-world.net/wp-content/uploads/2021/02/Docker-Logo-2015-2017.png" width = 200>](https://logos-world.net/wp-content/uploads/2021/02/Docker-Logo-2015-2017.png) [<img target="_blank" src="https://miro.medium.com/max/438/1*0G5zu7CnXdMT9pGbYUTQLQ.png" width = 200>](https://miro.medium.com/max/438/1*0G5zu7CnXdMT9pGbYUTQLQ.png) [<img target="_blank" src="https://logos-world.net/wp-content/uploads/2021/10/Python-Symbol.png" width = 200>](https://logos-world.net/wp-content/uploads/2021/10/Python-Symbol.png)

## Contributing

<p><b> ML Enginner </b> : Anuradha Dissanayake </p>
<p><b> Data Scientist </b>: Shashi Withanage </p>

## License

Copyright 2022 Anuradha Dissanayake and Shashi Withange

## Credits

1) https://github.com/aws/amazon-sagemaker-examples/blob/master/sagemaker-pipelines/tabular/abalone_build_train_deploy/sagemaker-pipelines-preprocess-train-evaluate-batch-transform.ipynb
2) https://docs.aws.amazon.com/sagemaker/



