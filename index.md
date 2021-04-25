## Introduction

Azure machine learning platform is a very strong platform that sometimes is undereastimated. There are may components that we have already familiar with, like experments, machine learning tools and interfaces of its GUIs. Actually as a data scientist, we may concern mroe about how to program by SDK. Azure machine learning workspace rightly is a workspace in which we can perform our assignments through SDK as well as the I/O in interfaces.

In this article, we will discuss about how to use SDKs to create and do some works related to workspace, experiment, and run on workspace , these are very fundamental concepts and frequenly used in scenarios. We also will review the structure of components in Azure.

## 1. Azure machine learning platform

Azure machine learning platform is supporting services for operating machine learning workloads, these operations are based on the cloud of Microsoft Azure. In building on the Azure cloud platform, it enables us to manage compute, data ETL, ML workflow management(orchestration), model registration and management, metrics management and monitoring and model deployment. Now, let's have a provision on what an Azure machine learning workspace looks like:

In tutorial, the definition of Azure workspace is that a context for the experiments, data, compute targets, and other assets associated with a machine learning workload. It can be also seen as the boundary for a set of related machine learning assets. The assets in a workspace include:

![image](https://user-images.githubusercontent.com/71245576/115993558-62c00080-a5a1-11eb-828d-f15fa09cbb8c.png)

You can see a framework about what can be included in a workspace:

![image](https://user-images.githubusercontent.com/71245576/115993702-f2fe4580-a5a1-11eb-90ff-856086568f5b.png)

The Azure resources created alongside a workspace include: a storage account which is used to store files used by the workspace, an application insights instance which is used to monitor predictive services in the workspace, an Azure Key Vault instance which is used to manage secrets such as authentication keys and credentials used by the workspace, and a container registry which is created as-needed to manage containers for deployed models.

There is a very useful assignment that you can implement: role-based access control. As the tutorial said, you can assign role-based authorization policies to a workspace, enabling you to manage permissions that restrict what actions specific Azure Active Directory (AAD) principals can perform.

## 2. Creating a workspace

Creating a workspace in Azure can be in any of two ways: you can either create a new machine learning resource in interface by specifying the subscription, resource group and workspace name or use the Azure machine learning Python SDK to run code that creates a workspace.

You can check your subscription and resource group information on your Azure portal and code like this:

```python
from azureml.core import Workspace

ws = Workspace.create(name='aml-workspace', 
                      subscription_id='21c3858b-da4e-...',
                      resource_group='ML',
                      create_resource_group=True,
                      location='East US 2'
                     )
```

Deploying all of components of this workspace took a few minutes:

![image](https://user-images.githubusercontent.com/71245576/115995497-8be48f00-a5a9-11eb-9009-c3b547aaf0c6.png)

Now you can go back to refresh to check if the resource is deployed: you can see the machine learn workspace named aml-workspace is deployed, alongside with some resources that we discessed, like the container registry which is created as-needed to manage containers for deployed models.

![image](https://user-images.githubusercontent.com/71245576/115995547-b0d90200-a5a9-11eb-887b-7ca2000ade8d.png)


## 3. Azure machine learning tools

We should be very familiar with the interface of Azure machine learning studio, you can see the studio like this:

![image](https://user-images.githubusercontent.com/71245576/115995915-390bd700-a5ab-11eb-8b57-70bb6690cde4.png)

On the left shelf of the interface, there are three parts: author, assets and manage. They include many tools to process machine learning assignments. You also can create new notebook, pipeline, dataset, compute instance and so on by "Create new". 

For the assets, yo ucan manage your assets in workspace in the Azure portal but we may involve in using more focused and dedicated interfaces in machine learninig operations.

Under the Author pane, Designer is a interface for "no code" machine learning pipeline implements. The automated machine learning is a wizard interface to train model with a combination of algroithms and techniques to find the best model for your data.

### 3.1 Machine Learning SDK

GUIs in Azure machine learning studio make it very easy to create and manage machine learning aassets. However, as a data scientist, we ofen use a code-based approach to managing resources. As my experience, code-based approach can be more helpfil to some tasks that are very specific and may not easily implement through GUIs like batching files and versioning controls.

By writing scripts to create and manage resources, you can run machine learning operations from preferred context, automate asset create and configuration, ensure consistency for resouces, and incorporate machine learning asset configuration into developer operations(DevOps) workflows.

Note that Azure machine learning also provide SDKs for R programming, but Python SDK has broader capabilities than R SDK.

Now, let's install the Azure machine learning SDK for Python by using the pip package:

```python
pip install azureml-sdk
pip install azureml-sdk[notebooks,automl,explain]
```

Notice that the version of Python may cause some libraries not be installed successfully, check more information:
https://docs.microsoft.com/en-us/python/api/overview/azure/ml/install?view=azure-ml-py.

The second line above is to include th extra named notebooks, AutoML and explan, in which notebooks includes widgets for displaying detailed output in Jupyter Notebooks, AutoML is for automated machine learning training, and the explain extra includes packages for generating model explannations.

### 3.2 Connecting to a workspace

After installing the SDK package in the context we can write code to connect to the workspace and perform machine learning operations. There is a way to connect a workspace: use a workspace configuration file which can be downloaded from the Overview page.

The JSON file could be like this:

![image](https://user-images.githubusercontent.com/71245576/115997437-18468000-a5b1-11eb-8110-846ddbb0158b.png)

To connect to the workspace using the configuration file you can use the from_config method of the Workspace class in the SDK, it looks for a file named config.json in the folder containig the Python code file, but you can specify another path if necessary.
```python
from azureml.core import Workspace

ws = Workspace.from_config()
```
The alternative to connect to the workspace using the configuration file is to use the get method of the Workspace classwith specified subscription, resource group and workspace details:

```python
from azureml.core import Workspace

ws = Workspace.get(name='aml-workspace',
                   subscription_id='1234567-abcde-890-fgh...',
                   resource_group='aml-resources')
```

The configuration file technique may be more preferred due to its greater flexibility when using multiple scripts. By the way if there is no current active session with your Azure subscription, you will be prompted to authenticate.

### 3.3 Working with the workspace class

Workspace class is the starting point for most code operations. You can use its compute_targets attribute to retrieve a dictionary object containing the compute targets defined in the workspace:

```python
for compute_name in ws.compute_targets:
    compute = ws.compute_targets[compute_name]
    print(compute.name, ":", compute.type)
```
It shows the compute target that I have:

![image](https://user-images.githubusercontent.com/71245576/115997714-33fe5600-a5b2-11eb-8291-8cfca158d30b.png)

The SDK contains a rich library of classes that you can use to create, manage and use mnay kinds of asset in an Azure machine learning workspace.

### 3.4 CLI extension

The Azure command-line interface(CLI) is a cross-platform command-line tool for managing Azure resources.

To install the CLI extension, you must first install the Azure CLI, please see the full installation instractions for all supported platforms.

To find the installed version and run az version:

![image](https://user-images.githubusercontent.com/71245576/115777686-9a516180-a383-11eb-935e-227ca8a4dac3.png)

For Windows, the Azure CLI is installed via a MSI, whici provides access to the CLI through the Windows Command Prompt or PowerShell.

Now download and install the current release of the Azure CLI. After the installation is complete, you will need to close and reopen any active Windows Command Prompt or PowerShell windows to use the Azure CLI.

![image](https://user-images.githubusercontent.com/71245576/115778759-20ba7300-a385-11eb-8441-9b32095f3e04.png)

After installing it, you can now run the Azure CLI with the az command from either Windows Command Prompt or Powershell. PowerShell offers some tab completion features not available from Windows Command Prompt.

First you should log in, it will open the browser to log you in. 

![image](https://user-images.githubusercontent.com/71245576/115780137-cd492480-a386-11eb-8da1-2c3c12d89403.png)

Now add the Azure Machine Learning CLI extension by running the following commands:
```bash
az extension add -n azure-cli-ml
```
Already installed:

![image](https://user-images.githubusercontent.com/71245576/115780238-e81b9900-a386-11eb-88cb-47bc6444ef3e.png)

To use the Azure Machine Learning CLI extension, run the az ml command with the appropriate parameters for the action you want to perform. For example, to list the compute targets in a workspace, run the following command:

```bash
az ml computetarget list -g 'aml-resources' -w 'aml-workspace'
```

Notice that -g parameter specifies the name of the resource group in which the Azure Machine Learning workspace specified in the -w parameter is defined. These parameters are shortened aliases for --resource-group and --workspace-name.

If there is an exception said that it is a unrecoginized arguments, you can have a try to wait it for a while. When I make sure that my workspace name and resource group name is correct, it still throws me this, I think it is on its way initializing the platform after recent installment.

See my result of compute target list:

![image](https://user-images.githubusercontent.com/71245576/115785633-b823c400-a38d-11eb-809a-57150f8e218d.png)

### 3.5 Compute instances

Azure machine learning includes the ability to create Compute instances in a workspace to provide a development environment that is managed with all of the other assets in the workspace.

You can create Compute instances in your workspace

![image](https://user-images.githubusercontent.com/71245576/115998137-e71b7f00-a5b3-11eb-9e93-7d935b2fce62.png)

The compute instances include Jupyter notebook and JupyterLab installations that you can use to write and run code that uses the Azure Machine Learning SDK to work with assets. You also can choose a compute instance image that provides the compute specification you need, from different CPU/GPU requirement.

You can store notebooks independently in workspace storage and open them in any compute instance.

### 3.6 Visual studio code

Visual Studio Code is a lightweight code editing environment for Microsoft Windows, Apple macOS, and Linux. It provides a visual interface for many kinds of code, including Microsoft C#, JavaScript, Python and others; as well as intellisense and syntax formatting for common data formats such as JSON and XML.

After configuration, you can run your code remotely in VS code:

![image](https://user-images.githubusercontent.com/71245576/115998936-ba696680-a5b7-11eb-840b-a2ce8c57fca3.png)

I think it better to use than Jupyter if you want to perform machine learning tasks in Azure platform. As the tutorial said: visual Studio Code's flexibility is based on the ability to install modular extensions that add syntax checking, debugging, and visual management interfaces for specific workloads. For example, the Microsoft Python extension for Visual Studio Code adds support for writing and running Python code in scripts or notebooks within the Visual Studio Code interface.

By the way, there are a lot of very user-friendly functionality in this extension:

![image](https://user-images.githubusercontent.com/71245576/115999043-32d02780-a5b8-11eb-9be1-d8b6ce8dd989.png)

So I stronly recommend using VS studio for Azure machine learning assignments.

## 4. Machine learning experiments

Running experiments can explore data or to build and evaluate predictive models. In Azure machine learning, we can run a script to run multiple times of a experiment, even with different data, code, or settings; and Azure machine learning tracks each run, enabling you to view run history and compare results for each run.

### 4.1 Creating an experiment
When you submit an experiment, you use its run context to intialize and end the experment run that is tracked in Azure machine learning like the following code:

```python
from azureml.core import Experiment

# create an experiment variable
experiment = Experiment(workspace = ws, name = "my-experiment")

# start the experiment
run = experiment.start_logging()

# experiment code goes here

# end the experiment
run.complete()
```

You can check on your experiment interface or notifications that will tell you the status of this run like this:

![image](https://user-images.githubusercontent.com/71245576/116004098-33c08380-a5cf-11eb-83a9-1ec9cac45045.png)

After the experiment run has completed, you can view the details of the run in the Experiments tab in Azure machine learning studio.

### 4.2 Logging metrics and creating outputs

The logging metrics can be tracked across runs of an experiment in Azure. Every experiment generates log files that include the messages during interactive execution. This enables you to use simple print statements to write messages to the log. Meanwhile, if you want to record named metrics for comparison across runs, you can do it by using the Run object; which provides a range of logging functions specifically for this purpose.

The functions include:

![image](https://user-images.githubusercontent.com/71245576/116004279-fad4de80-a5cf-11eb-8099-8e2480d4fceb.png)

For example, to records the number of observations in a CSV file like this:

```python
from azureml.core import Experiment
import pandas as pd


# Start logging data from the experiment
run = experiment.start_logging()

# load the dataset and count the rows
data = pd.read_csv('Users/daily-bike-share.csv')
row_count = (len(data))

# Log the row count
run.log('observations', row_count)

# Complete the experiment
run.complete()
```

Now let's retrieve and view logged metrics by using the RunDetails widget in a notebook:

```python
from azureml.widgets import RunDetails

RunDetails(run).show()
```

You can also retrieve the metrics using the Run object's get_metrics method, which retruns a JSON representation of the metrics:

```python
import json

# Get logged metrics
metrics = run.get_metrics()
print(json.dumps(metrics, indent=2))
```

The result is like this:

![image](https://user-images.githubusercontent.com/71245576/116005330-733d9e80-a5d4-11eb-9baf-108f713a6a25.png)

### 4.3 Experiment output files

Sometimes we need to generate output files. Often these are trained machine learning models but you can save any sort of files and make it available as an output of your experiment run.

The output files of an experiment are saved in its outputs folder. The technique you use to add files to the outputs of an experiment depend on how you are running the experiment. 

For example, you can upload local files to the run's pitputs folder by using the Run object's upload_file method in your experiment code:

```python
run.upload_file(name='outputs/sample.csv', path_or_stream='./sample.csv')
```

You need to wait a few minutes for its uploading, I am also confused why it needs to take a while...My .csv file is only 70 KB.

When running an experiment in a remote compute context, any files written to the outputs folder in the compute context are automatically uploaded to the run's outputs folder when the run complemets.

Now you can retrieve a list of output files from the Run object like this:

```python
import json

files = run.get_file_names()
print(json.dumps(files, indent=2))
```

It shows like this:

![image](https://user-images.githubusercontent.com/71245576/116006381-efd27c00-a5d8-11eb-8d2d-5f994e72a57d.png)

### 4.4 Running a script as an experiment

You can run an experiment inline using the start_logging method of the Experiment object, but it is more common to encapsulate the experiment logic in a script and run the script as an experiment.

To access the experiment run context which is needed to log metrics, the script must import the azureml.core.Run class and call its get_context method. Then script then can use the run context to log metrics, upload files and complete the experiment.

This is the script that we will run as the experiment.
```python
from azureml.core import Run
import pandas as pd
import matplotlib.pyplot as plt
import os

# Get the experiment run context
run = Run.get_context()

# load the diabetes dataset
data = pd.read_csv('data.csv')

# Count the rows and log the result
row_count = (len(data))
run.log('observations', row_count)

# Save a sample of the data
os.makedirs('outputs', exist_ok=True)
data.sample(100).to_csv("outputs/sample.csv", index=False, header=True)

# Complete the run
run.complete()
```
Now we must define a script configuration that defines the script to be run and the Python environment in which to run it. It will be accompliashed by using a ScriptRunConfig object.

For example, the following code could be used to run an experiment based on a script in the experiment_files folder, in which any files used by the script should be contained.

```python
from azureml.core import Experiment, ScriptRunConfig
import matplotlib.pyplot as plt

# Create a script config
script_config = ScriptRunConfig(source_directory='./',
                                script='experiment.ipynb') 

# submit the experiment
experiment = Experiment(workspace = ws, name = 'my-experiment')
run = experiment.submit(config=script_config)
run.wait_for_completion(show_output=True)
```
It took a few minute that is because it needs to new a run for the experiment and then complete it. Notice that the tutorial used the script as .py file but the kernel is very old so that so many libraries should be preinstalled and context should be further configured, so I recommend you to create the script as .ipynb file.

## Reference

Build AI solutions with Azure Machine Learning, retrieved from https://docs.microsoft.com/en-us/learn/paths/build-ai-solutions-with-azure-ml-service/

