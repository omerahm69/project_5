# ![CI logo](https://codeinstitute.s3.amazonaws.com/fullstack/ci_logo_small.png)

## How to use this repo

1. Fork this repo

2. In your forked  cherry leafs classifier repo click on the green Code button.

3. Then, from the Codespaces tab, click Create codespace on main.

4. Wait for the workspace to open. This can take a few minutes.

5. Open the jupyter_notebooks directory in the explorer sidebar, and click on the notebook you want to open

6. Click the kernel button and choose Python Environments.

7.Choose the kernel that says Python 3.12.1 as it inherits from the workspace, so it will be Python-3.12.1 as installed by Codespaces. To confirm this, you can use `! python --version` in a notebook code cell.

## Cloud IDE Reminders

To log into the Heroku toolbelt CLI:

1. Log in to your Heroku account and go to _Account Settings_ in the menu under your avatar.
2. Scroll down to the _API Key_ and click _Reveal_
3. Copy the key
4. In the terminal, run `heroku_config`
5. Paste in your API key when asked

You can now use the `heroku` CLI program - try running `heroku apps` to confirm it works. This API key is unique and private to you, so do not share it. If you accidentally make it public, then you can create a new one with _Regenerate API Key_.

## Dataset Content

- The dataset is sourced from [Kaggle](https://www.kaggle.com/codeinstitute/cherry-leaves). We then created a fictitious user story where predictive analytics can be applied in a real project in the workplace.
- The dataset contains +4 thousand images taken from the client's crop fields. The images show healthy cherry leaves and cherry leaves that have powdery mildew, a fungal disease that affects many plant species. The cherry plantation crop is one of the finest products in their portfolio, and the company is concerned about supplying the market with a compromised quality product.

## Business Case Requirements

General information:

The cherry plantation crop from Farmy & Foods, a large-scale cherry plantation operator, is facing a critical issue with powdery mildew, a fungal disease affecting cherry leaves. The current inspection process is manual, requiring an employee to spend 
~ 30 minutes per tree visually analyzing leaf samples. If mildew is found, a compound is applied in 1 minute. Given the thousands of trees across multiple farms, this approach is unscalable and labor-intensive.

Client Overview:
The client aims to improve operational effeciency through automation. Pawdery mildew compromises not only the health crops but also product quality. Asimilar issue and process exist for other crops, making this initiative strategically scalable.

Business reqirements:
- Conduct a visual study to differentiate healthy cherry leaves from those affected by powdery mildew.
- Develop a predictive model that can classifies leaf images as healthy or infected.
- Provide a user-friendly dashboard for field use.
- Achieve a minimum model accuracy of 97% to meet success criteria.
  
Coventional Data Analysis:
Traditional data analysis can support the visual study by:
- Identifying visual patterns in infected versus health leaves.
- Highlighting key visual features (e.g., color distribution, texture features, and leaf shape)
- Providing explainable insighs into the model's behavior.

  ML Solution
To save time in this process, the IT team suggested an ML system that detects instantly, using a leaf tree image, if it is healthy or has powdery mildew. A similar manual process is in place for other crops for detecting pests, and if this initiative is successful, there is a realistic chance to replicate this project for all other crops.

Dataset: The dataset is a collection of over 4,000 cherry leaf images (healthy and infected), provided by the client and also hosted on Kaggle.  

- 1 - The client is interested in conducting a study to visually differentiate a healthy cherry leaf from one with powdery mildew.
- 2 - The client is interested in predicting if a cherry leaf is healthy or contains powdery mildew.

## Hypothesis and validation
Hypothesis:
A Convolutional Neural Network (CNN) can accurately classify cherry leaf images into healthy or Powdery Mildew categories.  

Validation Steps:
-    Use a balanced dataset with sufficient examples from both classes.
-    Track performance metrics such as accuracy and precision
-    Evaluate against a seperate test set to ensure generalizability.


## Mapping Business Requirements to ML Tasks

Requuirement 1 - Visual Study 

Mapped to Data Visualisations Tasks:

- Display class-wise mean and standard deviation images.

- Show the difference between average images of both classes.

- Present image montages to visualize variation. 

Requirement 2 - Prediction
Mapped to ML Tasks:
    
- Train a binary classifier (Healthy vs. Powdery Mildew).

- Generate classification reports and evaluate with metrics.

- Deploy predictions through a web-based dashboard.


## ML Business Case

-We want an ML model to predict if an image is healthy or not based on historical image data. It is a supervised binary classification model that predicts whether a cherry leaf is healthy or infected.
The output includes:
-    A classification label
-    Associated probability/confidence score.


Success Metric:
Minimum 97% accuracy on unseen data.

Business Benefit:

-    Reduce inspection time from 30 minutes to seconds per tree.

-    Prevent supply chain contamination with compromised produce.

-    Enable future cross-crop scalability for other fungal or pest infections.


## Dashboard Design

Page 1: Quick Project Summary
- General information
    The cherry plantation crop from Farmy & Foods is facing a challenge where their cherry plantations have been presenting powdery mildew. Currently, the process is manual verification if a given cherry tree contains powdery mildew. An employee spends around 30 minutes in each tree, taking a few samples of tree leaves and verifying visually if the leaf tree is healthy or has powdery mildew. If there is powdery mildew, the employee applies a specific compound to kill the fungus. The time spent applying this compound is 1 minute. The company has thousands of cherry trees located on multiple farms across the country. As a result, this manual process is not scalable due to the time spent in the manual process inspection.
    To save time in this process, the IT team suggested an ML system that detects instantly, using a leaf tree image, if it is healthy or has powdery mildew. A similar manual process is in place for other crops for detecting pests, and if this initiative is successful, there is a realistic chance to replicate this project for all other crops. The dataset is a collection of cherry leaf images provided by Farmy & Foods, taken from their crops.
    Goal: Is to train a machine learning model to classify images into these two categories.
            1. Healthy
            2. Powdery Mildew

- Project Dataset
        The dataset used for this project is from [Kaggle](https://www.kaggle.com/codeinstitute/cherry-leaves).
        The dataset contains +4 thousand images taken from the client's
        crop fields.
        The images show healthy cherry leaves and cherry leaves
        that have powdery mildew, a fungal disease that affects many plant species.
        The cherry plantation crop is one of the finest products in their portfolio,
        and the company is concerned about supplying the market with a compromised quality product.
        
        The images show healthy cherry leaves and cherry leaves that
        have powdery mildew.

- Business requirements
        The project has 2 business requirements:
            1 - The client is interested in conducting a study to visually differentiate
                healthy cherry leaves from those with powdery mildew.
            2 - The client is interested in predicting if a cherry leaf
                is healthy or contains powdery mildew."
Page 2: Project Hypothesis and Validation
-   Block for each project hypothesis, describe the conclusion and how you validated it.

Page 3: Visual Study Findings
    It will answer business requirement 1
    Checkbox 1 - Difference between average and variability image
    Checkbox 2 - Differences between average
    Checkbox 3 - Image Montage

Page 4: ML Performance Metrics
    Label Frequencies for Train, Validation and Test Sets
    Model History - Accuracy and Losses
    Model evaluation result

Page 5: Live Predictions
    Upload a cherry leaf image (JPG/PNG) and the model will predict if it's Healthy or Powdery Mildew affected.


## Unfixed Bugs

- You will need to mention unfixed bugs and why they were unfixed. This section should include shortcomings of the frameworks or technologies used. Although time can be a significant variable for consideration, paucity of time and difficulty understanding implementation is not a valid reason to leave bugs unfixed.

## Deployment

### Heroku

- The App live link is: `https://YOUR_APP_NAME.herokuapp.com/`
- Set the runtime.txt Python version to a [Heroku-20](https://devcenter.heroku.com/articles/python-support#supported-runtimes) stack currently supported version.
- The project was deployed to Heroku using the following steps.

1. Log in to Heroku and create an App
2. At the Deploy tab, select GitHub as the deployment method.
3. Select your repository name and click Search. Once it is found, click Connect.
4. Select the branch you want to deploy, then click Deploy Branch.
5. The deployment process should happen smoothly if all deployment files are fully functional. Click the button Open App on the top of the page to access your App.
6. If the slug size is too large, then add large files not required for the app to the .slugignore file.

## Main Data Analysis and Machine Learning Libraries
-   Python
-   NumPy, Pandas, matplotlib, Tensorflow, seaborn
-   Streamlit
-   jupyter
-   GitHub
-   Heroku

## Credits

- The process from the Code Institute WalkthroughProject01 project was used to help create this project.
    In addition to the material from the course I had from Code Institute

## Acknowledgements (optional)

- I'm extremely grateful to the completion of my Diploma Full Stack Software Development, it would not have been possible without the support and nurturing of my mentor Mr. Daniel Hamilton.Thank you for Code Institute for giving me the opportunity to join the program and thanks to all of the staff, student care and student support for their patience and encouragement.
