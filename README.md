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

## Business Requirements

The cherry plantation crop from Farmy & Foods is facing a challenge where their cherry plantations have been presenting powdery mildew. Currently, the process is manual verification if a given cherry tree contains powdery mildew. An employee spends around 30 minutes in each tree, taking a few samples of tree leaves and verifying visually if the leaf tree is healthy or has powdery mildew. If there is powdery mildew, the employee applies a specific compound to kill the fungus. The time spent applying this compound is 1 minute. The company has thousands of cherry trees located on multiple farms across the country. As a result, this manual process is not scalable due to the time spent in the manual process inspection.

To save time in this process, the IT team suggested an ML system that detects instantly, using a leaf tree image, if it is healthy or has powdery mildew. A similar manual process is in place for other crops for detecting pests, and if this initiative is successful, there is a realistic chance to replicate this project for all other crops. The dataset is a collection of cherry leaf images provided by Farmy & Foods, taken from their crops.

- 1 - The client is interested in conducting a study to visually differentiate a healthy cherry leaf from one with powdery mildew.
- 2 - The client is interested in predicting if a cherry leaf is healthy or contains powdery mildew.

## Hypothesis and how to validate?

The Hypothesis of this project is that a deep learning model, particularly a Convolutional Neural Network (CNN), can effectively classify cherry leaf images into 'Healthy' and 'Powdery Mildew' with high accuracy.
The validation process involved the following:
- Using a abalanced dataset with sufficient examples of both classes
- List here your project hypothesis(es) and how you envision validating it (them).

## The rationale to map the business requirements to the Data Visualisations and ML tasks

-Business Requirement 1: Data Visualization

-   We will display the "mean" and "standard deviation" images for healthy and powdery_mildew.
-   We will display the difference between average healthy and powdery_mildew.
-   We will display an image montage for either     healthy and powdery_mildew.

Business Requirement 2: Classification

-   We want to predict if a given image is healthy and powdery_mildew.
-   We want to build a binary classifier and generate reports.

## ML Business Case

-We want an ML model to predict if an image is healthy or not based on historical image data. It is a supervised model, a 2-class, single-label classification model.
Our ideal outcome is to provide the the cherry plantation crop from Farmy & Foodsmedical with a faster and more reliable way to differentiate between healthy and powdery mildew leaves.


The model success metrics are

Accuracy of 65% or above on the test set.
The model output is defined as a flag, indicating if the image is healthy or not and the associated probability of that. As usual, the medical staff will do the blood smear workflow and upload the picture to the App. The prediction is made on the fly (not in batches).
Heuristics:
the process is manual verification if a given cherry tree contains powdery mildew. An employee spends around 30 minutes in each tree, taking a few samples of tree leaves and verifying visually if the leaf tree is healthy or has powdery mildew. If there is powdery mildew, the employee applies a specific compound to kill the fungus. The time spent applying this compound is 1 minute. The company has thousands of cherry trees located on multiple farms across the country. As a result, this manual process is not scalable due to the time spent in the manual process inspection.

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

### Content

- The text for the Home page was taken from Wikipedia Article A.
- Instructions on how to implement form validation on the Sign-Up page were taken from [Specific YouTube Tutorial](https://www.youtube.com/).
- The icons in the footer were taken from [Font Awesome](https://fontawesome.com/).

### Media

- The photos used on the home and sign-up page are from This Open-Source site.
- The images used for the gallery page were taken from this other open-source site.

## Acknowledgements (optional)

- Thank the people who provided support throughout this project.
