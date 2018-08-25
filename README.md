# Recipe Recommender

This is a small deep learning project that uses PyTorch to recognize food objects.

Food is one of the central aspects of our lives that computer software hasn't really revolutionized. There are a lot of problems that people have when it comes to food. There's a growing concern about healthy eating habits. We consume too much sugar, fats, and salt in our diets. There's also have a big problem with food waste - in America it's estimated that about 30-40% of food produced is wasted. There have been technology solutions - phone apps and trackers - that help people manage their diets and food waste, but there's a disconnect between the physical realm of food and the interface provided by the technology. Solutions often require the user to bridge this disconnect manually (e.g. by entering nutrition information into a phone app). Image recognition can provide a smoother interface between food and technology, making it more convenient to use and more effective.

In this project, I use deep learning to build a model that can recognize food items. This kind of technology may be seen in phone apps or smart fridges, helping us figure out what to do with our produce without throwing it away. I also made a simple web application to demonstrate the type of application this model can be used in. 

## Notebooks
Here are the notebooks created in this project, and their functions:
1. [preprocess](https://github.com/jonathanshuai/springboard-capstone2/blob/master/notebooks/1-preprocess.ipynb) - In this notebook, we look at the dataset and perform some cropping to make it suitable for use in deep learning.
2. [data-augmentation](https://github.com/jonathanshuai/springboard-capstone2/blob/master/notebooks/2-data-augmentation.ipynb) - In this notebook, we look at data augmentation techniques to apply when training our model.
3. [training](https://github.com/jonathanshuai/springboard-capstone2/blob/master/notebooks/3-training.ipynb) - In this notebook, we use PyTorch to train our model.
4. [application](https://github.com/jonathanshuai/springboard-capstone2/blob/master/notebooks/4-application.ipynb) - In this notebook, we apply the model to images multiple objects in them.

## Database
The database was too big to be put on GitHub. Here's a link to download the zip from Google Drive.

These are the files associated with the dataset.
* food-items.txt - List of food classes in alphabetical order.
* links.txt - List of links from which I downloaded the images from.

## Application
The web application was written with Angular for the front end and Flask for the back end. If you want to run it on your own machine you can follow these instructions:
1. Make sure you have all the required packages from `requirements.txt`.
2. From the `app/frontend` directory, run `ng serve`.
3. In `app/backend` configure `db.ini` to use your database.
4. From the `app/backend` directory, run `./bootstrap.sh`.
5. Navigate to `localhost:4200` to see your webpage.

Alternatively, you can just watch the [short demo on YouTube](https://youtu.be/I17lKgTJVYI).

## Acknowledgments
Thanks to my mentor Charlotte Werger for the idea and guiding me through the process.