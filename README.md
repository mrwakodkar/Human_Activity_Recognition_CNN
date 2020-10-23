# Human_Activity_Recognition_CNN
Human Activity Recognition for Elderly people for better understanding of their health status

Dataset link: https://www.cis.fordham.edu/wisdm/dataset.php

#Steps to run##<br>
1.clone the repo<br>
2.install the streamlit if not installed ```pip install streamlit```or ```pip3 install streamlit```<br>
3.run the app.py with the command ```streamlit run app.py```<br>
4.a tab will open in your default browser <br>
5.All the ml models will be running on your dashboard<br>

#Usecase<br>
we can monitor the elderly people health by classifying their activities and based on the time duraction they perform the specific activity we can categories the current health situation e.g, we have installed the video where the elderly people leaves for the long duration or we have putted the belts on their waist by this we can get the data and according to this data we can classify their movements in to Walking,Jogging,Sitting,Standing,Upstairs,Downstairs and then based on the time duration in a single day that they perform the particular activity we can categories their health into Healthy,Unhealthy.
#What i have did<br>
I have build ml model which can perdict the Activity performed by the person based on the given data (data can be of in any format either 1.video or 2.acclerometer or gyroscope data here i have used the 2 nd type of data for Activity recognition ) I will predict the Activity performed in later part we can categories the Activites+duration into unhealthy or healthy person so that we can keep eye on the elderly people.
