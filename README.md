# FreddieLoanDataAnalysis-Midterm

## Instructions for running docker file 

Pull the image using following command

docker pull ankit08015/ads-midterm

**Part 1:**

**Config file Format:-**

**Name-** config.ini

**Format-**

\[user.data\]

username = {username}

password = {password}

startYear= (year)

endYear= {year}

**Download the data using following command**

docker run -v
\~/desktop/config.ini:/src/Part\_2/Classification/config.ini
ankit08015/ads-midterm sh /src/Part\_1/runPart1.sh

Commit the container using

docker commit \<container\_id\> ankit08015/ads-midterm

**Now, run the Jupiter notebook from below command to see the analysis
and EDA.**

*Name- Part\_1/Part1-EDA.ipynb*

docker run -it -p 8888:8888 ankit08015/ads-midterm jupyter notebook \--ip
0.0.0.0 \--no-browser \--allow-root

**Part 2:**

**Config file Format:-**

**Name-** config.ini

**Format-**

\[user.data\]

username = {username}

password = {password}

trainQ= (TrainQuarter)

testQ= {TestQuarter}

**A. Prediction:**

In this part everythinh is done on jupyter notebook. So, there is no data to download.

*Path- /src/Part\_2/Prediction*

Run using following command-

docker run -it -p 8888:8888 ankit08015/ads-midterm jupyter notebook \--ip
0.0.0.0 \--no-browser \--allow-root


**B. Classification:**

Download the data for single quarter using following command

docker run -v
\~/desktop/config.ini:/src/Part\_2/Classification/config.ini
ankit08015/ads-midterm sh /src/Part\_2/Classification/runPart2.sh

For Multiple quarters use following command:

docker run -v
\~/desktop/config.ini:/src/Part\_2/Classification/config.ini
ankit08015/ads-midterm sh
/src/Part\_2/Classification/runPart2Multiple.sh

Now commit the container

docker commit \<container\_id\> ankit08015/ads-midterm

Now use following command to view analysis in notebooks.

*Path is- Part\_2/Classification*

docker run -it -p 8888:8888 ankit08015/ads-midterm jupyter notebook \--ip
0.0.0.0 \--no-browser \--allow-root
