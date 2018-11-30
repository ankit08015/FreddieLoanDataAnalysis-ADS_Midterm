# FreddieLoanDataAnalysis-Midterm

**[Instructions for running docker file]{.underline}**

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

docker run -it ankit08015/ads-midterm sh /src/Part\_1/runPart1.sh

Commit the container using

docker commit \<container\_id\> ankit08015/ads-test

**Now, run the Jupiter notebook from below command to see the analysis
and EDA.**

*Name- Part\_1/Part1-EDA.ipynb*

docker run -it -p 8888:8888 ankit08015/ads-test jupyter notebook \--ip
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

docker commit \<container\_id\> ankit08015/ads-test

Now use following command to view analysis in notebooks.

*Path is- Part\_2/Classification*

docker run -it -p 8888:8888 ankit08015/ads-test jupyter notebook \--ip
0.0.0.0 \--no-browser \--allow-root
