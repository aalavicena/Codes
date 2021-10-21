# Overview
THIS IS A UNIVERSITY PROJECT.

This model is based on Papia Nandi's codes in Towards Data Science on Medium. Link: https://towardsdatascience.com/cnns-for-audio-classification-6244954665ab
The dataset for this project is warblr10k from Bird Presence dataset available on Kaggle. See here: https://www.kaggle.com/christofhenkel/bird-presence
The dataset contains 3 parameters: itemid (filename), datasetid (dataset), and hasbird (bird yes or bird no indicated by 1 and 0, respectively). See the .csv metadata file for more info. 

# For Noah 
One particular problem that I encountered when trying to run the data is because the codes called for each audio file to signal start and end (see Line 102 and 52).
Whilst I don't need to cut them because it's just a binary classification, the original codes use a different dataset (Rainforest Connection) and has extra parameters. That is tstart (start of part of the sound that has sound in it) and tend (end of that part of the sound)
I don't know if it could run perfectly (haven't tried cuz I got no GPU), could you please help me check and modify anything that you think necessary? Thanks!
