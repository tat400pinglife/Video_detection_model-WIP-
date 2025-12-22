# AI_video_detection_Model

Session #1: setup\
Created dedicated conda environment, fully portable with yml file\
Starting testing frame extraction with temportal and frequency techniques\
No GPU implementation as of yet, future plans to use google cloud server or remote GPU server  \
Feel free to try to use your own nvidia cards, I only have AMD :p\

Session #2: creating model algorithm and sourcing data\
Created sample data and finsihed frame analysis script and modified to not save frames\
Script now saves data into tensors with data labels\
Still need to implement the model and try with big dataset\
currently do not have the space or resources to load in vidbench data\
Will need to look into usng profesors remote servor or buy external harddisk or SSD\
(probably not buying with these prices going up tho ...)\
Note to self: check if frames are sufficient\

Session #3: actually creating a small model that returns a reading of a video\
Script now only looks at the first 8 seconds, NOTE frame size and # of frames may need to be reduced\
or have some form of offloading to avoid having a gazillion frames on disk\
Next steps are to fine tune the algorithm but also look into a GPU (server, hugging face, or google)\
to look into big data training and making a model with more parameters than like a hundred\
opencv is giving me warnings but it runs so like lady justice i turn a blind eye\