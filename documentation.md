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

Session #4: research and model tweaks\
Model now also takes into account the prnu variance and mean of frames aswell\
Gate was included so that the prnu value doesnt weigh the same as other features\
Convolution sizes have been modified for the prnu and visualization now inlcludes the prnu frames\
Next try to find a way for the model to pinpoint artifacts or "suspicious frames" if it is determined to be fake\
Also as of now the model has exceeded 100mb so it will be excluded from future updates, make own local model through models\

Session #5: New models with frame visualization\
Additional models were made to pinpoint suspicious frames and show them\
One is general classifier via frequency and spatial analysis with low prnu checnks\
Additional now includes unet artifact hunter that returns frames\
Temporal that shows any spikes in suspicious frames\
Note that accuracy will be low, I was thinking i mislabled data but online models are certainly wrong\
on atleast some of the videos that did not have obvious watermarking\
Maybe consider combining the models but i don't want to overload one and make it unusable if reverted\
Next, analyze accuracy and increase dataset\

Session #6: Thinking and planning\
It has now occured to me now about a 2 weeks since the beginning of the project\
That maybe it shouldn't be doing a general classification and it should first identify the type of video\
prnu shouldn't apply to cgi or non-camera footage, and edited videos might skew temporal bias\
So new architecture would look like a single top level classifier that then weighs gates depending\
on the video type, where each branch would output a parameter that goes into a sigmoid to output a probability\
Each branch would still return evidence via heatmap, temporal map, and gradient cam, but final decision would\ 
fall to the sigmoid or algorithm so evidence might be choatic or messy but there still will be a answer\
This would now require training a classifier, which means more data that needs to be labled and stored\
Scratch that it will just learn the optimal weight\

Session#7: Adding cuda support and expanding analysis\
Added parameters to check for cuda compatible gpus and added an audio branch\
Next look into pinpointing audio discrepencies based on time and look into creating a docker\
and look into maybe porting to web app\
Also will need to sort files to make it more clean and need to modify data_augment for new tensors\

Session#8: Cleaned up structure and debugged artifact branch\
Added some more folders to tidy files up, looked into the artifact branch to make sure it was atually working\
as of now it can identify unusual artifacts like discoloration, weird blurring, etc. It can identify hotspots but\
struggles with videos it has never seen. This is expected to improve when the real big dataset comes in, but as of now\
it remains to be of low importance in the final determination of the video\