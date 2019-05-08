# Deep-Learning-for-VUI

Amazon Alexa is one of the most popular representatives of today’s voice-powered conversational agents. These kind of voice user interface (VUI) platforms allow the third-party developers to freely design and build their own voice applications. However, how to evaluate these voice applications still needs exploring. As indicated in [1], topic Analysis for these voice applications is crucial in evaluating them. In this project, I deploy deep learning methods (e.g. DAN, text CNN, skip-gram) to conduct topic analysis based on my self-collected Alexa skill dataset. This project uses a machine learning algorithm (LR) to generate baseline. Both DAN & text CNN have better performance (with fastText embedding) , as well as DAN with self-generated skip-gram embedding matrix.

## Data Collection -- Alexa Crawler

In order efficiently conduct data collection, I developed a crawler to automate the task of collecting a large sample of skills’ responses to voice commands. This crawler follows the basic Alexa interaction mode of "open-command-stop" to initiate the skill and exit. As indicated in the follow figure, this crawler simulates the voice interaction between users and Alexa devices.

![Screenshot](figures/crawler.jpeg)

The crawler iterated through over 40,000 Alexa skills to perform this reponses data collection process for each skill, which is described pragmatically as Algorithm 1. 

![Screenshot](figures/algorithm.jpeg)
