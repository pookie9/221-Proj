# 221 project! 
## Josh King, Sydney Li, Peter Wang

Play around with `python run.py` to test it out! 

#Setting up for Text to Speech
In order to generate speech from text, we must first label the the training audio. 
We are using pocketsphinx, part of the Sphinx project developed at CMU in order to generate a time stamped phoneme. 

##Installing Sphinx
1. Update Brew 
'''
brew update
'''
2. Use the following tap (following instructions on http://www.moreiscode.com/getting-started-with-cmu-sphinx-on-mac-os-x/)
'''
brew tap watsonbox/cmu-sphinx
'''
3. Install both sphinxbase and pocketsphinx
'''
brew install --HEAD watsonbox/cmu-sphinx/cmu-sphinxbase
brew install --HEAD watsonbox/cmu-sphinx/cmu-pocketsphinx
'''
