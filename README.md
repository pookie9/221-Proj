# 221 project! 
## Josh King, Sydney Li, Peter Wang

Play around with `python run.py` to test it out! 

#Setting up for Text to Speech
In order to generate speech from text, we must first label the the training audio. 
We are using pocketsphinx, part of the Sphinx project developed at CMU in order to generate a time stamped phoneme. 

##Installing Sphinx
Update Brew 
```
brew update
```
Use the following tap (following instructions on http://www.moreiscode.com/getting-started-with-cmu-sphinx-on-mac-os-x/)
```
brew tap watsonbox/cmu-sphinx
```
Install both sphinxbase and pocketsphinx
```
brew install --HEAD watsonbox/cmu-sphinx/cmu-sphinxbase
brew install --HEAD watsonbox/cmu-sphinx/cmu-pocketsphinx
```

Now you should be able to run pocketsphinx to transcribe wav files

```
pocketsphinx_continuous -infile sample.wav
```

For the full manual on pocketsphinx_continous, see this site:
http://manpages.org/pocketsphinx_continuous

To test time-stamped phoneme labels without specifying language models:
```
pocketsphinx_continuous -logfn log.txt -time yes -allphone -infile sample.wav
```
