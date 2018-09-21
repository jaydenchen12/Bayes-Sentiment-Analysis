#created by Jayden Chen
# 11/28/2017
# only run this script with python 3
# need to pip install echant library for this script to work also

import math, os, pickle, re, enchant

class Bayes_Classifier:
   def __init__(self, trainDirectory = "./"):
      '''This method initializes and trains the Naive Bayes Sentiment Classifier.  If a
      cache of a trained classifier has been stored, it loads this cache.  Otherwise,
      the system will proceed through training.  After running this method, the classifier
      is ready to classify input text.'''
      self.positiveDict = {}
      self.negativeDict = {}
      if (os.path.isfile(trainDirectory + "negativeDictionary_best2.txt") and os.path.isfile(trainDirectory + "positiveDictionary_best2.txt")):
           self.positiveDict = self.load("positiveDictionary_best2.txt")
           self.negativeDict = self.load("negativeDictionary_best2.txt")
      else:
           self.train()

   def train(self):
       '''Trains the Naive Bayes Sentiment Classifier.'''
       iFileList = []

       for fFileObj in os.walk("db_txt_files/"):
            iFileList = fFileObj[2]
            break
       for file in iFileList:
            inFile = self.loadFile(file)
            wordList = self.tokenize(inFile)
            parseName = file.split("-")
            if parseName[1] == "5":
                for word in wordList:
                    if word not in self.positiveDict:
                        self.positiveDict[word] = 0
                    self.positiveDict[word] += 1
            elif parseName[1] == "1":
                for word in wordList:
                    if word not in self.negativeDict:
                        self.negativeDict[word] = 0
                    self.negativeDict[word] += 1

       self.save(self.negativeDict, "negativeDictionary_best2.txt")
       self.save(self.positiveDict, "positiveDictionary_best2.txt")


   def classify(self, sText):
      '''Given a target string sText, this function returns the most likely document
      class to which the target string belongs. This function should return one of three
      strings: "positive", "negative" or "neutral".
      '''
      texts = self.tokenize(sText);
      pPos = 0.5
      pNeg = 0.5
      probabilitiesPos = []
      probabilitiesNeg = []
      probabilitiesPos.append(math.log(pPos))
      probabilitiesNeg.append(math.log(pNeg))
      for text in texts:
          #calulating the chance that it is positive and negative
          numberPos = self.positiveDict.get(text)
          numberNeg = self.negativeDict.get(text)
          if numberPos is None and numberNeg is None:
              numberPos = 1
              numberNeg = 1
              continue
          elif numberPos is None:
              numberPos = 1
          elif numberNeg is None:
              numberNeg = 1
          numberPos = float(numberPos)
          numberNeg = float(numberNeg)
          pWordPos = numberPos / sum(self.positiveDict.values())
          pWordNeg = numberNeg / sum(self.negativeDict.values())
          condProbPos = numberPos / ( numberNeg + numberPos)
          condProbPosOut = (condProbPos * pWordPos) / pPos
          probabilitiesPos.append(math.log(condProbPosOut))

          condProbNeg = numberNeg / ( numberNeg + numberPos)
          condProbNegOut = (condProbNeg * pWordNeg) / pNeg
          probabilitiesNeg.append(math.log(condProbNegOut))

      totalProbNeg = sum(probabilitiesNeg)
      totalProbPos = sum(probabilitiesPos)
      if totalProbPos > totalProbNeg:
          return "positive"
      elif totalProbPos < totalProbNeg:
          return "negative"
      else:
          print(totalProbNeg, totalProbPos)
          return "neutral"

   def loadFile(self, sFilename):
      '''Given a file name, return the contents of the file as a string.'''

      f = open("db_txt_files/"+ sFilename, "r")
      sTxt = f.read()
      f.close()
      return sTxt

   def save(self, dObj, sFilename):
      '''Given an object and a file name, write the object to the file using pickle.'''

      f = open(sFilename, "wb")
      p = pickle.Pickler(f)
      p.dump(dObj)
      f.close()

   def load(self, sFilename):
      '''Given a file name, load and return the object stored in the file.'''

      f = open(sFilename, "rb")
      u = pickle.Unpickler(f)
      dObj = u.load()
      f.close()
      return dObj

   def tokenize(self, sText):
      '''Given a string of text sText, returns a list of the individual tokens that
      occur in that string (in order).'''
      sText = sText.lower()
      lTokens = []
      sToken = ""
      for c in sText:
         if re.match("[a-zA-Z0-9]", str(c)) != None or c == "\'" or c == "_" or c == '-':
               sToken += c
         else:
            if sToken != "":
               lTokens.append(sToken)
               sToken = ""
            if c.strip() != "" and c.strip() != "," and c.strip() != "." and c.strip() != "?" and c.strip() != "!":
               lTokens.append(str(c.strip()))

      if sToken != "":
         lTokens.append(sToken)
      modList = []
      for index in range(len(lTokens)):
         try:
             modList.append(lTokens[index] + " " + lTokens[index+1])
         except IndexError:
             modList.append(lTokens[index])
      return modList

bayes = Bayes_Classifier()
test = []
truePos = 0
falsePos = 0
falseNeg = 0
iFileList = []
filecount = 0
for fFileObj in os.walk("movies_reviews/"):
    iFileList = fFileObj[2]
    break

for file in iFileList:
    inFile = bayes.loadFile(file)
    wordList = bayes.tokenize(inFile)
    parseName = file.split("-")
    if parseName[1] == "5":
        #first is true value, second is classified value
        trial = ("positive", bayes.classify(inFile))
        test.append(trial)
    elif parseName[1] == "1":
        trial = ("negative", bayes.classify(inFile))
        test.append(trial)
    print(trial)
    filecount += 1
    print(filecount)
for item in test:
    if item[0] == "positive" and item[1] == "positive":
        truePos += 1
    elif item[0] == "positive" and item[1] == "negative":
        falseNeg += 1
    elif item[0] == "negative" and item[1] == "positive":
        falsePos += 1
precision = float(truePos)/(truePos + falsePos)
recall = float(truePos)/(truePos + falseNeg)
fmeasure = (2 * precision * recall) / (precision + recall)
print("Precision: ")
print(precision)
print("Recall: ")
print(recall)
print("f-measure: ")
print(fmeasure)

