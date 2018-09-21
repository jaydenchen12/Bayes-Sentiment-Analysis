import math, os, pickle, re

class Bayes_Classifier:
   def __init__(self, trainDirectory = "./"):
      '''This method initializes and trains the Naive Bayes Sentiment Classifier.  If a
      cache of a trained classifier has been stored, it loads this cache.  Otherwise,
      the system will proceed through training.  After running this method, the classifier
      is ready to classify input text.'''
      self.positiveDict = {}
      self.negativeDict = {}
      if (os.path.isfile(trainDirectory + "negativeDictionary.txt") and os.path.isfile(trainDirectory + "positiveDictionary.txt")):
           self.positiveDict = self.load("positiveDictionary.txt")
           self.negativeDict = self.load("negativeDictionary.txt")
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
       self.save(self.negativeDict, "negativeDictionary.txt")
       self.save(self.positiveDict, "positiveDictionary.txt")


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
          pWordPos = numberPos / sum(self.positiveDict.itervalues())
          pWordNeg = numberNeg / sum(self.negativeDict.itervalues())
          condProbPos = numberPos / ( numberNeg + numberPos)
          condProbPosOut = (condProbPos * pWordPos) / pPos
          probabilitiesPos.append(math.log(condProbPosOut))

          condProbNeg = numberNeg / ( numberNeg + numberPos)
          condProbNegOut = (condProbNeg * pWordNeg) / pNeg
          probabilitiesNeg.append(math.log(condProbNegOut))
      totalProbNeg = sum(probabilitiesNeg)
      totalProbPos = sum(probabilitiesPos)
      if math.exp(totalProbPos) > math.exp(totalProbNeg):
          return "positive"
      elif math.exp(totalProbPos) < math.exp(totalProbNeg):
          return "negative"
      else:
          return "neutral"

   def loadFile(self, sFilename):
      '''Given a file name, return the contents of the file as a string.'''

      f = open("db_txt_files/"+ sFilename, "r")
      sTxt = f.read()
      f.close()
      return sTxt

   def save(self, dObj, sFilename):
      '''Given an object and a file name, write the object to the file using pickle.'''

      f = open(sFilename, "w")
      p = pickle.Pickler(f)
      p.dump(dObj)
      f.close()

   def load(self, sFilename):
      '''Given a file name, load and return the object stored in the file.'''

      f = open(sFilename, "r")
      u = pickle.Unpickler(f)
      dObj = u.load()
      f.close()
      return dObj

   def tokenize(self, sText):
      '''Given a string of text sText, returns a list of the individual tokens that
      occur in that string (in order).'''
      lTokens = []
      sToken = ""
      for c in sText:
         if re.match("[a-zA-Z0-9]", str(c)) != None or c == "\'" or c == "_" or c == '-':
            sToken += c
         else:
            if sToken != "":
               lTokens.append(sToken)
               sToken = ""
            if c.strip() != "":
               lTokens.append(str(c.strip()))

      if sToken != "":
         lTokens.append(sToken)

      return lTokens
bayes = Bayes_Classifier()
test = []
truePos = 0
falsePos = 0
falseNeg = 0
iFileList = []
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
print "Precision: "
print precision
print "Recall: "
print recall
print "f-measure: "
print fmeasure
