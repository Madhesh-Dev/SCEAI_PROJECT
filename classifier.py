import data
from random import randint
import numpy as np
from sklearn.metrics import accuracy_score

from functools import reduce


mutProb = .15
tournamentSize = 5
elitism = True


WEIGHT_ERROR = 0.2

alpha_cut = 0

nbRules = 10



truthCache = {}
inferenceCache = {}


class Triangle:
    def __init__(self,min,max):
        self.min = min
        self.max = max
        self.alpha_cut = alpha_cut
        
    def valAt(self,x): 
        alpha = 2/(self.max-self.min)
        f = lambda x: alpha*(x-self.min)
        g = lambda x: alpha*(-x+self.max)
        mid = (self.min+self.max)/2
        if(x<self.min or x>self.max):
            return 0.0
        elif(x<mid):
            return f(x)
        else:
            return g(x)
    def at(self,x):
        val = self.valAt(x)
        return val if val >= alpha_cut else 0

smallTriangle = Triangle(-0.5,0.5)
medTriangle = Triangle(0,1)
largeTriangle = Triangle(0.5,1.5)


    
class Indiv:
    def __init__(self):
        self.rules = []
        for i in range(nbRules):
            self.rules.append(generateRule())
    def __str__(self):
        s = ""
        for i in range(nbRules):
            s +="Rule "+str(i)+": "+ self.rules[i]+"\n"
        return s

def generateRule():
    randBits = []
    randRule = randint(0,pow(2,12)-1)
    rule = "{0:b}".format(randRule)

    randClass = randint(0,2)
    
    for i in range(12-len(rule)):
        randBits.append(0)
    for i in range(len(rule)):
        randBits.append(int(rule[i]))
    return randBits



def getCompetitionStrength(rule):
    competitionStrength = [0,0,0]
    
    for classNumber in range(3):
        classArray = data.getClassArray(classNumber)
        competitionStrength[classNumber] = sum( [ getMuA(rule,row) for _,row in classArray.iterrows()] )

    return competitionStrength
    

def getConf(rule):
    hashedRule = "".join(str(i) for i in rule)
    if hashedRule in truthCache:
        return truthCache[hashedRule]
    else:
        if hashedRule == "000000000000":
            maxIndex,maxValue = (-1,-1)
        else:
            
            competitionStrength = getCompetitionStrength(rule)
            
            strSum = sum(competitionStrength)
            if strSum != 0:
                truthDegree = [i/strSum for i in competitionStrength]
                
                maxIndex,maxValue = max(enumerate(truthDegree),key=lambda x:x[1])
            else:
                maxIndex,maxValue = (-1,0) #No classes are recognized
        truthCache[hashedRule] = (maxIndex,maxValue)
        return (maxIndex,maxValue)

def getConfVect(rules):
    return [getConf(rule) for rule in rules]



def toCacheString(rule,data_row):
    strRule = "".join(str(i) for i in rule)
    strRow = ""
    for x in range(len(data_row)):
        strRow += "%0.3f" %data_row[x]
    return strRule+strRow


def getMuA(rule,data_row):
    cacheString = toCacheString(rule,data_row)
    if cacheString in inferenceCache:
        return inferenceCache[cacheString]
    else:
        maxArray = []
        ruleCounter = 0
        for x in range(0, len(data_row)):
            datum = data_row[x]
            if rule[ruleCounter:ruleCounter+3] != [0,0,0]:
                small = 0
                medium = 0
                large = 0
                if rule[ruleCounter] == 1:
                    small = smallTriangle.at(datum)
                if rule[ruleCounter+1] == 1:
                    medium = medTriangle.at(datum)
                if rule[ruleCounter+2] == 1:
                    large = largeTriangle.at(datum)
                maxArray.append(max(small,medium,large))
            ruleCounter += 3
        if maxArray == []:
            muA = 0
        else:
            muA = min(maxArray)
        inferenceCache[cacheString] = muA
        return muA

def getMuAVect(rules,data_row):
    return [getMuA(rule,data_row) for rule in rules]


def getPredictedConfVect(confVect,muAVect):
    predictedConfVect = [0,0,0]
    cnt = [1,1,1]


    for i in range(len(confVect)):
        ruleClass,ruleConf = confVect[i]
        if ruleClass != -1:
            predictedConfVect[ruleClass] += muAVect[i]*ruleConf
            cnt[ruleClass] += 1

    averagedPredictedConfVect = [predictedConfVect[i]/cnt[i] for i in range(3)]
    return averagedPredictedConfVect

def getPredictedClass(rules,data_row):
    predictedConfVect = getPredictedConfVect(getConfVect(rules),getMuAVect(rules,data_row))
    predictedClass,predictedConf = max(enumerate(predictedConfVect),key=lambda x:x[1])
    return predictedClass,predictedConf


def getPredictedClasses(indiv,data):
    predictedClassArray = []
    for _,data_row in data.iterrows():
        predictedClass,predictedConf = getPredictedClass(indiv.rules,data_row)
        predictedClassArray.append(predictedClass)
    return predictedClassArray


def getAccuracy(indiv):
    predictedClassArray = getPredictedClasses(indiv,data.X_test)
    score = accuracy_score(data.y_test,predictedClassArray)
    return score

def checkRules(indiv):
    confVect = getConfVect(indiv.rules)
    goodRulesNb = 0
    badRulesNb = 0
    for classNb,conf in confVect:
        if classNb == -1:
            if conf == 0:
                badRulesNb += 1
            elif conf == -1:
                goodRulesNb += 1
            else:
                print("This is really weird in classifier:checkrules")
    return goodRulesNb,badRulesNb


def calcComplexity(indiv,dontCare=True):
    complexity = 0
    for rule in indiv.rules:
        if dontCare:
            for i in range(0, 3):
                if not (rule[i] == rule[i+1] == rule[i+2] == 0):
                    complexity += 1
                    i += 3
        else:
            complexity += sum(rule)        
    return complexity
 
def getMuAPast(muArray,rule):
    maxArray = []
    
    for i in [0,3,6,9]:
        if rule[i:i+3] != [0,0,0]:
            maxArray.append(max([ muArray[j]*rule[j] for j in range(i,i+3)]))
    
    if maxArray == []:
        return -1
    muA = min(maxArray)
    return muA
               

def getClassFromRule(rule):
    competitionStrength = [0,0,0]
    for classNumber in range(3):
        classArray = data.getClassArray(classNumber)
        competitionStrength[classNumber] += np.dot(classArray,rule)
    maxIndex = getMaxIndex(competitionStrength)
    print("The class of this rule is class " + str(maxIndex) + \
          " with a score of: " + str(competitionStrength[maxIndex]))
    return maxIndex,competitionStrength

def getMaxIndex(l):
    if len(l)==1:
        return 0
    else:
        max = l[0]
        index = 0
        for i in range(1,len(l)):
            if l[i] > max:
                max = l[i]
                index = i
        return index




def getMuArray(row):
    muArray = []
    x1 = row['SepalLength']
    x2 = row['SepalWidth']
    x3 = row['PetalLength']
    x4 = row['PetalWidth']

    muArray += [smallTriangle.at(x1),medTriangle.at(x1),largeTriangle.at(x1)]
    muArray += [smallTriangle.at(x2),medTriangle.at(x2),largeTriangle.at(x2)]
    muArray += [smallTriangle.at(x3),medTriangle.at(x3),largeTriangle.at(x4)]
    muArray += [smallTriangle.at(x4),medTriangle.at(x4),largeTriangle.at(x4)]
    return muArray







def getTruth(rule):
    ruleString = str(rule)
    if ruleString in truthCache:
        return truthCache[ruleString] 

    competitionStrength = getCompetitionStrength(rule)
    sumComp = sum(competitionStrength)
    if sumComp == 0:
        truthCache[ruleString] = [-1,0]
        return [-1,0]
    else:
        index, value = max(enumerate([competitionStrength[0]/sumComp,competitionStrength[1]/sumComp,competitionStrength[2]/sumComp]), key = lambda e: e[1])
        truthCache[ruleString] = [index, value]
        return [index, value]







def simple_infer(rules):
    
    inferences = []
    for index,datum in data.X_train.iterrows():
        classes = [[],[],[]]
        inf = [0,0,0]
        
        for rule in rules:

            if cache not in inferenceCache:
                inferenceCache[cache] = getMuA(getMuArray(datum),rule)
            inferred = inferenceCache[cache]

            inferred = getMuA(getMuArray(datum),rule)
            
            confidence = getTruth(rule)
            classes[confidence[0]].append(inferred)
        for i in range(len(classes)):
            if classes[i] != []:
                inf[i] += sum(classes[i])/len(classes[i])
        inferences.append(max(enumerate(inf), key=lambda x:x[1])[0])
    return inferences



def infer(rules, forAccuracy = False):
    class1 = []
    class2 = []
    class3 = []

    classes = [class1, class2, class3]
    inferences = []

    for index,datum in data.X_test.iterrows():
        for rule in rules:
            ruleString = str(rule)
            if ruleString not in inferenceCache:
                inferenceCache[ruleString] = getMuA(getMuArray(datum),rule)
                
            inferred = inferenceCache[ruleString]
            confidence = getTruth(rule)
            classes[confidence[0]].append(inferred*confidence[1])
        
        if class1:
            class1_inferred = reduce(lambda x, y: x + y, classes[0]) / len(classes[0])
        else:
            class1_inferred = 0
        if class2:
            class2_inferred = reduce(lambda x, y: x + y, classes[1]) / len(classes[1])
        else:
            class2_inferred = 0
        if class3:
            class3_inferred = reduce(lambda x, y: x + y, classes[2]) / len(classes[2])
        else:
            class3_inferred = 0

        l = [class1_inferred, class2_inferred, class3_inferred]

        if forAccuracy:
            inferences.append(l)
        else:
            inferences.append(l[data.y_train.iloc[index]]) 
        class1 = []
        class2 = []
        class3 = []


    return inferences


def getAccuracyPast(inferences):
    total = len(inferences)
    correct = 0
    weightedCorrect = 0.0
    for i in range(len(inferences)):
        index, value = max(enumerate(inferences[i]), key = lambda e: e[1])
        if index == data.y_train.iloc[i]:
            correct += 1
            weightedCorrect += value
    accuracy = float(correct) / float(total)
    weightedAccuracy = weightedCorrect / float(total)
    print("Accuracy: ", accuracy)
    print("Weighted Accuracy: ", weightedAccuracy)


def computeFitness(inferences):
    fitDatum = []
    for confidence in inferences:
        fitDatum.append(confidence)
    return sum(fitDatum)


def applyRule(rule, data):
    ruleCounter = 0
    calcedRuleArray = []
    for x in range(0, len(data)):
        datum = data[x]
        small = 0
        medium = 0
        large = 0
        

        if rule[ruleCounter] == 1:
            small = smallTriangle.at(datum)
        if rule[ruleCounter+1] == 1:
            medium = medTriangle.at(datum)
        if rule[ruleCounter+2] == 1:
           large = largeTriangle.at(datum)
        calcedRuleArray.append(max(small, medium, large))
        ruleCounter += 3
    calcedRule = min(calcedRuleArray[0], calcedRuleArray[1], calcedRuleArray[2])
    finalInference = calcedRule
    return finalInference

