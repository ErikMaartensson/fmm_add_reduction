#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 29 10:20:57 2024
"""

import numpy as np
import copy
#import sys
import re
from pathlib import Path
import time
import random
#import datetime

# Checks the dimensions of the matrices
def correctDimensions(A, B, C, n, m, k):
    # Check that each matrix has the right dimension
    if len(A[0]) != n*m:
        return False
    if len(B[0]) != m*k:
        return False
    if len(C) != n*k:
        return False
    
    # Check that each matrix has the same number of terms
    if len(A) != len(B) or len(B) != len(C[0]):
        
        return False
    
    return True

def dimensionsFromFileName(fileName):
    splitName = fileName.split('-')
    n = int(splitName[1][0])
    m = int(splitName[1][1])
    k = int(splitName[1][2])
    isBinary = splitName[-1][0:4] == "mod2"
    return n, m, k, isBinary

# Verifies that an exact scheme for matrix multiplication of an [n*m] matrix times an [m*k] matrix works correctly
# The function is a slightly tweaked version from the code of https://github.com/arbenson/fast-matmul/blob/master/codegen/verify.py
def verifyScheme(A, B, C, n, m, k, isBinary = False):
    if not correctDimensions(A, B, C, n, m, k):
        print("Incorrect matrix dimensions")
        return False
    
    q = len(A)
    totalChecks = 0
    failedChecks = 0
    flag = True
    
    for a in range(n): 
        for b in range(m):
            # a and b describe the position in a row of A
            rA = a*m+b
            for c in range(m):
                for d in range(k):
                    # c and d describe the position in a row of B
                    rB = c*k+d
                    for e in range(n):
                        for f in range(k):
                            # e and f describe the position in a row of C
                            rC = e*k+f
                            # compute the contribution
                            currentSum = 0
                            totalChecks += 1
                            for i in range(q):
                                currentSum += A[i][rA]*B[i][rB]*C[rC][i]
                            if isBinary: 
                                currentSum %= 2 # Reduce sum modulo 2 in the binary case
                            if a == e and b == c and d == f:
                                # should be a 1
                                if currentSum != 1:
                                    #print("Trouble at", a, b, c, d, e, f, "sum should be 1, is ", currentSum)
                                    #print("\tcheck row",rA,"of A, row",rB,"of B, and row",rC,"of C")
                                    failedChecks += 1
                                    flag = False
                            else:
                                # should be a 1
                                if currentSum != 0:
                                    #print("Trouble at", a, b, c, d, e, f, "sum should be 0, is ", currentSum)
                                    #print("\tcheck row",a*n+b,"of A, row",c*k+d,"of B, and row",e*k+f,"of C")
                                    failedChecks += 1
                                    flag = False
    
    #if failedChecks > 0:
        #print("\nFailed",failedChecks,"out of",totalChecks,"checks!")
    #print(totalChecks)
    #print(failedChecks)
    
    return flag, failedChecks

def verifyAdditions(G, patternsWithSign):
    for pattern in patternsWithSign:
        performAction(G, pattern[0], pattern[1])
    return additionsDone(G)

def verifyAdditionsTriple(A, patternsWithSignA, B, patternsWithSignB, C, patternsWithSignC):

    return verifyAdditions(A, patternsWithSignA) and verifyAdditions(B, patternsWithSignB) and verifyAdditions(C, patternsWithSignC)

# Translates a grid to binary
def gridToBinary(G):
    for i in range(len(G)):
        for j in range(len(G[0])):
            G[i][j] = G[i][j]%2
    return G

def tripleToBinary(A,B,C):
    gridToBinary(A)
    gridToBinary(B)
    gridToBinary(C)

# Decides whether two grids are identical
def compareGrid(G1, G2):

    n = len(G1)
    m = len(G1[0])
    
    if n != len(G2) or m != len(G2[0]):
        return False # The dimensions do not match
    
    flag = True
    
    for i in range(n):
        for j in range(m):
            g1 = int(G1[i][j])
            g2 = int(G2[i][j])
            if g1 != g2:
                print("Mismatch for (i,j)=(",i,",",j,")")
                flag = False
                # return False # The grid values do not match
    
    return flag

# Detects if a grid contains entries that are not in {-1, 0, 1}
def containsNonStandardEntries(G):
    for row in G:
        for entry in row:
            if entry != -1 and entry != 0 and entry != 1:
                return True
    return False

def tripleContainsNonStandardEntries(A, B, C):
    return containsNonStandardEntries(A) or containsNonStandardEntries(B) or containsNonStandardEntries(C)

def printGrid(G):
    for row in G:
        for entry in row:
            # print(entry, end = " ")
            print(f"{entry: 2d}", end = "  ")
        print("\n")

# Re-orders the rows of the C grid correctly after reading grids in Moosbauer's format
def reOrderGrid(C, n):
    C2 = C[::n]
    for i in range(1,n):
        C2 = C2 + C[i::n]
    return C2

# Various functions for parsing files
        
def processQuadruple(quadruple, n, m):
    i = int(quadruple[2])
    j = int(quadruple[3])
    value = 1
    if quadruple[0] == '-':
        value = -1
    index = m*i + j - m - 1
    return index, value

# Used to flip around the B factors to make Moosbauer and Adaptive follow the standard form of Benson-Grey
def flipFactor(B, m, k):
    B = np.array(B).reshape(m,k).tolist() # Turn the vector into a matrix
    # B = np.array(B).reshape(k,m).tolist() # Turn the vector into a matrix
    B = list(map(list, zip(*B))) # Transpose the matrix
    B = sum(B, []) # Turn the transposed matrix back to a vector
    return B

# Parts of reading a file on 'exp' format
def processFactor(expression, n, m):
    a = n*m*[0]
    numberOfTerms = len(expression)//4
    for k in range(numberOfTerms):
        quadruple = expression[4*k:4*k + 4]
        index, value = processQuadruple(quadruple, n, m)
        a[index] = value
    return a
    
# Parts of reading a file on 'exp' format
def cleanFactor(expression):
    expression = expression.strip() # Trim white space
    if expression[0] == '(':
        expression = expression[1:-1]
    if expression[0] != '-':
        expression = '+' + expression # Clarify the first term of the factor
    return expression

# Reads the three grids given in human-readable form as in Kauers-Moosbauer
def readGridsExpInner(fileName, n = 2, m = 2, k = 2):
    file = open(fileName, 'r')
    Lines = file.readlines()
    A = []
    B = []
    C = []
    for line in Lines:
        
        splitLine = line.split('*')
        aFactor = cleanFactor(splitLine[0])
        bFactor = cleanFactor(splitLine[1])
        cFactor = cleanFactor(splitLine[2])

        aFactor = processFactor(aFactor, n, m)
        bFactor = processFactor(bFactor, m, k)
        cFactor = processFactor(cFactor, k, n)
        
        # To do: Flip around the bFactor?
        
        A.append(aFactor)
        B.append(bFactor)
        C.append(cFactor)

    C = list(map(list, zip(*C))) # Transpose the grid for C
    C = reOrderGrid(C, n)

    file.close()
    return A, B, C

# Reads a grid on expression format
def readGridsExp(fileName):
    splitName = fileName.split('-')
    n = int(splitName[1][0])
    m = int(splitName[1][1])
    k = int(splitName[1][2])
    return readGridsExpInner(fileName, n, m, k)

# Replaces non-integer characters with empty string
def extractFactor(expression, n, m):
    cleanLine = re.sub('[{},]', '', expression)
    cleanLine = cleanLine.split()
    factor = n*m*[0]
    for k in range(n*m):
        factor[k] = int(cleanLine[k])
    return factor
    

def readGridsMInner(fileName, n, m, k):
    
    txt = Path(fileName).read_text()
    txt = txt.replace('\n', '')
    Lines = txt.split('}}},')
    
    A = []
    B = []
    C = []
    
    for i in range(len(Lines)):
        splitLine = Lines[i].split('}},')

        aFactor = extractFactor(splitLine[0], n, m)
        bFactor = extractFactor(splitLine[1], m, k)
        cFactor = extractFactor(splitLine[2], n, k)
        
        A.append(aFactor)
        B.append(bFactor)
        C.append(cFactor)
        
    C = list(map(list, zip(*C))) # Transpose the grid for C
    C = reOrderGrid(C, n) # Re-order the rows of C correctly
    
    return A,B,C

# Reads a grid on m format
def readGridsM(fileName):
    splitName = fileName.split('-')
    n = int(splitName[1][0])
    m = int(splitName[1][1])
    k = int(splitName[1][2])
    return readGridsMInner(fileName, n, m, k)

# Reads the three grids given on the Benson-Grey format
def readGrids(fileName):
    numberOfGridsRead = 0
    A = []
    B = []
    C = []
    file = open(fileName, 'r')
    Lines = file.readlines()
    for line in Lines:
        lineToChars = line.split()
        if lineToChars[0] == '#':
            numberOfGridsRead = numberOfGridsRead + 1
            continue
        newRow = [int(char) for char in lineToChars]
        # print(newRow)
        if numberOfGridsRead == 0:
            A.append(newRow)
        elif numberOfGridsRead == 1:
            B.append(newRow)
        else:
            C.append(newRow)
    A = list(map(list, zip(*A))) # Transpose the grid for A
    B = list(map(list, zip(*B))) # Transpose the grid for B
    file.close()
    return A, B, C

# Reads a scheme from a specified file
def readGridsGeneral(fileName):
    fileType = fileName.split('.')[1]
    if fileType == "txt":
        A,B,C = readGrids(fileName)
    elif fileType == "exp":
        A,B,C = readGridsExp(fileName)
    elif fileType == "m":
        A,B,C = readGridsM(fileName)
    else:
        A,B,C = [],[],[]
        print("Unsupported file type")

    n, m, k, isBinary = dimensionsFromFileName(fileName)
    flag, errors = verifyScheme(A, B, C, n, m, k, isBinary)
    
    if flag:
        print("Scheme correctly performs multiplication of", n, "by", m, "times", m, "by", k, "matrices.\n")
    else:
        print("Verification of the scheme failed, with a total of", errors, "errors!\n")

    return A,B,C

# Gives the grids for the multiplication algorithm by Laderman 
def getGridsLaderman():
    A = [[1, 1, 1, -1, -1, 0, 0, -1, -1],
    [1, 0, 0, -1, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 1, 0, 0, 0, 0],
    [-1, 0, 0, 1, 1, 0, 0, 0, 0],
    [0, 0, 0, 1, 1, 0, 0, 0, 0],
    [1, 0, 0, 0, 0, 0, 0, 0, 0],
    [-1, 0, 0, 0, 0, 0, 1, 1, 0],
    [-1, 0, 0, 0, 0, 0, 1, 0, 0],
    [0, 0, 0, 0, 0, 0, 1, 1, 0],
    [1, 1, 1, 0, -1, -1, -1, -1, 0],
    [0, 0, 0, 0, 0, 0, 0, 1, 0],
    [0, 0, -1, 0, 0, 0, 0, 1, 1],
    [0, 0, 1, 0, 0, 0, 0, 0, -1],
    [0, 0, 1, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 1, 1],
    [0, 0, -1, 0, 1, 1, 0, 0, 0],
    [0, 0, 1, 0, 0, -1, 0, 0, 0],
    [0, 0, 0, 0, 1, 1, 0, 0, 0],
    [0, 1, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 1, 0, 0, 0],
    [0, 0, 0, 1, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 1, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 1]]


    B = [[0, 0, 0, 0, 1, 0, 0, 0, 0],
    [0, -1, 0, 0, 1, 0, 0, 0, 0],
    [-1, 1, 0, 1, -1, -1, -1, 0, 1],
    [1, -1, 0, 0, 1, 0, 0, 0, 0],
    [-1, 1, 0, 0, 0, 0, 0, 0, 0],
    [1, 0, 0, 0, 0, 0, 0, 0, 0],
    [1, 0, -1, 0, 0, 1, 0, 0, 0],
    [0, 0, 1, 0, 0, -1, 0, 0, 0],
    [-1, 0, 1, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 1, 0, 0, 0],
    [-1, 0, 1, 1, -1, -1, -1, 1, 0],
    [0, 0, 0, 0, 1, 0, 1, -1, 0],
    [0, 0, 0, 0, 1, 0, 0, -1, 0],
    [0, 0, 0, 0, 0, 0, 1, 0, 0],
    [0, 0, 0, 0, 0, 0, -1, 1, 0],
    [0, 0, 0, 0, 0, 1, 1, 0, -1],
    [0, 0, 0, 0, 0, 1, 0, 0, -1],
    [0, 0, 0, 0, 0, 0, -1, 0, 1],
    [0, 0, 0, 1, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 1, 0],
    [0, 0, 1, 0, 0, 0, 0, 0, 0],
    [0, 1, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 1]]

    C = [[0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0],
    [1, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 1, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 1, 1, 0, 1, 1, 0, 0, 0, 1, 0, 1, 0, 1, 0, 0, 0, 0, 0],
    [0, 1, 1, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 1, 0, 0, 0, 0, 0, 0],
    [0, 1, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 1, 1, 0, 0, 1, 0, 0],
    [0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 1, 0],
    [0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1]]
    
    return A, B, C



# A lower limit for the achievable circuit depth
def minCircuitDepth(actions, atomic):
    nodes = atomic*[(0,0)] + actions
    depths = (atomic + len(actions))*[0]
    
    for k in range(atomic, len(depths)):
        depths[k] = 1 + min(depths[nodes[k][0]], depths[nodes[k][1]])
    return max(depths)

# Calculates the naive circuit depth of a set of actions
def circuitDepth(actions, atomic):
    nodes = atomic*[(0,0)] + actions
    depths = (atomic + len(actions))*[0]
    
    for k in range(atomic, len(depths)):
        depths[k] = 1 + max(depths[nodes[k][0]], depths[nodes[k][1]])
    return max(depths)

# The naive number of additions for a grid G
def naiveAdditions(G):
    numAdditions = 0
    for row in G:
        numAdditions = numAdditions + sum(r != 0 for r in row)
    return numAdditions - len(G)

#def NumAdditionsFromScores(scores):
#    return sum(scores) 

def additionsDone(G):
    for row in G:
        if sum(r != 0 for r in row) > 1:
            return False
    return True

# Permutes the list of actions
def permuteActionList(savingScores, savingActions):
    seed = 0
    seed = 2
    random.seed(seed) # For reproducible randomness
    # random.seed(datetime.now().timestamp()) # For non-reproducible randomness
    permutation = list(range(len(savingScores)))
    random.shuffle(permutation)
    savingScores = [savingScores[k] for k in permutation]
    savingActions = [savingActions[k] for k in permutation]
    return savingScores, savingActions

# Input - a matrix G, an action (i,j), grid dimensions m, n  
def checkActionSum(G, i, j):
    # m = len(G) # Number of rows of the grid
    numberOfPatterns = 0
    for row in G:
        # print(str(row[i]) + " " + str(row[j]))
        if row[i] != 0 and row[i] == row[j]:
            numberOfPatterns = numberOfPatterns + 1
    
    return numberOfPatterns

# Input - a matrix G, an action (i,j), grid dimensions m, n  
def checkActionDifference(G, i, j):
    # m = len(G) # Number of rows of the grid
    numberOfPatterns = 0
    for row in G:
        if row[i] != row[j] and abs(row[i]) == abs(row[j]):
            numberOfPatterns = numberOfPatterns + 1
            # print(str(row[i]) + " " + str(row[j]))
    
    return numberOfPatterns

def checkAction(G, i, j, isSum):
    if isSum:
        return checkActionSum(G, i, j)
    else:
        return checkActionDifference(G, i, j)

# Tracks the changes to G that subtracting columns i and j lead to
def trackChangesSubtract(G, i, j):
    changes = []
    for k in range(len(G)):
        row = G[k]
        if (row[i] == 1 and row[j] == -1) or (row[i] == -1 and row[j] == 1):
            changes.append((k, (row[i], row[j])))
    return changes

# Tracks the changes to G that adding columns i and j lead to
def trackChangesSum(G, i, j):
    changes = []
    for k in range(len(G)):
        row = G[k]
        if (row[i] == 1 and row[j] == 1) or (row[i] == -1 and row[j] == -1):
            changes.append((k, (row[i], row[j])))
    return changes

# Tracks the changes to G that a given action leads to
def trackChanges(G, action):
    if action[1] == 1:
        return trackChangesSum(G, action[0][0], action[0][1])
    else:
        return trackChangesSubtract(G, action[0][0], action[0][1])

# Reverts the changes to G as specified in changes
def revertChanges(G, pattern, changes):
    for row in G:
        row.pop() # Remove every element of the last column of G
    i = pattern[0]
    j = pattern[1]
    for change in changes:
        k = change[0]
        G[k][i] = change[1][0]
        G[k][j] = change[1][1]

# Returns a list of all actions that save additions
def getAllSavingActions(G):
    n = len(G[0])
    actions = []
    scores = []
    for i in range(n):
        for j in range(i + 1, n):
            score = checkActionSum(G, i, j)
            if score > 1:
                scores.append(score)
                actions.append(((i,j),1))
            score = checkActionDifference(G, i, j)
            if score > 1:
                scores.append(score)
                actions.append(((i,j),0))
    return scores, actions

def checkActionSumImproved(G, i, j):
    Gcopy = copy.deepcopy(G)
    score = performActionSum(Gcopy, (i, j))
    scores, actions = getAllSavingActions(Gcopy)
    return score, len(scores)

def checkActionDifferenceImproved(G, i, j):
    Gcopy = copy.deepcopy(G)
    score = performActionSubtract(Gcopy, (i, j))
    scores, actions = getAllSavingActions(Gcopy)
    return score, len(scores)

# Greedy vanilla -
# Searches for the current best action - chooses the lexicographically first one in terms of a tie and sum before difference
def greedySearch(G):
    bestPattern = (0,0)
    bestScore = 0
    isSum = 1
    n = len(G[0])
    for i in range(n):
        for j in range(i + 1, n):
            score = checkActionSum(G, i, j)
            if score > bestScore:
                bestScore = score
                isSum = 1
                bestPattern = (i,j)
            score = checkActionDifference(G, i, j)
            if score > bestScore:
                bestScore = score
                isSum = 0
                bestPattern = (i,j)
    return bestPattern, bestScore, isSum

# Slight improvement over Greedy vanilla's search for the best next action 
def greedySearchImproved(G):
    bestPattern = (0,0)
    bestScore = 0
    bestOptions = 0
    isSum = 1
    n = len(G[0])
    for i in range(n):
        for j in range(i + 1, n):
            score, numOptions = checkActionSumImproved(G, i, j)
            if score > bestScore or (score == bestScore and numOptions > bestOptions):
                bestScore = score
                bestOptions = numOptions
                isSum = 1
                bestPattern = (i,j)
            score, numOptions = checkActionDifferenceImproved(G, i, j)
            if score > bestScore or (score == bestScore and numOptions > bestOptions):
                bestScore = score
                bestOptions = numOptions
                isSum = 0
                bestPattern = (i,j)
    return bestPattern, bestScore, isSum

# Performs an action and updates the new list of saving actions
def performActionAndUpdateSavingActions(G, scores, actions, currentAction):
    newScores = []
    newActions = []
    performAction(G, currentAction[0], currentAction[1])
    newIndex = len(G[0]) - 1
    i = currentAction[0][0]
    j = currentAction[0][1]
    potentialColumns = [[False,False]]*newIndex
    for m in range(len(actions)):
        action = actions[m]
        k = action[0][0]
        l = action[0][1]
        if k != i and k != j and l != i and l != j: # If neither index is part of the current pattern (i,j), then the scores are unchanged
            newActions.append(action)
            newScores.append(scores[m])
            continue
        isSum = action[1]
#        potentialColumns[k][isSum] = True
#        potentialColumns[l][isSum] = True
        #score = checkAction(G, k, l, isSum)
        if k != i and k != j:
            potentialColumns[k][isSum] = True
        if l != i and l != j:
            potentialColumns[l][isSum] = True
        score = 0 # For the current action itself we don't need to compute the score
        if not (k == i and l == j and isSum == currentAction[1]):
            score = checkAction(G, k, l, isSum)
        if score > 1:
            newScores.append(score)
            newActions.append(((k, l),isSum))
    # Check the combination of any column together with the new column
    for m in range(newIndex):
        if potentialColumns[m][1]:
            score = checkAction(G, m, newIndex, 1)
            if score > 1:
                newScores.append(score)
                newActions.append(((m, newIndex), 1))
        if potentialColumns[m][0]:
            score = checkAction(G, m, newIndex, 0)
            if score > 1:
                newScores.append(score)
                newActions.append(((m, newIndex), 0))
        
        
    return newScores, newActions

# Finds the best action w.r.t. the metric of Greedy potential
def greedySearchWithPotential(G, actions, scores, alpha):
    bestPattern = (0,0)
    bestScore = 0
    isSum = 1

    for i in range(len(scores)):
        score = scores[i]
        action = actions[i]
        changes = trackChanges(G, action) # tracks the changes that an action leads to
        #Gcopy = copy.deepcopy(G) # skip copying the whole matrix?
        #performAction(Gcopy, action[0], action[1])
        #newScores, newActions = getAllSavingActions(Gcopy)
        #newScores, newActions = performActionAndUpdateSavingActions(Gcopy, scores, actions, action)
        newScores, newActions = performActionAndUpdateSavingActions(G, scores, actions, action)
        revertChanges(G, action[0], changes) # reverts the changes to G that the action lead to
        score = score + alpha*(sum(newScores) - len(newScores)) # Should we subtract by 1 here? It shouldn't change anything, right?
        # score = sum(newScores) - len(newScores) # Only use remaining potential
        if score > bestScore:
            bestScore = score
            bestPattern = action[0]
            isSum = action[1]
    return bestPattern, bestScore, isSum

def performActionSum(G, pattern):
    score = 0
    for row in G:
        if row[pattern[0]] == 1 and row[pattern[1]] == 1:
            row[pattern[0]] = 0
            row[pattern[1]] = 0
            row.append(1)
            score = score + 1
        elif row[pattern[0]] == -1 and row[pattern[1]] == -1:
            row[pattern[0]] = 0
            row[pattern[1]] = 0
            row.append(-1)
            score = score + 1
        else:
            row.append(0)
    # return G, score
    return score

def performActionSubtract(G, pattern):
    score = 0
    for row in G:
        if row[pattern[0]] == 1 and row[pattern[1]] == -1:
            row[pattern[0]] = 0
            row[pattern[1]] = 0
            row.append(1)
            score = score + 1
        elif row[pattern[0]] == -1 and row[pattern[1]] == 1:
            row[pattern[0]] = 0
            row[pattern[1]] = 0
            row.append(-1)
            score = score + 1
        else:
            row.append(0)
    # return G, score
    return score
    
def performAction(G, pattern, isSum):
    if isSum:
        return performActionSum(G, pattern)
    else:
        return performActionSubtract(G, pattern)

# A function that performs trivial additions. It assumes that there are no saving actions left.
def performTrivialAdditions(G):
    scores = []
    patternsWithSign = []

    for k in range(len(G)):
        row = G[k]
        firstNonZeroFound = False
        firstNonZero = -1
        secondNonZero = -1
        i = 0

        while i < len(row):
            if row[i] != 0:
                if not firstNonZeroFound:
                    firstNonZeroFound = True
                    firstNonZero = i
                else:
                    secondNonZero = i
                    if row[firstNonZero] == row[secondNonZero]:
                        score = performActionSum(G, (firstNonZero, secondNonZero))
                        patternsWithSign.append(((firstNonZero, secondNonZero), 1))
                    else:
                        score = performActionSubtract(G, (firstNonZero, secondNonZero))
                        patternsWithSign.append(((firstNonZero, secondNonZero), 0))
                    scores.append(score)
                    firstNonZeroFound = False
                    row = G[k]
            i += 1
    return scores, patternsWithSign

# Greedy vanilla - chooses the action that saves the most additions, chooses the first found action in the case of ties
def greedySolution(G):
    scores = []
    patternsWithSign = []
    
    while 1:
        bestPattern, score, isSum = greedySearch(G)
        score = performAction(G, bestPattern, isSum)
        patternsWithSign.append((bestPattern, isSum))
        scores.append(score)
        if additionsDone(G):
            return scores, patternsWithSign

# Greedy vanilla plus marginally smarter tie-breaks - not in the paper
def greedySolutionImproved(G):
    scores = []
    patternsWithSign = []
    
    while 1:
        bestPattern, score, isSum = greedySearchImproved(G)
        print(bestPattern)
        score = performAction(G, bestPattern, isSum)
        patternsWithSign.append((bestPattern, isSum))
        scores.append(score)
        if additionsDone(G):
            return scores, patternsWithSign

# Greedy with potential - optimizes based on how many additions an action saves + how much remaining potential is left
def greedySolutionWithPotential(G, alpha, printActions = True):  
    scores = []
    patternsWithSign = []
    savingScores, savingActions = getAllSavingActions(G) # Compute these the slow way once
    if False:
        savingScores, savingActions = permuteActionList(savingScores, savingActions)
    while 1:
        
        # For convenience for the combination with the greedy solver, use an inner function and call the trivial stuff from an outer function?
        if len(savingScores) == 0:
            #trivialScores, trivialPatternsWithSign = greedySolution(G)
            trivialScores, trivialPatternsWithSign = performTrivialAdditions(G)
            return scores + trivialScores, patternsWithSign + trivialPatternsWithSign
            
        bestPattern, score, isSum = greedySearchWithPotential(G, savingActions, savingScores, alpha)
        if printActions:
            print(bestPattern, isSum, score)
        score = checkAction(G, bestPattern[0], bestPattern[1], isSum)
        savingScores, savingActions = performActionAndUpdateSavingActions(G, savingScores, savingActions, (bestPattern, isSum))
        patternsWithSign.append((bestPattern, isSum))
        scores.append(score)
        if additionsDone(G):
            return scores, patternsWithSign

# A function that finds the best alpha for greedy with potential - the inner part
def optimizeAlphaForGreedyWithPotentialInner(G, alphaLow, scoresLow, patternsWithSignLow, alphaHigh, scoresHigh, patternsWithSignHigh, epsilon, printActions = False):
    Gcopy = copy.deepcopy(G)
    additionsLow = len(scoresLow)
    additionsHigh = len(scoresHigh)
    
    alphaMid = (alphaLow + alphaHigh)/2
    print("Tests alpha equal to", alphaMid)
    scoresMid, patternsWithSignMid = greedySolutionWithPotential(Gcopy, alphaMid, printActions)
    additionsMid = len(scoresMid)
    
    # We are within the desired level of accuracy to the optimal solution
    if alphaMid - alphaLow < epsilon:
        if additionsHigh < additionsMid and additionsHigh < additionsLow:
            return alphaHigh, scoresHigh, patternsWithSignHigh
        if additionsMid < additionsLow:
            return alphaMid, scoresMid, patternsWithSignMid
        return alphaLow, scoresLow, patternsWithSignLow
    
    # Compute the best solution to the left of the middle point
    if additionsLow >= additionsMid and additionsHigh < additionsMid: # The optimal solution will not be in the left half
        print("Optimal alpha is more than", alphaMid)
        alphaLeft = alphaMid
        scoresLeft = scoresMid
        patternsWithSignLeft = patternsWithSignMid
    else: # Find the optimal solution to the left of the middle point
        alphaLeft, scoresLeft, patternsWithSignLeft = optimizeAlphaForGreedyWithPotentialInner(G, alphaLow, scoresLow, patternsWithSignLow, alphaMid, scoresMid, patternsWithSignMid, epsilon, printActions)
    
    # Compute the best solution to the right of the middle point
    if additionsHigh >= additionsMid and additionsLow < additionsMid: # The optimal solution will not be in the right half
        print("Optimal alpha is less than", alphaMid)
        alphaRight = alphaMid
        scoresRight = scoresMid
        patternsWithSignRight = patternsWithSignMid
    else:
        alphaRight, scoresRight, patternsWithSignRight = optimizeAlphaForGreedyWithPotentialInner(G, alphaMid, scoresMid, patternsWithSignMid, alphaHigh, scoresHigh, patternsWithSignHigh, epsilon, printActions)
    
    if len(scoresRight) < len(scoresLeft): # The right half solution uses fewer additions
        return alphaRight, scoresRight, patternsWithSignRight
    return alphaLeft, scoresLeft, patternsWithSignLeft # The left half solution uses at least as few solutions as the right half one

# A function that finds the best alpha for greedy with potential
def optimizeAlphaForGreedyWithPotential(G, alphaLow, alphaHigh, epsilon, printActions = False):
    Gcopy = copy.deepcopy(G)
    print("Tests alpha equal to", alphaLow)
    scoresLow, patternsWithSignLow = greedySolutionWithPotential(Gcopy, alphaLow, printActions)
    Gcopy = copy.deepcopy(G)
    print("Tests alpha equal to", alphaHigh)
    scoresHigh, patternsWithSignHigh = greedySolutionWithPotential(Gcopy, alphaHigh, printActions)
            
    alphaOptimal, scoresOptimal, patternsWithSignOptimal =  optimizeAlphaForGreedyWithPotentialInner(G, alphaLow, scoresLow, patternsWithSignLow, alphaHigh, scoresHigh, patternsWithSignHigh, epsilon, printActions)
    
    # if not (sum(scoresOptimal) - len(scoresOptimal) <) # We are unlikely to have found the optimal solution!
    
    return alphaOptimal, scoresOptimal, patternsWithSignOptimal

# Optimizes based on how many additions an action saves + how much remaining potential is left - with different alpha for different steps 
# Under development
def greedySolutionWithArithmeticPotential(G, alpha0, alphaDiff):
    scores = []
    patternsWithSign = []
    alpha = alpha0
    
    while 1:
        savingScores, savingActions = getAllSavingActions(G)
        if len(savingScores) == 0:
            trivialScores, trivialPatternsWithSign = greedySolution(G)
            return scores + trivialScores, patternsWithSign + trivialPatternsWithSign
            
        bestPattern, score, isSum = greedySearchWithPotential(G, alpha)
        alpha = max(alpha + alphaDiff, 0.001)
        print(bestPattern, isSum, score)
        #print(bestPattern)
        score = performAction(G, bestPattern, isSum)
        patternsWithSign.append((bestPattern, isSum))
        scores.append(score)
        #printGrid(G)
        if additionsDone(G):
            return scores, patternsWithSign

# For finding which indices to search on depth 0 when using multiple cores for the brute-force approach
def findIndices(numActions, numCores, core):
    r = numActions % numCores
    share = numActions//numCores
    startIndex = 0
    k = 0
    while k < core:
        startIndex += share
        if k < r:
            startIndex += 1
        k += 1
    endIndex = startIndex + share
    if core < r:
        endIndex += 1
    print(startIndex)
    print(endIndex)
    return list(range(startIndex,endIndex))

#def bruteforceSolutionInner(G, scores, actions):
def bruteforceSolutionInner(G, savingScores, savingActions, minimumPotentialSavableAdditions, savedAdditions, depth, verbosity = 2):
    # Use a preliminary limit of the number of additions that we can save as input
    if not savingScores:
        return [], [] # The base case where we have no saving actions
    bestScore = 0
    bestActions = []
    bestScores = []
    
    # If it's not possible to beat the limit, then we just discard this branch
    if savedAdditions + sum(savingScores) - len(savingScores) <= minimumPotentialSavableAdditions:
        return [], []
    
    indices = list(range(len(savingScores)))
    
    numCores = 4 # The number of available cores
    core = 3 # Which core to use - starts counting at 0
    splitWork = False
    #splitWork = True
    if depth == 0 and splitWork:
        randomizeActionList = False
        randomizeActionList = True
        if randomizeActionList:
            savingScores, savingActions = permuteActionList(savingScores, savingActions)
        indices = findIndices(len(savingScores), numCores, core)
        print(indices)
    
    for k in indices:
        if depth <= verbosity:
            print("Depth: " + str(depth) + ", index: " + str(k))
        #Gcopy = copy.deepcopy(G)
        score = savingScores[k]
        action = savingActions[k]
        changes = trackChanges(G, action) # tracks the changes that an action leads to
        #performAction(Gcopy, action[0], action[1])
        #scores, actions = bruteforceSolutionInner(Gcopy, depth + 1, verbosity)
        #performAction(G, action[0], action[1])
        newScores, newActions = performActionAndUpdateSavingActions(G, savingScores, savingActions, action)
        # When calling the inner function, also add potential limit and number of saved additions
        scores, actions = bruteforceSolutionInner(G, newScores, newActions, minimumPotentialSavableAdditions, savedAdditions + score - 1, depth + 1, verbosity)
        currentScore = score - 1 + sum(scores) - len(scores)
        revertChanges(G, action[0], changes) # reverts the changes to G that the action lead to
        if currentScore > bestScore: 
            bestScore = currentScore
            bestScores = [score] + scores
            bestActions = [action] + actions
    return bestScores, bestActions
        

def bruteforceSolution(G, verbosity = 2):
    
    savingScores, savingActions = getAllSavingActions(G)
    depth = 0
    savedAdditions = 0
    alpha = 0.2 # Parameter for the greedy approach
    
    # Use a brute greedy solution as a preliminary limit for what we can at least achieve
    Gcopy = copy.deepcopy(G)
    greedyScores, greedyPatternsWithSign = greedySolutionWithPotential(Gcopy, alpha)
    minimumPotentialSavableAdditions = sum(greedyScores) - len(greedyScores) # We can at least do this well
    print("Greedy achieves:", minimumPotentialSavableAdditions)
    
    print("\nThe full grid has", len(savingActions), "saving actions.\n")
    
    scores, actions = bruteforceSolutionInner(G, savingScores, savingActions, minimumPotentialSavableAdditions, savedAdditions, depth, verbosity)
    
    if minimumPotentialSavableAdditions > sum(scores) - len(scores):
        scores = greedyScores
        actions = greedyPatternsWithSign
    
    for action in actions:
        performAction(G, action[0], action[1])
    if additionsDone(G):
        return scores, actions
    trivialScores, trivialActions = greedySolution(G)
    
    return scores + trivialScores, actions + trivialActions

# Moosbauer mod 0 schemes
#fileName = "algorithms/moosbauer/Moosbauer-223-11-mod0.m" # BFS (0, 0, 0)
#fileName = "algorithms/moosbauer/Moosbauer-224-14-mod0.m"# BFS ( 0.0 , 0.335 , 0.005 )
#fileName = "algorithms/moosbauer/Moosbauer-225-18-mod0.m" # BFS ( 0.0 , 0.255 , 0.0 )
#fileName = "algorithms/moosbauer/Moosbauer-233-15-mod0.m" # BFS (0, 0, 0), also works when transforming the system to binary
#fileName = "algorithms/moosbauer/Moosbauer-234-20-mod0.m" # BFS ( 0.0 , 0.145 , 0.005 )
#fileName = "algorithms/moosbauer/Moosbauer-235-25-mod0.m" # BFS ( 0.25 , 0.0 , 0.0 )
#fileName = "algorithms/moosbauer/Moosbauer-244-26-mod0.m" # BFS ( 0.0 , 0.0 , 0.005 )
##fileName = "algorithms/moosbauer/Moosbauer-245-33-mod0.exp" # Currently can't process this one properly!
#fileName = "algorithms/moosbauer/Moosbauer-255-40-mod0.m" # BFS ( 0.135 , 0.005 , 0.055 )
#fileName = "algorithms/moosbauer/Moosbauer-266-56-mod0.m" # some optimal alpha values are (0.01, 0.07890625000000001, 0.04828125) # BFS ( 0.0 , 0.07 , 0.05 )
#fileName = "algorithms/moosbauer/Moosbauer-334-29-mod0.m" # some optimal alpha values are (0.01, 0.5, 0.01)
#fileName = "algorithms/moosbauer/Moosbauer-335-36-mod0.m" # some optimal alpha values are (0.01, 0.01, 0.07890625000000001) # BFS ( 0.005 , 0.005 , 0.09 )
#fileName = "algorithms/moosbauer/Moosbauer-336-42-mod0.m" # some optimal alpha values are (0.2396875, 0.09421875000000002, 0.0559375) # BFS ( 0.23750000000000002 , 0.105 , 0.0575 )
#fileName = "algorithms/moosbauer/Moosbauer-344-38-mod0.m" # some optimal alpha values are (0.17843750000000003, 0.18609375, 0.21671875) # BFS ( 0.18 , 0.185 , 0.2 )
#fileName = "algorithms/moosbauer/Moosbauer-345-47-mod0.m" # some optimal alpha values are (0.09421875000000002, 0.01, 0.07125000000000001) # BFS ( 0.0 , 0.005 , 0.07 )
#fileName = "algorithms/moosbauer/Moosbauer-346-56-mod0.m" # some optimal alpha values are (0.224375, 0.01, 0.07125000000000001) # BFS ( 0.225 , 0.005 , 0.07 )
#fileName = "algorithms/moosbauer/Moosbauer-355-58-mod0.m" # some optimal alpha values are (0.01, 0.1478125, 0.07890625000000001) # BFS ( 0.005 , 0.145 , 0.075 )
#fileName = "algorithms/moosbauer/Moosbauer-356-71-mod0.m" # some optimal alpha values are (0.12484375, 0.01, 0.10187500000000001) # BFS ( 0.12 , 0.005 , 0.105 )
#fileName = "algorithms/moosbauer/Moosbauer-444-49-mod0.m" # some optimal alpha values are (0.01, 0.08656250000000001, 0.01) # BFS ( 0.0005 , 0.084 , 0.0005 )
#fileName = "algorithms/moosbauer/Moosbauer-445-62-mod0.m" # some optimal alpha values are (0.17078125000000002, 0.1171875, 0.0559375) # BFS (0.17171717171717174, 0.11616161616161617, 0.05555555555555556)
#fileName = "algorithms/moosbauer/Moosbauer-446-74-mod0.m" # some optimal alpha values are (0.01, 0.10953125, 0.04828125) # BFS  0.005, 0.12, 0.05
#fileName = "algorithms/moosbauer/Moosbauer-455-76-mod0.m" # some optimal alpha values are (0.1478125, 0.10953125, 0.01) # BFS ( 0.145 , 0.11 , 0.005 )
#fileName = "algorithms/moosbauer/Moosbauer-456-93-mod0.m" # some optimal alpha values are (0.18609375, 0.14015625, 0.032968750000000005) BFS ( 0.1825 , 0.1275 , 0.032 )
#fileName = "algorithms/moosbauer/Moosbauer-555-97-mod0.m" # has coefficients up to 6 # some optimal alpha values are (0.06359375 0.040625, 0.07125000000000001) # BFS ( 0.14 , 0.115 , 0.07 )
#fileName = "algorithms/moosbauer/Moosbauer-555-97-mod0a.m" # has coefficients up to 15 # some optimal alpha values are (0.10953125, 0.1325, 0.01) # BFS ( 0.065 , 0.085 , 0.025 )
#fileName = "algorithms/moosbauer/Moosbauer-556-116-mod0.m" # some optimal alpha values are (0.06359375, 0.10953125, 0.040625) # BFS ( 0.14 , 0.08 , 0.045 ) is worse by 2 than previous result

## Moosbauer mod 2 schemes
#fileName = "algorithms/moosbauer/Moosbauer-226-21-mod2.exp" # BFS ( 0.0 , 0.0 , 0.105 )
#fileName = "algorithms/moosbauer/Moosbauer-236-30-mod2.exp" # BFS ( 0.365 , 0.095 , 0.08 )
#fileName = "algorithms/moosbauer/Moosbauer-246-39-mod2.exp" # BFS ( 0.0 , 0.005 , 0.065 )
#fileName = "algorithms/moosbauer/Moosbauer-256-48-mod2.exp" # BFS ( 0.21 , 0.105 , 0.08 )
#fileName = "algorithms/moosbauer/Moosbauer-266-56-mod2.exp" # some optimal alpha values are (0.01, 0.01, 0.0559375) # BFS ( 0.0 , 0.0 , 0.05 )
#fileName = "algorithms/moosbauer/Moosbauer-336-42-mod2.exp" # some optimal alpha values are (0.255, 0.09421875000000002, 0.06359375) # BFS ( 0.255 , 0.09 , 0.06 )
#fileName = "algorithms/moosbauer/Moosbauer-346-56-mod2.exp" # some optimal alpha values are (0.18609375, 0.10187500000000001, 0.06359375) # BFS ( 0.18 , 0.105 , 0.06 )
#fileName = "algorithms/moosbauer/Moosbauer-356-71-mod2.exp" # some optimal alpha values are (0.2396875, 0.10187500000000001, 0.0559375) # BFS ( 0.24 , 0.105 , 0.055 )
#fileName = "algorithms/moosbauer/Moosbauer-366-85-mod2.exp" # some optimal alpha values are (0.224375, 0.0559375, 0.0253125) # BFS values ( 0.22 , 0.06 , 0.025 )
#fileName = "algorithms/moosbauer/Moosbauer-444-47-mod2.exp" # some optimal alpha values are (0.27796875, 0.1325, 0.01) # BFS ( 0.278 , 0.13 , 0.0005 )
#fileName = "algorithms/moosbauer/Moosbauer-445-60-mod2.exp" # some optimal alpha values are (0.17078125000000002, 0.1325, 0.040625) # BFS ( 0.167 , 0.132 , 0.046 )
#fileName = "algorithms/moosbauer/Moosbauer-446-74-mod2.exp" # some optimal alpha values are (0.07890625000000001, 0.10187500000000001, 0.0559375) # BFS (0.08, 0.065, 0.055)    
#fileName = "algorithms/moosbauer/Moosbauer-456-93-mod2.exp" # some optimal alpha values are (0.21671875, 0.12484375, 0.040625) # BFS ( 0.23 , 0.125 , 0.04 )
#fileName = "algorithms/moosbauer/Moosbauer-466-116-mod2.exp" # some optimal alpha values are (0.01, 0.09421875000000002, 0.032968750000000005) # BFS ( 0.0002 , 0.094 , 0.0312 )
#fileName = "algorithms/moosbauer/Moosbauer-555-95-mod2.exp" # some optimal alpha values are (0.17078125000000002, 0.20906249999999998, 0.01) # BFS ( 0.17 , 0.21 , 0.005 )
#fileName = "algorithms/moosbauer/Moosbauer-556-116-mod2.exp" # some optimal alpha values are (0.1325, 0.01, 0.032968750000000005) # BFS  ( 0.13 , 0.005 , 0.06 )
#fileName = "algorithms/moosbauer/Moosbauer-566-144-mod2.exp" # some optimal alpha values are (0.04828125, 0.040625, 0.032968750000000005) # BFS ( 0.05 , 0.095 , 0.035 )
#fileName = "algorithms/moosbauer/Moosbauer-666-164-mod2.exp" # some optimal alpha values are (0.1478125, 0.10187500000000001, 0.032968750000000005) # BFS ( 0.1296, 0.103, 1/30)

# Adaptive flip graph schemes
#fileName = "algorithms/other/Adaptive-455-73-mod2.m" # some optimal alpha values are (0.01, 0.07890625000000001, 0.032968750000000005) # BFS (0.005, 0.075, 0.03)
#fileName = "algorithms/other/Adaptive-555-94-mod2.m" # some optimal alpha values are (0.10953125, 0.1325, 0.0253125) # BFS with ( 0.11 , 0.132 , 0.025500000000000002 )

# Various schemes found in the Ballard-Grey paper

#fileName = "algorithms/other/Strassen-222-7-18.txt"
#fileName = "algorithms/other/Strassen-222-7-24.txt"
#fileName = "algorithms/grey/Grey-322-11-50.txt"
#fileName = "algorithms/grey/Grey-422-14-84.txt"
#fileName = "algorithms/grey/Grey-323-15-103.txt"
#fileName = "algorithms/grey/Grey-522-18-99.txt"
#fileName = "algorithms/grey/Grey-423-20-144.txt" # We can get 58 -> 57 additions by going to binary - 5400 s with brute-force
#fileName = "algorithms/grey/Grey-234-20-144.txt" # We can get 62 -> 61 additions by going to binary
#fileName = "algorithms/grey/Grey-433-29-234.txt" # some optimal alpha values are (0.01, 0.01, 0.224375)
#fileName = "algorithms/grey/Grey-343-29-234.txt" # some optimal alpha values are (0.01, 0.3009375, 0.01)

# 333 schemes
#fileName = "algorithms/grey/Grey-333-23-221.txt"
#fileName = "algorithms/grey/Grey-333-23-152.txt"
#fileName = "algorithms/grey/Grey-333-23-144.txt"
#fileName = "algorithms/grey/Grey-333-23-143.txt"
#fileName = "algorithms/grey/Grey-333-23-142.txt"
#fileName = "algorithms/other/Smirnov-333-23-139.txt"
fileName = "algorithms/other/Laderman-333-23-98.txt"

A2, B2, C2 = readGridsGeneral(fileName)



####### Old special case code for reading Laderman's algorithm
#A2, B2, C2 = getGridsLaderman()
#flag, errors = verifyScheme(A2, B2, C2, 3, 3, 3, False)
#if not flag:
#    print("Laderman scheme error!")
#else:
#    print("Laderman scheme correctly formed!")



forceBinary = True
forceBinary = False

if tripleContainsNonStandardEntries(A2, B2, C2):
    print("Contains entries not in {-1, 0, 1}!")
    print("Transforms the system to binary.\n")
    tripleToBinary(A2,B2,C2)
else:
    print("All entries are in {-1, 0, 1}.\n")
    if forceBinary:
        tripleToBinary(A2,B2,C2) # Turn the system into binary even if it is not required
        print("Still turns the system into binary")

A = copy.deepcopy(A2)
B = copy.deepcopy(B2)
C = copy.deepcopy(C2)

#algorithmChoice = 0 # Vanilla greedy
#algorithmChoice = 1 # Slightly better greedy - not covered in the paper
algorithmChoice = 2 # Greedy with potential
#algorithmChoice = 3 # Brute force
#algorithmChoice = 4 # Greedy with varying alpha values (under development)
#algorithmChoice = 5 # Greedy with structured search for optimal alpha values
#algorithmChoice = 6 # Greedy with brute-forcing for optimal alpha values

verbosity = 0 # For the brute-force algorithm print-outs

print("File name:", fileName)

print("Naive number of additions:")
print(naiveAdditions(A2) + naiveAdditions(B2) + naiveAdditions(C2))
print((naiveAdditions(A2), naiveAdditions(B2), naiveAdditions(C2)),"\n")

# The Greedy vanilla algorithm - corresponds to setting alla alpha values equal to 0
if algorithmChoice == 0:
    scoresA2, patternsWithSignA2 = greedySolution(A2)
    scoresB2, patternsWithSignB2 = greedySolution(B2)
    scoresC2, patternsWithSignC2 = greedySolution(C2)

# A marginally better greedy appraoch - not covered in the paper
elif algorithmChoice == 1:
    scoresA2, patternsWithSignA2 = greedySolutionImproved(A2)
    scoresB2, patternsWithSignB2 = greedySolutionImproved(B2)
    scoresC2, patternsWithSignC2 = greedySolutionImproved(C2)
    
# The Greedy with potential algorithm
elif algorithmChoice == 2:
    printActions = True
    #alphaA = 0.1296 # Optimal for the Moosbauer-666 case
    #alphaB = 0.103 # Optimal for the Moosbauer-666 case
    #alphaC = 1/30 # Optimal for the Moosbauer-666 case
    #alphaA = 0.11 # Optimal for the Adaptive-555 case
    #alphaB = 0.132 # Optimal for the Adaptive-555 case
    #alphaC = 0.0255 # Optimal for the Adaptive-555 case
    alphaA = 0.2 # Some default value
    alphaB = 0.2 # Some default value
    alphaC = 0.2 # Some default value
    #printActions = False
    start = time.time()
    print("Actions and scores for A\n")
    scoresA2, patternsWithSignA2 = greedySolutionWithPotential(A2, alphaA, printActions)
    print("After solving for A", time.time() - start, "seconds have passed.")
    print("\nActions and scores for B\n")
    scoresB2, patternsWithSignB2 = greedySolutionWithPotential(B2, alphaB, printActions)
    print("After solving for B", time.time() - start, "seconds have passed.")
    print("\nActions and scores for C\n")
    scoresC2, patternsWithSignC2 = greedySolutionWithPotential(C2, alphaC, printActions)
    print("After solving for C", time.time() - start, "seconds have passed.")

# Brute-forces for an optimal algorithm - slow for larger dimensions
elif algorithmChoice == 3:
    start = time.time()
    print("Actions and scores for A\n")
    scoresA2, patternsWithSignA2 = bruteforceSolution(A2, verbosity)
    print("After solving for A", time.time() - start, "seconds have passed.")
    print("\nActions and scores for B\n")
    scoresB2, patternsWithSignB2 = bruteforceSolution(B2, verbosity)
    print("After solving for B", time.time() - start, "seconds have passed.")
    print("\nActions and scores for C\n")
    scoresC2, patternsWithSignC2 = bruteforceSolution(C2, verbosity)
    print("After solving for C", time.time() - start, "seconds have passed.")

# An attempt at using different alpha values for different steps - under development
elif algorithmChoice == 4:
    alpha0 = 0.25
    alphaDiff = -0.02
    scoresA2, patternsWithSignA2 = greedySolutionWithArithmeticPotential(A2, alpha0, alphaDiff)
    scoresB2, patternsWithSignB2 = greedySolutionWithArithmeticPotential(B2, alpha0, alphaDiff)
    scoresC2, patternsWithSignC2 = greedySolutionWithArithmeticPotential(C2, alpha0, alphaDiff)

# Perform greedy with potential where we optimize for alpha
elif algorithmChoice == 5:
    alphaLow = 0.01
    alphaHigh = 0.5
    epsilon = 0.001
    start = time.time()
    
    alphaA2, scoresA2, patternsWithSignA2 = optimizeAlphaForGreedyWithPotential(A2, alphaLow, alphaHigh, epsilon)
    print("\nOptimal alpha for A is:", alphaA2)
    print("After solving for A", time.time() - start, "seconds have passed.")
    print("Number of additions needed for A is", len(scoresA2), "\n")
    alphaB2, scoresB2, patternsWithSignB2 = optimizeAlphaForGreedyWithPotential(B2, alphaLow, alphaHigh, epsilon)
    print("\nOptimal alpha for B is:", alphaB2)
    print("After solving for B", time.time() - start, "seconds have passed.")
    print("Number of additions needed for B is", len(scoresB2), "\n")
    alphaC2, scoresC2, patternsWithSignC2 = optimizeAlphaForGreedyWithPotential(C2, alphaLow, alphaHigh, epsilon)
    print("\nOptimal alpha for C is:", alphaC2)
    print("After solving for C", time.time() - start, "seconds have passed.")
    print("Number of additions needed for A is", len(scoresA2), "\n")

# Tests a range of alpha values evenly spread out
elif algorithmChoice == 6:
    
    start = time.time()
    
    #dataPoints = 201
    dataPoints = 101 
    #dataPoints = 101
    #dataPoints = 11
    
    isDefault = True
    isDefault = False
    
    alphaLow = 0
    #alphaLow = 0.01
    alphaHigh = 0.5    
    
    if isDefault:
        alphaLowA = alphaLow
        alphaLowB = alphaLow
        alphaLowC = alphaLow
    
        alphaHighA = alphaHigh
        alphaHighB = alphaHigh
        alphaHighC = alphaHigh
    else: # Allow for different search intervals for different alpha values
        #numCores = 4 # Number of available cores
        numCores = 1 # Number of available cores
        core = 0 # The core number to use - starts counting from 0
        
        alphaLowA = 0.11
        alphaHighA = 0.15
        alphaDeltaA = (alphaHighA - alphaLowA)/(dataPoints - 1)/numCores
        alphaLowA += alphaDeltaA*core
        alphaHighA += alphaDeltaA*core
        
        alphaLowB = 0
        alphaHighB = 0.05
        alphaDeltaB = (alphaHighB - alphaLowB)/(dataPoints - 1)/numCores
        alphaLowB += alphaDeltaB*core
        alphaHighB += alphaDeltaA*core
        
        alphaLowC = 0
        alphaHighC = 0.1
        alphaDeltaC = (alphaHighC - alphaLowC)/(dataPoints - 1)/numCores
        alphaLowC += alphaDeltaC*core
        alphaHighC += alphaDeltaA*core
    
    numberAdditionsA = [0]*dataPoints
    numberAdditionsB = [0]*dataPoints
    numberAdditionsC = [0]*dataPoints
    
    leastAdditionsA = 10000
    leastAdditionsB = 10000
    leastAdditionsC = 10000
    
    alphaRangeA = np.linspace(alphaLowA, alphaHighA, dataPoints)
    alphaRangeB = np.linspace(alphaLowB, alphaHighB, dataPoints)
    alphaRangeC = np.linspace(alphaLowC, alphaHighC, dataPoints)
    
    alphaA = 0
    alphaB = 0
    alphaC = 0
    
    for k in range(len(alphaRangeA)):
        alphaACurrent = alphaRangeA[k]
        alphaBCurrent = alphaRangeB[k]
        alphaCCurrent = alphaRangeC[k]
        print("Tests alphaA =", alphaACurrent)
        currentScoresA, currentPatternsWithSignA = greedySolutionWithPotential(A2, alphaACurrent, printActions = False)
        print("Tests alphaB =", alphaBCurrent)
        currentScoresB, currentPatternsWithSignB = greedySolutionWithPotential(B2, alphaBCurrent, printActions = False)
        print("Tests alphaC =", alphaCCurrent)
        currentScoresC, currentPatternsWithSignC = greedySolutionWithPotential(C2, alphaCCurrent, printActions = False)
        
        numberAdditionsA[k] = len(currentScoresA)
        numberAdditionsB[k] = len(currentScoresB)
        numberAdditionsC[k] = len(currentScoresC)
        
        if len(currentScoresA) < leastAdditionsA:
            leastAdditionsA = len(currentScoresA)
            scoresA2 = currentScoresA
            patternsWithSignA2 = currentPatternsWithSignA
            print("alpha =", alphaACurrent, "is best so far for A with", len(scoresA2), "additions.")
            alphaA = alphaACurrent
            
        if len(currentScoresB) < leastAdditionsB:
            leastAdditionsB = len(currentScoresB)
            scoresB2 = currentScoresB
            patternsWithSignB2 = currentPatternsWithSignB
            print("alpha =", alphaBCurrent, "is best so far for B with", len(scoresB2), "additions.")
            alphaB = alphaBCurrent
            
        if len(currentScoresC) < leastAdditionsC:
            leastAdditionsC = len(currentScoresC)
            scoresC2 = currentScoresC
            patternsWithSignC2 = currentPatternsWithSignC
            print("alpha =", alphaCCurrent, "is best so far for C with", len(scoresC2), "additions.")
            alphaC = alphaCCurrent
        
        A2 = copy.deepcopy(A)
        B2 = copy.deepcopy(B)
        C2 = copy.deepcopy(C)
        
    print("\nBest values of alpha are (", alphaA, ",", alphaB, ",", alphaC, ")\n")
    print("After solving for all grids", time.time() - start, "seconds have passed.")

print("\nOptimized number of additions:")
print(len(scoresA2) + len(scoresB2) + len(scoresC2))
print((len(scoresA2), len(scoresB2), len(scoresC2)))

if verifyAdditionsTriple(copy.deepcopy(A), patternsWithSignA2, copy.deepcopy(B), patternsWithSignB2, copy.deepcopy(C), patternsWithSignC2):
    print("\nOptimization of additions successful!")
else:
    print("\nOptimization of additions failed!")

# Depth of the addition circuit if performed exactly as specified with no attempts at optimization
print("\nNaive circuit depths:", circuitDepth([action[0] for action in patternsWithSignA2], len(A[0])),
circuitDepth([action[0] for action in patternsWithSignB2], len(B[0])),
circuitDepth([action[0] for action in patternsWithSignC2], len(C[0])))

# Lower limit of the depth of the addition circuit if performing optimization w.r.t. circuit depth
print("Minimum circuit depths:",minCircuitDepth([action[0] for action in patternsWithSignA2], len(A[0])),
minCircuitDepth([action[0] for action in patternsWithSignB2], len(B[0])),
minCircuitDepth([action[0] for action in patternsWithSignC2], len(C[0])))
