import numpy as np
import os
import cv2
import sys
import random
import re

def isTraining(trainPatients, nameFile):
	for patient in trainPatients:
		if "Subject" + str(patient) + "_" in nameFile and 'MRI' in nameFile:
			return True
	return False

def findGt(numPatient, numSlices, lstFiles):
	for l in lstFiles:
		if "Subject" + str(numPatient) + "_" in l and 'Gtruth' in l and numSlices in l:
			return l
	return None

filesTrain_save = sys.argv[2]
filesTest_save = sys.argv[3]

trainPatients = []
file_numpatientsTraining = open(sys.argv[1], "r")
lines = file_numpatientsTraining.readlines();
trainPatients = [int(x) for x in lines]


file_nameTrain = open(filesTrain_save, "w")
file_nameTest = open(filesTest_save, "w")


# read all nii files in the directory
lstPngFiles = []  # create an empty list
for dirName, subdirList, fileList in os.walk('./'):
    for filename in fileList:
        if ".png" in filename.lower():  # check whether the file's png
            lstPngFiles.append(os.path.join(dirName,filename))



for nameFile in lstPngFiles:
	slice_num = re.search(r'slice\d+', nameFile).group()
	patientNum = re.search(r'\d+', nameFile).group()
	if isTraining(trainPatients, nameFile):
		gt_name = findGt(patientNum, slice_num, lstPngFiles) 
		file_nameTrain.write(nameFile + '\t' + gt_name + '\n')


	elif 'MRI' in nameFile:
		gt_name = findGt(patientNum, slice_num, lstPngFiles) 
		file_nameTest.write(nameFile + '\t' + gt_name + '\n')


file_nameTest.close()
file_nameTrain.close()
