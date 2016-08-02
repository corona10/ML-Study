import numpy

def devide(feature, label, testSetWeight=0.2):
	if(testSetWeight >= 1):
		print "devide(feature, label, testSetWeight(<1)"
		return
	totalLength = len(feature)
	testLength = int(round(totalLength * testSetWeight))
	trainLength = totalLength - testLength

	return feature[:trainLength], label[:trainLength], \
			feature[trainLength:], label[trainLength:]
