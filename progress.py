import sys

BAR_LENGTH = 30

def progress(total, step, cost):
	completeBarLength = int(round(BAR_LENGTH*step / float(total)))
	percent = round(100 * step / float(total), 1)
	bar = completeBarLength * '=' + (30-completeBarLength) * '-'
	sys.stdout.write('[%s] %s%s cost: %s\r' %(bar, percent, '%', cost))
	sys.stdout.flush()

def complete():
	print "[==========COMPLETE!!==========]"
