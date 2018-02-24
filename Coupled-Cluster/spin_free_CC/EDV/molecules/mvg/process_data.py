# This should print the optical rotation values in degrees from any file.
import sys
import os
input_file = sys.argv[1] + '_' + sys.argv[2] + '.dat'
mkdir = 'mkdir results/%s;'%(sys.argv[1]) 
os.system(mkdir)
output_file = 'results/' + sys.argv[1] + '/' + sys.argv[2] + '.dat'
bashCommand = """ grep trace_3_actual_optrot %s > %s """ % (input_file, output_file)
#print(bashCommand)
#print(bashCommand.split())
os.system(bashCommand)
omega = 0.07735713394560646
prefactor = -6465712.506336764
prefactor *= omega
#fin_prefactors = {'h2o2': , 'h2_4': , 'h2_5': , 'h2_6': , 'h2_7': , 'fluorooxirane': , '(S)-mox': }
Mass = {'h2o2': 34.00547930326, 'h2_4': 8.062600256560001, 'h2_5': 10.078250320700002, 'h2_6': 12.093900384840001, 'h2_7': 14.109550448980002, 'fluorooxirane':'a' , '(S)-mox':'a' }
f = open(output_file, 'r')
for line in f:
    a = line.split('=') 
    scale = prefactor/Mass[sys.argv[1]]    
    print(scale * float(a[1]))


# I am going to fix this Later!! 
#import subprocess
#process = subprocess.Popen(bashCommand.split(), stdout=subprocess.PIPE, shell=True)
#process = subprocess.Popen(bashCommand, stdout=subprocess.PIPE)
#output, error = process.communicate()
#print(output)

