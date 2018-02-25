# Example PBS cluster job submission in Python

from subprocess import Popen, PIPE
import time
import sys

# If you want to be emailed by the system, include these in job_string:
#PBS -M your_email@address
#PBS -m abe  # (a = abort, b = begin, e = end)
#num_threads = [1, 4, 8, 16, 24]
# Loop over your jobs
#for i in num_threads:

# Open a pipe to the qsub or interactive command.
proc = Popen('qsub', shell=True, stdin=PIPE, stdout=PIPE, stderr=PIPE, close_fds=True)

# Customize your options here
mol_name = sys.argv[1]
density_type = sys.argv[2]
filt_singles = sys.argv[3]
if filt_singles == "true":
    output_file =  sys.argv[1] + '_' + sys.argv[2] + '_singles_frozen' + '.dat'
else:
    output_file =  sys.argv[1] + '_' + sys.argv[2] + '.dat'
processors = "nodes=1:ppn=24"
queue = sys.argv[4]
if queue == "dev_q":
    walltime = "2:00:00"
else:
    walltime = """%s:00:00"""%sys.argv[5]
#command = "/home/akumar1/newriver/installed/psi4/bin/psi4  -n {} -o output_{}".format(i,i) 

job_string = """#!/bin/bash
#PBS -N %s
#PBS -l walltime=%s
#PBS -l %s
#PBS -A crawdad
#PBS -q %s 
#PBS -W group_list=dragonstooth
source /groups/crawdad_lab/opt/etc/crawdad_psi4vars.sh -python 3
echo "psi4 at"
which psi4
echo ""
cd $PBS_O_WORKDIR
python3 input_lg.py %s %s %s &>> %s 
""" % (mol_name, walltime, processors, queue, mol_name, density_type, filt_singles, output_file)
# Send job_string to qsub
if (sys.version_info > (3, 0)):
    print(job_string)
    proc.stdin.write(job_string.encode('utf-8'))
else:
    proc.stdin.write(job_string)

out, err = proc.communicate()

# Print your job and the system response to the screen as it's submitted
print(job_string)
print(out)

