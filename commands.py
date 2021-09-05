# Paul Amoruso
# Senior design 2 (EEL 4915L)
# Instructure's Lei Wei, and Samuel Richie, and Dr. Kar

# Python program to explain os.system() method
# This code will be used as a startup program so that when the nano starts up
# it well go straigt to detecting the birds and defending agains squirrels.
	
# importing os module
import os
from subprocess import Popen, PIPE

# Command to execute
# Using Ubuntu OS command

#goAndRunDocker = 'cd ~/jetson-inference/ && docker/run.sh --volume ~/jetson-inference/python/training/detection/ssd/:/jetson-inference/python/training/detection/ssd'

# Use this command to run the object detection program on local os, not Docker container.
runPython = 'cd ~/jetson-inference/python/training/detection/ssd/ && python3 detectnet.py --model=models/cvat-training-v2-out/ssd-mobilenet.onnx --labels=models/cvat-training-v2-out/labels.txt --input-blob=input_0 --output-cvg=scores --output-bbox=boxes csi://0'


# Using os.system() method
p = os.system(runPython)


# The following is various attemts used when utilizing docker.
#p = os.popen("sudo -S %s"%(goAndRunDocker + runPython2), 'w').write('passwords')
#p = Popen(['sudo', '-S'] + goAndRunDocker, stdin=PIPE, stderr=PIPE, universal_newlines=True)
#sudo_prompt = p.communicate(sudoPassword + '\n')[1]
#p = os.system('echo %s|sudo -S %s' % (sudoPassword, goAndRunDocker))

#sudo docker exec -it 487aba5bb344 bash
#os.system('sudo docker exec -it 487aba5bb344 commands.py' + '&&' +runPython2)
