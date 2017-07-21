import os
import subprocess
import sys
            
subprocess.call(['java', '-Xmx2000m', '-cp',
 		       os.path.join(sys.argv[1], 'matsim-0.8.1.jar'),
	    	   'org.matsim.run.Controler', sys.argv[2]])