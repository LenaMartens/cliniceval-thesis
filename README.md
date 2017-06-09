# cliniceval-thesis

Code for master thesis: extraction of temporal information from medical text.

A complete run of the code is done by running `python code/complete_procedure`. Most configuration is stored in this file and can be changed there. There are two kinds of procedures that can be run: a base\_procedure() which is the local classifier model with structured prediction and a transitive\_procedure() which is the transition-based parser with a neural network. 
It makes a Procedure class (BaseProcedure or TransitiveProcedure). These classes are defined in the code/procedure.py file. After a procedure is instantiated it is ready to predict annotations for a directory of files. They are the glue of the system: they call on all the other modules and tie it all together.

The code/configuration.INI file specifies the system paths where data can be read from or written to.

The code/requirements.txt file contains the dependencies of this code. Install with `pip install -r code\requirements.txt`
