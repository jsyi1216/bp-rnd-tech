@echo off
call activate tensorflow_cpu 
start python trend.py 


REM absolute path version of Run_program.bat
REM --------
REM @echo off
REM call "D:\Anaconda\Scripts\activate.bat"
REM start D:\Anaconda\envs\tensorflow_cpu\python D:\workspace_VS\bp-rnd-tech\trend.py 


REM PATH need to be added for Run_Program.bat
REM --------
REM D:\Anaconda\envs\tensorflow_cpu     #for "python" cmd
REM D:\Anaconda\Scripts                 #activation batchfile "activate"
REM D:\Anaconda\envs                    #for activate conda virtual env,"tensorflow_cpu" for this case 
REM D:\workspace_VS\bp-rnd-tech\        #absolute path contains "trend.py"
