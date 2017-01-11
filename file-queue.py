"""
this is called when something changes in the search directory.

* it will run the first file that looks like "run*.py",
* put the output in the results subdirectory,
* and move the file to the done subdirectory
"""
import os
import re
import shutil
import subprocess
import time

from watchdog import observers
from watchdog.events import FileModifiedEvent, FileSystemEventHandler

# folder to watch for files
SEARCH_PATH = 'tasks'
# folder to store output of scripts
# this will be placed under the search path
RESULTS_SUBDIR = 'results'
# folder to store done scripts
# this will be placed under the search path
COMPLETE_SUBDIR = 'done'
abs_search_path = os.path.abspath(SEARCH_PATH)


class RunFile(FileSystemEventHandler):
    def __init__(self):
        super().__init__()

    def on_modified(self, event: FileModifiedEvent):
        files = [f for f in os.listdir(abs_search_path) if re.match('.*run.*\.py$', f)]
        if len(files) == 0:
            return

        # just take the first for now
        file = files[0]
        script_filepath = os.path.join(abs_search_path, file)
        log_filepath = '{}-results-{}.txt'.format(time.strftime('%y%m%d-%H%M%S'), file)
        log_filepath = os.path.join(abs_search_path, RESULTS_SUBDIR, log_filepath)
        print('Running: {}'.format(file))
        print('Saving output to: {}'.format(log_filepath))

        with open(log_filepath, 'w') as log:
            subprocess.call(['python', script_filepath], stderr=subprocess.STDOUT, stdout=log)
        shutil.move(script_filepath, os.path.join(abs_search_path, COMPLETE_SUBDIR, file))

        print('Done')


# make sure the results and done subdirs exist
os.makedirs(os.path.join(abs_search_path, RESULTS_SUBDIR), exist_ok=True)
os.makedirs(os.path.join(abs_search_path, COMPLETE_SUBDIR), exist_ok=True)

obv = observers.Observer()
obv.schedule(RunFile(), SEARCH_PATH, recursive=False)
obv.start()
try:
    while obv.is_alive():
        time.sleep(2)
except KeyboardInterrupt:
    pass
finally:
    obv.stop()
    obv.join()
