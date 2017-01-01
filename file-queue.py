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


class RunFile(FileSystemEventHandler):
    def __init__(self):
        super().__init__()

    def on_modified(self, event: FileModifiedEvent):
        files = [f for f in os.listdir(SEARCH_PATH) if re.match('.*run.*\.py$', f)]
        if len(files) == 0:
            return

        file = files[0]
        filepath = os.path.join(SEARCH_PATH, file)
        log = '{} results {}.txt'.format(time.strftime('%y%m%d %H%M%S'), file)
        log = os.path.join(SEARCH_PATH, RESULTS_SUBDIR, log)
        print('Running: {}'.format(file))

        with open(log, 'w') as log:
            subprocess.call(['python', filepath], stderr=subprocess.PIPE, stdout=log)
        shutil.move(filepath, os.path.join(SEARCH_PATH, COMPLETE_SUBDIR, file))

        print('Done')


# make sure the results and done subdirs exist
os.makedirs(os.path.join(SEARCH_PATH, RESULTS_SUBDIR), exist_ok=True)
os.makedirs(os.path.join(SEARCH_PATH, COMPLETE_SUBDIR), exist_ok=True)

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
