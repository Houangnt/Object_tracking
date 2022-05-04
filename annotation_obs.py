import os.path
import time
import shutil
import json
import base64

import cv2
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler

input_dir = "C:/Users/GHTK/Downloads/5/imgs"
checkname = ''


class Watcher:
    DIRECTORY_TO_WATCH = "C:/Users/GHTK/Downloads/5/imgs"

    def __init__(self):
        self.observer = Observer()

    def run(self):
        event_handler = Handler()
        self.observer.schedule(event_handler, self.DIRECTORY_TO_WATCH, recursive=True)
        self.observer.start()
        try:
            while True:
                time.sleep(5)
        except:
            self.observer.stop()
            print("Error")

        self.observer.join()


class Handler(FileSystemEventHandler):

    @staticmethod
    def on_any_event(event):
        global checkname
	
        #print(event.is_directory)
        #print(event.event_type)
	
        #print(event.src_path)
        if event.is_directory:
            return None
	
        elif event.event_type == 'modified':
            time.sleep(0.5)
            if event.src_path == checkname:
                checkname = ''
            else:
                # Take any action here when a file is first created.
                file_name = os.path.basename(event.src_path)
                # data = json.load(open(event.src_path, encoding="utf8"))
                data = json.load(open(event.src_path, encoding="utf-8"))

                next_name = str(int(file_name[:-5]) + 1).rjust(6, '0') + '.json'
                new_path = os.path.join(input_dir, next_name)
                image = cv2.imread(new_path.replace('.json', '.jpg'))
                ret, buff = cv2.imencode('.jpg', image)
                jpg_as_text = base64.b64encode(buff)
                data['imageData'] = jpg_as_text.decode('utf-8')
                with open(new_path, "w") as write_file:
                    json.dump(data, write_file, indent=4)
                checkname = new_path


if __name__ == '__main__':
    w = Watcher()
    w.run()
