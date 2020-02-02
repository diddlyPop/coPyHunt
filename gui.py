############################################################################################################
#
#   coPy - Scavenger Hunt game vs pre-trained image classifier
#
#   by Kyle Guss
#
#   known problems:
#       FIXED - 'display_item' size constrained when initialized, item sometimes blank due to size constraints
#       size of entire window changes with dynamic image size
#           need to either scale image down or find another way to keep window size somewhat static
#
#
#############################################################################################################

import PySimpleGUI as sg
from PIL import ImageGrab
import os
import random
import yolo


class Game:

    def __init__(self):
        labels_path = os.path.sep.join(["yolo-coco", "coco.names"])          # Pre-trained class labels
        self.labels = open(labels_path).read().strip().split("\n")

        self.current_item = random.choice(self.labels)
        self.current_round = 0
        self.points = 0

        self.display_header = sg.Text('Welcome to coPy. Find the item',
                                      key='_HEAD_',
                                      background_color='Black')
        self.display_image = sg.Image(filename="hunt.png")
        self.display_round = sg.Text(f"Level: {self.current_round}",
                                     key='_ROUND_',
                                     size=(15, 1),
                                     background_color='Black')
        self.display_item = sg.Text(f"Item: {self.current_item}",
                                    key='_ITEM_',
                                    size=(15, 1),
                                    background_color='Black')

        self.display_points = sg.Text(f"Points: {self.points}",
                                      key='_POINTS_',
                                      size=(15, 1),
                                      background_color='Black')

        self.yolo_classifier = yolo.ImageClassifier()
        self.curr_round_classes = []

    def new_round(self):
        self.current_round += 1
        self.current_item = random.choice(self.labels)
        self.display_image.update(filename="image.png")  # Ideally  only be updated when unique photo is found
        self.display_round.update(f"Level: {self.current_round}")
        self.display_item.update(f"Item: {self.current_item}")
        self.display_points.update(f"Points: {self.points}")

    def scan_image(self):  # Takes image from clipboard, compares to temp. If UNIQUE then save photo.
        print(self.current_item)
        try:
            img = ImageGrab.grabclipboard()  # Take image from clipboard
            img.save("image.png", format='PNG')
            self.yolo_classifier.classify()
            self.curr_round_classes = self.yolo_classifier.listClassesFromClassify()
        except AttributeError as e:  # Catches if clipboard contents not Image
            print(e)

    def determine_points(self):
        if self.current_item in self.curr_round_classes:
            self.points += 1
        else:
            print(f"{self.current_item} not found")
            print(self.curr_round_classes)

    def start(self):
        sg.theme_background_color('Black')

        # All the stuff inside your window.
        layout = [[self.display_header],
                  [self.display_round, self.display_points],
                  [self.display_item],
                  [self.display_image],
                  [sg.Button('Scan', key='_SCAN_')],
                  [sg.Button('Cancel')]]

        # Create the Window
        window = sg.Window('coPy', layout)
        # Event Loop to process "events" and get the "values" of the inputs
        while True:
            event, values = window.read()
            if event in (None, 'Cancel'):  # If user closes window or clicks cancel
                break
            elif event == '_SCAN_':
                self.scan_image()
                self.determine_points()
                self.new_round()
                window['_ITEM_'].update(value=f"Item: {self.current_item}")

        window.close()


if __name__ == '__main__':
    coPy = Game()
    coPy.start()
