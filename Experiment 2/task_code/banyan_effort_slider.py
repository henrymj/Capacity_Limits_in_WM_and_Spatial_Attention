from psychopy import visual, event, core, gui, tools, monitors, parallel
import pandas as pd
import numpy as np
import eyelinker
import sys
import os
import csv
import errno
import json

monitor_details = {"name": "Monitor01", "distance": 80, "resolution": [1920, 1080], "width": 56.9}

participant_info = {"Subject": "0"}


image_size = .7  # DVA
text_size = 0.2
connecter_width = 0.1
slider_length = 15
slider_height = 0.2
slider_y = -10
y_buffer = 0.1 + image_size + text_size


def color_convert(color):
    return [round(((n / 127.5) - 1), 3) for n in color]


class EffortSlider:

    def __init__(self, monitor_details):
        self.monitor_details = monitor_details

        self.data = []

    def open_window(self, **kwargs):
        self.monitor = monitors.Monitor(
            self.monitor_details["name"], width=self.monitor_details["width"], distance=self.monitor_details["distance"]
        )
        self.monitor.setSizePix(self.monitor_details["resolution"])
        self.win = visual.Window(
            monitor=self.monitor, fullscr=True, units="deg", checkTiming=True, **kwargs
        )  # TODO, checkTiming

        self.winx, self.winy = self.win.size / 2

    def quit_experiment(self):
        """Completes anything that must occur when the experiment ends."""
        if self.win:
            self.win.close()
        print("Slider has been terminated.")
        sys.exit(0)

    def chdir(self):
        """Changes the directory to where the data will be saved."""

        try:
            os.makedirs("./subject_data")
        except OSError as e:
            if e.errno != errno.EEXIST:
                raise

        os.chdir("./subject_data")

    def get_participant_info_from_dialog(self, screen=0):
        """Gets subject info from dialog box.

        Parameters:
        additional_fields_dict -- An optional dictionary containing more
            fields for the dialog box and output dictionary.
        screen -- an int describing the screen you want the dialog to appear on
        """

        # Modifies experiment_info dict directly
        out = gui.DlgFromDict(
            participant_info, title="Cassia", tip={"Unique Subject Identifier": "From the cronus log"}, screen=screen
        )
        self.subject_str = participant_info["Subject"]
        self.rng = np.random.RandomState(int(self.subject_str))

        return out.OK

    def run(self):

        self.chdir()
        ok = self.get_participant_info_from_dialog()
        if not ok:
            print("Slider has been terminated.")
            sys.exit(1)

        condition_images = {
            "COLORS: 0": "../cond_images/SS0.png",
            "COLORS: 1": "../cond_images/SS1.png",
            "COLORS: 2": "../cond_images/SS2.png",
            "COLORS: 3": "../cond_images/SS3.png",
            "COLORS: 4": "../cond_images/SS4.png",
            "COLORS: 6": "../cond_images/SS6.png",
            "COLORS: 8": "../cond_images/SS8.png",
            "DIGITS: 0": "../cond_images/SS0.png",
            "DIGITS: 1": "../cond_images/SS1.png",
            "DIGITS: 2": "../cond_images/SS2.png",
            "DIGITS: 3": "../cond_images/SS3.png",
            "DIGITS: 4": "../cond_images/SS4.png",
            "DIGITS: 6": "../cond_images/SS6.png",
            "DIGITS: 8": "../cond_images/SS8.png",
        }

        self.open_window(screen=0)

        conds = list(condition_images.keys())

        y_dists = np.arange(slider_y + y_buffer, slider_y + (len(conds) + 0.1) * y_buffer, y_buffer)

        # randomly shift the order so either colors or digits appears first
        y_dists = np.roll(y_dists, self.rng.choice([0, 7]))

        # randomly order 2 colors for the 2 tasks
        border_colors = self.rng.choice(["gold", "green"], size=2, replace=False)

        # draw the slider bar
        slider_bar = visual.Rect(
            self.win, width=slider_length, height=slider_height, lineColor="black", fillColor="black", pos=(0, slider_y)
        )
        slider_bar.draw()
        slider_min = -slider_length / 2
        slider_max = slider_length / 2
        # add text on each end
        left_text = visual.TextStim(self.win, text="No Effort", pos=(slider_min - 1.5, slider_y), height=text_size)
        right_text = visual.TextStim(
            self.win, text="Max Effort\nPossible", pos=(slider_max + 1.5, slider_y), height=text_size
        )
        left_text.draw()
        right_text.draw()
        # top text
        q_text = visual.TextStim(
            self.win,
            text="How much mental effort did you extend on each task?\nUse the mouse to drag the images horizontally across the slider.\nPress 'Enter' once complete.",
            pos=(0, 10),
            height=text_size,
        )
        q_text.draw()

        # For each condition, draw the image with text floating above it, and a connecting line down to the slider
        border_dict = {}
        image_dict = {}
        text_dict = {}
        connecter_dict = {}
        for c, cond in enumerate(condition_images):
            border = visual.Rect(
                self.win,
                width=image_size * 1.1,
                height=image_size * 1.1,
                lineColor=border_colors[int(np.floor(c / 7))],
                fillColor=border_colors[int(np.floor(c / 7))],
                pos=(0, y_dists[c]),
            )
            image = visual.ImageStim(
                self.win, image=condition_images[cond], size=(image_size, image_size), pos=(0, y_dists[c])
            )

            text = visual.TextStim(
                self.win, text=cond, pos=(0, y_dists[c] + image_size / 2 + text_size / 2), height=text_size
            )
            connecter = visual.Rect(
                self.win,
                width=connecter_width,
                height=y_dists[c] - slider_y,
                lineColor="black",
                fillColor="black",
                pos=(0, (y_dists[c] + slider_y) / 2),
            )
            border_dict[cond] = border
            image_dict[cond] = image
            text_dict[cond] = text
            connecter_dict[cond] = connecter

        # draw everything in order
        for connecter in connecter_dict.values():
            connecter.draw()
        for border in border_dict.values():
            border.draw()
        for image in image_dict.values():
            image.draw()
        for text in text_dict.values():
            text.draw()

        self.win.flip()

        # enter while loop, waiting for enter key to end
        mouse = event.Mouse(visible=True)
        keys = event.getKeys()
        while "return" not in keys:

            # update things if they've been clicked on

            for cond, image in image_dict.items():
                img_selected = False
                if mouse.isPressedIn(image) or ((mouse.getPressed()[0] == 1) and img_selected):
                    img_selected = True
                    # update the x position of the image to the mouse position
                    new_x = min(max(mouse.getPos()[0], slider_min), slider_max)
                    image.pos = new_x, image.pos[1]
                    connecter_dict[cond].pos = new_x, connecter_dict[cond].pos[1]
                    text_dict[cond].pos = new_x, text_dict[cond].pos[1]
                img_selected = False

            # draw everything
            slider_bar.draw()
            left_text.draw()
            right_text.draw()
            q_text.draw()
            for connecter in connecter_dict.values():
                connecter.draw()
            for border in border_dict.values():
                border.draw()
            for image in image_dict.values():
                image.draw()
            for text in text_dict.values():
                text.draw()
            self.win.flip()

            keys = event.getKeys()

        # save out as json
        out_dict = {}
        for cond, image in image_dict.items():
            out_dict[cond] = image.pos[0]
        out_dict["slider_min"] = slider_min
        out_dict["slider_max"] = slider_max
        json.dump(out_dict, open(f"./{self.subject_str}_effort.json", "w"))


if __name__ == "__main__":
    slider = EffortSlider(monitor_details)
    slider.run()
