"""
A change detection experiment that presents lateralized stimuli with and without placeholders
Written by Darius Suplica - dsuplica@uchicago.edu
2023-08-18

Reviewed & Edited by Henry Jones - henryjones@uchicago.edu
2023-08-31

"""

import numpy as np
import pandas as pd
import os
import sys
import csv
import errno

import psychopy.core
import psychopy.event
import psychopy.visual
import psychopy.sound
import psychopy.tools.monitorunittools
import psychopy.gui
import psychopy.parallel
import eyelinker


EYE_MAX_DIST = 1.5  # in DVA

STIM_TIME = 0.2
DELAY_TIME = 0.8
BASELINE_TIME = 0.3
ITI_TIME = [0.8, 0.9]  # min, max

TRIALS_PER_BLOCK = 80  # must be a multiple of 16
NUM_BLOCKS = 20  # 100 per condition, total 1600

trial_types = {
    "2N": {"code": 12, "n_targets": 2, "n_placeholders": 0},
    "4N": {"code": 14, "n_targets": 4, "n_placeholders": 0},
    "6N": {"code": 16, "n_targets": 6, "n_placeholders": 0},
    "8N": {"code": 18, "n_targets": 8, "n_placeholders": 0},
    "2P": {"code": 22, "n_targets": 2, "n_placeholders": 8},
    "4P": {"code": 24, "n_targets": 4, "n_placeholders": 6},
    "6P": {"code": 26, "n_targets": 6, "n_placeholders": 4},
    "8P": {"code": 28, "n_targets": 8, "n_placeholders": 2},
}


# Color values
stim_colors = [
    "#AF0000",  # dark red
    "#FF00FF",  # pink
    "#FFFFFF",  # white
    "#FFFF00",  # yellow
    "#00FFFF",  # cyan
    "#008C00",  # dark green
    "#FF8000",  # orange
    "#0000FF",  # blue
    "#000000",  # black
]
ph_color = "#929292"  # gray, luminance matched to average of colors for the placeholders
bg_color = [-0.25] * 3  # from -1 to 1

# All sizes of stimuli & underlying grid are in DVA
stim_side_length = 1
stim_radius = stim_side_length / np.sqrt(2)  # distance from square center to corner
stim_area = stim_side_length**2

ph_radius_inner = 0.1
ph_radius_outer = np.sqrt((stim_area / np.pi) + (ph_radius_inner**2))

# buffers to avoid crowding
grid_padding = 0.5  # closest a stimulus can get to its cell's edge
stim_buffer = stim_side_length / 2 + grid_padding  # closest the center of the stimulus can get to its cell's edge
ph_buffer = ph_radius_outer / 2 + grid_padding


# screen coordinates for underlying grid
spatial_bin_size = np.array([4, 3])  # row, col sizes
n_rows = 3
n_cols = 5

# find when the rows and columns start, relative to the center of the screen (0,0)
# these define the top left corner of the grid cells
row_starts = np.linspace(spatial_bin_size[0] * n_rows, spatial_bin_size[0], n_rows) - (spatial_bin_size[0] * n_rows / 2)  # center vertically
col_starts = np.linspace(-spatial_bin_size[1] * n_cols, -spatial_bin_size[1], n_cols) + (spatial_bin_size[1] * n_cols / 2)  # center horizontally

# find which bins either 1) contain the fixation point or 2) are adjacent to the fixation point
# to be excluded from the spatial bins
row_center = [np.floor((n_rows / 2))] if (n_rows % 2 == 1) else [(n_rows / 2) - 1, (n_rows / 2)]
col_center = [np.floor((n_cols / 2))] if (n_cols % 2 == 1) else [(n_cols / 2) - 1, (n_cols / 2)]
fixation_bins = [(col_starts[int(c)], row_starts[int(r)]) for r in row_center for c in col_center]

# build the spatial bins, excluding the fixation bins
spatial_bins = [(c, r) for r in row_starts for c in col_starts if (c, r) not in fixation_bins]


monitor_details = {"name": "Monitor01", "distance": 82, "resolution": [1920, 1080], "width": 53}  # checked distance 2023-09-01

keys = ["z", "slash"]

data_fields = [
    "Subject",
    "Block",
    "Trial",
    "block_condition" "trial_condition",
    "n_targets",
    "n_placeholders",
    "stim_colors",
    "spt_bins",
    "spt_coords",
    "test_stim",
    "test_coords",
    "change",
    "ACC",
    "RT",
    "REJ",
]


user_input = {"Subject": "0", "Unique ID": "000000", "Age": "0", "Gender": "X"}

instructions = [
    (
        "This experiment will test your ability to remember items in your working memory\n\n"
        "Keep your eyes on the central fixation point at all times during the trial.\n"
        "Then, you will see a series of colored squares on the screen.\n"
        "Remember as many of the colors and locations as you can.\n"
        "In some trials, there will also be gray placeholder circles. Do not remember these.\n\n"
        "Then, after a delay, a test color will appear.\n"
        "If it is the same as the color that appeared in that location previously, press z.\n"
        "If it is not, press /.\n"
        "Once you give your response, the next trial will automatically start, so be ready to keep your eyes on the fixation point."
    )
]

experiment_name = "supracapacity"


class WM_supracapacity:

    """
    Docstring_temp
    """

    def __init__(
        self,
        trials_per_block=TRIALS_PER_BLOCK,
        num_blocks=NUM_BLOCKS,
        instructions=instructions,
        experiment_name=experiment_name,
        trial_types=trial_types,
        monitor_details=monitor_details,
        stim_time=STIM_TIME,
        delay_time=DELAY_TIME,
        baseline_time=BASELINE_TIME,
        iti_time=ITI_TIME,
        eye_max_dist=EYE_MAX_DIST,
        keys=keys,
        data_fields=data_fields,
        user_input=user_input,
        stim_size=stim_radius,
        stim_colors=stim_colors,
        ph_color=ph_color,
        bg_color=bg_color,
        spatial_bins=spatial_bins,
        spatial_bin_size=spatial_bin_size,
        stim_buffer=stim_buffer,
        ph_buffer=ph_buffer,
    ):
        # value checks

        if trials_per_block % 16:  # TODO: check if should be 16
            raise ValueError("Number of trials must be a multiple of 32 for appropriate balancing")
        self.experiment_name = experiment_name
        self.trials_per_block = trials_per_block
        self.num_blocks = num_blocks
        self.trial_types = trial_types
        self.monitor_details = monitor_details
        self.stim_time = stim_time
        self.baseline_time = baseline_time
        self.delay_time = delay_time
        self.iti_time = iti_time

        self.eye_max_dist = eye_max_dist
        self.keys = keys
        self.spatial_bins = spatial_bins
        self.spatial_bin_size = spatial_bin_size
        self.stim_buffer = stim_buffer
        self.ph_buffer = ph_buffer
        self.exp_info = user_input
        self.stim_colors = stim_colors
        self.stim_size = stim_size
        self.ph_radius_inner = ph_radius_inner
        self.ph_radius_outer = ph_radius_outer
        self.ph_color = ph_color
        self.bg_color = bg_color

        self.instructions = instructions

        self.row_rejected = np.zeros(5)
        self.experiment_data = pd.DataFrame(columns=data_fields)

        self.block_cur = 1

        self.et_rt = False  # this is enabled when we want to record realtime eyetracking
        self.rejection_counter = {key: 0 for key in trial_types.keys()}
        # counter of rejected values per condition

    # experiment window handling - from Colin Quirk's templateexperiments module

    def open_window(self, **kwargs):
        """Opens the psychopy window.

        Additional keyword arguments are sent to psychopy.visual.Window().
        """
        self.experiment_monitor = psychopy.monitors.Monitor(
            self.monitor_details["name"], width=self.monitor_details["width"], distance=self.monitor_details["distance"]
        )
        self.experiment_monitor.setSizePix(self.monitor_details["resolution"])
        self.experiment_window = psychopy.visual.Window(
            monitor=self.experiment_monitor, fullscr=True, color=self.bg_color, colorSpace="rgb", units="deg", allowGUI=False, **kwargs
        )

    def quit_experiment(self):
        """Completes anything that must occur when the experiment ends."""
        if self.experiment_window:
            self.experiment_window.close()
        print("The experiment has ended")
        sys.exit(0)

    def get_experiment_info_from_dialog(self, screen=0):
        """Gets subject info from dialog box.

        Parameters:
        additional_fields_dict -- An optional dictionary containing more
            fields for the dialog box and output dictionary.
        screen -- an int describing the screen you want the dialog to appear on
        """

        # Modifies experiment_info dict directly
        exp_info = psychopy.gui.DlgFromDict(self.exp_info, title=self.experiment_name, tip={"Unique Subject Identifier": "From the cronus log"}, screen=screen)
        self.exp_info = exp_info.dictionary
        np.random.seed(int(self.exp_info["Unique ID"]))

        self.subject_num = int(self.exp_info["Subject"])

        with open(f"{self.exp_info['Subject']}_info.csv", "w") as f:
            writer = csv.DictWriter(f, fieldnames=self.exp_info.keys())
            writer.writeheader()
            writer.writerow(self.exp_info)

        return exp_info.OK

    def display_text_screen(self, text="", text_color=[-1, -1, -1], text_height=36, bg_color=None, keyList=["space"], **kwargs):
        """Takes a string as input and draws it centered on the screen.

        Allows for simple writing of text to a screen with a background color
        other than the normal one. Switches back to the default background
        color after any keyboard input.

        This works by drawing a rect on top of the background
        that fills the whole screen with the selected color.

        Parameters:
        text -- A string containing the text to be displayed.
        text_color -- A list of 3 values between -1 and 1
            (default is [-1, -1, -1]).
        text_height --- An int that defines the height of the text in pix.
        bg_color -- A list of 3 values between -1 and 1 (default is default
            background color).


        Additional keyword arguments are sent to psychopy.visual.TextStim().
        """

        if bg_color is None:
            bg_color = self.bg_color

        backgroundRect = psychopy.visual.Rect(self.experiment_window, fillColor=bg_color, units="norm", width=2, height=2)

        textObject = psychopy.visual.TextStim(
            self.experiment_window,
            text=text,
            color=text_color,
            units="pix",
            height=text_height,
            alignHoriz="center",
            alignVert="center",
            wrapWidth=round(0.8 * self.experiment_window.size[0]),
            **kwargs,
        )

        backgroundRect.draw()
        textObject.draw()
        self.experiment_window.flip()

        keys = []
        psychopy.core.wait(0.2)  # Prevents accidental key presses
        keys = psychopy.event.waitKeys(keyList=keyList)
        self.experiment_window.flip()
        return keys

    # METHODS TO SET UP EYETRACKING. Adapted from Will Thyer's Cannonball experiment

    def et_initialize(self):
        """
        Initializes eyetracking if it's enabled
        """
        self.tracker = eyelinker.EyeLinker(self.experiment_window, "SCR" + self.exp_info["Subject"] + ".edf", "BOTH")

        self.tracker.initialize_graphics()
        self.tracker.open_edf()
        self.tracker.initialize_tracker()
        self.tracker.send_tracking_settings()

    def et_instruct(self):
        """displays ET instructions"""
        self.tracker.display_eyetracking_instructions()
        self.tracker.setup_tracker()

    def et_conclude(self):
        """Conclude experiment and transfers the edf file"""

        self.tracker.set_offline_mode()
        self.tracker.close_edf()
        self.tracker.transfer_edf()
        self.tracker.close_connection()

    def realtime_eyetrack(self, condition, wait_time, s_rate=0.01, dummy=False):
        """
        gets realtime eyetracking data and determines whether to reject the trial
        Options:
        condition - current trial condition
        time - time to collect for
        dummy - if true, always return false (don't reject)
        """

        if dummy:
            psychopy.core.wait(wait_time)
            return False

        start_time = psychopy.core.getTime()
        while psychopy.core.getTime() < start_time + wait_time:
            left, right = self.tracker.gaze_data_both

            if left is not None and right is not None:
                lx, ly = left
                rx, ry = right
                x = np.nanmean([lx, rx])
                y = np.nanmean([ly, ry])
            elif left is not None:
                x, y = left
            elif right is not None:
                x, y = right
            else:  # if no eye data do not reject
                return False, None

            # get the x,y pixel values relative to the center of the screen
            winx, winy = self.experiment_window.size / 2
            x -= winx
            y -= winy

            dist = np.linalg.norm(np.array([x, y]))

            radius = psychopy.tools.monitorunittools.deg2pix(self.eye_max_dist, self.experiment_monitor)

            if dist > radius:
                self.rejection_counter[condition] += 1
                print(f"# of rejected {condition} trials:{self.rejection_counter[condition]}")

                self.tracker.stop_recording()
                self.display_eyemovement_feedback((x, y))

                return True

            psychopy.core.wait(s_rate)

        # if we haven't rejected yet, return false
        return False

    def display_eyemovement_feedback(self, eyes):
        """
        show the position of the eye movement
        """

        psychopy.visual.TextStim(win=self.experiment_window, text="Eye Movement Detected.\nPress space to continue.", pos=[0, 1], color=[1, -1, -1]).draw()
        self.draw_fixation()

        psychopy.visual.Circle(win=self.experiment_window, radius=8, pos=eyes, fillColor="red", units="pix").draw()

        self.experiment_window.flip()
        psychopy.event.waitKeys(keyList=["space"])

    def do_rejection(self, reject):
        self.row_rejected = np.roll(self.row_rejected, 1)
        self.row_rejected[0] = reject

        if np.sum(self.row_rejected) == 5:
            self.row_rejected = np.zeros(5)
            self.display_text_screen(text="Rejected 5 in row. Continue?", keyList=["y"], bg_color=[0, 0, 255], text_color=[1, 1, 1])

    # helper functions for EEG
    def send_synced_event(self, code, keyword="SYNC"):
        """Send port code to EEG and eyetracking message for later synchronization

        Parameters:
        code-- Digits to send
        keyword-- Accompanying sync keyword (matters for later EEGLAB preprocessing)
        """

        message = keyword + " " + str(code)

        if self.port:
            self.port.setData(code)
            psychopy.core.wait(0.005)
            self.port.setData(0)
            self.tracker.send_message(message)

    def chdir(self):
        """Changes the directory to where the data will be saved."""

        try:
            os.makedirs("./subject_data")
        except OSError as e:
            if e.errno != errno.EEXIST:
                raise

        os.chdir("./subject_data")

    def setup_eeg(self):
        """Connects the parallel port for EEG port code"""
        try:
            self.port = psychopy.parallel.ParallelPort(address=53328)
        except Exception as e:
            self.port = None
            print("No parallel port connected. Port codes will not send!")
            raise e

    def instruct(self):  # TODO: add instructions
        """displays instructions for the experiment"""

        for instruction in self.instructions:
            print(instruction)

            self.display_text_screen(text=instruction, keyList=["space"])

        self.display_text_screen(text=("These are the visual stimuli you will be seeing. Press space to preview each one"), keyList=["space"])

        for stim_color in self.stim_colors:
            self.draw_square(color=stim_color, pos=(0, 0))
            self.experiment_window.flip()
            psychopy.core.wait(0.2)  # Prevents accidental key presses
            psychopy.event.waitKeys(keyList=["space"])

        self.display_text_screen(
            text="Gray circles are irrelevant. Do not remember it when it comes up. You will never be tested on it. Press space to continue", keyList=["space"]
        )
        self.draw_donut(color=self.ph_color, pos=(0, 0))
        self.experiment_window.flip()
        psychopy.core.wait(0.2)  # Prevents accidental key presses
        psychopy.event.waitKeys(keyList=["space"])

        self.experiment_window.flip()

    """block creation and assignment"""

    def assign_stimuli(self, trial_data):
        # assign colors, adding placeholders if conditioned
        trial_data["stim_colors"] = list(np.random.choice(self.stim_colors, trial_data["n_targets"], replace=False))
        trial_data["stim_colors"] = trial_data["stim_colors"] + [self.ph_color] * trial_data["n_placeholders"]

        # assign a bin for each stimulus at random
        trial_data["spt_bins"] = np.random.choice(np.arange(0, len(self.spatial_bins)), trial_data["n_targets"] + trial_data["n_placeholders"], replace=False)
        trial_data["spt_coords"] = np.array(self.spatial_bins)[trial_data["spt_bins"]].astype(float)  # get top left corner of bins
        # add jitter to the coordinates, moving them from the top left corner towards the center of the bin, with some buffer to avoid the grid edges/each other
        spatial_buffer = ([self.stim_buffer] * trial_data["n_targets"]) + ([self.ph_buffer] * trial_data["n_placeholders"])
        # jitter across x axis - positive to shift rightward
        trial_data["spt_coords"][:, 0] += np.random.uniform(spatial_buffer, self.spatial_bin_size[0] - spatial_buffer)
        # jitter across y axis - negative to shift downward
        trial_data["spt_coords"][:, 1] -= np.random.uniform(spatial_buffer, self.spatial_bin_size[1] - spatial_buffer)

        return trial_data

    def assign_test(self, trial_data):
        """
        assigns test data to a trial based on its condition and change status.
        Arguments:
        """

        # if change, assign a stimulus that was not used in one of the stimulis locations
        if trial_data["change"]:
            trial_data["test_stim"] = np.random.choice(np.setdiff1d(self.stim_colors, trial_data["stim_colors"]))
            target_locs = trial_data["spt_coords"][: trial_data["n_targets"]]
            trial_data["test_coords"] = np.random.permutation(target_locs)[0]

        # if no change, use one of the stimuli presented with its same location
        else:
            idx = np.random.choice(trial_data["n_targets"])
            trial_data["test_stim"] = trial_data["stim_colors"][idx]
            trial_data["test_coords"] = trial_data["spt_coords"][idx]

        return trial_data

    # TODO - set up WM and attn blocks
    def make_blocks(self, n_blocks, n_per_block, conditions=None):
        """
        generator that creates blocks
        trial_condition: overrides to make all trials in the block this condition
        """

        if conditions is None:
            conditions = list(self.trial_types.keys())

        assert n_per_block % len(conditions) == 0

        for _ in range(n_blocks):
            block_data = pd.DataFrame(columns=self.experiment_data.keys())

            # if we're just starting, block 0, else append by 1

            # populate rest of fields
            block_data["Trial"] = np.arange(0, n_per_block)
            block_data["Subject"] = self.subject_num

            # assign trial conditions and changes

            n_per_cond = n_per_block // len(conditions)
            remaining_trials = np.arange(n_per_block)
            for cond in conditions:
                curr_trials = np.random.choice(remaining_trials, n_per_cond, replace=False)
                remaining_trials = np.setdiff1d(remaining_trials, curr_trials)
                change = np.random.choice([True, False], len(curr_trials), replace=True)
                block_data.loc[curr_trials, "trial_condition"] = cond
                block_data.loc[curr_trials, "change"] = change
                block_data.loc[curr_trials, "n_targets"] = self.trial_types[cond]["n_targets"]
                block_data.loc[curr_trials, "n_placeholders"] = self.trial_types[cond]["n_placeholders"]

            block_data = block_data.apply(self.assign_stimuli, axis=1)

            block_data = block_data.apply(self.assign_test, axis=1)

            block_data.to_csv("tmp.csv")
            yield block_data

    # DISPLAY FUNCTIONS

    def display_message(self, text, text_color=[-1, -1, -1], bg_color=None, text_height=36, **kwargs):
        """Takes a string as input and draws it centered on the screen.

        Allows for simple writing of text to a screen with a background color
        other than the normal one. Continues when space is pressed

        This works by drawing a rect on top of the background
        that fills the whole screen with the selected color.



        Arguments:
        text -- A string containing the text to be displayed.
        text_color -- list of 3 values between [-1,-1,-1] and [1,1,1]
        text_height --- An int that defines the height of the text in pix.
        bg_color -- List of 3 color values

        Additional keyword arguments are sent to psychopy.visual.TextStim().


        """

        if bg_color is None:
            bg_color = self.bg_color
        else:
            bg_color = bg_color

        backgroundRect = psychopy.visual.Rect(self.experiment_window, fillColor=bg_color, units="norm", width=2, height=2)

        textObject = psychopy.visual.TextStim(
            self.experiment_window,
            text=text,
            color=text_color,
            units="pix",
            height=text_height,
            alignHoriz="center",
            alignVert="center",
            wrapWidth=round(0.8 * self.experiment_window.size[0]),
            **kwargs,
        )

        backgroundRect.draw()
        textObject.draw()
        self.experiment_window.flip()

        psychopy.core.wait(0.2)  # Prevents accidental key presses
        _ = psychopy.event.waitKeys(keyList=["space"])
        self.experiment_window.flip()

    def display_start_block_screen(self):
        self.display_text_screen(text=f"This is block {self.block_cur}.\nPress space to begin the block.", keyList=["space"])

    def display_fixation(self, wait_time=None, keyList=None, realtime_eyetrack=False, trial=None):
        """Displays a fixation cross. A helper function for self.run_trial.

        Parameters:
        wait_time -- The amount of time the fixation should be displayed for.
        keyList -- If keyList is given, will wait until key press
        trial -- Trial object needed for realtime eyetracking functionality.
        real_time_eyetracking -- Bool for if you want to do realtime eyetracking or not


        """
        self.draw_fixation()

        self.experiment_window.flip()

        if realtime_eyetrack:
            reject = self.realtime_eyetrack(wait_time=wait_time, condition=trial["trial_condition"])
            return reject
        else:
            if keyList:
                resp = psychopy.event.waitKeys(maxWait=wait_time, keyList=keyList)
                if resp == ["p"]:
                    self.display_text_screen(text="Paused", keyList=["space"])
                    self.display_fixation(wait_time=1)
                elif resp == ["o"]:
                    self.tracker.calibrate()
                    self.display_fixation(wait_time=1)
                elif resp == ["escape"]:
                    resp = self.display_text_screen(text="Are you sure you want to exit?", keyList=["y", "n"])
                    if resp == ["y"]:
                        self.experiment_window.saveFrameIntervals()
                        print("Overall, %i frames were dropped." % self.experiment_window.nDroppedFrames)

                        self.tracker.transfer_edf()
                        self.quit_experiment()
                    else:
                        self.display_fixation(wait_time=1)
            else:
                psychopy.core.wait(wait_time)

    def init_fixation(self):
        # set up fixation shapes, based on Thaler et al., 2013
        self.fix_outer = psychopy.visual.Circle(
            self.experiment_window, lineColor=None, fillColor=[-1, -1, -1], fillColorSpace="rgb", radius=0.25, units="deg", interpolate=True
        )
        self.fix_cross = psychopy.visual.ShapeStim(
            self.experiment_window,
            vertices=((0, -0.26), (0, 0.26), (0, 0), (-0.26, 0), (0.26, 0)),
            lineWidth=psychopy.tools.monitorunittools.deg2pix(0.16, self.experiment_monitor),
            closeShape=False,
            lineColor=self.bg_color,
        )
        self.fix_inner = psychopy.visual.Circle(
            self.experiment_window, lineColor=None, fillColor=[-1, -1, -1], fillColorSpace="rgb", radius=0.075, units="deg", interpolate=True
        )

    def draw_fixation(self):
        self.fix_outer.draw()
        self.fix_cross.draw()
        self.fix_inner.draw()

    def draw_trak(self, x=930, y=510):
        trak = psychopy.visual.Circle(self.experiment_window, lineColor=None, fillColor=[1, 1, 1], fillColorSpace="rgb", radius=20, pos=[x, y], units="pix")
        trak.draw()

    def draw_square(self, color, pos):
        psychopy.visual.Polygon(radius=self.stim_size, ori=45, edges=4, color=color, pos=pos, win=self.experiment_window).draw()

    def draw_donut(self, color, pos):
        psychopy.visual.Circle(radius=self.ph_radius_outer, color=color, pos=pos, win=self.experiment_window).draw()
        psychopy.visual.Circle(radius=self.ph_radius_inner, color=self.bg_color, pos=pos, win=self.experiment_window).draw()

    def present_stim(self, trial, realtime_eyetrack=False):
        """
        presents a stimulus array of colored squares (and gray circles)

        Arguments:
        trial - data for the trial
        realtime_eyetracking - bool if to use realtime eyetracking
        """

        self.draw_fixation()
        self.draw_trak()
        for istim, stim_color in enumerate(trial["stim_colors"]):
            if stim_color == self.ph_color:
                self.draw_donut(stim_color, trial["spt_coords"][istim])
            else:
                self.draw_square(stim_color, trial["spt_coords"][istim])

        self.send_synced_event(self.trial_types[trial["trial_condition"]]["code"])
        self.experiment_window.flip()

        if realtime_eyetrack:
            reject = self.realtime_eyetrack(wait_time=self.stim_time, condition=trial["trial_condition"])
            return reject
        else:
            psychopy.core.wait(self.stim_time)

    def present_test(self, trial):
        """
        Presents the test stimulus

        Arguments:
        trial - trial details
        """
        self.send_synced_event(9)

        self.draw_fixation()
        print(trial["test_coords"])
        self.draw_square(trial["test_stim"], trial["test_coords"])

        self.experiment_window.flip()

    def get_response(self):
        """
        Gets the participant response
        """

        rt_timer = psychopy.core.MonotonicClock()

        psychopy.core.wait(0.1)

        resp = psychopy.event.waitKeys(keyList=self.keys, timeStamped=rt_timer)

        return resp[0][0], resp[0][1] * 1000  # key and rt in milliseconds

    def run_trial(self, trial, realtime_eyetrack=False):
        """
        Runs a trial

        Trial condition codes:

        1 (start) / stim code (stim 1) / stim code (stim 2) / 2 (ITI) / 3 (test) / 4 (end)

        Arguments:
        trial - trial data
        realtime_eyetrack - whether or not to use realtime eyetracking
        """

        if self.tracker.mock:
            realtime_eyetrack = False

        print(trial["trial_condition"])

        # ITI
        iti_time = np.random.random() * (self.iti_time[1] - self.iti_time[0]) + self.iti_time[0]
        self.display_fixation(wait_time=iti_time, trial=trial, keyList=["p", "escape", "o"], realtime_eyetrack=False)

        # start trial
        status = f"Block {trial['Block']+1}, Trial {trial['Trial']+1}"
        print(status)
        self.tracker.send_status(status)
        self.tracker.start_recording()

        # Start eyetracking, baseline period, send code 1 to begin the trial
        self.send_synced_event(1)

        reject = self.display_fixation(wait_time=self.baseline_time, trial=trial, realtime_eyetrack=realtime_eyetrack)
        if reject:
            self.do_rejection(1)
            return None, None

        # display the  stim (code sent in function)
        reject = self.present_stim(trial, realtime_eyetrack=realtime_eyetrack)
        if reject:
            self.do_rejection(1)
            return None, None

        # send code 2, begin the delay period
        self.send_synced_event(2)
        reject = self.display_fixation(self.delay_time, trial=trial, realtime_eyetrack=realtime_eyetrack)
        if reject:
            self.do_rejection(1)
            return None, None

        # display the test array (sends code 9)
        self.present_test(trial)
        resp, rt = self.get_response()

        # conclude the trial, send code 4
        self.send_synced_event(4)
        self.tracker.stop_recording()
        self.do_rejection(0)

        if trial["change"] and resp == self.keys[0]:
            acc = 0
        elif trial["change"] and resp == self.keys[1]:
            acc = 1
        elif not trial["change"] and resp == self.keys[0]:
            acc = 1
        elif not trial["change"] and resp == self.keys[1]:
            acc = 0
        else:
            raise ValueError("Invalid response")

        print(f"Acc:{acc}, RT:{rt:.3f}")

        return acc, rt

    # START RUNNING THE EXPERIMENT

    def run(self):
        """
        Run the experiment!
        """
        self.chdir()
        ok = self.get_experiment_info_from_dialog()

        if not ok:
            print("Experiment has been terminated.")
            sys.exit(1)

        if os.path.isfile(f"{self.exp_info['Subject']}_data.csv"):
            overwrite_dlg = psychopy.gui.Dlg("Overwrite?", labelButtonOK="Overwrite", labelButtonCancel="Exit", screen=0)
            overwrite_dlg.addText("File already exists. Overwrite?")
            overwrite_dlg.show()

            if overwrite_dlg.OK is None:
                print("Experiment has been terminated.")
                sys.exit(1)

        self.open_window(screen=0)
        self.init_fixation()  # set up fixation cross

        # self.instruct()

        self.et_initialize()
        self.et_instruct()

        """
        Run practice blocks - don't send condition codes or record data
        Runs one trial per condition
        """
        self.port = None
        prac = self.display_text_screen(text="Practice block?", keyList=["y", "n"])
        while prac == ["y"]:
            for block in self.make_blocks(n_blocks=1, n_per_block=16):
                accs = []

                for trial_num, trial in block.iterrows():
                    if trial_num == 0:
                        self.display_start_block_screen()
                    acc, rt = self.run_trial(trial, realtime_eyetrack=False)
                    accs.append(acc)

                self.display_text_screen(text=f"Block Accuracy: {round(np.mean(accs)*100)}\n\n\n\nPress space to continue.", keyList=["space"])
                prac = self.display_text_screen(text="Practice block?", keyList=["y", "n"])

        """
        Run the main experiment blocks with realtime eyetracking - all normal
        """

        self.display_text_screen(text="Starting main experiment. Press space to continue", keyList=["space"])
        self.setup_eeg()

        for block in self.make_blocks(n_blocks=self.num_blocks, n_per_block=self.trials_per_block):
            self.tracker.calibrate()

            accs = []
            for trial_num, trial in block.iterrows():
                if trial_num == 0:
                    self.display_start_block_screen()
                if trial_num % 5 == 0:
                    self.tracker.drift_correct()  # drift correct every 5

                acc, rt = self.run_trial(trial, realtime_eyetrack=True)  # run each trial with realtime eyetracking
                if acc is not None and rt is not None:  # trial was NOT rejected
                    accs.append(acc)
                    trial["ACC"] = acc
                    trial["RT"] = rt
                    trial["REJ"] = False
                else:
                    trial["ACC"] = np.nan
                    trial["RT"] = np.nan
                    trial["REJ"] = True
                block.loc[trial_num] = trial

            block["Block"] = self.block_cur

            self.experiment_data.append(block)
            self.experiment_data = self.experiment_data[block.columns]

            if self.block_cur == 1:
                block.to_csv(f"{self.exp_info['Subject']}_data.csv", header=True, index=False, mode="w+")
            else:
                block.to_csv(f"{self.exp_info['Subject']}_data.csv", header=False, index=False, mode="a")

            self.block_cur += 1
            self.display_text_screen(text=f"Block Accuracy: {round(100*np.nanmean(accs))}\n\n\n\nPress space to continue.", keyList=["space"])

        # makeup blocks

        while any([True for c in trial_types.keys() if self.rejection_counter[c] > 10]):
            makeup_block = pd.DataFrame(columns=self.experiment_data.columns)

            # build up mega make-up block
            for cond in self.rejection_counter.keys():
                for block in self.make_blocks(n_blocks=1, n_per_block=self.rejection_counter[cond], conditions=[cond]):
                    self.rejection_counter[cond] = 0
                    makeup_block = pd.concat((makeup_block, block)).sample(frac=1)

            makeup_block = makeup_block.sample(frac=1).reset_index(drop=True)  # shuffle the trial types together

            accs = []

            self.tracker.calibrate()
            self.rejection_tracker = np.zeros(5)
            for trial_num, trial in makeup_block.iterrows():  # TODO: break this into smaller blocks
                if trial_num == 0:
                    self.display_start_block_screen()
                if trial_num > 0 and not trial_num % self.trials_per_block:  # give them breaks every 80
                    self.display_text_screen(text=f"Take a short break. Press space to continue", keyList=["space"])

                if trial_num % 5 == 0:
                    self.tracker.drift_correct()  # drift correct every 5
                acc, rt = self.run_trial(trial, realtime_eyetrack=True)  # run each trial with realtime eyetracking
                if acc is not None and rt is not None:  # trial was NOT rejected
                    accs.append(acc)
                    trial["ACC"] = acc
                    trial["RT"] = rt
                    trial["REJ"] = False
                else:
                    trial["ACC"] = np.nan
                    trial["RT"] = np.nan
                    trial["REJ"] = True
                makeup_block.loc[trial_num] = trial

            makeup_block["Block"] = self.block_cur

            self.experiment_data.append(makeup_block)
            self.experiment_data = self.experiment_data[makeup_block.columns]
            self.experiment_data.to_csv(f"{self.exp_info['Subject']}_data.csv", header=False, mode="a")  # update the csv file

            self.display_text_screen(text=f"Block Accuracy: {round(100*np.nanmean(accs))}\n\n\n\nPress space to continue.", keyList=["space"])
        """
        End of Experiment
        """
        self.display_text_screen("The experiment is now over, please get your experimenter.", bg_color=[0, 0, 0], text_color=[1, 1, 1])

        self.et_conclude()
        self.quit_experiment()


# If you call this script directly, the task will run with your defaults
if __name__ == "__main__":
    exp = WM_supracapacity()

    try:
        exp.run()
    except Exception as e:
        exp.et_conclude()
        raise e
