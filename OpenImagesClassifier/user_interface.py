"""Graphical user interface for ImagesClassifier"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from OpenImagesClassifier import image_classifier_runtime as runtime
from OpenImagesClassifier import config
from OpenImagesClassifier import data_processing as data
from OpenImagesClassifier import setup
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

from enum import Enum
from tkinter import messagebox
from tkinter import ttk
from tkinter import filedialog
from PIL import ImageTk, Image

import tensorflow as tf
import tkinter as tk
import numpy as np
import os
import threading
import sqlite3
import time
import datetime


class SetupState(Enum):
    NONE = 0
    DATA_FETCHED = 1
    READY_FOR_TRAIN = 2


class ModelWrapper:
    def __init__(self, model_type=runtime.ModelType.SMALL_RESNET, batch_size=256):
        self.graph = tf.Graph()
        self.model_type = model_type
        with self.graph.as_default():
            self.model = runtime.ImageClassifier(model_type, batch_size)

    def train(self, cycles):
        with self.graph.as_default():
            return self.model.train(cycles)

    def validate(self, number_of_batches=1, aggregated=True, write_summary=True):
        with self.graph.as_default():
            return self.model.validate(number_of_batches, aggregated, write_summary)

    def test(self, number_of_batches=1, aggregated=True, write_summary=True):
        with self.graph.as_default():
            return self.model.test(number_of_batches, aggregated, write_summary)

    def predict(self, file_list):
        with self.graph.as_default():
            return self.model.predict(file_list)


class Controller(tk.Frame):
    def __init__(self, master):
        super(Controller, self).__init__(master)
        self._ready_to_run = False
        self.batch_size = 256
        self.inserted_batch_size = self.batch_size
        self.configure(bd='2', relief='groove', padx=4, pady=4)
        self.small_resnet = None
        self.trained_network = None
        self.current_model = None
        self._process_active = False

        tk.Label(self, text='General Model Control').grid(row=0, columnspan=3)
        # choose a network
        self.v = tk.IntVar()
        self.v.trace("w", lambda x, y, z, v=self.v: self._radio_updated(v))
        self.v.set(1)
        self.rd_resnet = tk.Radiobutton(self, text="Small Resnet", variable=self.v, value=1)
        self.rd_resnet.grid(row=1, column=0, sticky='W')
        self.rd_trained = tk.Radiobutton(self, text="Trained Model", variable=self.v, value=2)
        self.rd_trained.grid(row=1, column=1, sticky='W')

        # batch size
        tk.Label(self, text="Batch Size:").grid(row=2, column=0, sticky='W')

        sv = tk.StringVar()
        sv.trace("w", lambda x, y, z, sv=sv: self._batch_inserted(sv))
        sv.set(self.inserted_batch_size)
        self.batch_entry = tk.Entry(self, textvariable=sv, width=5)
        self.batch_entry.grid(row=2, column=1, sticky='W')
        self.batches_button = tk.Button(self, text='Apply Batch Size', command=self._apply_batch_size)
        self.batches_button.grid(row=2, column=2, sticky='E')

        # training panel
        tk.Label(self, text="Training").grid(row=4, columnspan=3)
        self.panels = []
        panel = ActionPanel(master=self, action=self._run_train, controller=self, name='Train')
        panel.grid(row=5, columnspan=3)
        self.panels.append(panel)

        # validation panel
        tk.Label(self, text="Validation").grid(row=6, columnspan=3)
        panel = ActionPanel(master=self, action=self._run_validation, controller=self,
                            name='Validation', bool_option_name="Aggregated results")
        panel.grid(row=7, columnspan=3)
        self.panels.append(panel)

        # testing panel
        tk.Label(self, text="Testing").grid(row=8, columnspan=3)
        panel = ActionPanel(master=self, action=self._run_test, controller=self,
                            name='Test', bool_option_name="Aggregated results")
        panel.grid(row=9, columnspan=3)
        self.panels.append(panel)

        # prediction panel
        tk.Label(self, text="Prediction").grid(row=10, columnspan=3)
        panel = PredictionPanel(self, self._run_predict)
        panel.grid(row=11, columnspan=3)
        self.panels.append(panel)

        # self.result_controller = result_controller
        # self.result_history = self.result_controller.result_history
        self.result_history = ResultHistory(self)

        self._disable_action()
        self._propagate_batch_size()
        self._should_stop_action = False

    def _setup_models(self):
        if self._ready_to_run:
            self.small_resnet = ModelWrapper(model_type=runtime.ModelType.SMALL_RESNET, batch_size=self.batch_size)
            self.trained_network = ModelWrapper(model_type=runtime.ModelType.TRAINED_MODEL, batch_size=self.batch_size)
            self._radio_updated(self.v)

    def _radio_updated(self, v):
        if v.get() == 1:
            self.current_model = self.small_resnet
        else:
            self.current_model = self.trained_network

    def stop_action(self):
        self._should_stop_action = True

    def completed_process(self):
        self._process_active = False
        self._should_stop_action = False
        self.enable_action()

    def _apply_batch_size(self):
        if self.batch_size != self.inserted_batch_size:
            self.batch_size = self.inserted_batch_size
            self._setup_models()
            self._propagate_batch_size()

    def _batch_inserted(self, sv):
        if sv.get() != '':
            self.inserted_batch_size = int(sv.get())

    def set_runstate(self, state):
        if self._ready_to_run != state:
            self._ready_to_run = state
            self.enable_action()
            self._setup_models()

    def _run_train(self, cycles, caller, early_stop=False):
        self._disable_action()
        if not self._process_active:
            self.result_history.reset_training_results()
            t = threading.Thread(target=self._thread_train, args=(cycles, caller, early_stop))
            t.start()
            self._process_activated()

    def _thread_train(self, cycles, caller, early_stop):
        caller.is_stoppable()
        for i in range(0, cycles, 10):
            if self._should_stop_action:
                break
            cycles_now = min(10, cycles - i)
            result = self.current_model.train(cycles_now)
            caller.update_progress(cycles_now)

            self.result_history.add_training_result(result)

            if early_stop:
                if self.should_stop_early(result):
                    break
        caller.completed_process()
        self.completed_process()

    def should_stop_early(self, result):
        # todo early stop mgm
        return False

    def _run_validation(self, cycles, caller, aggregated=False):
        self._disable_action()
        if not self._process_active:
            self.result_history.reset_validation_results()
            t = threading.Thread(target=self._thread_validation, args=(cycles, caller, aggregated))
            t.start()
            self._process_activated()

    def _thread_validation(self, cycles, caller, aggregated):
        if aggregated:
            caller.progress_unknown()
            result = self.current_model.validate(number_of_batches=cycles, aggregated=True)
            self.result_history.add_validation_results(result)
        else:
            caller.is_stoppable()
            for i in range(0, cycles, 10):
                if self._should_stop_action:
                    break
                cycles_now = min(10, cycles - i)
                result = self.current_model.validate(cycles_now, aggregated=False)
                caller.update_progress(cycles_now)
                self.result_history.add_validation_results(result)

        caller.completed_process()
        self.completed_process()

    def _run_test(self, cycles, caller, aggregated=False):
        self._disable_action()
        if not self._process_active:
            self.result_history.reset_test_results()
            t = threading.Thread(target=self._thread_test, args=(cycles, caller, aggregated))
            t.start()
            self._process_activated()

    def _thread_test(self, cycles, caller, aggregated):
        if aggregated:
            caller.progress_unknown()
            result = self.current_model.test(number_of_batches=cycles, aggregated=True)
            self.result_history.add_test_results(result)
        else:
            caller.is_stoppable()
            for i in range(0, cycles, 10):
                if self._should_stop_action:
                    break
                cycles_now = min(10, cycles - i)
                result = self.current_model.test(cycles_now, aggregated=False)
                caller.update_progress(cycles_now)
                self.result_history.add_test_results(result)

        caller.completed_process()
        self.completed_process()

    def _run_predict(self, filename, caller):
        self._disable_action()
        if not self._process_active:
            t = threading.Thread(target=self._thread_predict, args=(filename, caller))
            t.start()
            self._process_activated()

    def _thread_predict(self, filename, caller):
        caller.progress_unknown()
        result = self.current_model.predict([filename])
        caller.completed_process()
        caller.transmit_result(result)
        self.completed_process()

    def _process_activated(self):
        self._process_active = True
        self._disable_action()

    def _disable_action(self):
        self.batches_button.configure(state='disabled')
        self.rd_resnet.configure(state='disabled')
        self.rd_trained.configure(state='disabled')
        self.batch_entry.configure(state='disabled')
        for panel in self.panels:
            panel.disable_control()

    def enable_action(self):
        self.batches_button.configure(state='normal')
        self.rd_resnet.configure(state='normal')
        self.rd_trained.configure(state='normal')
        self.batch_entry.configure(state='normal')
        for panel in self.panels:
            panel.enable_control()

    def _propagate_batch_size(self):
        for panel in self.panels:
            panel.set_batch_size(self.batch_size)

    def get_runstate(self):
        return self._ready_to_run


class ResultController(tk.Frame):
    def __init__(self, master):
        super(ResultController, self).__init__(master)
        self.result_history = ResultHistory()


class ResultHistory:
    def __init__(self, controller):
        self.controller = controller
        self.train_results = {'small_resnet': {'training': [], 'validation': []},
                              'trained_model': {'training': [], 'validation': []}}
        self.validation_results = {'small_resnet': [], 'trained_model': []}
        self.test_results = {'small_resnet': [], 'trained_model': []}

    def _get_model_key(self):
        if self.controller.current_model.model_type == runtime.ModelType.SMALL_RESNET:
            return 'small_resnet'
        if self.controller.current_model.model_type == runtime.ModelType.TRAINED_MODEL:
            return 'trained_model'

    def add_training_result(self, result):
        model_key = self._get_model_key()
        self.train_results[model_key]['training'].extend(result['training'])
        self.train_results[model_key]['validation'].append(result['validation'])

    def reset_training_results(self):
        model_key = self._get_model_key()
        self.train_results[model_key] = {'training': [], 'validation': []}

    def add_validation_results(self, results):
        model_key = self._get_model_key()
        self.validation_results[model_key].extend(results)

    def reset_validation_results(self):
        model_key = self._get_model_key()
        self.validation_results[model_key] = []

    def add_test_results(self, results):
        model_key = self._get_model_key()
        self.test_results[model_key].extend(results)
        print(results)

    def reset_test_results(self):
        model_key = self._get_model_key()
        self.test_results[model_key] = []


class ResultPane(tk.Frame):
    def __init__(self, master, result_manager):
        super(ResultPane, self).__init__(master)
        self.result_manager = result_manager


class Panel(tk.Frame):
    def __init__(self, master):
        super(Panel, self).__init__(master)
        self.batch_size = 0

    def disable_control(self):
        return

    def enable_control(self):
        return

    def completed_process(self):
        return

    def update_progress(self, steps):
        return

    def progress_unknown(self):
        return

    def set_batch_size(self, size):
        return

    def is_stoppable(self):
        return


class PredictionPanel(Panel):
    def __init__(self, master, action):
        super(PredictionPanel, self).__init__(master)
        self.overall_steps = 100
        self.action = action

        self.configure(bd='2', relief='groove', padx=4, pady=4)
        tk.Label(self, text="Select Image...").grid(column=0, row=0, sticky="W")
        self.open_button = tk.Button(self, command=self._open_file, text="Open and Predict")
        self.open_button.grid(column=1, row=0, padx=10)

        self.progress_bar = ttk.Progressbar(self, orient="horizontal", length=100, mode='determinate')
        self.progress_bar.grid(row=0, column=2)

        self.image_label = tk.Label(self)
        self.image_label.grid(row=1, column=0)

        self.filename = None
        self.image = None

    def _open_file(self):
        self.filename = filedialog.askopenfilename(initialdir="./", title="Select Image",
                                                   filetypes=(('JPEG Image', '*.jpg'),))
        if os.path.exists(self.filename):
            self.action(self.filename, self)
            self.progress_bar.configure(mode='indeterminate', maximum=100)
            self.progress_bar.start(50)

            image = Image.open(self.filename)
            image = image.resize(size=(256, 256))
            self.tk_image = ImageTk.PhotoImage(image)
            # self.image_label.image = self.photo
            self.image_label.configure(image=self.tk_image)

    def transmit_result(self, result):
        with sqlite3.connect(config.DATABASE['filename']) as conn:
            c = conn.cursor()
            res = c.execute("SELECT DisplayLabelName FROM Dict ORDER BY ClassNumber")
            names = np.array(res.fetchall())
            names = names.reshape([len(names)])

            fig = Figure(figsize=(2, 3), facecolor="white")
            axis = fig.add_subplot(111)
            y_pos = np.arange(len(names))
            x = np.array(result[0]['prediction']).reshape([10])

            axis.barh(y_pos, x, align='center', color='blue')
            axis.set_yticks(y_pos)
            axis.set_yticklabels(names)
            axis.invert_yaxis()
            axis.set_xlabel('Prediction')
            for i, v in enumerate(x):
                axis.text(v + 3, i, " "+str(v), va='center', color='black', fontweight='bold')
            fig.tight_layout()
            canvas = FigureCanvasTkAgg(fig, master=self)
            canvas.show()
            canvas.get_tk_widget().grid(row=1, column=1, columnspan=2, padx=10, pady=10)
        return

    def disable_control(self):
        self.open_button.configure(state="disabled")

    def enable_control(self):
        self.open_button.configure(state="normal")

    def completed_process(self):
        self.progress_bar.stop()
        self.progress_bar.configure(maximum=self.overall_steps, value=self.overall_steps, mode='determinate')


class ActionPanel(Panel):
    def __init__(self, action, master, controller, name='Train', bool_option_name='Early Stopping'):
        super(ActionPanel, self).__init__(master)
        self.overall_steps = 0
        self.current_step = 0
        self.action = action
        self.controller = controller

        self.configure(bd='2', relief='groove', padx=4, pady=4)

        lbl_cycles = tk.Label(self, text="Cycles:")
        lbl_cycles.grid(column=0, row=0, sticky="W")

        sv = tk.StringVar()
        self.cycles_entry = tk.Entry(self, width=5, textvariable=sv)
        self.cycles_entry.grid(column=1, row=0)

        self.calc_label = tk.Label(self, text='= 0 Images')
        self.calc_label.grid(column=2, row=0)
        sv.trace("w", lambda x, y, z, sv=sv: self._update_image_count(sv))
        self.cycles_entry.insert(0, "10")

        self.train_button = tk.Button(self, text=name, command=self._run_process, width=15)
        self.train_button.grid(column=4, row=0, sticky="E")

        self.stop_button = tk.Button(self, text='Stop', command=self._stop, state='disabled')
        self.stop_button.grid(column=4, row=1, sticky="E")

        self.bool_var = tk.IntVar()
        self.bool_flag = tk.Checkbutton(self, text=bool_option_name, var=self.bool_var)
        self.bool_flag.deselect()
        self.bool_flag.grid(column=2, columnspan=2, row=1)

        lbl_step_txt = tk.Label(self, text="Batch:")
        lbl_step_txt.grid(column=0, row=2, sticky="W")
        self.label_step = tk.Label(self, text="0/0", width=10)
        self.label_step.grid(column=1, row=2)
        self.progress_bar = ttk.Progressbar(self, orient="horizontal", length=300, mode='determinate')
        self.progress_bar.grid(row=2, column=2, columnspan=3, rowspan=2, pady=4)

        tk.Label(self, text='Duration:').grid(row=3, column=0)

        self.duration_label = tk.Label(self)
        self.duration_label.grid(row=3, column=1)

        self.start_time = None
        self._in_action = False

    def _update_image_count(self, sv):
        if sv.get() != '':
            self.overall_steps = int(sv.get())
            self.calc_label.configure(text='= {} Images'.format((self.overall_steps * self.batch_size)))

    def _stop(self):
        self.controller.stop_action()

    def disable_control(self):
        self.train_button.configure(state='disabled')
        self.cycles_entry.configure(state='disabled')
        self.bool_flag.configure(state='disabled')
        self.stop_button.configure(state='disabled')

    def enable_control(self):
        self.train_button.configure(state='normal')
        self.cycles_entry.configure(state='normal')
        self.bool_flag.configure(state='normal')
        self.stop_button.configure(state='disabled')

    def is_stoppable(self):
        self.stop_button.configure(state='normal')

    def _run_process(self):
        self.current_step = 0
        self.progress_bar.stop()
        self.progress_bar.configure(maximum=self.overall_steps, value=self.current_step, mode='determinate')
        self.label_step.configure(text="{}/{}".format(self.current_step, self.overall_steps))
        self.start_time = datetime.datetime.now()
        self._in_action = True
        self.time_thread = threading.Thread(target=self._time_upd)
        self.time_thread.start()

        bool_flag = bool(self.bool_var.get())
        self.action(self.overall_steps, self, bool_flag)

    def _time_upd(self):
        while True:
            if not self._in_action:
                break
            self.duration_label.configure(text=str(datetime.datetime.now() - self.start_time).split('.', 2)[0])
            time.sleep(1)

    def update_progress(self, steps):
        self.current_step = self.current_step + steps
        self.progress_bar.configure(value=self.current_step)
        self.label_step.configure(text="{}/{}".format(self.current_step, self.overall_steps))

    def completed_process(self):
        self.progress_bar.stop()
        self.progress_bar.configure(maximum=self.overall_steps, value=self.overall_steps, mode='determinate')
        self.current_step = 0
        self._in_action = False

    def set_batch_size(self, size):
        self.batch_size = size
        self.calc_label.configure(text='= {} Images'.format((self.overall_steps * self.batch_size)))

    def progress_unknown(self):
        self.progress_bar.configure(mode='indeterminate', maximum=100)
        self.progress_bar.start(50)
        self.label_step.configure(text="?")


class Application(tk.Frame):
    def __init__(self, master=None):
        super(Application, self).__init__(master)

        self.controller = Controller(master=self)
        self.controller.pack(side='bottom')

        self.status = StatusFrame(master=self, controller=self.controller)
        self.status.pack(side='top')

        self.model_small_resnet = None
        self.model_trained = None
        self.current_model = None


class StatusFrame(tk.Frame):
    def __init__(self, controller, master=None):
        super(StatusFrame, self).__init__(master)
        self.controller = controller
        self.configure(bd='2', relief='groove', padx=2, pady=2)

        self.state_label_basic = tk.Label(self, text="(Ready)", width='14', fg="white")
        self.state_label_basic.grid(column=0, row=0)
        label_basic = tk.Label(self, text="Download Dataset")
        label_basic.grid(column=1, row=0)
        self.button_basic = tk.Button(self, text="Execute", command=self._command_download_basic)
        self.button_basic.grid(column=2, row=0)

        self.state_label_tfrecods = tk.Label(self, text="(Ready)", width='14', fg="white")
        self.state_label_tfrecods.grid(column=0, row=1)
        label_basic = tk.Label(self, text="Prepare for run training")
        label_basic.grid(column=1, row=1)
        self.button_tfrecords = tk.Button(self, text="Execute", command=self._command_build_tfrecords)
        self.button_tfrecords.grid(column=2, row=1)

        spacing = tk.Label(self, width=5)
        spacing.grid(column=4, row=0)
        label = tk.Label(self, text="Number of Images:")
        label.grid(column=5, row=0, sticky='W')
        self.lbl_image_count = tk.Label(self, text="0")
        self.lbl_image_count.grid(column=6, row=0)

        label = tk.Label(self, text="Number of Labels:")
        label.grid(column=5, row=1, sticky="W")
        self.lbl_label_count = tk.Label(self, text="0")
        self.lbl_label_count.grid(column=6, row=1)

        self._set_states()
        self._update_dataset_meta()

    def _set_states(self):
        state = self._check_setup_state()
        if state == SetupState.NONE:
            self.state_label_basic.configure(text="(Required)", bg="red", fg="white")
            self.state_label_tfrecods.configure(text="(Required)", bg="red", fg="white")
            self.button_tfrecords.configure(state="disabled")
        if state == SetupState.DATA_FETCHED:
            self.state_label_basic.configure(text="(Ready)", bg="green", fg="white")
            self.state_label_tfrecods.configure(text="(Required)", bg="red", fg="white")
            self.button_tfrecords.configure(state="disabled")
        if state == SetupState.READY_FOR_TRAIN:
            self.state_label_basic.configure(text="(Ready)", bg="green", fg="white")
            self.state_label_tfrecods.configure(text="(Ready)", bg="green", fg="white")

    def _update_dataset_meta(self):
        if self._check_setup_state() != SetupState.NONE:
            with sqlite3.connect(config.DATABASE['filename']) as conn:
                c = conn.cursor()
                result = c.execute("SELECT count(*) FROM Images")
                self.lbl_image_count.configure(text='{}'.format(result.fetchone()[0]))
                result = c.execute("SELECT count(*) FROM Labels")
                self.lbl_label_count.configure(text='{}'.format(result.fetchone()[0]))

    def _command_download_basic(self):
        result = tk.messagebox.askyesno("Download Dataset", "Do you want to download the dataset?"
                                                            "\n- Needs ~16 GB disk space.\n- Removes "
                                                            "content of folder ./data/Images ")
        if result:
            self.state_label_basic.configure(text="(In Process)", bg="yellow", fg="black")
            self.button_basic.configure(state='disabled')
            t = threading.Thread(target=self._run_download_basic)
            t.start()
        return

    def _run_download_basic(self):
        setup.setup()
        self._set_states()
        self.button_basic.configure(state='normal')
        self._update_dataset_meta()

    def _command_build_tfrecords(self):
        result = tk.messagebox.askyesno("Prepare for Run", "Do you want prepare for run?\n"
                                                           "- Builds .tfrecords files.\n"
                                                           "- Needs ~24 GB disk space.\n"
                                                           "- Removes content of folder ./data/ImagesRaw ")
        if result:
            self.state_label_tfrecods.configure(text="(In Process)", bg="yellow", fg="black")
            self.button_tfrecords.configure(state='disabled')
            t = threading.Thread(target=self._run_build_tfrecords)
            t.start()
        return

    def _run_build_tfrecords(self):
        data.preprocess_all_sets()
        self._set_states()
        self.button_tfrecords.configure(state='normal')

    def _check_setup_state(self):
        state = SetupState.NONE
        if os.path.exists(config.DATA_DIRECTORY + '/Images'):
            state = SetupState.DATA_FETCHED
        if os.path.exists(config.DATA_DIRECTORY + '/ImagesRaw/train.tfrecords') and \
                os.path.exists(config.DATA_DIRECTORY + '/ImagesRaw/validation.tfrecords') and \
                os.path.exists(config.DATA_DIRECTORY + '/ImagesRaw/test.tfrecords'):
            state = SetupState.READY_FOR_TRAIN
            self.controller.set_runstate(True)
        return state


def on_closing():
    if messagebox.askokcancel("Quit", "Do you want to quit?"):
        root.destroy()


root = None


def main():
    global root
    root = tk.Tk()
    root.title("Open Images Classifier")
    root.protocol("WM_DELETE_WINDOW", on_closing)
    app = Application(master=root)
    app.pack()
    app.mainloop()
    os._exit(0)


if __name__ == '__main__':
    main()
