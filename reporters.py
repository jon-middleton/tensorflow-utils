"""Generic bot that can be used to broadcast messages to a Microsoft Teams webhook."""

import pymsteams
import subprocess
from numpy import round
from Jon.utils.dictionaries.reporting import BOT_URL_DICT
from tensorflow.keras.callbacks import Callback


class Bot(Callback):
    """Abstract class for a reporting bot. Inherit from this class to report for a given model by
    overriding Bot.set_message and Bot.on_epoch_end.
    """
    def __init__(self, bot_name="jon"):
        """Initializes a bot.

        Args:
            bot_name (str):
        """
        super().__init__()
        if bot_name.lower() not in BOT_URL_DICT.keys():
            raise Exception(f"Invalid bot name. Bot name must be one of {BOT_URL_DICT.keys()}.")
        else:
            self.name_for_bot = bot_name.lower()

        self.url = BOT_URL_DICT[self.name_for_bot]
        self.message = self.set_message()

    @property
    def server_name(self):
        """Grabs the server's name from Linux stdout, formats this name, and returns it.

        Returns:
            str
        """
        server_name = subprocess.check_output(["uname", "-n"])
        return str(server_name)[2:-3]

    def set_message(self):
        """Sets the first line of the report message of the bot.

        Returns:
            str
        """
        message = f"\nServer: {self.server_name}\n"
        if self.model is not None:
            return "".join([message, f"\nModel: {self.model.name}\n"])
        else:
            return message

    def on_epoch_end(self, epoch, logs=None):
        """Grabs the learning rate, training metric values, and validation metric values, formats
        them, joins them to self.message, and pushes the message to Microsoft Teams via self.send.

        Args:
            epoch (int): index of epoch.
            logs (dict): metric results for this training epoch, and for the validation epoch if
            validation is performed. Validation result keys are prefixed with `val_`.

        Returns:
            None
        """
        learning_rate = round(float(self.model.optimizer.lr), 4)
        self.message = "".join([self.message, f"\nEpoch: {epoch + 1}\n"])
        self.message = "".join([self.message, f"\nLearning rate: {learning_rate}\n"])

        metric_dict = self.model.history.history
        for metric_name in metric_dict:
            metric_value = round(metric_dict[metric_name][-1], 4)
            new_string = f"\n{metric_name}: {metric_value}\n"
            self.message = "".join([self.message, new_string])

        self.send()

    def send(self):
        """Uses pymsteams to push self.message to Microsoft Teams, then resets self.message.

        Returns:
            None
        """
        connector_card = pymsteams.connectorcard(self.url)
        connector_card.text(self.message)
        connector_card.send()

        self.message = self.set_message()
