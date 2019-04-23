import os
import configparser


SETTINGS_FILENAME = 'settings.cfg'

VIEW_SECTION = 'view'
VIEW_HQ_OPTION = 'high_quality'
LIVE_IMAGE_PREVIEW_OPTION = 'live_preview'
PREVIEW_ON_TOP_OPTION = 'preview_on_top'
STYLE = 'style'

ELEMENTS_SECTION = 'elements'
EXPERIMENTAL_ELEMENTS ='experimental'

UPDATES_SECTION = "updates"
UPDATE_DONT_REMIND_VERSION = "dont_remind_version"

DEFAULTS = {
    VIEW_SECTION: {
        VIEW_HQ_OPTION: 'False',
        LIVE_IMAGE_PREVIEW_OPTION: 'True',
        PREVIEW_ON_TOP_OPTION: 'True',
        STYLE: 'default',
    },
    ELEMENTS_SECTION: {
        EXPERIMENTAL_ELEMENTS: 'False',
    },
    UPDATES_SECTION: {
        UPDATE_DONT_REMIND_VERSION: "0"
    },
}


class ConfigWrapper(configparser.SafeConfigParser):

    cfg = None

    def get_with_default(self, section, name):
        value = self.get(section, name)
        if value is not None:
            return value
        return DEFAULTS[section][name]

    def get(self, section, option, **kwargs):
        try:
            value = configparser.SafeConfigParser.get(self, section, option, **kwargs)
        except (configparser.NoOptionError, configparser.NoSectionError):
            value = None
        return value

    def set(self, section, option, value=None):
        if section not in self.sections():
            self.add_section(section)
        configparser.SafeConfigParser.set(self, section, option, str(value))
        self._save_settings()

    def remove_option(self, section, option):
        configparser.SafeConfigParser.remove_option(self, section, option)
        self._save_settings()

    def _save_settings(self):
        # Create directory if not exists
        settings_path = self.get_settings_path()
        settings_dir = os.path.dirname(settings_path)
        if not os.path.exists(settings_dir):
            os.makedirs(settings_dir)
        # Save settings
        with open(settings_path, 'w') as configfile:
            self.write(configfile)

    @classmethod
    def get_settings(cls):
        if ConfigWrapper.cfg is None:
            ConfigWrapper.cfg = ConfigWrapper()
            ConfigWrapper.cfg.read(cls.get_settings_path())
        return ConfigWrapper.cfg

    @classmethod
    def get_settings_path(cls):
        # For Windows
        if os.name == 'nt':
            return os.path.join(os.environ['appdata'], 'CVLab', SETTINGS_FILENAME)
        # For Unix
        else:
            return os.path.expanduser(os.path.join('~', '.cvlab', SETTINGS_FILENAME))
