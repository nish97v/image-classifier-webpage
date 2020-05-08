from os import getcwd


class Config(object):
    DEBUG = True
    DEVELOPMENT = True
    SECRET_KEY = 'ky261MEQChMSvkCYKXbCBg'
    FLASK_SECRET = SECRET_KEY
    UPLOAD_FOLDER = getcwd()
    ALLOWED_EXTENSIONS = {'png', 'bmp', 'jpg', 'jpeg', 'gif'}
    IMAGE_LABELS = ['cat', 'dog']
    # FLASK_HTPASSWD_PATH = '/secret/.htpasswd'
    # DB_HOST = 'database' # a docker link


class ProductionConfig(Config):
    DEVELOPMENT = False
    DEBUG = False
    # DB_HOST = 'my.production.database' # not a docker link