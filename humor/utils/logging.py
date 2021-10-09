
import os, re, datetime, shutil

class Logger(object):
    '''
    "Static" class to handle logging.
    '''
    log_file = None

    @staticmethod
    def init(log_path):
        Logger.log_file = log_path

    @staticmethod
    def log(write_str):
        print(write_str)
        if not Logger.log_file:
            print('Logger must be initialized before logging!')
            return
        time_str = datetime.datetime.now().strftime("%Y-%m-%d_%H:%M:%S")
        with open(Logger.log_file, 'a') as f:
            f.write(time_str + '  ')
            f.write(str(write_str) + '\n')

def class_name_to_file_name(class_name):
    '''
    Converts class name to the containing file name.
    Assumes class name is in CamelCase and file name is the same
    name but in snake_case.
    Can be used for models and datasets.
    '''
    toks = re.findall('[A-Z][^A-Z]*', class_name)
    toks = [tok.lower() for tok in toks]
    file_name = '_'.join(toks)
    return file_name

def mkdir(dir_name):
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)

def cp_files(dir_out, file_list):
    ''' copies a list of files '''
    if not os.path.exists(dir_out):
        print('Cannot copy to nonexistent directory ' + dir_out)
        return
    for f in file_list:
        shutil.copy(f, dir_out)