import glob
import os
import configparser

def select_and_exec_workflow(basedir='.', ext='py', verbose=True):
    """
    Reads a configuration file and returns an instance of ConfigParser:
    First, looks for files in *basedir* with extension *ext*.
    Asks user to select a file if several files are found,
    and parses it using ConfigParser module.
    @rtype: L{ConfigParser.ConfigParser}
    """
    work_flow_files = glob.glob(os.path.join(basedir, u'*.{}'.format(ext)))


    if not work_flow_files:
        raise Exception("No configuration file found!")

    if len(work_flow_files) == 1:
        # only one configuration file
        wf_file = work_flow_files[0]
    else:
        print("Select a configuration file:")
        for i, f in enumerate(work_flow_files, start=1):
            print("{} - {}".format(i, f))
        res = int(input(''))
        wf_file = work_flow_files[res - 1]

    if verbose:
        print("Reading workflow file: {}".format(wf_file))

    conf = configparser.ConfigParser(allow_no_value=True)
    conf.read(wf_file)

    return conf

select_and_parse_config_file()