import xml.etree.ElementTree as ET
import shutil
import numpy as np
import os
import gzip


def open_xml(path):
    # Open xml and xml.gz files into ElementTree
    if path.endswith('.gz'):
        return ET.parse(gzip.open(path))
    else:
        return ET.parse(path)


def insert_DTD(xml_file):

    # Insert DTD
    s = '<?xml version=\'1.0\' encoding=\'UTF-8\'?> \n\
    <!DOCTYPE config SYSTEM "http://www.matsim.org/files/dtd/config_v1.dtd">'

    # Needed to check whether DTD is already there
    first_line = '<?xml version=\'1.0\' encoding=\'UTF-8\'?>'

    with open(xml_file, 'r+') as f:
        content = f.read()
        f.seek(0, 0)

        if not f.readline() == first_line:
            f.seek(0, 0)
            f.write(s + '\n' + content)


def close_xml(tree, name):
    """
    Close xml and insert MATsim DTD because ET.write removes is.
    :param tree: xml we intend to close.
    :param name: filename to save the xml after closing it.
    """
    tree.write(name)
    insert_DTD(name)


def clean_output_folder(output_dir, files_to_keep=[]):
    """
    Removes all undesired files from MATsim output directory
    :param files_to_keep: List of files that should be kept
    :param output_dir: Matsim output directory that we want to clean
    """
    files_to_keep.append('output_config.xml.gz')
    files_to_keep.append('output_events.xml.gz')
    files_to_keep.append('.DS_Store')
    dir_content = os.listdir(output_dir)

    # Safety check
    common_output_items = ['scorestats.txt',
                           'stopwatch.png',
                           'output_plans.xml.gz']
    safety_list = [el for el in dir_content if el in common_output_items]

    if len(safety_list) == len(common_output_items):
        for item in [el for el in dir_content if el not in files_to_keep]:
            full_path = os.path.join(output_dir, item)
            if os.path.isfile(full_path):
                os.remove(full_path)
            if os.path.isdir(full_path):
                shutil.rmtree(full_path)
    else:
        print("Error while cleaning output directory: {} does not seem "
              "to be a MATsim output directory".format(output_dir))


def daily_mode_share(events):

    # Initialize
    shares = {'car': 0,
              'pt': 0,
              'walk': 0}

    # Open events
    tree = open_xml(events)
    root = tree.getroot()

    # Count
    for ev in root.findall("./event[@type='departure']"):
        if ev.attrib['legMode'] in ['car']:
            shares['car'] += 1
        elif ev.attrib['legMode'] in ['transit_walk', 'walk']:
            shares['walk'] += 1
        elif ev.attrib['legMode'] in ['pt']:
            shares['pt'] += 1
        else:
            print('\n{} is not counted in mode shares!!'.format(
                ev.attrib['legMode']))
    return shares

def hourly_mode_share(events):

    # Initialize
    shares = {}
    for i in range(0,30):
        shares['car{}'.format(str(i))] = 0
        shares['pt{}'.format(str(i))] = 0
        shares['walk{}'.format(str(i))] = 0

    # Open events
    tree = open_xml(events)
    root = tree.getroot()

    # Count
    for ev in root.findall("./event[@type='departure']"):
        hour = str(int(float(ev.attrib['time'])/3600))
        if ev.attrib['legMode'] in ['car']:
            shares['car'+hour] += 1
        elif ev.attrib['legMode'] in ['transit_walk', 'walk']:
            shares['walk'+hour] += 1
        elif ev.attrib['legMode'] in ['pt']:
            shares['pt'+hour] += 1
        else:
            print('\n{} is not counted in mode shares!!'.format(
                ev.attrib['legMode']))
    return shares

def dic2vec(dic, map):
    vec = np.empty(len(dic))
    for key, value in dic.items():
        vec[map[key]] = value
    return vec

