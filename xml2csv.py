import os
import glob
import pandas as pd
import xml.etree.ElementTree as ET

dir_name = 'default_rotate_light'

def xml_to_csv(path):
    xml_list = []
    for xml_file in glob.glob(path + '/*.xml'):
        print(xml_file)
        tree = ET.parse(xml_file)
        root = tree.getroot()
        for member in root.findall('object'):
            value = (root.find('filename').text,
                     int(root.find('size')[0].text),
                     int(root.find('size')[1].text),
                     member[0].text,
                     int(member[4][0].text),
                     int(member[4][1].text),
                     int(member[4][2].text),
                     int(member[4][3].text)
                     )
            xml_list.append(value)
    column_name = ['filename', 'width', 'height', 'class', 'xmin', 'ymin', 'xmax', 'ymax']
    xml_df = pd.DataFrame(xml_list, columns=column_name)
    return xml_df


def main():
    try:
        if not os.path.exists('./data/' + dir_name + '/'):
            os.makedirs('./data/' + dir_name + '/')
    except OSError:
        print ('Error: Creating directory. ' + './data/' + dir_name + '/')

    for directory in ['train', 'test']:
        image_path = os.path.join(os.getcwd(), 'images/' + dir_name + '/{}'.format(directory))
        xml_df = xml_to_csv(image_path)
        xml_df.to_csv('data/' + dir_name + '/{}_labels.csv'.format(directory), index=None)
    print('Successfully converted xml to csv.')


if __name__ == '__main__':
    main()